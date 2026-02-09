from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from typing import Any, Optional, cast

import hydra
import mlflow
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader

from analysis import generate_2d_plots, create_loss_plots_from_mlflow
from data import TrajectoryDataset, TrajectoryGenerator, make_collate_fn, get_dataset_name
from schedulers import ConvergenceChecker, GECO


def train_epoch(
    model: nn.Module,
    data_loader: Iterator[tuple[torch.Tensor, torch.Tensor]],
    loss_fn,
    loss_cfg: DictConfig,
    optimizer: torch.optim.Optimizer,
    geco_pos: GECO,
    geco_norm: GECO,
    iterations: int
) -> tuple[float, dict[str, float]]:
    """Train for one epoch.

    Args:
        model: Model to train.
        train_loader: Training data loader.
        loss_fn: Loss function taking (model_output, batch) and returning
            (total_loss, components_dict).
        optimizer: Optimizer.
        device: Device to train on.

    Returns:
        Tuple of (average_loss, average_components).
    """
    model.train()
    total_loss = 0.0
    component_sums: dict[str, float] = {}

    for _ in range(iterations):
        batch, shift_batch = next(data_loader)

        optimizer.zero_grad()
        z, _ = model(batch, norm=False)
        z_shift, _ = model(shift_batch, norm=False)
        loss, components = loss_fn(
            z,
            z_shift,
            batch,
            geco_pos.lambda_val,
            geco_norm.lambda_val,
            loss_cfg
        )
        loss.backward()
        optimizer.step()

        geco_pos.step(components["positivity"])
        geco_norm.step(components["norm"])

        total_loss += loss.item()
        for k, v in components.items():
            component_sums[k] = component_sums.get(k, 0.0) + v

    avg_components = {k: v / iterations for k, v in component_sums.items()}
    return total_loss / iterations, avg_components


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: Iterator[tuple[torch.Tensor, torch.Tensor]],
    loss_fn,
    loss_cfg: DictConfig,
    geco_pos: GECO,
    geco_norm: GECO,
    iterations: int
) -> tuple[float, dict[str, float]]:
    """Evaluate model on validation set.

    Args:
        model: Model to evaluate.
        val_loader: Validation data loader.
        loss_fn: Loss function taking (model_output, batch) and returning
            (total_loss, components_dict).
        device: Device to evaluate on.

    Returns:
        Tuple of (average_loss, average_components).
    """
    model.eval()
    total_loss = 0.0
    component_sums: dict[str, float] = {}

    for _ in range(iterations):
        batch, shift_batch = next(data_loader)

        z, _ = model(batch, norm=False)
        z_shift, _ = model(shift_batch, norm=False)
        loss, components = loss_fn(
            z,
            z_shift,
            batch,
            geco_pos.lambda_val,
            geco_norm.lambda_val,
            loss_cfg
        )

        total_loss += loss.item()
        for k, v in components.items():
            component_sums[k] = component_sums.get(k, 0.0) + v

    avg_components = {k: v / iterations for k, v in component_sums.items()}
    return total_loss / iterations, avg_components


def train(cfg: DictConfig, continue_id: Optional[str] = None) -> None:
    """Run training with the given config. Assumes MLflow run is already active."""
    mlflow.log_dict(cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True)), "config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = hydra.utils.instantiate(cfg.model).to(device)
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    geco_pos = hydra.utils.instantiate(cfg.regularization.positivity)
    geco_norm = hydra.utils.instantiate(cfg.regularization.norm)
    loss_fn = hydra.utils.get_method(cfg.loss._target_)

    checker = None
    if cfg.convergence is not None and cfg.convergence.enabled:
        checker = ConvergenceChecker(
            window=cfg.convergence.window,
            patience=cfg.convergence.patience,
            threshold=cfg.convergence.threshold,
            smoothing=cfg.convergence.smoothing,
        )

    for k in range(cfg.training.K):
        optimizer.state.clear()
        geco_pos.reset(cfg.regularization.positivity.lambda_init)
        geco_norm.reset(cfg.regularization.norm.lambda_init)

        if cfg.training.vary_data:
            train_path = os.path.join(cfg.data.save_path, get_dataset_name(cfg.data), "train", f"k_{k}")
        else:
            train_path = os.path.join(cfg.data.save_path, get_dataset_name(cfg.data), "train", "k_0")
        if not os.path.exists(train_path):
            generator = TrajectoryGenerator()
            generator.generate_dataset(
                train_path,
                num_sequences=5000000//cfg.data.seq_len,
                sequence_length=cfg.data.seq_len,
                box_width=cfg.data.box_width,
                box_height=cfg.data.box_height,
                n_shift=cfg.data.n_shift,
                sigma_shift=cfg.data.sigma_shift,
            )
            
        val_path = os.path.join(cfg.data.save_path, get_dataset_name(cfg.data), "val")
        if not os.path.exists(val_path):
            generator = TrajectoryGenerator()
            generator.generate_dataset(
                val_path,
                num_sequences=500000//cfg.data.seq_len,
                sequence_length=cfg.data.seq_len,
                box_width=cfg.data.box_width,
                box_height=cfg.data.box_height,
                n_shift=cfg.data.n_shift,
                sigma_shift=cfg.data.sigma_shift,
            )

        train_dataset = TrajectoryDataset(train_path)
        train_loader = iter(DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=False,
                                    collate_fn=make_collate_fn(device)))
        val_dataset = TrajectoryDataset(val_path)
        val_loader = iter(DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=True,
                                    collate_fn=make_collate_fn(device)))
        
        start_epoch = 0
        if continue_id:
            model, train_loader, val_loader, geco_pos, geco_norm, start_epoch = continue_train(continue_id, cfg, k, model, train_loader, val_loader, geco_pos, geco_norm)

        if start_epoch >= cfg.training.epochs:
            print(f"[k={k}] Already completed {start_epoch} epochs, skipping")
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            epoch_pbar = tqdm(range(start_epoch + 1, cfg.training.epochs + 1), desc=f"[k={k}] Training")
            for epoch in epoch_pbar:
                train_loss, loss_components = train_epoch(
                    model,
                    train_loader,
                    loss_fn,
                    cfg.loss,
                    optimizer,
                    geco_pos,
                    geco_norm,
                    cfg.training.train_iterations,
                )

                val_loss, _ = evaluate(
                    model,
                    val_loader,
                    loss_fn,
                    cfg.loss,
                    geco_pos,
                    geco_norm,
                    cfg.training.val_iterations,
                )

                metrics = {f"k{k}/train_loss": train_loss, f"k{k}/val_loss": val_loss}
                metrics.update({f"k{k}/{name}": val for name, val in loss_components.items()})
                metrics[f"k{k}/lambda_pos"] = geco_pos.lambda_val
                metrics[f"k{k}/lambda_norm"] = geco_norm.lambda_val
                metrics[f"k{k}/positivity_geco"] = geco_pos.L
                metrics[f"k{k}/norm_geco"] = geco_norm.L
                mlflow.log_metrics(metrics, step=epoch)
                epoch_pbar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")

                if checker is not None:
                    if checker.step(**loss_components):
                        tqdm.write(f"[k={k}] Converged at epoch {epoch}")
                        tqdm.write(f"[k={k}] Final smoothed: {checker.current_smoothed}")
                        break

                checkpoint_path = os.path.join(tmpdir, f"checkpoint_k_{k}_epoch_{epoch}.pt")
                torch.save({
                    "k": k,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }, checkpoint_path)
                mlflow.log_artifact(checkpoint_path, artifact_path=f"checkpoints/k{k}")

        state_dict_path = os.path.join(tempfile.gettempdir(), f"model_k{k}_state_dict.pt")
        torch.save(model.state_dict(), state_dict_path)
        mlflow.log_artifact(state_dict_path, artifact_path="models")

        generate_2d_plots(model, k=k)
        create_loss_plots_from_mlflow(k=k)

def continue_train(
    run_id: str,
    cfg: DictConfig,
    k: int,
    model: nn.Module,
    train_loader: Iterator,
    val_loader: Iterator,
    geco_pos: GECO,
    geco_norm: GECO
) -> tuple[nn.Module, Iterator, Iterator, GECO, GECO, int]:
    """Continue training from a previous MLflow run.

    Args:
        run_id: The MLflow run ID to continue from.
        cfg: Current config. Must match the saved config except for training.epochs and training.K.
        k: The k iteration to continue.
        model: The model to load weights into.
        train_loader: Training data loader iterator.
        val_loader: Validation data loader iterator.
        geco_pos: GECO scheduler for positivity regularization.
        geco_norm: GECO scheduler for norm regularization.

    Returns:
        Tuple of (model, train_loader, val_loader, geco_pos, geco_norm, last_epoch)
        with restored state from the checkpoint. If run doesn't exist or
        no checkpoints are available, returns inputs unchanged with last_epoch=0.

    Raises:
        TypeError: If loaded config is not a DictConfig.
        ValueError: If critical config sections don't match.
    """
    client = MlflowClient()

    # Check if run exists
    try:
        client.get_run(run_id)
    except Exception:
        print(f"Run {run_id} not found, starting fresh")
        return model, train_loader, val_loader, geco_pos, geco_norm, 0

    # Load config from the run
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = client.download_artifacts(run_id, "config.yaml", tmpdir)
            saved_cfg = OmegaConf.load(config_path)
    except Exception:
        print(f"Could not load config from run {run_id}, starting fresh")
        return model, train_loader, val_loader, geco_pos, geco_norm, 0

    if not isinstance(saved_cfg, DictConfig):
        raise TypeError(f"Expected DictConfig, got {type(saved_cfg).__name__}")

    # Validate that critical config sections match (resolve interpolations before comparing)
    critical_sections = ["data", "model", "optimizer", "regularization", "loss"]
    for section in critical_sections:
        current = OmegaConf.to_container(cfg[section], resolve=True)
        saved = OmegaConf.to_container(saved_cfg[section], resolve=True)
        if current != saved:
            raise ValueError(
                f"Config section '{section}' differs from saved run. "
                f"Current: {current}, Saved: {saved}"
            )

    # Find the latest checkpoint for k
    artifacts = client.list_artifacts(run_id, f"checkpoints/k{k}")
    checkpoint_files = [a.path for a in artifacts if a.path.endswith(".pt")]
    if not checkpoint_files:
        print(f"No checkpoints found for k={k} in run {run_id}, starting fresh")
        return model, train_loader, val_loader, geco_pos, geco_norm, 0

    # Sort by epoch number and get the latest
    def get_epoch(path: str) -> int:
        return int(path.split("_epoch_")[1].replace(".pt", ""))

    latest_checkpoint = max(checkpoint_files, key=get_epoch)
    last_epoch = get_epoch(latest_checkpoint)

    # Load the checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = client.download_artifacts(run_id, latest_checkpoint, tmpdir)
        checkpoint = torch.load(checkpoint_path, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    # Forward Dataloaders    
    for _ in range(last_epoch*cfg.training.train_iterations):
        next(train_loader)
    for _ in range(last_epoch*cfg.training.val_iterations):
        next(val_loader)

    # Load lambda metrics and set geco values
    run = client.get_run(run_id)
    metrics = run.data.metrics
    geco_pos.lambda_val = metrics[f"k{k}/lambda_pos"]
    geco_pos.L = metrics[f"k{k}/positivity_geco"]
    geco_norm.lambda_val = metrics[f"k{k}/lambda_norm"]
    geco_norm.L = metrics[f"k{k}/norm_geco"]

    return model, train_loader, val_loader, geco_pos, geco_norm, last_epoch


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("grid-representations")

    continue_id: Optional[str] = cfg.get("continue_id", None)
    with open_dict(cfg):
        cfg.pop("continue_id", None)

    with mlflow.start_run(run_id=continue_id):
        train(cfg, continue_id=continue_id)


if __name__ == "__main__":
    main()
