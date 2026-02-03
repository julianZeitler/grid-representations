from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator

import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from analysis import generate_2d_plots, create_loss_plots_from_mlflow
from data import TrajectoryDataset, TrajectoryGenerator, make_collate_fn
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


def train(cfg: DictConfig) -> None:
    """Run training with the given config. Assumes MLflow run is already active."""
    mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config.yaml")

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

        if cfg.training.vary_seed:
            model.reset_parameters(seed=k)
        else:
            model.reset_parameters(seed=42)

        if cfg.training.vary_data:
            train_path = f"data/train/dim_{cfg.data.dim}_box_{cfg.data.box_width}x{cfg.data.box_height}_seq_len_{cfg.data.seq_len}_n_shift_{cfg.data.n_shift}_sigma_shift_{cfg.data.sigma_shift}_k_{k}"
        else:
            train_path = f"data/train/dim_{cfg.data.dim}_box_{cfg.data.box_width}x{cfg.data.box_height}_seq_len_{cfg.data.seq_len}_n_shift_{cfg.data.n_shift}_sigma_shift_{cfg.data.sigma_shift}_k_0"
        if not os.path.exists(train_path):
            generator = TrajectoryGenerator()
            generator.generate_dataset(
                train_path,
                num_sequences=100000,
                sequence_length=cfg.data.seq_len,
                box_width=cfg.data.box_width,
                box_height=cfg.data.box_height,
                n_shift=cfg.data.n_shift,
                sigma_shift=cfg.data.sigma_shift,
            )

        val_path = f"data/val/dim_{cfg.data.dim}_box_{cfg.data.box_width}x{cfg.data.box_height}_seq_len_{cfg.data.seq_len}_n_shift_{cfg.data.n_shift}_sigma_shift_{cfg.data.sigma_shift}"
        if not os.path.exists(val_path):
            generator = TrajectoryGenerator()
            generator.generate_dataset(
                val_path,
                num_sequences=10000,
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

        with tempfile.TemporaryDirectory() as tmpdir:
            for epoch in range(1, cfg.training.epochs + 1):
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
                print(f"[k={k}] Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

                if checker is not None:
                    if checker.step(**loss_components):
                        print(f"[k={k}] Converged at epoch {epoch}")
                        print(f"[k={k}] Final smoothed: {checker.current_smoothed}")
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


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("grid-representations")

    with mlflow.start_run():
        train(cfg)


if __name__ == "__main__":
    main()
