from __future__ import annotations

import os
import tempfile

import hydra
import mlflow
import mlflow.pytorch
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from data import TrajectoryDataset
from schedulers import ConvergenceChecker, GECO
from losses import global_loss


def train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn,
    loss_cfg: DictConfig,
    optimizer: torch.optim.Optimizer,
    geco_pos: GECO,
    geco_norm: GECO,
    iterations: int,
    epoch: int = 0
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

    for _ in range(epoch*iterations, (epoch+1)*iterations):
        batch, shift_batch = next(iter(data_loader))

        optimizer.zero_grad()
        z, _ = model(batch)
        z_shift, _ = model(shift_batch)
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
    data_loader: DataLoader,
    loss_fn,
    loss_cfg: DictConfig,
    geco_pos: GECO,
    geco_norm: GECO,
    iterations: int,
    epoch: int = 0
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

    for _ in range(epoch*iterations, (epoch+1)*iterations):
        batch, shift_batch = next(iter(data_loader))

        z, _ = model(batch)
        z_shift, _ = model(shift_batch)
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


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn,
    geco_pos: GECO,
    geco_norm: GECO,
    optimizer: torch.optim.Optimizer,
    train_cfg: DictConfig,
    loss_cfg: DictConfig,
    convergence_cfg: DictConfig | None = None,
    k: int = 0,
) -> dict:
    """Full training loop.

    Args:
        model: Model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        loss_fn: Loss function returning (total_loss, loss_dict) where loss_dict
            contains individual loss components for convergence checking.
        optimizer: Optimizer.
        device: Device to train on.
        epochs: Number of epochs.
        checkpoint_interval: Save checkpoint every N epochs (0 to disable).
        convergence_cfg: Convergence config with enabled, window, patience, threshold, smoothing.

    Returns:
        Dictionary with training history.
    """
    history = {"train_loss": [], "val_loss": []}

    # Setup convergence checker
    checker = None
    if convergence_cfg is not None and convergence_cfg.enabled:
        checker = ConvergenceChecker(
            window=convergence_cfg.window,
            patience=convergence_cfg.patience,
            threshold=convergence_cfg.threshold,
            smoothing=convergence_cfg.smoothing,
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        for epoch in range(1, train_cfg.epochs + 1):
            train_loss, loss_components = train_epoch(
                model,
                train_loader,
                loss_fn,
                loss_cfg,
                optimizer,
                geco_pos,
                geco_norm,
                train_cfg.train_iterations,
                epoch
            )

            val_loss, _ = evaluate(
                model,
                val_loader,
                loss_fn,
                loss_cfg,
                geco_pos,
                geco_norm,
                train_cfg.val_iterations,
                epoch
            )

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            metrics = {f"k{k}/train_loss": train_loss, f"k{k}/val_loss": val_loss}
            metrics.update({f"k{k}/{name}": val for name, val in loss_components.items()})
            mlflow.log_metrics(metrics, step=epoch)
            print(f"[k={k}] Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            # Convergence check
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

    return history


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Set MLflow tracking to local directory
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("grid-representations")

    with mlflow.start_run():
        # Log full config as YAML artifact (preserves hierarchy)
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config.yaml")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Instantiate model from config
        model = hydra.utils.instantiate(cfg.model).to(device)

        # Instantiate optimizer (pass params explicitly since not in config)
        optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
        geco_pos = hydra.utils.instantiate(cfg.regularization.positivity)
        geco_norm = hydra.utils.instantiate(cfg.regularization.norm)

        for k in range(cfg.training.K):
            optimizer.state.clear()
            geco_pos.reset(cfg.regularization.positivity.lambda_init)
            geco_norm.reset(cfg.regularization.norm.lambda_init)
            
            if cfg.training.vary_seed:
                model.reset_parameters(seed=k)
            else:
                model.reset_parameters(seed=42)

            if cfg.training.vary_data:
                train_path = f"data/train/dim_{cfg.data.dim}_box_{cfg.data.box_width}x{cfg.data.box_height}_seq_len_{cfg.data.seq_len}_k_{k}"
            else:
                train_path = f"data/train/dim_{cfg.data.dim}_box_{cfg.data.box_width}x{cfg.data.box_height}_seq_len_{cfg.data.seq_len}"
            if not os.path.exists(train_path):
                print(f"Generating dataset...")
                # TODO implement generator
            
            val_path = f"data/val/dim_{cfg.data.dim}_box_{cfg.data.box_width}x{cfg.data.box_height}_seq_len_{cfg.data.seq_len}"
            if not os.path.exists(val_path):
                print(f"Generating validation dataset...")
                # TODO implement generator

            train_dataset = TrajectoryDataset(train_path)
            train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=False,
                                      collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))
            val_dataset = TrajectoryDataset(val_path)
            val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=True,
                                      collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))

            history = train(
                model,
                train_loader,
                val_loader,
                global_loss,
                geco_pos,
                geco_norm,
                optimizer,
                cfg.training,
                cfg.loss,
                convergence_cfg=cfg.convergence,
                k=k,
            )

            mlflow.pytorch.log_model(model, f"{k}/model")


if __name__ == "__main__":
    main()
