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

from schedulers import ConvergenceChecker


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
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

    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        output = model(batch)
        loss, components = loss_fn(output, batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        for k, v in components.items():
            component_sums[k] = component_sums.get(k, 0.0) + v

    n = len(train_loader)
    avg_components = {k: v / n for k, v in component_sums.items()}
    return total_loss / n, avg_components


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn,
    device: torch.device,
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

    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(batch)
        loss, components = loss_fn(output, batch)
        total_loss += loss.item()
        for k, v in components.items():
            component_sums[k] = component_sums.get(k, 0.0) + v

    n = len(val_loader)
    avg_components = {k: v / n for k, v in component_sums.items()}
    return total_loss / n, avg_components


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    checkpoint_interval: int = 0,
    convergence_cfg: DictConfig | None = None,
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
        for epoch in range(1, epochs + 1):
            train_loss, loss_components = train_epoch(model, train_loader, loss_fn, optimizer, device)
            val_loss, _ = evaluate(model, val_loader, loss_fn, device)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss, **loss_components}, step=epoch)
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            # Convergence check
            if checker is not None:
                if checker.step(**loss_components):
                    print(f"Converged at epoch {epoch}")
                    print(f"Final smoothed: {checker.current_smoothed}")
                    break

            if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
                checkpoint_path = os.path.join(tmpdir, f"checkpoint_epoch_{epoch}.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }, checkpoint_path)
                mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")

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

        # TODO: Create DataLoaders and loss_fn (returning (loss, components_dict)), then call:
        # history = train(
        #     model, train_loader, val_loader, loss_fn, optimizer, device,
        #     epochs=cfg.training.iterations,
        #     checkpoint_interval=cfg.training.checkpoint_iters,
        #     convergence_cfg=cfg.convergence,
        # )

        # Log model
        mlflow.pytorch.log_model(model, "model")


if __name__ == "__main__":
    main()
