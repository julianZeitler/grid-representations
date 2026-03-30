"""
Energy-Based CAN with Local Volume Regularization using ActionableRGM encoder.

A denoising map g_θ: R^D → R^D is trained in RGM latent space with objective:

    L = (1/N) Σ_n ||z^(n) - g_θ(z^(n))||²  -  (λ/N) Σ_n log|det(I - J_{g_θ}(z^(n)))|

where:
  - z^(n) = z(x^(n)) are L2-normalised RGM latent representations of 2D positions x^(n)
  - J_{g_θ}(z) is the Jacobian of g_θ at z
  - λ controls the volume regularisation strength

The first term pulls energy down at visited states (makes them near-fixed-points of g_θ).
The second term penalises large basin volume via the local quadratic approximation:

    log Z_n ≈ const − log|det(I − J_{g_θ}(z^(n)))|

so minimising −log|det(I−J)| tightens the basin around each visited state.

Dynamics: z_{t+1} = g_θ(z_t), converges to fixed-point set M = {z : g_θ(z) = z}.
Energy: E(z) = ||z - g_θ(z)||², zero on M and positive elsewhere.
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch.func import vmap, jacrev

from rgm import ActionableRGM, PositionDecoder


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRACKING_URI = f"sqlite:///{os.path.join(_PROJECT_ROOT, 'mlruns.db')}"


# ============================================================================
# 1. Model loading  (same pattern as other can_*_rgm experiments)
# ============================================================================

def load_model(exp_id: str, k: int) -> ActionableRGM:
    """Load ActionableRGM weights from an MLflow run at iteration k."""
    mlflow.set_tracking_uri(_TRACKING_URI)
    client = MlflowClient()

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = client.download_artifacts(exp_id, "config.yaml", tmpdir)
        run_cfg = OmegaConf.load(config_path)

    model = ActionableRGM(
        input_size=run_cfg.model.input_size,
        latent_size=run_cfg.model.latent_size,
        om_init_scale=run_cfg.model.om_init_scale,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        state_dict_path = client.download_artifacts(
            exp_id, f"models/model_k{k}_state_dict.pt", tmpdir
        )
        state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

    model.eval()
    # torch.compile fails when the project path contains spaces ("Pilot Decoder").
    model._scan_linear_transforms = model._scan_linear_transforms_impl
    return model


# ============================================================================
# 2. Latent representations
# ============================================================================

def get_z(model: ActionableRGM, x: torch.Tensor) -> torch.Tensor:
    """Return L2-normalised latent z for 2D position x (shape [2])."""
    x_input = x.unsqueeze(0).unsqueeze(0)  # [1, 1, 2]
    _, z = model(x_input, norm=False)
    z = z.squeeze(0)  # [D]
    return z / (torch.linalg.norm(z) + 1e-5)


def get_z_batch(model: ActionableRGM, xs: torch.Tensor) -> torch.Tensor:
    """Return L2-normalised latents for a batch of 2D positions, shape [B, D]."""
    x_input = xs.unsqueeze(1)  # [B, 1, 2]
    _, z = model(x_input, norm=False)  # z: [B, D]
    norms = torch.linalg.norm(z, dim=1, keepdim=True)
    return z / (norms + 1e-5)


# ============================================================================
# 3. Denoising network g_θ
# ============================================================================

class DenoisingMLP(nn.Module):
    """Learned denoising map g_θ: R^D → R^D."""

    def __init__(self, dim: int, hidden_dim: int, n_layers: int):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ============================================================================
# 4. Collect visited latents from the 2D arena
# ============================================================================

def sample_visited_latents(
    model: ActionableRGM,
    n_samples: int,
    box_width: float,
    box_height: float,
    seed: int = 0,
) -> torch.Tensor:
    """
    Sample 2D positions uniformly from the arena and encode with RGM.

    Returns:
        zs: shape [n_samples, D], L2-normalised latent vectors
    """
    rng = np.random.default_rng(seed)
    xs_np = np.stack(
        [rng.uniform(0.0, box_width, size=n_samples),
         rng.uniform(0.0, box_height, size=n_samples)],
        axis=1,
    ).astype(np.float32)
    xs = torch.from_numpy(xs_np)

    with torch.no_grad():
        zs = get_z_batch(model, xs)  # [N, D]

    return zs


# ============================================================================
# 5. Loss: energy + volume regularisation
# ============================================================================

def compute_loss(
    g: DenoisingMLP,
    z_batch: torch.Tensor,
    lam: float,
) -> tuple[torch.Tensor, dict]:
    """
    Compute the energy-volume loss on a mini-batch of visited latents:

        L = ||z - g(z)||²  -  λ · log|det(I - J_g(z))|

    The Jacobian J_g(z) ∈ R^{D×D} is computed via vmap + jacrev, which is
    differentiable w.r.t. the parameters of g (enabling the required
    second-order gradient through the Jacobian).

    Args:
        g: denoising network
        z_batch: [B, D] batch of visited latents
        lam: volume regularisation coefficient

    Returns:
        loss: scalar tensor with grad_fn
        metrics: dict of float scalars for logging
    """
    D = z_batch.shape[1]

    # --- Energy term: ||z - g(z)||² ---
    g_z = g(z_batch)  # [B, D]
    energy = ((z_batch - g_z) ** 2).sum(dim=1).mean()

    # --- Volume term: mean log|det(I - J_g(z))| ---
    # vmap maps jacrev over the batch dimension, yielding [B, D, D]
    J_batch = vmap(jacrev(g))(z_batch)  # [B, D, D]
    I = torch.eye(D, device=z_batch.device).unsqueeze(0)  # [1, D, D]
    I_minus_J = I - J_batch  # [B, D, D]
    log_det = torch.linalg.slogdet(I_minus_J).logabsdet  # [B]
    volume_term = log_det.mean()

    loss = energy - lam * volume_term

    return loss, {
        "loss": loss.item(),
        "energy": energy.item(),
        "volume_term": volume_term.item(),
        "log_det": log_det.mean().item(),
    }


# ============================================================================
# 6. Training loop
# ============================================================================

def train(
    g: DenoisingMLP,
    zs: torch.Tensor,
    cfg: DictConfig,
) -> list[dict]:
    """Train the denoising map g_θ. Returns per-epoch metric history."""
    optimizer = torch.optim.Adam(g.parameters(), lr=cfg.train.lr)

    N = len(zs)
    batch_size = cfg.train.batch_size
    n_epochs = cfg.train.n_epochs
    lam = cfg.train.lam
    grad_clip = cfg.train.grad_clip
    log_interval = cfg.train.log_interval

    history: list[dict] = []

    for epoch in range(n_epochs):
        idx = torch.randperm(N)
        agg: dict[str, float] = {k: 0.0 for k in ("loss", "energy", "volume_term", "log_det")}
        n_batches = 0

        for start in range(0, N, batch_size):
            batch = zs[idx[start : start + batch_size]]
            optimizer.zero_grad()
            loss, metrics = compute_loss(g, batch, lam)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(g.parameters(), grad_clip)
            optimizer.step()

            for key, val in metrics.items():
                agg[key] += val
            n_batches += 1

        avg = {key: val / n_batches for key, val in agg.items()}
        avg["epoch"] = epoch
        history.append(avg)

        if (epoch + 1) % log_interval == 0:
            print(
                f"  Epoch {epoch + 1:>4}/{n_epochs}: "
                f"loss={avg['loss']:.5f}  "
                f"energy={avg['energy']:.5f}  "
                f"vol_term={avg['volume_term']:.5f}  "
                f"log|det|={avg['log_det']:.5f}"
            )
            mlflow.log_metrics(
                {
                    "train/loss": avg["loss"],
                    "train/energy": avg["energy"],
                    "train/volume_term": avg["volume_term"],
                    "train/log_det": avg["log_det"],
                },
                step=epoch,
            )

    return history


# ============================================================================
# 7. Energy and Jacobian evaluation at 2D positions
# ============================================================================

def energy(x: np.ndarray, g: DenoisingMLP, model: ActionableRGM) -> float:
    """Denoising energy E(x) = ||z(x) - g(z(x))||² at 2D position x."""
    with torch.no_grad():
        z = get_z(model, torch.tensor(x, dtype=torch.float32))
        g_z = g(z.unsqueeze(0)).squeeze(0)
        return ((z - g_z) ** 2).sum().item()


def log_det_I_minus_J(x: np.ndarray, g: DenoisingMLP, model: ActionableRGM) -> float:
    """log|det(I - J_g(z(x)))| at 2D position x."""
    z = get_z(model, torch.tensor(x, dtype=torch.float32))
    z = z.unsqueeze(0)  # [1, D]
    J = vmap(jacrev(g))(z)  # [1, D, D]
    D = z.shape[1]
    I = torch.eye(D).unsqueeze(0)
    log_det = torch.linalg.slogdet(I - J).logabsdet
    return log_det.item()


def compute_energy_grid(
    model: ActionableRGM,
    g: DenoisingMLP,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
) -> np.ndarray:
    """Evaluate E(x) on a 2D grid, shape [len(x_vals), len(y_vals)]."""
    E_grid = np.zeros((len(x_vals), len(y_vals)))
    for i, xi in enumerate(x_vals):
        for j, yi in enumerate(y_vals):
            E_grid[i, j] = energy(np.array([xi, yi]), g, model)
    return E_grid


# ============================================================================
# 8. Position decoder
# ============================================================================

def train_decoder(
    decoder: PositionDecoder,
    model: ActionableRGM,
    cfg: DictConfig,
) -> None:
    """Train the position decoder z → x with the RGM encoder frozen."""
    print("Training position decoder ...")
    rng = np.random.default_rng(1)
    n = cfg.decoder.n_train
    box_w, box_h = cfg.decoder.box_width, cfg.decoder.box_height

    xs_np = np.stack(
        [rng.uniform(0.0, box_w, size=n),
         rng.uniform(0.0, box_h, size=n)],
        axis=1,
    ).astype(np.float32)
    xs = torch.from_numpy(xs_np)

    with torch.no_grad():
        zs_dec = get_z_batch(model, xs)

    n_val = int(0.2 * n)
    n_train = n - n_val
    split_idx = torch.randperm(n)
    train_idx, val_idx = split_idx[:n_train], split_idx[n_train:]
    zs_train, xs_train = zs_dec[train_idx], xs[train_idx]
    zs_val, xs_val = zs_dec[val_idx], xs[val_idx]

    optimizer = torch.optim.Adam(decoder.parameters(), lr=cfg.decoder.lr)
    batch_size = cfg.decoder.batch_size
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(cfg.decoder.n_epochs):
        idx = torch.randperm(n_train)
        for start in range(0, n_train, batch_size):
            batch_z = zs_train[idx[start : start + batch_size]]
            batch_x = xs_train[idx[start : start + batch_size]]
            optimizer.zero_grad()
            loss = ((decoder(batch_z) - batch_x) ** 2).mean()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            train_mse = ((decoder(zs_train) - xs_train) ** 2).mean().item()
            val_mse = ((decoder(zs_val) - xs_val) ** 2).mean().item()
        train_losses.append(train_mse)
        val_losses.append(val_mse)

        if (epoch + 1) % cfg.decoder.log_interval == 0:
            print(
                f"  Decoder epoch {epoch + 1:>4}/{cfg.decoder.n_epochs}: "
                f"train_MSE={train_mse:.6f}  val_MSE={val_mse:.6f}"
            )
            mlflow.log_metrics(
                {"decoder/train_mse": train_mse, "decoder/val_mse": val_mse},
                step=epoch,
            )

    final_val_mse = val_losses[-1]
    print(f"  Decoder final val MSE: {final_val_mse:.6f}")
    mlflow.log_metric("decoder/final_val_mse", final_val_mse)

    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = list(range(1, cfg.decoder.n_epochs + 1))
    ax.plot(epochs, train_losses, label="Train MSE")
    ax.plot(epochs, val_losses, label="Val MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Position decoder training curves")
    ax.legend()
    plt.tight_layout()
    mlflow.log_figure(fig, "decoder_loss_curves.png")
    plt.close()
    print("  Saved decoder_loss_curves.png")


def decode_z(z: torch.Tensor, decoder: PositionDecoder) -> tuple[float, float]:
    """Decode latent z to 2D position via the learned decoder."""
    with torch.no_grad():
        x_hat = decoder(z.unsqueeze(0)).squeeze(0)
    return float(x_hat[0].item()), float(x_hat[1].item())


# ============================================================================
# 9. Demos
# ============================================================================

def demo_training_curves(history: list[dict]) -> None:
    """Plot training loss, energy term, and volume term over epochs."""
    print("=" * 60)
    print("DEMO 1: Training Curves")
    print("=" * 60)

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, [h["loss"] for h in history])
    axes[0].set_title("Total loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("L")

    axes[1].plot(epochs, [h["energy"] for h in history])
    axes[1].set_title("Energy term  ‖z − g(z)‖²")
    axes[1].set_xlabel("Epoch")

    axes[2].plot(epochs, [h["volume_term"] for h in history])
    axes[2].set_title("Volume term  log|det(I−J)|")
    axes[2].set_xlabel("Epoch")

    plt.tight_layout()
    mlflow.log_figure(fig, "training_curves.png")
    plt.close()
    print("  Saved training_curves.png")


def demo_energy_landscape(
    model: ActionableRGM,
    g: DenoisingMLP,
    cfg: DictConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute and visualise E(x) = ||z(x) - g(z(x))||² over the 2D arena."""
    print("=" * 60)
    print("DEMO 2: Energy Landscape  E(x) = ‖z(x) − g(z(x))‖²")
    print("=" * 60)

    n_vis = cfg.n_vis_steps
    box_w, box_h = cfg.box_width, cfg.box_height
    x_vals = np.linspace(0, box_w, n_vis)
    y_vals = np.linspace(0, box_h, n_vis)

    E_grid = compute_energy_grid(model, g, x_vals, y_vals)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(E_grid.T, origin="lower", extent=(0, box_w, 0, box_h), cmap="hot_r")
    ax.contour(x_vals, y_vals, E_grid.T, levels=20, colors="k", linewidths=0.5, alpha=0.4)
    ax.set_title("Energy  E(x) = ‖z(x) − g(z(x))‖²")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    plt.colorbar(im, ax=ax, label="E(x)")
    plt.tight_layout()
    mlflow.log_figure(fig, "energy_landscape.png")
    plt.close()
    print("  Saved energy_landscape.png")

    return x_vals, y_vals, E_grid


def demo_logdet_landscape(
    model: ActionableRGM,
    g: DenoisingMLP,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    cfg: DictConfig,
) -> None:
    """Visualise log|det(I − J_g(z(x)))| over the 2D arena."""
    print("=" * 60)
    print("DEMO 3: log|det(I − J_g)| Landscape")
    print("=" * 60)

    box_w, box_h = cfg.box_width, cfg.box_height
    n_vis = len(x_vals)
    LD_grid = np.zeros((n_vis, n_vis))
    for i, xi in enumerate(x_vals):
        for j, yi in enumerate(y_vals):
            LD_grid[i, j] = log_det_I_minus_J(np.array([xi, yi]), g, model)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(LD_grid.T, origin="lower", extent=(0, box_w, 0, box_h), cmap="RdBu_r")
    ax.contour(x_vals, y_vals, LD_grid.T, levels=20, colors="k", linewidths=0.5, alpha=0.4)
    ax.set_title("log|det(I − J_g(z(x)))|  (lower = tighter basin)")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    plt.colorbar(im, ax=ax, label="log|det(I−J)|")
    plt.tight_layout()
    mlflow.log_figure(fig, "logdet_landscape.png")
    plt.close()
    print("  Saved logdet_landscape.png")


def demo_iteration(
    model: ActionableRGM,
    g: DenoisingMLP,
    decoder: PositionDecoder,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    E_grid: np.ndarray,
    cfg: DictConfig,
) -> None:
    """
    Gradient descent on E(z) = ||z - g(z)||² from random starting positions.

    Direct iteration z_{t+1} = g(z_t) requires g to be a global contraction,
    which is incompatible with a continuous attractor (manifold of fixed points):
    by the Banach fixed-point theorem a global contraction has a unique fixed
    point, not a manifold.  Gradient descent on E instead converges to the
    nearest local minimum of E, i.e. the nearest point on the attractor
    manifold, without requiring global contractivity of g.

    Step: z_{t+1} = z_t − α · ∇_z E(z_t)
    where ∇_z E = 2(I − J_gᵀ)(z − g(z))
    """
    print("=" * 60)
    print("DEMO 4: Gradient Descent on E(z) = ‖z − g(z)‖²")
    print("=" * 60)

    box_w, box_h = cfg.box_width, cfg.box_height
    n_steps = cfg.iteration.n_steps
    n_init = cfg.iteration.n_init
    alpha = cfg.iteration.alpha

    rng_seed = np.random.default_rng(42)
    start_xs = rng_seed.uniform(0.05 * box_w, 0.95 * box_w, size=n_init)
    start_ys = rng_seed.uniform(0.05 * box_h, 0.95 * box_h, size=n_init)

    fig_traj, ax_traj = plt.subplots(figsize=(7, 6))
    fig_energy, ax_energy = plt.subplots(figsize=(7, 4))

    ax_traj.imshow(
        E_grid.T, origin="lower", extent=(0, box_w, 0, box_h), cmap="hot_r", alpha=0.6
    )

    for i, (x0, y0) in enumerate(zip(start_xs, start_ys)):
        with torch.no_grad():
            z_init = get_z(model, torch.tensor([x0, y0], dtype=torch.float32))
        z = z_init.clone()

        pos_traj = [[x0, y0]]
        energies = []

        for _ in range(n_steps):
            z = z.detach().requires_grad_(True)
            g_z = g(z.unsqueeze(0)).squeeze(0)
            E = ((z - g_z) ** 2).sum()
            energies.append(E.item())
            (dE_dz,) = torch.autograd.grad(E, z)
            z = (z - alpha * dE_dz).detach()
            pos_traj.append(list(decode_z(z, decoder)))

        pos_traj = np.array(pos_traj)
        ax_traj.plot(pos_traj[:, 0], pos_traj[:, 1], linewidth=0.8, alpha=0.7)
        ax_traj.plot(*pos_traj[0], "go", markersize=4)
        ax_traj.plot(*pos_traj[-1], "rs", markersize=4)
        ax_energy.plot(energies, alpha=0.7, linewidth=0.9, label=f"init {i}")

    ax_traj.set_xlim(0, box_w)
    ax_traj.set_ylim(0, box_h)
    ax_traj.set_title("GD on E(z), trajectories decoded to 2D")
    ax_traj.set_xlabel("x₁")
    ax_traj.set_ylabel("x₂")

    ax_energy.set_title("Denoising energy  E(z_t) = ‖z_t − g(z_t)‖²")
    ax_energy.set_xlabel("Iteration t")
    ax_energy.set_ylabel("E")
    ax_energy.set_yscale("log")
    ax_energy.legend(fontsize=7, ncol=2)

    fig_traj.tight_layout()
    fig_energy.tight_layout()
    mlflow.log_figure(fig_traj, "iteration_trajectories.png")
    mlflow.log_figure(fig_energy, "iteration_energy.png")
    plt.close(fig_traj)
    plt.close(fig_energy)
    print("  Saved iteration_trajectories.png, iteration_energy.png")


def demo_alpha_sweep(
    model: ActionableRGM,
    g: DenoisingMLP,
    decoder: PositionDecoder,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    E_grid: np.ndarray,
    cfg: DictConfig,
) -> None:
    """Gradient-descent trajectories on E(z) for different step sizes α."""
    print("=" * 60)
    print("DEMO 5: α Sweep")
    print("=" * 60)

    box_w, box_h = cfg.box_width, cfg.box_height
    alphas = list(cfg.alpha_sweep.alphas)
    n_steps = cfg.alpha_sweep.n_steps
    n_init = cfg.alpha_sweep.n_init

    rng_seed = np.random.default_rng(42)
    starts = np.stack(
        [rng_seed.uniform(0.05 * box_w, 0.95 * box_w, size=n_init),
         rng_seed.uniform(0.05 * box_h, 0.95 * box_h, size=n_init)],
        axis=1,
    )  # [n_init, 2]

    fig, axes = plt.subplots(len(alphas), 2, figsize=(12, 3.5 * len(alphas)))
    if len(alphas) == 1:
        axes = axes[np.newaxis, :]

    for row, alpha in enumerate(alphas):
        print(f"  alpha = {alpha} ...")
        ax_traj, ax_energy = axes[row, 0], axes[row, 1]

        ax_traj.imshow(
            E_grid.T, origin="lower", extent=(0, box_w, 0, box_h), cmap="hot_r", alpha=0.5
        )

        for x0 in starts:
            with torch.no_grad():
                z_init = get_z(model, torch.tensor(x0, dtype=torch.float32))
            z = z_init.clone()

            pos_traj = [x0.copy()]
            energies = []

            for _ in range(n_steps):
                z = z.detach().requires_grad_(True)
                g_z = g(z.unsqueeze(0)).squeeze(0)
                E = ((z - g_z) ** 2).sum()
                energies.append(E.item())
                (dE_dz,) = torch.autograd.grad(E, z)
                z = (z - alpha * dE_dz).detach()
                pos_traj.append(list(decode_z(z, decoder)))

            pos_traj = np.array(pos_traj)
            ax_traj.plot(pos_traj[:, 0], pos_traj[:, 1], linewidth=0.8, alpha=0.7)
            ax_traj.plot(*pos_traj[0], "go", markersize=4)
            ax_traj.plot(*pos_traj[-1], "rs", markersize=4)
            ax_energy.plot(energies, alpha=0.6, linewidth=0.8)

        ax_traj.set_xlim(0, box_w)
        ax_traj.set_ylim(0, box_h)
        ax_traj.set_ylabel(f"α={alpha}\nx₂")
        ax_traj.set_xlabel("x₁")
        ax_energy.set_ylabel(f"α={alpha}\nE(z)")
        ax_energy.set_xlabel("Step")
        ax_energy.set_yscale("log")
        if row == 0:
            ax_traj.set_title("Trajectories on energy landscape")
            ax_energy.set_title("E(z_t) = ‖z_t − g(z_t)‖²  — must decrease")

    fig.suptitle("GD on E(z): varying α", fontsize=13)
    plt.tight_layout()
    mlflow.log_figure(fig, "alpha_sweep.png")
    plt.close()
    print("  Saved alpha_sweep.png")


def demo_lambda_sweep(
    model: ActionableRGM,
    zs: torch.Tensor,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    cfg: DictConfig,
) -> None:
    """
    Train independent g_θ networks for several λ values and compare energy landscapes.

    λ=0 → pull-only (trivial identity fixed point)
    Intermediate λ → correct spectral structure for a continuous attractor
    λ→∞ → maximally contractive, point-like attractors
    """
    print("=" * 60)
    print("DEMO 5: λ Sweep")
    print("=" * 60)

    box_w, box_h = cfg.box_width, cfg.box_height
    lambdas = list(cfg.lambda_sweep.lambdas)
    n_epochs = cfg.lambda_sweep.n_epochs

    fig, axes = plt.subplots(1, len(lambdas), figsize=(5 * len(lambdas), 4), sharey=True)
    if len(lambdas) == 1:
        axes = [axes]

    for ax, lam in zip(axes, lambdas):
        print(f"  λ = {lam} ...")
        g_sweep = DenoisingMLP(
            dim=model.latent_size,
            hidden_dim=cfg.model.hidden_dim,
            n_layers=cfg.model.n_layers,
        )
        opt = torch.optim.Adam(g_sweep.parameters(), lr=cfg.train.lr)

        N = len(zs)
        for _ in range(n_epochs):
            idx = torch.randperm(N)
            for start in range(0, N, cfg.train.batch_size):
                batch = zs[idx[start : start + cfg.train.batch_size]]
                opt.zero_grad()
                loss, _ = compute_loss(g_sweep, batch, lam)
                loss.backward()
                nn.utils.clip_grad_norm_(g_sweep.parameters(), cfg.train.grad_clip)
                opt.step()

        E_grid = compute_energy_grid(model, g_sweep, x_vals, y_vals)
        im = ax.imshow(E_grid.T, origin="lower", extent=(0, box_w, 0, box_h), cmap="hot_r")
        ax.set_title(f"λ = {lam}")
        ax.set_xlabel("x₁")
        if ax is axes[0]:
            ax.set_ylabel("x₂")
        plt.colorbar(im, ax=ax)

    fig.suptitle("Energy landscape E(x) for varying λ", fontsize=13)
    plt.tight_layout()
    mlflow.log_figure(fig, "lambda_sweep.png")
    plt.close()
    print("  Saved lambda_sweep.png")


# ============================================================================
# Main
# ============================================================================

@hydra.main(version_base=None, config_path="config", config_name="can_energy_volume_rgm")
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri(_TRACKING_URI)
    mlflow.set_experiment("can_energy_volume_rgm")

    print("Energy-Based CAN with Local Volume Regularisation")
    print("=" * 60)
    print(f"Loading model from run {cfg.exp_id}, iteration k={cfg.k}")

    model = load_model(cfg.exp_id, cfg.k)
    D = model.latent_size
    print(f"Model loaded: latent_size={D}, M={model.M}")

    print(f"Sampling {cfg.n_samples} visited latents ...")
    zs = sample_visited_latents(model, cfg.n_samples, cfg.box_width, cfg.box_height)
    print(f"Latents shape: {zs.shape}")

    g = DenoisingMLP(
        dim=D,
        hidden_dim=cfg.model.hidden_dim,
        n_layers=cfg.model.n_layers,
    )
    n_params = sum(p.numel() for p in g.parameters())
    print(
        f"DenoisingMLP: hidden_dim={cfg.model.hidden_dim}, "
        f"n_layers={cfg.model.n_layers}, params={n_params}"
    )

    with mlflow.start_run():
        mlflow.log_text(OmegaConf.to_yaml(cfg, resolve=True), "config.yaml")
        mlflow.log_params({
            "exp_id": cfg.exp_id,
            "k": cfg.k,
            "latent_size": D,
            "lam": cfg.train.lam,
            "n_samples": cfg.n_samples,
            "hidden_dim": cfg.model.hidden_dim,
            "n_layers": cfg.model.n_layers,
            "n_epochs": cfg.train.n_epochs,
            "lr": cfg.train.lr,
        })

        print("\nTraining position decoder ...")
        decoder = PositionDecoder(latent_size=D, hidden_dim=cfg.decoder.hidden_dim, n_layers=cfg.decoder.n_layers)
        train_decoder(decoder, model, cfg)

        print(f"\nTraining for {cfg.train.n_epochs} epochs  (λ={cfg.train.lam}) ...")
        history = train(g, zs, cfg)

        demo_training_curves(history)
        x_vals, y_vals, E_grid = demo_energy_landscape(model, g, cfg)
        demo_logdet_landscape(model, g, x_vals, y_vals, cfg)
        demo_iteration(model, g, decoder, x_vals, y_vals, E_grid, cfg)
        demo_alpha_sweep(model, g, decoder, x_vals, y_vals, E_grid, cfg)
        demo_lambda_sweep(model, zs, x_vals, y_vals, cfg)

    print("\n" + "=" * 60)
    print("All demos complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
