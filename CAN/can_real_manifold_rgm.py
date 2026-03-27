"""
CAN with correct on-manifold energy gradient descent using a learned ActionableRGM.

Position: x (2D)
Latent representation: z (obtained by iterating RGM one step to position x)

The on-manifold energy is:
    E(x) = -1/2 z(x)^T A z(x)

Its gradient w.r.t. position x is:
    dE/dx = -z(x)^T A (dz/dx)

So the correct gradient descent dynamics are:
    dx/dt = -dE/dx = z(x)^T A (dz/dx)

This is guaranteed to decrease E(x) at every step (for small enough alpha),
and fixed points are exactly the local minima of E(x).
"""

from __future__ import annotations

import os
import sys
import tempfile

# Ensure project root is on path for rgm import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf
import torch

from rgm import ActionableRGM


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRACKING_URI = f"sqlite:///{os.path.join(_PROJECT_ROOT, 'mlruns.db')}"


# ============================================================================
# 1. Model loading
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
    # torch.compile triggers C++ JIT compilation which fails when the project
    # path contains spaces ("Pilot Decoder").  For CAN inference we don't need
    # the compiled version, so replace it with the plain implementation.
    model._scan_linear_transforms = model._scan_linear_transforms_impl
    return model


# ============================================================================
# 2. Latent representations via RGM
# ============================================================================

def get_z(model: ActionableRGM, x: torch.Tensor) -> torch.Tensor:
    """
    Get latent representation z for 2D position x by iterating the RGM one step.

    The RGM is started at z0 and iterated once with displacement x, yielding
    z = RGM([x]).  The output is L2-normalised to unit length.

    Args:
        model: ActionableRGM (must be in eval mode)
        x: 2D position tensor, shape [2]

    Returns:
        z: latent representation, shape [D]
    """
    x_input = x.unsqueeze(0).unsqueeze(0)  # [1, 1, 2]
    _, z = model(x_input, norm=False)
    z = z.squeeze(0)  # [D]
    return z / (torch.linalg.norm(z) + 1e-5)


# ============================================================================
# 3. Weight matrix
# ============================================================================

def build_A(
    model: ActionableRGM,
    n_steps: int,
    box_width: float,
    box_height: float,
) -> torch.Tensor:
    """
    Build autoassociative weight matrix A from RGM representations via Riemann sum.

    A = (1/area) * integral z_bar(x) z_bar(x)^T dx   (diagonal zeroed)
    where z_bar(x) = z(x) - mean(z(x))
    """
    D = model.latent_size
    A = torch.zeros(D, D)
    dx = box_width / n_steps
    dy = box_height / n_steps

    x_vals = torch.linspace(dx / 2, box_width - dx / 2, n_steps)
    y_vals = torch.linspace(dy / 2, box_height - dy / 2, n_steps)

    with torch.no_grad():
        for xi in x_vals:
            for yi in y_vals:
                x = torch.stack([xi, yi])
                z = get_z(model, x)
                z_bar = z - z.mean()
                A += torch.outer(z_bar, z_bar) * dx * dy

    A.fill_diagonal_(0.0)
    return A


# ============================================================================
# 4. On-manifold gradient dynamics
# ============================================================================

def energy(x: np.ndarray, A: torch.Tensor, model: ActionableRGM) -> float:
    """Compute on-manifold energy E(x) = -1/2 z(x)^T A z(x)."""
    x_t = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        z = get_z(model, x_t)
    return (-0.5 * z @ A @ z).item()


def energy_gradient(
    x: np.ndarray, A: torch.Tensor, model: ActionableRGM
) -> tuple[float, np.ndarray]:
    """
    Compute energy E(x) and its on-manifold gradient dE/dx via autograd.

    dE/dx = -z(x)^T A (dz/dx)
    """
    x_t = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    z = get_z(model, x_t)
    E = -0.5 * z @ A @ z
    (dE_dx,) = torch.autograd.grad(E, x_t)
    return E.item(), dE_dx.numpy()


def step_gradient(
    x: np.ndarray,
    A: torch.Tensor,
    model: ActionableRGM,
    alpha: float = 1.0,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
    box_width: float = 2.0,
    box_height: float = 2.0,
) -> tuple[np.ndarray, float]:
    """
    Perform one on-manifold gradient descent step.

    Returns:
        x_new: updated 2D position
        E: energy before the step
    """
    rng = rng or np.random.default_rng()
    E, dE_dx = energy_gradient(x, A, model)
    delta = -alpha * dE_dx
    if noise_std > 0:
        delta += rng.normal(0, noise_std, size=2)
    x_new = x + delta
    x_new[0] = np.clip(x_new[0], 0.0, box_width)
    x_new[1] = np.clip(x_new[1], 0.0, box_height)
    return x_new, E


# ============================================================================
# 5. Demos
# ============================================================================

def demo_embedding(model: ActionableRGM, cfg: DictConfig) -> None:
    """Visualise 2D similarity kernel and mean latent activity."""
    print("=" * 60)
    print("DEMO 1: Embedding and Similarity Kernel")
    print("=" * 60)

    n_vis = cfg.n_vis_steps
    box_w, box_h = cfg.box_width, cfg.box_height
    x_vals = np.linspace(0, box_w, n_vis)
    y_vals = np.linspace(0, box_h, n_vis)

    D = model.latent_size
    Z = np.zeros((n_vis, n_vis, D))
    with torch.no_grad():
        for i, xi in enumerate(x_vals):
            for j, yi in enumerate(y_vals):
                x = torch.tensor([xi, yi], dtype=torch.float32)
                Z[i, j] = get_z(model, x).numpy()

    # Cosine similarity to the centre reference point
    ci, cj = n_vis // 2, n_vis // 2
    z_ref = Z[ci, cj]
    z_norms = np.linalg.norm(Z, axis=2)
    sim = np.einsum("ijk,k->ij", Z, z_ref) / (
        z_norms * np.linalg.norm(z_ref) + 1e-8
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axes[0].imshow(
        sim.T, origin="lower", extent=(0, box_w, 0, box_h), cmap="hot", vmin=-1, vmax=1
    )
    axes[0].plot(
        x_vals[ci], y_vals[cj], "g+", markersize=15, markeredgewidth=2, label="reference"
    )
    axes[0].set_title("Similarity kernel K(Δx) from centre")
    axes[0].set_xlabel("x₁")
    axes[0].set_ylabel("x₂")
    plt.colorbar(im0, ax=axes[0])
    axes[0].legend()

    mean_act = np.mean(np.abs(Z), axis=2)
    im1 = axes[1].imshow(
        mean_act.T, origin="lower", extent=(0, box_w, 0, box_h), cmap="viridis"
    )
    axes[1].set_title("Mean |z| per position")
    axes[1].set_xlabel("x₁")
    axes[1].set_ylabel("x₂")
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    mlflow.log_figure(fig, "embedding.png")
    plt.close()


def demo_energy_landscape(
    model: ActionableRGM, A: torch.Tensor, cfg: DictConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute and visualise the 2D energy landscape."""
    print("=" * 60)
    print("DEMO 2: Energy Landscape")
    print("=" * 60)

    n_vis = cfg.n_vis_steps
    box_w, box_h = cfg.box_width, cfg.box_height
    x_vals = np.linspace(0, box_w, n_vis)
    y_vals = np.linspace(0, box_h, n_vis)

    E_grid = np.zeros((n_vis, n_vis))
    for i, xi in enumerate(x_vals):
        for j, yi in enumerate(y_vals):
            E_grid[i, j] = energy(np.array([xi, yi]), A, model)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(E_grid.T, origin="lower", extent=(0, box_w, 0, box_h), cmap="RdBu_r")
    ax.contour(
        x_vals, y_vals, E_grid.T, levels=20, colors="k", linewidths=0.5, alpha=0.4
    )
    ax.set_title("Energy landscape E(x) = -½ z(x)ᵀ A z(x)")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    plt.colorbar(im, ax=ax, label="E(x)")
    plt.tight_layout()
    mlflow.log_figure(fig, "energy_landscape.png")
    plt.close()

    return x_vals, y_vals, E_grid


def demo_alpha_sweep(
    model: ActionableRGM,
    A: torch.Tensor,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    E_grid: np.ndarray,
    cfg: DictConfig,
) -> None:
    """Visualise gradient-descent trajectories for different alpha values."""
    print("=" * 60)
    print("DEMO 3: Alpha Sweep")
    print("=" * 60)

    box_w, box_h = cfg.box_width, cfg.box_height
    alphas = list(cfg.alpha_sweep.alphas)
    n_steps = cfg.alpha_sweep.n_steps
    n_init = cfg.alpha_sweep.n_init

    rng_seed = np.random.default_rng(42)
    starts = np.stack(
        [
            rng_seed.uniform(0.05 * box_w, 0.95 * box_w, size=n_init),
            rng_seed.uniform(0.05 * box_h, 0.95 * box_h, size=n_init),
        ],
        axis=1,
    )  # [n_init, 2]

    fig, axes = plt.subplots(len(alphas), 2, figsize=(12, 3.5 * len(alphas)))
    if len(alphas) == 1:
        axes = axes[np.newaxis, :]

    for row, alpha in enumerate(alphas):
        print(f"  alpha = {alpha} ...")
        ax_traj, ax_energy = axes[row, 0], axes[row, 1]

        ax_traj.imshow(
            E_grid.T, origin="lower", extent=(0, box_w, 0, box_h), cmap="RdBu_r", alpha=0.5
        )

        for x0 in starts:
            x = x0.copy()
            traj = [x.copy()]
            energies = [energy(x, A, model)]
            for _ in range(n_steps):
                x, _ = step_gradient(x, A, model, alpha=alpha, box_width=box_w, box_height=box_h)
                traj.append(x.copy())
                energies.append(energy(x, A, model))
            traj = np.array(traj)
            ax_traj.plot(traj[:, 0], traj[:, 1], linewidth=0.8, alpha=0.7)
            ax_traj.plot(*traj[0], "go", markersize=4)
            ax_traj.plot(*traj[-1], "rs", markersize=4)
            ax_energy.plot(energies, alpha=0.6, linewidth=0.8)

        ax_traj.set_xlim(0, box_w)
        ax_traj.set_ylim(0, box_h)
        ax_traj.set_ylabel(f"α={alpha}\nx₂")
        ax_traj.set_xlabel("x₁")
        ax_energy.set_ylabel(f"α={alpha}\nE(x)")
        ax_energy.set_xlabel("Step")
        if row == 0:
            ax_traj.set_title("Trajectories on energy landscape")
            ax_energy.set_title("Energy E(x) — must decrease")

    fig.suptitle("On-manifold gradient descent: varying α", fontsize=13)
    plt.tight_layout()
    mlflow.log_figure(fig, "alpha_sweep.png")
    plt.close()


def demo_noise(
    model: ActionableRGM,
    A: torch.Tensor,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    E_grid: np.ndarray,
    cfg: DictConfig,
) -> None:
    """Visualise gradient-descent trajectories for different noise levels."""
    print("=" * 60)
    print("DEMO 4: Noise Robustness")
    print("=" * 60)

    box_w, box_h = cfg.box_width, cfg.box_height
    alpha = cfg.noise.alpha
    noise_levels = list(cfg.noise.noise_levels)
    n_steps = cfg.noise.n_steps
    n_init = cfg.noise.n_init

    rng_seed = np.random.default_rng(42)
    starts = np.stack(
        [
            rng_seed.uniform(0.05 * box_w, 0.95 * box_w, size=n_init),
            rng_seed.uniform(0.05 * box_h, 0.95 * box_h, size=n_init),
        ],
        axis=1,
    )

    fig, axes = plt.subplots(len(noise_levels), 2, figsize=(12, 3.5 * len(noise_levels)))
    if len(noise_levels) == 1:
        axes = axes[np.newaxis, :]

    for row, noise in enumerate(noise_levels):
        print(f"  noise = {noise} ...")
        ax_traj, ax_energy = axes[row, 0], axes[row, 1]

        ax_traj.imshow(
            E_grid.T, origin="lower", extent=(0, box_w, 0, box_h), cmap="RdBu_r", alpha=0.5
        )

        for x0 in starts:
            rng = np.random.default_rng(42)
            x = x0.copy()
            traj = [x.copy()]
            energies = [energy(x, A, model)]
            for _ in range(n_steps):
                x, _ = step_gradient(
                    x, A, model, alpha=alpha, noise_std=noise,
                    rng=rng, box_width=box_w, box_height=box_h,
                )
                traj.append(x.copy())
                energies.append(energy(x, A, model))
            traj = np.array(traj)
            ax_traj.plot(traj[:, 0], traj[:, 1], linewidth=0.8, alpha=0.7)
            ax_traj.plot(*traj[0], "go", markersize=4)
            ax_traj.plot(*traj[-1], "rs", markersize=4)
            ax_energy.plot(energies, alpha=0.6, linewidth=0.8)

        ax_traj.set_xlim(0, box_w)
        ax_traj.set_ylim(0, box_h)
        ax_traj.set_ylabel(f"noise={noise}\nx₂")
        ax_traj.set_xlabel("x₁")
        ax_energy.set_ylabel(f"noise={noise}\nE(x)")
        ax_energy.set_xlabel("Step")
        if row == 0:
            ax_traj.set_title("Trajectories on energy landscape")
            ax_energy.set_title("Energy E(x)")

    fig.suptitle(f"Noise robustness (α={alpha})", fontsize=13)
    plt.tight_layout()
    mlflow.log_figure(fig, "noise_robustness.png")
    plt.close()


def demo_trajectories(
    model: ActionableRGM,
    A: torch.Tensor,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    E_grid: np.ndarray,
    cfg: DictConfig,
) -> None:
    """Show individual trajectories overlaid on the normalised energy landscape."""
    print("=" * 60)
    print("DEMO 5: Trajectories on Energy Landscape")
    print("=" * 60)

    box_w, box_h = cfg.box_width, cfg.box_height
    starts = [np.array(s, dtype=float) for s in cfg.trajectories.start_positions]
    noise_levels = list(cfg.trajectories.noise_levels)
    alpha = cfg.trajectories.alpha
    n_steps = cfg.trajectories.n_steps

    E_norm = (E_grid - E_grid.mean()) / (E_grid.std() + 1e-10)

    n_rows = len(noise_levels)
    n_cols = len(starts)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), sharex=True, sharey=True
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for row, noise in enumerate(noise_levels):
        print(f"  noise = {noise} ...")
        for col, x0 in enumerate(starts):
            ax = axes[row, col]
            ax.imshow(
                E_norm.T, origin="lower", extent=(0, box_w, 0, box_h),
                cmap="RdBu_r", alpha=0.6,
            )
            ax.contour(
                x_vals, y_vals, E_norm.T, levels=15, colors="k",
                linewidths=0.4, alpha=0.3,
            )

            rng = np.random.default_rng(42)
            x = x0.copy()
            traj = [x.copy()]
            for _ in range(n_steps):
                x, _ = step_gradient(
                    x, A, model, alpha=alpha, noise_std=noise,
                    rng=rng, box_width=box_w, box_height=box_h,
                )
                traj.append(x.copy())
            traj = np.array(traj)

            ax.scatter(traj[:, 0], traj[:, 1], s=18, color="tomato", alpha=0.3, zorder=2)
            ax.set_xlim(0, box_w)
            ax.set_ylim(0, box_h)

            if row == 0:
                ax.set_title(f"x₀ = [{x0[0]:.1f}, {x0[1]:.1f}]", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"noise={noise}\nx₂", fontsize=9)
            if row == n_rows - 1:
                ax.set_xlabel("x₁")

    fig.suptitle(f"Trajectories on energy landscape (α={alpha})", fontsize=12)
    plt.tight_layout()
    mlflow.log_figure(fig, "trajectories.png")
    plt.close()


# ============================================================================
# Main
# ============================================================================

@hydra.main(version_base=None, config_path="config", config_name="can_real_manifold_rgm")
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri(_TRACKING_URI)
    mlflow.set_experiment("can_real_manifold_rgm")

    print("On-Manifold CAN with Learned ActionableRGM")
    print("=" * 60)
    print(f"Loading model from run {cfg.exp_id}, iteration k={cfg.k}")

    model = load_model(cfg.exp_id, cfg.k)
    print(f"Model loaded: latent_size={model.latent_size}, M={model.M}")

    print("Building weight matrix A ...")
    A = build_A(model, cfg.n_weight_steps, cfg.box_width, cfg.box_height)
    print(f"A shape: {A.shape}, Frobenius norm: {torch.linalg.norm(A):.4f}")

    with mlflow.start_run():
        mlflow.log_text(OmegaConf.to_yaml(cfg, resolve=True), "config.yaml")
        mlflow.log_params({"exp_id": cfg.exp_id, "k": cfg.k, "latent_size": model.latent_size})

        demo_embedding(model, cfg)
        x_vals, y_vals, E_grid = demo_energy_landscape(model, A, cfg)
        demo_alpha_sweep(model, A, x_vals, y_vals, E_grid, cfg)
        demo_noise(model, A, x_vals, y_vals, E_grid, cfg)
        demo_trajectories(model, A, x_vals, y_vals, E_grid, cfg)

    print("\n" + "=" * 60)
    print("All demos complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
