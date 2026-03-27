"""
CAN with Hebbian learning of the weight matrix A, using a learned ActionableRGM.

The Hebbian rule updates A online from random position samples:
    A ← (1 - γ) A + η * z̄(x) z̄(x)ᵀ    (diagonal kept zero)

where z̄(x) = z(x) - mean(z(x)) is the mean-centred latent representation.

The on-manifold energy is:
    E(x) = -½ z(x)ᵀ A z(x)

Its gradient w.r.t. position x is:
    dE/dx = -z(x)ᵀ A (dz/dx)

Key demo: how the energy landscape evolves as A is learned over training.
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

from rgm import ActionableRGM


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRACKING_URI = f"sqlite:///{os.path.join(_PROJECT_ROOT, 'mlruns.db')}"


# ============================================================================
# 1. Model loading  (identical to can_real_manifold_rgm.py)
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
    model._scan_linear_transforms = model._scan_linear_transforms_impl
    return model


# ============================================================================
# 2. Latent representations
# ============================================================================

def get_z(model: ActionableRGM, x: torch.Tensor) -> torch.Tensor:
    """Return L2-normalised latent z for 2-D position x (shape [2])."""
    x_input = x.unsqueeze(0).unsqueeze(0)  # [1, 1, 2]
    _, z = model(x_input, norm=False)
    z = z.squeeze(0)  # [D]
    return z / (torch.linalg.norm(z) + 1e-5)


def get_z_batch(
    model: ActionableRGM, xs: torch.Tensor
) -> torch.Tensor:
    """Return normalised latents for a batch of positions, shape [B, D]."""
    # xs: [B, 2]
    x_input = xs.unsqueeze(1)  # [B, 1, 2]
    _, z = model(x_input, norm=False)  # z: [B, D]
    norms = torch.linalg.norm(z, dim=1, keepdim=True)
    return z / (norms + 1e-5)


# ============================================================================
# 3. Hebbian learning of A
# ============================================================================

def hebbian_step(
    A: torch.Tensor,
    z: torch.Tensor,
    lr: float,
    weight_decay: float,
) -> torch.Tensor:
    """
    One Hebbian update for a single latent vector z (shape [D]).

    A ← (1 - γ) A + η z̄ z̄ᵀ    (diagonal zeroed afterwards)
    """
    z_bar = z - z.mean()
    A = (1.0 - weight_decay) * A + lr * torch.outer(z_bar, z_bar)
    A.fill_diagonal_(0.0)
    return A


def hebbian_step_batch(
    A: torch.Tensor,
    zs: torch.Tensor,
    lr: float,
    weight_decay: float,
) -> torch.Tensor:
    """
    Batched Hebbian update (average outer product over a mini-batch).

    zs: [B, D]
    """
    z_bar = zs - zs.mean(dim=1, keepdim=True)          # [B, D]
    outer_sum = torch.einsum("bi,bj->ij", z_bar, z_bar) / z_bar.shape[0]
    A = (1.0 - weight_decay) * A + lr * outer_sum
    A.fill_diagonal_(0.0)
    return A


def train_hebbian(
    model: ActionableRGM,
    cfg: DictConfig,
    snapshot_iters: list[int],
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """
    Learn A via online Hebbian updates, returning the final A and
    snapshots of A at the requested iterations.

    Positions are drawn uniformly at random from the box at each step.
    """
    D = model.latent_size
    box_w, box_h = cfg.box_width, cfg.box_height
    lr = cfg.hebbian.lr
    weight_decay = cfg.hebbian.weight_decay
    n_iter = cfg.hebbian.n_iter
    batch_size = cfg.hebbian.batch_size

    A = torch.zeros(D, D)
    snapshots: dict[int, torch.Tensor] = {}
    rng = np.random.default_rng(0)

    snapshot_set = set(snapshot_iters)

    with torch.no_grad():
        for t in range(1, n_iter + 1):
            # Sample a mini-batch of random positions
            xs_np = np.stack(
                [rng.uniform(0, box_w, size=batch_size),
                 rng.uniform(0, box_h, size=batch_size)],
                axis=1,
            ).astype(np.float32)
            xs = torch.from_numpy(xs_np)
            zs = get_z_batch(model, xs)  # [B, D]
            A = hebbian_step_batch(A, zs, lr, weight_decay)

            if t in snapshot_set:
                snapshots[t] = A.clone()
                print(f"  Snapshot at iteration {t:>6d} | "
                      f"‖A‖_F = {torch.linalg.norm(A):.4f}")

    # Always include the final state
    snapshots[n_iter] = A.clone()
    return A, snapshots


# ============================================================================
# 4. Energy & gradient (shared with the original script)
# ============================================================================

def energy(x: np.ndarray, A: torch.Tensor, model: ActionableRGM) -> float:
    """Compute on-manifold energy E(x) = -½ z(x)ᵀ A z(x)."""
    x_t = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        z = get_z(model, x_t)
    return (-0.5 * z @ A @ z).item()


def energy_gradient(
    x: np.ndarray, A: torch.Tensor, model: ActionableRGM
) -> tuple[float, np.ndarray]:
    """Compute E(x) and dE/dx = -z(x)ᵀ A (dz/dx) via autograd."""
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
    """One on-manifold gradient-descent step."""
    rng = rng or np.random.default_rng()
    E, dE_dx = energy_gradient(x, A, model)
    delta = -alpha * dE_dx
    if noise_std > 0:
        delta += rng.normal(0, noise_std, size=2)
    x_new = x + delta
    x_new[0] = np.clip(x_new[0], 0.0, box_width)
    x_new[1] = np.clip(x_new[1], 0.0, box_height)
    return x_new, E


def compute_energy_grid(
    model: ActionableRGM,
    A: torch.Tensor,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
) -> np.ndarray:
    """Evaluate E(x) on a 2-D grid, shape [len(x_vals), len(y_vals)]."""
    E_grid = np.zeros((len(x_vals), len(y_vals)))
    for i, xi in enumerate(x_vals):
        for j, yi in enumerate(y_vals):
            E_grid[i, j] = energy(np.array([xi, yi]), A, model)
    return E_grid


# ============================================================================
# 5. Demos
# ============================================================================

def demo_landscape_evolution(
    model: ActionableRGM,
    snapshots: dict[int, torch.Tensor],
    cfg: DictConfig,
) -> None:
    """
    Main demo: show how the energy landscape changes as A is learned.

    One column per snapshot, showing:
      - Row 0: energy landscape (imshow + contour)
      - Row 1: gradient-descent trajectories from fixed starting points
    """
    print("=" * 60)
    print("DEMO: Energy Landscape Evolution During Hebbian Learning")
    print("=" * 60)

    n_vis = cfg.n_vis_steps
    box_w, box_h = cfg.box_width, cfg.box_height
    x_vals = np.linspace(0, box_w, n_vis)
    y_vals = np.linspace(0, box_h, n_vis)
    alpha = cfg.evolution.alpha
    n_traj_steps = cfg.evolution.n_traj_steps
    n_init = cfg.evolution.n_init

    sorted_iters = sorted(snapshots.keys())
    n_cols = len(sorted_iters)

    rng_seed = np.random.default_rng(42)
    starts = np.stack(
        [rng_seed.uniform(0.05 * box_w, 0.95 * box_w, size=n_init),
         rng_seed.uniform(0.05 * box_h, 0.95 * box_h, size=n_init)],
        axis=1,
    )

    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 9))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    # Compute grids for all snapshots first so we can share colour scale
    grids: dict[int, np.ndarray] = {}
    for t in sorted_iters:
        print(f"  Computing energy grid at iteration {t} ...")
        grids[t] = compute_energy_grid(model, snapshots[t], x_vals, y_vals)

    # Use a shared colour range centred on 0
    all_vals = np.concatenate([g.ravel() for g in grids.values()])
    vmax = np.quantile(np.abs(all_vals), 0.99)
    vmin = -vmax

    for col, t in enumerate(sorted_iters):
        A_t = snapshots[t]
        E_grid = grids[t]
        ax_land = axes[0, col]
        ax_traj = axes[1, col]

        # --- Row 0: landscape ---
        im = ax_land.imshow(
            E_grid.T, origin="lower", extent=(0, box_w, 0, box_h),
            cmap="RdBu_r", vmin=vmin, vmax=vmax,
        )
        ax_land.contour(
            x_vals, y_vals, E_grid.T, levels=12,
            colors="k", linewidths=0.4, alpha=0.35,
        )
        ax_land.set_title(f"iter {t}", fontsize=10)
        ax_land.set_xlabel("x₁")
        if col == 0:
            ax_land.set_ylabel("x₂\n(energy)")
        plt.colorbar(im, ax=ax_land, fraction=0.046, pad=0.04)

        # --- Row 1: trajectories ---
        ax_traj.imshow(
            E_grid.T, origin="lower", extent=(0, box_w, 0, box_h),
            cmap="RdBu_r", vmin=vmin, vmax=vmax, alpha=0.5,
        )
        for x0 in starts:
            x = x0.copy()
            traj = [x.copy()]
            for _ in range(n_traj_steps):
                x, _ = step_gradient(
                    x, A_t, model, alpha=alpha,
                    box_width=box_w, box_height=box_h,
                )
                traj.append(x.copy())
            traj = np.array(traj)
            ax_traj.plot(traj[:, 0], traj[:, 1], lw=0.9, alpha=0.8)
            ax_traj.plot(*traj[0], "go", ms=4)
            ax_traj.plot(*traj[-1], "rs", ms=4)

        ax_traj.set_xlim(0, box_w)
        ax_traj.set_ylim(0, box_h)
        ax_traj.set_xlabel("x₁")
        if col == 0:
            ax_traj.set_ylabel("x₂\n(trajectories)")

    fig.suptitle(
        "Hebbian learning of A — energy landscape evolution\n"
        f"(α={alpha}, {n_traj_steps} GD steps per trajectory)",
        fontsize=12,
    )
    plt.tight_layout()
    mlflow.log_figure(fig, "landscape_evolution.png")
    plt.close()
    print("  Saved landscape_evolution.png")


def demo_frobenius_norm(snapshots: dict[int, torch.Tensor]) -> None:
    """Plot ‖A‖_F over training to show convergence."""
    print("=" * 60)
    print("DEMO: Frobenius Norm of A Over Training")
    print("=" * 60)

    iters = sorted(snapshots.keys())
    norms = [torch.linalg.norm(snapshots[t]).item() for t in iters]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iters, norms, marker="o", markersize=4, linewidth=1.5)
    ax.set_xlabel("Hebbian iteration")
    ax.set_ylabel("‖A‖_F")
    ax.set_title("Frobenius norm of A during Hebbian learning")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    mlflow.log_figure(fig, "frobenius_norm.png")
    plt.close()
    print("  Saved frobenius_norm.png")


def demo_final_trajectories(
    model: ActionableRGM,
    A: torch.Tensor,
    cfg: DictConfig,
) -> None:
    """Show detailed gradient-descent trajectories on the final energy landscape."""
    print("=" * 60)
    print("DEMO: Trajectories on Final Energy Landscape")
    print("=" * 60)

    n_vis = cfg.n_vis_steps
    box_w, box_h = cfg.box_width, cfg.box_height
    x_vals = np.linspace(0, box_w, n_vis)
    y_vals = np.linspace(0, box_h, n_vis)
    alpha = cfg.final_trajectories.alpha
    n_steps = cfg.final_trajectories.n_steps
    starts = [np.array(s, dtype=float) for s in cfg.final_trajectories.start_positions]

    E_grid = compute_energy_grid(model, A, x_vals, y_vals)
    E_norm = (E_grid - E_grid.mean()) / (E_grid.std() + 1e-10)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        E_norm.T, origin="lower", extent=(0, box_w, 0, box_h),
        cmap="RdBu_r", alpha=0.7,
    )
    ax.contour(
        x_vals, y_vals, E_norm.T, levels=15,
        colors="k", linewidths=0.4, alpha=0.3,
    )

    for x0 in starts:
        x = x0.copy()
        traj = [x.copy()]
        energies = []
        for _ in range(n_steps):
            x, E = step_gradient(x, A, model, alpha=alpha, box_width=box_w, box_height=box_h)
            traj.append(x.copy())
            energies.append(E)
        traj = np.array(traj)
        ax.plot(traj[:, 0], traj[:, 1], lw=1.2, alpha=0.85)
        ax.plot(*traj[0], "go", ms=5, label="start" if x0 is starts[0] else "")
        ax.plot(*traj[-1], "rs", ms=5, label="end" if x0 is starts[0] else "")

    ax.set_xlim(0, box_w)
    ax.set_ylim(0, box_h)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_title(f"Final Hebbian A — gradient descent (α={alpha})")
    ax.legend(loc="upper right")
    plt.colorbar(im, ax=ax, label="E(x) [normalised]")
    plt.tight_layout()
    mlflow.log_figure(fig, "final_trajectories.png")
    plt.close()
    print("  Saved final_trajectories.png")


def demo_weight_matrix(snapshots: dict[int, torch.Tensor]) -> None:
    """Visualise the weight matrix A at selected snapshots."""
    print("=" * 60)
    print("DEMO: Weight Matrix A Evolution")
    print("=" * 60)

    sorted_iters = sorted(snapshots.keys())
    n = len(sorted_iters)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    all_vals = torch.cat([snapshots[t].flatten() for t in sorted_iters])
    vmax = torch.quantile(all_vals.abs(), 0.99).item()

    for ax, t in zip(axes, sorted_iters):
        A_np = snapshots[t].numpy()
        im = ax.imshow(A_np, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"iter {t}")
        ax.set_xlabel("neuron j")
        ax.set_ylabel("neuron i")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Weight matrix A during Hebbian learning", fontsize=12)
    plt.tight_layout()
    mlflow.log_figure(fig, "weight_matrix_evolution.png")
    plt.close()
    print("  Saved weight_matrix_evolution.png")


# ============================================================================
# Main
# ============================================================================

@hydra.main(version_base=None, config_path="config", config_name="can_hebbian_manifold_rgm")
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri(_TRACKING_URI)
    mlflow.set_experiment("can_hebbian_manifold_rgm")

    print("On-Manifold CAN with Hebbian Learning of A")
    print("=" * 60)
    print(f"Loading model from run {cfg.exp_id}, iteration k={cfg.k}")

    model = load_model(cfg.exp_id, cfg.k)
    print(f"Model loaded: latent_size={model.latent_size}, M={model.M}")

    # Build snapshot schedule
    n_iter = cfg.hebbian.n_iter
    n_snapshots = cfg.hebbian.n_snapshots
    # Log-spaced snapshots so we see early rapid changes
    snapshot_iters = np.unique(
        np.round(np.geomspace(1, n_iter, n_snapshots)).astype(int)
    ).tolist()

    print(f"\nHebbian training for {n_iter} iterations ...")
    print(f"  lr={cfg.hebbian.lr}, weight_decay={cfg.hebbian.weight_decay}, "
          f"batch_size={cfg.hebbian.batch_size}")
    print(f"  Snapshots at: {snapshot_iters}")

    A_final, snapshots = train_hebbian(model, cfg, snapshot_iters)

    with mlflow.start_run():
        mlflow.log_text(OmegaConf.to_yaml(cfg, resolve=True), "config.yaml")
        mlflow.log_params({
            "exp_id": cfg.exp_id,
            "k": cfg.k,
            "latent_size": model.latent_size,
            "hebbian_lr": cfg.hebbian.lr,
            "hebbian_weight_decay": cfg.hebbian.weight_decay,
            "hebbian_n_iter": cfg.hebbian.n_iter,
            "hebbian_batch_size": cfg.hebbian.batch_size,
        })

        demo_frobenius_norm(snapshots)
        demo_weight_matrix(snapshots)
        demo_landscape_evolution(model, snapshots, cfg)
        demo_final_trajectories(model, A_final, cfg)

    print("\n" + "=" * 60)
    print("All demos complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
