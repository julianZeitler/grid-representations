"""
CANPathIntegrator: unified CAN attractor + ActionableRGM path integration.

State: (z, x) where
    z ∈ R^D  — running RGM latent (accumulated via T applications)
    x ∈ R^2  — explicit 2D position

Per-step dynamics:
    dE/dx = gradient of E(x) = -½ z(x)ᵀ A z(x)  w.r.t. x  (via autograd)

    Δx_CAN   = -alpha_can * dE/dx         (attractor correction)
    Δx_total = Δx_PI + Δx_CAN             (sum in x-space)

    z_{t+1} = T(Δx_total) @ z_t           (RGM latent update)
    x_{t+1} = clip(x_t + Δx_total)        (explicit position update)

Note: z(x) used for computing E(x) is the *fixed encoding function*
    z(x) = T(x) z₀ / ‖…‖
and is distinct from the running latent state z_t.
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hydra
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn

from rgm import ActionableRGM
from data import TrajectoryGenerator


_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
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
    # torch.compile fails when the project path contains spaces ("Pilot Decoder")
    model._scan_linear_transforms = model._scan_linear_transforms_impl
    return model


# ============================================================================
# 2. CAN helpers (pure PyTorch)
# ============================================================================

def get_z(model: ActionableRGM, x: torch.Tensor) -> torch.Tensor:
    """
    Fixed encoding function: z(x) = T(x) z₀ / ‖…‖.

    Args:
        model: ActionableRGM in eval mode
        x: 2D position, shape [2]

    Returns:
        z: unit-normalised latent, shape [D]
    """
    x_input = x.unsqueeze(0).unsqueeze(0)  # [1, 1, 2]
    _, z = model(x_input, norm=False)
    z = z.squeeze(0)  # [D]
    return z / (torch.linalg.norm(z) + 1e-5)


def energy(x: torch.Tensor, A: torch.Tensor, model: ActionableRGM) -> float:
    """Compute E(x) = -½ z(x)ᵀ A z(x)."""
    with torch.no_grad():
        z = get_z(model, x)
    return (-0.5 * z @ A @ z).item()


def energy_gradient(
    x: torch.Tensor, A: torch.Tensor, model: ActionableRGM
) -> torch.Tensor:
    """
    Compute on-manifold gradient dE/dx via autograd.

    Uses torch.enable_grad() so it works correctly even when called from
    inside a torch.no_grad() context (e.g. during inference demos).

    Args:
        x: 2D position, shape [2]

    Returns:
        dE/dx, shape [2]
    """
    with torch.enable_grad():
        x_t = x.detach().requires_grad_(True)
        z = get_z(model, x_t)
        E = -0.5 * z @ A @ z
        (dE_dx,) = torch.autograd.grad(E, x_t)
    return dE_dx.detach()


def build_A(
    model: ActionableRGM,
    n_steps: int,
    box_width: float,
    box_height: float,
) -> torch.Tensor:
    """
    Build autoassociative weight matrix A via Riemann sum over the arena.

    A = (1/area) ∫ z̄(x) z̄(x)ᵀ dx   (diagonal zeroed)
    where z̄(x) = z(x) - mean(z(x))
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
# 3. CANPathIntegrator
# ============================================================================

class CANPathIntegrator(nn.Module):
    """
    Unified CAN attractor + ActionableRGM path integration.

    The model maintains two coupled states:
      z ∈ R^D  — running RGM latent, updated by T(Δx_total)
      x ∈ R^2  — explicit 2D position, updated by Δx_total

    At each step:
      Δx_CAN   = -alpha_can * dE/dx|_x   (gradient of CAN energy)
      Δx_total = Δx_PI + Δx_CAN          (path integration + attractor)
      z'       = T(Δx_total) @ z
      x'       = clip(x + Δx_total, arena)
    """

    def __init__(
        self,
        model: ActionableRGM,
        A: torch.Tensor,
        alpha_can: float,
        box_width: float = 2.0,
        box_height: float = 2.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.A: torch.Tensor
        self.register_buffer("A", A)
        self.alpha_can = alpha_can
        self.box_width = box_width
        self.box_height = box_height

    def step(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
        dx_pi: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single dynamics step.

        Args:
            z:     [B, D] running latent state
            x:     [B, 2] current position
            dx_pi: [B, 2] external path-integration displacement

        Returns:
            z_new: [B, D]
            x_new: [B, 2]
        """
        # CAN gradient correction — loop over batch (vmap can be added later)
        dx_can = torch.stack([
            -self.alpha_can * energy_gradient(x[b], self.A, self.model)
            for b in range(x.shape[0])
        ])  # [B, 2]

        dx_total = dx_pi + dx_can  # sum in x-space

        # RGM latent update via get_T — no full forward pass needed
        T = self.model.get_T(dx_total.unsqueeze(1))  # [B, 1, D, D]
        z_new = torch.einsum("bij,bj->bi", T[:, 0], z)

        # Explicit position update with arena clipping
        x_new = x + dx_total
        x_new = torch.stack([
            x_new[:, 0].clamp(0.0, self.box_width),
            x_new[:, 1].clamp(0.0, self.box_height),
        ], dim=1)

        return z_new, x_new

    def forward(
        self,
        dx_pi_seq: torch.Tensor,
        z0: torch.Tensor,
        x0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run dynamics over a full velocity sequence.

        Args:
            dx_pi_seq: [B, T, 2] external displacement sequence
            z0:        [B, D]   initial latent state
            x0:        [B, 2]   initial position

        Returns:
            z_seq: [B, T, D]
            x_seq: [B, T, 2]
        """
        z, x = z0, x0
        z_list, x_list = [], []
        for t in range(dx_pi_seq.shape[1]):
            z, x = self.step(z, x, dx_pi_seq[:, t])
            z_list.append(z)
            x_list.append(x)
        return torch.stack(z_list, dim=1), torch.stack(x_list, dim=1)


# ============================================================================
# 4. Demos
# ============================================================================

def _make_z0(model: ActionableRGM, batch_size: int) -> torch.Tensor:
    """Replicate the model's learned initial state across a batch."""
    z0 = torch.nn.functional.softplus(model.z0)
    z0 = z0 / torch.linalg.norm(z0)
    return z0.unsqueeze(0).expand(batch_size, -1).detach()


def demo_energy_landscape(
    model: ActionableRGM, A: torch.Tensor, cfg: DictConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute and visualise the 2D energy landscape."""
    print("=" * 60)
    print("DEMO 1: Energy Landscape")
    print("=" * 60)

    n_vis = cfg.can.n_vis_steps
    box_w, box_h = cfg.data.box_width, cfg.data.box_height
    x_vals = np.linspace(0, box_w, n_vis)
    y_vals = np.linspace(0, box_h, n_vis)

    E_grid = np.zeros((n_vis, n_vis))
    for i, xi in enumerate(x_vals):
        for j, yi in enumerate(y_vals):
            x = torch.tensor([xi, yi], dtype=torch.float32)
            E_grid[i, j] = energy(x, A, model)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(E_grid.T, origin="lower", extent=(0, box_w, 0, box_h), cmap="RdBu_r")
    ax.contour(x_vals, y_vals, E_grid.T, levels=20, colors="k", linewidths=0.5, alpha=0.4)
    ax.set_title("Energy landscape E(x) = -½ z(x)ᵀ A z(x)")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    plt.colorbar(im, ax=ax, label="E(x)")
    plt.tight_layout()
    mlflow.log_figure(fig, "energy_landscape.png")
    plt.close()

    return x_vals, y_vals, E_grid


def demo_alpha_sweep(
    integrator: CANPathIntegrator,
    model: ActionableRGM,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    E_grid: np.ndarray,
    cfg: DictConfig,
) -> None:
    """Gradient-descent-only trajectories for different alpha_can values."""
    print("=" * 60)
    print("DEMO 2: Alpha Sweep (pure CAN, dx_pi=0)")
    print("=" * 60)

    box_w, box_h = cfg.data.box_width, cfg.data.box_height
    alphas = list(cfg.can.alpha_sweep.alphas)
    n_steps = cfg.can.alpha_sweep.n_steps
    n_init = cfg.can.alpha_sweep.n_init

    rng = np.random.default_rng(42)
    starts = np.stack([
        rng.uniform(0.05 * box_w, 0.95 * box_w, size=n_init),
        rng.uniform(0.05 * box_h, 0.95 * box_h, size=n_init),
    ], axis=1)  # [n_init, 2]

    fig, axes = plt.subplots(len(alphas), 2, figsize=(12, 3.5 * len(alphas)))
    if len(alphas) == 1:
        axes = axes[np.newaxis, :]

    dx_pi = torch.zeros(n_init, 2)
    z0 = _make_z0(model, n_init)
    dx_pi_seq = dx_pi.unsqueeze(1).expand(-1, n_steps, -1)

    for row, alpha in enumerate(alphas):
        print(f"  alpha_can = {alpha} ...")
        integrator.alpha_can = alpha
        ax_traj, ax_energy = axes[row, 0], axes[row, 1]
        ax_traj.imshow(E_grid.T, origin="lower", extent=(0, box_w, 0, box_h),
                       cmap="RdBu_r", alpha=0.5)

        x0 = torch.tensor(starts, dtype=torch.float32)

        with torch.no_grad():
            _, x_seq = integrator.forward(dx_pi_seq, z0.clone(), x0)

        x_seq_np = x_seq.numpy()  # [n_init, n_steps, 2]
        for b in range(n_init):
            traj = np.concatenate([starts[b:b+1], x_seq_np[b]], axis=0)
            ax_traj.plot(traj[:, 0], traj[:, 1], linewidth=0.8, alpha=0.7)
            ax_traj.plot(*traj[0], "go", markersize=4)
            ax_traj.plot(*traj[-1], "rs", markersize=4)
            e_traj = [energy(torch.tensor(traj[t], dtype=torch.float32), integrator.A, model)
                      for t in range(len(traj))]
            ax_energy.plot(e_traj, alpha=0.6, linewidth=0.8)

        ax_traj.set_xlim(0, box_w)
        ax_traj.set_ylim(0, box_h)
        ax_traj.set_ylabel(f"α_can={alpha}\nx₂")
        ax_traj.set_xlabel("x₁")
        ax_energy.set_ylabel(f"α_can={alpha}\nE(x)")
        ax_energy.set_xlabel("Step")
        if row == 0:
            ax_traj.set_title("Trajectories on energy landscape (dx_pi=0)")
            ax_energy.set_title("Energy E(x) — must decrease")

    fig.suptitle("Pure CAN gradient descent: varying alpha_can", fontsize=13)
    plt.tight_layout()
    mlflow.log_figure(fig, "alpha_sweep.png")
    plt.close()

    integrator.alpha_can = cfg.can.alpha_can


def demo_noise(
    integrator: CANPathIntegrator,
    model: ActionableRGM,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    E_grid: np.ndarray,
    cfg: DictConfig,
) -> None:
    """Noise robustness with pure CAN dynamics."""
    print("=" * 60)
    print("DEMO 3: Noise Robustness")
    print("=" * 60)

    box_w, box_h = cfg.data.box_width, cfg.data.box_height
    alpha_can = cfg.can.noise.alpha_can
    noise_levels = list(cfg.can.noise.noise_levels)
    n_steps = cfg.can.noise.n_steps
    n_init = cfg.can.noise.n_init

    integrator.alpha_can = alpha_can

    rng_seed = np.random.default_rng(42)
    starts = np.stack([
        rng_seed.uniform(0.05 * box_w, 0.95 * box_w, size=n_init),
        rng_seed.uniform(0.05 * box_h, 0.95 * box_h, size=n_init),
    ], axis=1)

    fig, axes = plt.subplots(len(noise_levels), 2, figsize=(12, 3.5 * len(noise_levels)))
    if len(noise_levels) == 1:
        axes = axes[np.newaxis, :]

    z0 = _make_z0(model, n_init)
    x0 = torch.tensor(starts, dtype=torch.float32)

    for row, noise in enumerate(noise_levels):
        print(f"  noise = {noise} ...")
        rng = np.random.default_rng(42)
        ax_traj, ax_energy = axes[row, 0], axes[row, 1]
        ax_traj.imshow(E_grid.T, origin="lower", extent=(0, box_w, 0, box_h),
                       cmap="RdBu_r", alpha=0.5)

        noise_seq = torch.tensor(
            rng.normal(0, noise, size=(n_init, n_steps, 2)), dtype=torch.float32
        )

        with torch.no_grad():
            _, x_seq = integrator.forward(noise_seq, z0.clone(), x0.clone())

        x_seq_np = x_seq.numpy()
        for b in range(n_init):
            traj = np.concatenate([starts[b:b+1], x_seq_np[b]], axis=0)
            ax_traj.plot(traj[:, 0], traj[:, 1], linewidth=0.8, alpha=0.7)
            ax_traj.plot(*traj[0], "go", markersize=4)
            ax_traj.plot(*traj[-1], "rs", markersize=4)
            e_traj = [energy(torch.tensor(traj[t], dtype=torch.float32), integrator.A, model)
                      for t in range(len(traj))]
            ax_energy.plot(e_traj, alpha=0.6, linewidth=0.8)

        ax_traj.set_xlim(0, box_w)
        ax_traj.set_ylim(0, box_h)
        ax_traj.set_ylabel(f"noise={noise}\nx₂")
        ax_traj.set_xlabel("x₁")
        ax_energy.set_ylabel(f"noise={noise}\nE(x)")
        ax_energy.set_xlabel("Step")
        if row == 0:
            ax_traj.set_title("Trajectories on energy landscape")
            ax_energy.set_title("Energy E(x)")

    fig.suptitle(f"Noise robustness (alpha_can={alpha_can})", fontsize=13)
    plt.tight_layout()
    mlflow.log_figure(fig, "noise_robustness.png")
    plt.close()

    integrator.alpha_can = cfg.can.alpha_can


def demo_trajectories(
    integrator: CANPathIntegrator,
    model: ActionableRGM,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    E_grid: np.ndarray,
    cfg: DictConfig,
) -> None:
    """Individual trajectories overlaid on the normalised energy landscape."""
    print("=" * 60)
    print("DEMO 4: Trajectories on Energy Landscape")
    print("=" * 60)

    box_w, box_h = cfg.data.box_width, cfg.data.box_height
    starts = [np.array(s, dtype=float) for s in cfg.can.trajectories.start_positions]
    noise_levels = list(cfg.can.trajectories.noise_levels)
    alpha_can = cfg.can.trajectories.alpha_can
    n_steps = cfg.can.trajectories.n_steps
    n_cols = len(starts)
    n_rows = len(noise_levels)

    integrator.alpha_can = alpha_can
    E_norm = (E_grid - E_grid.mean()) / (E_grid.std() + 1e-10)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows),
                             sharex=True, sharey=True)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for row, noise in enumerate(noise_levels):
        print(f"  noise = {noise} ...")
        for col, x0_np in enumerate(starts):
            ax = axes[row, col]
            ax.imshow(E_norm.T, origin="lower", extent=(0, box_w, 0, box_h),
                      cmap="RdBu_r", alpha=0.6)
            ax.contour(x_vals, y_vals, E_norm.T, levels=15, colors="k",
                       linewidths=0.4, alpha=0.3)

            rng = np.random.default_rng(42)
            x0 = torch.tensor(x0_np, dtype=torch.float32).unsqueeze(0)  # [1, 2]
            z0 = _make_z0(model, 1)
            noise_seq = torch.tensor(
                rng.normal(0, noise, size=(1, n_steps, 2)), dtype=torch.float32
            )

            with torch.no_grad():
                _, x_seq = integrator.forward(noise_seq, z0, x0)

            traj = np.concatenate([x0_np[np.newaxis], x_seq.squeeze(0).numpy()], axis=0)
            ax.scatter(traj[:, 0], traj[:, 1], s=18, color="purple", alpha=0.3, zorder=2)
            ax.set_xlim(0, box_w)
            ax.set_ylim(0, box_h)

            if row == 0:
                ax.set_title(f"x₀ = [{x0_np[0]:.1f}, {x0_np[1]:.1f}]", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"noise={noise}\nx₂", fontsize=9)
            if row == n_rows - 1:
                ax.set_xlabel("x₁")

    fig.suptitle(f"Trajectories on energy landscape (alpha_can={alpha_can})", fontsize=12)
    plt.tight_layout()
    mlflow.log_figure(fig, "trajectories.png")
    plt.close()

    integrator.alpha_can = cfg.can.alpha_can


def _generate_trajectories(
    box_w: float, box_h: float, n_steps: int, n_traj: int, cfg: DictConfig
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate trajectories using TrajectoryGenerator and convert to CAN coordinates.

    TrajectoryGenerator produces positions centered at 0 in [-box_w/2, box_w/2].
    CAN arena is [0, box_w], so we offset by [box_w/2, box_h/2].

    Returns:
        x0:    [n_traj, 2]             initial positions in CAN coordinates
        dx:    [n_traj, n_steps-1, 2]  per-step displacements
    """
    gen = TrajectoryGenerator(
        sigma=cfg.data.sigma,
        b=cfg.data.b,
        dt=cfg.data.dt,
        mu=cfg.data.mu,
    )
    # positions: [n_traj, n_steps, 2], centered at 0
    np.random.seed(42)
    positions = gen.generate_trajectory(box_w, box_h, n_steps, batch_size=n_traj)
    offset = np.array([box_w / 2, box_h / 2])
    x0 = (positions[:, 0] + offset).astype(np.float32)
    dx = np.diff(positions, axis=1).astype(np.float32)
    return x0, dx


def demo_path_integration(
    integrator: CANPathIntegrator,
    model: ActionableRGM,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    E_grid: np.ndarray,
    cfg: DictConfig,
) -> None:
    """Pure path integration with alpha_can=0 using realistic trajectories."""
    print("=" * 60)
    print("DEMO 5: Pure Path Integration (alpha_can=0)")
    print("=" * 60)

    box_w, box_h = cfg.data.box_width, cfg.data.box_height
    n_steps = cfg.can.path_integration.n_steps
    n_traj = cfg.can.path_integration.n_trajectories
    old_alpha = integrator.alpha_can
    integrator.alpha_can = cfg.can.path_integration.alpha_can

    x0_np, dx_np = _generate_trajectories(box_w, box_h, n_steps, n_traj, cfg)
    x0 = torch.tensor(x0_np)
    dx_pi_seq = torch.tensor(dx_np)  # [n_traj, n_steps-1, 2]
    z0 = _make_z0(model, n_traj)

    with torch.no_grad():
        _, x_seq = integrator.forward(dx_pi_seq, z0, x0)

    E_norm = (E_grid - E_grid.mean()) / (E_grid.std() + 1e-10)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(E_norm.T, origin="lower", extent=(0, box_w, 0, box_h), cmap="RdBu_r", alpha=0.4)

    x_seq_np = x_seq.numpy()
    for b in range(n_traj):
        traj = np.concatenate([x0_np[b:b+1], x_seq_np[b]], axis=0)
        ax.plot(traj[:, 0], traj[:, 1], linewidth=0.9, alpha=0.8)
        ax.plot(*traj[0], "go", markersize=5)
        ax.plot(*traj[-1], "rs", markersize=5)

    ax.set_xlim(0, box_w)
    ax.set_ylim(0, box_h)
    ax.set_title("Pure path integration (alpha_can=0)")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    plt.tight_layout()
    mlflow.log_figure(fig, "path_integration.png")
    plt.close()

    integrator.alpha_can = old_alpha


def demo_combined(
    integrator: CANPathIntegrator,
    model: ActionableRGM,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    E_grid: np.ndarray,
    cfg: DictConfig,
) -> None:
    """
    Combined CAN + path integration: n_can_steps CAN-only iterations (dx_pi=0)
    followed by one path integration step per trajectory timestep.

    The velocity is integrated into a single displacement applied after the CAN
    relaxation, so the position update reflects the full movement for that step.
    """
    print("=" * 60)
    print("DEMO 6: Combined CAN + Path Integration")
    print("=" * 60)

    box_w, box_h = cfg.data.box_width, cfg.data.box_height
    n_steps = cfg.can.combined.n_steps
    n_traj = cfg.can.combined.n_trajectories
    n_can_steps = cfg.can.combined.n_can_steps
    alphas = list(cfg.can.combined.alphas)
    old_alpha = integrator.alpha_can

    x0_np, dx_np = _generate_trajectories(box_w, box_h, n_steps, n_traj, cfg)
    # dx_np = np.concatenate([dx_np, np.zeros((n_traj, 20, 2), dtype=dx_np.dtype)], axis=1)
    x0 = torch.tensor(x0_np)
    z0 = _make_z0(model, n_traj)
    dx_zero = torch.zeros(n_traj, 2)

    E_norm = (E_grid - E_grid.mean()) / (E_grid.std() + 1e-10)
    fig, axes = plt.subplots(1, len(alphas), figsize=(6 * len(alphas), 6), sharey=True)
    if len(alphas) == 1:
        axes = [axes]

    for ax, alpha in zip(axes, alphas):
        print(f"  alpha_can = {alpha} ...")
        integrator.alpha_can = alpha

        z, x = z0.clone(), x0.clone()
        x_hist = [x0_np.copy()]

        with torch.no_grad():
            for t in range(dx_np.shape[1]):
                # CAN relaxation: n_can_steps with dx_pi = 0
                for _ in range(n_can_steps):
                    z, x = integrator.step(z, x, dx_zero)
                # Path integration: apply the trajectory displacement for this step
                dx_pi = torch.tensor(dx_np[:, t])
                z, x = integrator.step(z, x, dx_pi)
                x_hist.append(x.numpy().copy())

        x_hist_np = np.stack(x_hist, axis=1)  # [n_traj, n_steps, 2]

        ax.imshow(E_norm.T, origin="lower", extent=(0, box_w, 0, box_h),
                  cmap="RdBu_r", alpha=0.4)
        ax.contour(x_vals, y_vals, E_norm.T, levels=15, colors="k",
                   linewidths=0.4, alpha=0.3)

        for b in range(n_traj):
            traj = x_hist_np[b]  # [n_steps, 2]
            ax.plot(traj[:, 0], traj[:, 1], linewidth=0.9, alpha=0.8)
            ax.plot(*traj[0], "go", markersize=5)
            ax.plot(*traj[-1], "rs", markersize=5)

        ax.set_xlim(0, box_w)
        ax.set_ylim(0, box_h)
        ax.set_title(f"α={alpha}")
        ax.set_xlabel("x₁")

    axes[0].set_ylabel("x₂")
    fig.suptitle(f"Combined CAN ({n_can_steps} inner steps) + path integration", fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    mlflow.log_figure(fig, "combined.png")
    plt.close()

    integrator.alpha_can = old_alpha


# ============================================================================
# Main
# ============================================================================

@hydra.main(version_base=None, config_path="config", config_name="can_path_integration")
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri(_TRACKING_URI)
    mlflow.set_experiment("can_path_integration")

    print("CANPathIntegrator: CAN attractor + RGM path integration")
    print("=" * 60)
    print(f"Loading model from run {cfg.exp_id}, iteration k={cfg.k}")

    model = load_model(cfg.exp_id, cfg.k)
    print(f"Model loaded: latent_size={model.latent_size}, M={model.M}")

    print("Building weight matrix A ...")
    A = build_A(model, cfg.can.n_weight_steps, cfg.data.box_width, cfg.data.box_height)
    print(f"A shape: {A.shape}, Frobenius norm: {torch.linalg.norm(A):.4f}")

    integrator = CANPathIntegrator(
        model=model,
        A=A,
        alpha_can=cfg.can.alpha_can,
        box_width=cfg.data.box_width,
        box_height=cfg.data.box_height,
    )

    with mlflow.start_run():
        mlflow.log_text(OmegaConf.to_yaml(cfg, resolve=True), "config.yaml")
        mlflow.log_params({
            "exp_id": cfg.exp_id,
            "k": cfg.k,
            "latent_size": model.latent_size,
            "alpha_can": cfg.can.alpha_can,
        })

        x_vals, y_vals, E_grid = demo_energy_landscape(model, A, cfg)
        # demo_alpha_sweep(integrator, model, x_vals, y_vals, E_grid, cfg)
        # demo_noise(integrator, model, x_vals, y_vals, E_grid, cfg)
        # demo_trajectories(integrator, model, x_vals, y_vals, E_grid, cfg)
        # demo_path_integration(integrator, model, x_vals, y_vals, E_grid, cfg)
        demo_combined(integrator, model, x_vals, y_vals, E_grid, cfg)

    print("\n" + "=" * 60)
    print("All demos complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
