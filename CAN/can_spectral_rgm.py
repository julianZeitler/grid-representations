"""
CAN with spectral parameterisation of the recurrent weight matrix,
using an ActionableRGM encoder.

The weight matrix is:
    W = Q Λ Qᵀ,   Q orthogonal,   Λ = diag(λ₁,…,λ_D),   λᵢ = σ(λ̃ᵢ) ∈ (0,1)

with σ the sigmoid.  The on-manifold energy is:
    E(z) = -½ zᵀ W z = -½ ‖Qᵀz‖²_Λ

Low energy ↔ z is strongly aligned with high-λ eigenvectors of W.

Training minimises:
    L = -(1/2N) Σₙ z⁽ⁿ⁾ᵀ W z⁽ⁿ⁾  +  λ_reg · R(Λ)

where R is a spectral regulariser that penalises large eigenvalues:
  "nuclear"  :  R = Σᵢ λᵢ
  "logdet"   :  R = −Σᵢ log(1 − λᵢ)   ← partition-function penalty,
                    derived from  log Z ≈ const − ½ Σᵢ log(1−λᵢ),
                    diverges automatically as λᵢ → 1 (natural barrier).

Q is parameterised via the matrix exponential of a skew-symmetric matrix A:
    Q = exp(A),   A = −Aᵀ   (upper-triangle stored as free parameters)
This keeps Q exactly orthogonal throughout training.

Dynamics: sphere-projected gradient descent on E(z) in latent space:

    z_{t+1} = (z_t + α W z_t) / ‖z_t + α W z_t‖

This is equivalent to power iteration with (I + αW), which contracts
transverse components (small λᵢ) relative to manifold components (large λᵢ).
States starting from different z(x₀) converge to different points in the
high-λ eigenspace — the continuous attractor manifold.
Trajectories are decoded to 2D position space via nearest-neighbour lookup
on a pre-computed latent grid.
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

from rgm import ActionableRGM, PositionDecoder


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRACKING_URI = f"sqlite:///{os.path.join(_PROJECT_ROOT, 'mlruns.db')}"


# ============================================================================
# 1. Model loading  (identical to other can_*_rgm experiments)
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
# 3. Spectral CAN: W = Q Λ Qᵀ
# ============================================================================

class SpectralCAN(nn.Module):
    """
    Recurrent weight matrix W = Q Λ Qᵀ with:
      Λ = diag(σ(λ̃ᵢ)) ∈ (0,1)^D         (sigmoid-constrained eigenvalues)
      Q = exp(A),  A = −Aᵀ               (orthogonal via skew-symmetric exp)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        n_upper = dim * (dim - 1) // 2
        self.A_upper = nn.Parameter(torch.zeros(n_upper))
        self.lambda_raw = nn.Parameter(torch.zeros(dim))

    def get_Q(self) -> torch.Tensor:
        """Orthogonal matrix Q = exp(A) where A is skew-symmetric."""
        # Recompute indices on correct device — avoids register_buffer typing issues
        idx = torch.triu_indices(self.dim, self.dim, offset=1, device=self.A_upper.device)
        A = torch.zeros(self.dim, self.dim, device=self.A_upper.device)
        A[idx[0], idx[1]] = self.A_upper
        A = A - A.T  # enforce skew-symmetry
        return torch.linalg.matrix_exp(A)

    def get_lam(self) -> torch.Tensor:
        """Eigenvalues λᵢ = σ(λ̃ᵢ) ∈ (0, 1)."""
        return torch.sigmoid(self.lambda_raw)

    def get_W(self) -> torch.Tensor:
        """Weight matrix W = Q Λ Qᵀ."""
        Q = self.get_Q()
        return Q @ torch.diag(self.get_lam()) @ Q.T

    def energy_batch(self, zs: torch.Tensor) -> torch.Tensor:
        """E(z) = −½ zᵀWz for each row of zs.  Returns shape [N]."""
        W = self.get_W()
        return -0.5 * (zs @ W * zs).sum(dim=1)


# ============================================================================
# 4. Collect visited latents
# ============================================================================

def sample_visited_latents(
    model: ActionableRGM,
    n_samples: int,
    box_width: float,
    box_height: float,
    seed: int = 0,
) -> torch.Tensor:
    """Sample 2D positions uniformly and encode with RGM. Returns [N, D]."""
    rng = np.random.default_rng(seed)
    xs_np = np.stack(
        [rng.uniform(0.0, box_width, size=n_samples),
         rng.uniform(0.0, box_height, size=n_samples)],
        axis=1,
    ).astype(np.float32)
    xs = torch.from_numpy(xs_np)
    with torch.no_grad():
        zs = get_z_batch(model, xs)
    return zs


# ============================================================================
# 5. Loss
# ============================================================================

def compute_loss(
    can: SpectralCAN,
    zs: torch.Tensor,
    lam_reg: float,
    reg_type: str,
) -> tuple[torch.Tensor, dict]:
    """
    L = -(1/2N) Σₙ z⁽ⁿ⁾ᵀ W z⁽ⁿ⁾  +  λ_reg · R(Λ)

    reg_type:
      "nuclear" :  R = Σᵢ λᵢ
      "logdet"  :  R = −Σᵢ log(1 − λᵢ)   (partition-function penalty)
    """
    data_term = can.energy_batch(zs).mean()

    lam = can.get_lam()
    if reg_type == "nuclear":
        regularizer = lam.sum()
    elif reg_type == "logdet":
        regularizer = -torch.log(1.0 - lam + 1e-8).sum()
    else:
        raise ValueError(f"Unknown reg_type: {reg_type!r}")

    loss = data_term + lam_reg * regularizer

    return loss, {
        "loss": loss.item(),
        "data_term": data_term.item(),
        "regularizer": regularizer.item(),
        "mean_lam": lam.mean().item(),
        "max_lam": lam.max().item(),
    }


# ============================================================================
# 6. Training
# ============================================================================

def train(
    can: SpectralCAN,
    zs: torch.Tensor,
    cfg: DictConfig,
) -> list[dict]:
    """Train SpectralCAN. Returns per-epoch metric history."""
    optimizer = torch.optim.Adam(can.parameters(), lr=cfg.train.lr)

    N = len(zs)
    batch_size = cfg.train.batch_size
    n_epochs = cfg.train.n_epochs
    lam_reg = cfg.train.lam_reg
    reg_type = cfg.train.reg_type
    log_interval = cfg.train.log_interval

    history: list[dict] = []

    for epoch in range(n_epochs):
        idx = torch.randperm(N)
        agg = {k: 0.0 for k in ("loss", "data_term", "regularizer", "mean_lam", "max_lam")}
        n_batches = 0

        for start in range(0, N, batch_size):
            batch = zs[idx[start : start + batch_size]]
            optimizer.zero_grad()
            loss, metrics = compute_loss(can, batch, lam_reg, reg_type)
            loss.backward()
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
                f"data={avg['data_term']:.5f}  "
                f"reg={avg['regularizer']:.5f}  "
                f"λ̄={avg['mean_lam']:.4f}  "
                f"λ_max={avg['max_lam']:.4f}"
            )
            mlflow.log_metrics(
                {
                    "train/loss": avg["loss"],
                    "train/data_term": avg["data_term"],
                    "train/regularizer": avg["regularizer"],
                    "train/mean_lam": avg["mean_lam"],
                    "train/max_lam": avg["max_lam"],
                },
                step=epoch,
            )

    return history


# ============================================================================
# 7. Z-space dynamics helpers
# ============================================================================

def step_z(z: torch.Tensor, W: torch.Tensor, alpha: float) -> tuple[torch.Tensor, float]:
    """
    One step of sphere-projected GD on E(z) = −½ zᵀWz.

    GD on E: z ← z − α∇_z E = z + αWz = (I + αW)z
    Project back to unit sphere: z ← z / ‖z‖

    The relative scaling of component i after the unnormalised step is
    (1 + αλᵢ).  Components aligned with high-λ eigenvectors grow faster,
    so normalisation contracts the transverse (low-λ) components relative
    to the manifold (high-λ) components — exactly the CAN attractor dynamic.

    Returns:
        z_new: updated unit-norm latent
        E: energy before the step
    """
    with torch.no_grad():
        E = (-0.5 * z @ W @ z).item()
        z_new = z + alpha * (W @ z)
        z_new = z_new / (torch.linalg.norm(z_new) + 1e-8)
    return z_new, E


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


def compute_energy_grid(
    model: ActionableRGM,
    W: torch.Tensor,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
) -> np.ndarray:
    """Evaluate E(x) = −½ z(x)ᵀ W z(x) on a 2D grid."""
    E_grid = np.zeros((len(x_vals), len(y_vals)))
    with torch.no_grad():
        for i, xi in enumerate(x_vals):
            for j, yi in enumerate(y_vals):
                z = get_z(model, torch.tensor([xi, yi], dtype=torch.float32))
                E_grid[i, j] = (-0.5 * z @ W @ z).item()
    return E_grid


# ============================================================================
# 8. Demos
# ============================================================================

def demo_training_curves(history: list[dict]) -> None:
    """Plot loss, data term, regularizer, and eigenvalue statistics."""
    print("=" * 60)
    print("DEMO 1: Training Curves")
    print("=" * 60)

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    axes[0].plot(epochs, [h["loss"] for h in history])
    axes[0].set_title("Total loss")
    axes[0].set_xlabel("Epoch")

    axes[1].plot(epochs, [h["data_term"] for h in history])
    axes[1].set_title("Data term  −½ zᵀWz")
    axes[1].set_xlabel("Epoch")

    axes[2].plot(epochs, [h["regularizer"] for h in history])
    axes[2].set_title("Regularizer  R(Λ)")
    axes[2].set_xlabel("Epoch")

    axes[3].plot(epochs, [h["mean_lam"] for h in history], label="mean λ")
    axes[3].plot(epochs, [h["max_lam"] for h in history], label="max λ")
    axes[3].set_title("Eigenvalue statistics")
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylim(0, 1)
    axes[3].legend()

    plt.tight_layout()
    mlflow.log_figure(fig, "training_curves.png")
    plt.close()
    print("  Saved training_curves.png")


def demo_eigenvalue_spectrum(can: SpectralCAN) -> None:
    """Bar plot of learned eigenvalues λᵢ sorted descending."""
    print("=" * 60)
    print("DEMO 2: Eigenvalue Spectrum")
    print("=" * 60)

    with torch.no_grad():
        lam = can.get_lam().numpy()

    lam_sorted = np.sort(lam)[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].bar(np.arange(len(lam_sorted)), lam_sorted, width=1.0)
    axes[0].axhline(0.5, color="r", linestyle="--", linewidth=0.8, label="λ = 0.5")
    axes[0].set_title("Learned eigenvalues λᵢ (sorted)")
    axes[0].set_xlabel("Index")
    axes[0].set_ylabel("λᵢ = σ(λ̃ᵢ)")
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    axes[1].plot(np.arange(len(lam_sorted)), np.cumsum(lam_sorted) / lam_sorted.sum())
    axes[1].axhline(0.9, color="r", linestyle="--", linewidth=0.8, label="90% of Σλᵢ")
    axes[1].set_title("Cumulative eigenvalue fraction")
    axes[1].set_xlabel("Number of components")
    axes[1].set_ylabel("Fraction of Σλᵢ")
    axes[1].legend()

    plt.tight_layout()
    mlflow.log_figure(fig, "eigenvalue_spectrum.png")
    plt.close()
    print("  Saved eigenvalue_spectrum.png")


def demo_energy_landscape(
    model: ActionableRGM,
    W: torch.Tensor,
    cfg: DictConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute and visualise E(x) = −½ z(x)ᵀ W z(x) over the 2D arena."""
    print("=" * 60)
    print("DEMO 3: Energy Landscape  E(x) = −½ z(x)ᵀ W z(x)")
    print("=" * 60)

    n_vis = cfg.n_vis_steps
    box_w, box_h = cfg.box_width, cfg.box_height
    x_vals = np.linspace(0, box_w, n_vis)
    y_vals = np.linspace(0, box_h, n_vis)

    E_grid = compute_energy_grid(model, W, x_vals, y_vals)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(E_grid.T, origin="lower", extent=(0, box_w, 0, box_h), cmap="RdBu_r")
    ax.contour(x_vals, y_vals, E_grid.T, levels=20, colors="k", linewidths=0.5, alpha=0.4)
    ax.set_title("Energy  E(x) = −½ z(x)ᵀ W z(x)")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    plt.colorbar(im, ax=ax, label="E(x)")
    plt.tight_layout()
    mlflow.log_figure(fig, "energy_landscape.png")
    plt.close()
    print("  Saved energy_landscape.png")

    return x_vals, y_vals, E_grid


def demo_alpha_sweep(
    model: ActionableRGM,
    W: torch.Tensor,
    decoder: PositionDecoder,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    E_grid: np.ndarray,
    cfg: DictConfig,
) -> None:
    """Z-space sphere-projected GD trajectories for different step sizes α."""
    print("=" * 60)
    print("DEMO 4: α Sweep")
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
    )

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
            with torch.no_grad():
                z = get_z(model, torch.tensor(x0, dtype=torch.float32))

            pos_traj = [x0.copy()]
            energies = []

            for _ in range(n_steps):
                z, e = step_z(z, W, alpha)
                energies.append(e)
                pos_traj.append(decode_z(z, decoder))

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
        if row == 0:
            ax_traj.set_title("Trajectories (decoder z→x)")
            ax_energy.set_title("E(z_t) = −½ z_tᵀ W z_t  — must decrease")

    fig.suptitle("Sphere-projected GD on E(z): varying α", fontsize=13)
    plt.tight_layout()
    mlflow.log_figure(fig, "alpha_sweep.png")
    plt.close()
    print("  Saved alpha_sweep.png")


def demo_reg_sweep(
    model: ActionableRGM,
    zs: torch.Tensor,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    cfg: DictConfig,
) -> None:
    """Train independent SpectralCANs for several λ_reg values and compare
    the energy landscape and eigenvalue spectrum."""
    print("=" * 60)
    print("DEMO 5: Regularisation Sweep")
    print("=" * 60)

    box_w, box_h = cfg.box_width, cfg.box_height
    lam_regs = list(cfg.reg_sweep.lam_regs)
    n_epochs = cfg.reg_sweep.n_epochs
    reg_type = cfg.train.reg_type

    n = len(lam_regs)
    fig_land, axes_land = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    fig_spec, axes_spec = plt.subplots(1, n, figsize=(5 * n, 3), sharey=True)
    if n == 1:
        axes_land = [axes_land]
        axes_spec = [axes_spec]

    D = model.latent_size
    for ax_l, ax_s, lam_reg in zip(axes_land, axes_spec, lam_regs):
        print(f"  λ_reg = {lam_reg} ...")
        can_sweep = SpectralCAN(dim=D)
        opt = torch.optim.Adam(can_sweep.parameters(), lr=cfg.train.lr)

        for _ in range(n_epochs):
            idx = torch.randperm(len(zs))
            for start in range(0, len(zs), cfg.train.batch_size):
                batch = zs[idx[start : start + cfg.train.batch_size]]
                opt.zero_grad()
                loss, _ = compute_loss(can_sweep, batch, lam_reg, reg_type)
                loss.backward()
                opt.step()

        W_sweep = can_sweep.get_W().detach()
        E_grid = compute_energy_grid(model, W_sweep, x_vals, y_vals)
        im = ax_l.imshow(E_grid.T, origin="lower", extent=(0, box_w, 0, box_h), cmap="RdBu_r")
        ax_l.set_title(f"λ_reg = {lam_reg}")
        ax_l.set_xlabel("x₁")
        if ax_l is axes_land[0]:
            ax_l.set_ylabel("x₂")
        plt.colorbar(im, ax=ax_l)

        with torch.no_grad():
            lam_sorted = np.sort(can_sweep.get_lam().numpy())[::-1]
        ax_s.bar(np.arange(len(lam_sorted)), lam_sorted, width=1.0)
        ax_s.set_title(f"λ_reg = {lam_reg}")
        ax_s.set_xlabel("Index")
        ax_s.set_ylim(0, 1)
        if ax_s is axes_spec[0]:
            ax_s.set_ylabel("λᵢ")

    fig_land.suptitle(f"Energy landscape for varying λ_reg  ({reg_type})", fontsize=13)
    fig_spec.suptitle(f"Eigenvalue spectrum for varying λ_reg  ({reg_type})", fontsize=13)
    fig_land.tight_layout()
    fig_spec.tight_layout()
    mlflow.log_figure(fig_land, "reg_sweep_landscape.png")
    mlflow.log_figure(fig_spec, "reg_sweep_spectrum.png")
    plt.close(fig_land)
    plt.close(fig_spec)
    print("  Saved reg_sweep_landscape.png, reg_sweep_spectrum.png")


# ============================================================================
# Main
# ============================================================================

@hydra.main(version_base=None, config_path="config", config_name="can_spectral_rgm")
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri(_TRACKING_URI)
    mlflow.set_experiment("can_spectral_rgm")

    print("CAN with Spectral Parameterisation of W = QΛQᵀ")
    print("=" * 60)
    print(f"Loading model from run {cfg.exp_id}, iteration k={cfg.k}")

    model = load_model(cfg.exp_id, cfg.k)
    D = model.latent_size
    print(f"Model loaded: latent_size={D}, M={model.M}")

    print(f"Sampling {cfg.n_samples} visited latents ...")
    zs = sample_visited_latents(model, cfg.n_samples, cfg.box_width, cfg.box_height)
    print(f"Latents shape: {zs.shape}")

    can = SpectralCAN(dim=D)
    n_params = sum(p.numel() for p in can.parameters())
    print(f"SpectralCAN: dim={D}, params={n_params}  (A_upper: {D*(D-1)//2}, λ̃: {D})")

    with mlflow.start_run():
        mlflow.log_text(OmegaConf.to_yaml(cfg, resolve=True), "config.yaml")
        mlflow.log_params({
            "exp_id": cfg.exp_id,
            "k": cfg.k,
            "latent_size": D,
            "lam_reg": cfg.train.lam_reg,
            "reg_type": cfg.train.reg_type,
            "n_samples": cfg.n_samples,
            "n_epochs": cfg.train.n_epochs,
            "lr": cfg.train.lr,
        })

        print("\nTraining position decoder ...")
        decoder = PositionDecoder(latent_size=D, hidden_dim=cfg.decoder.hidden_dim, n_layers=cfg.decoder.n_layers)
        train_decoder(decoder, model, cfg)

        print(f"\nTraining for {cfg.train.n_epochs} epochs  "
              f"(λ_reg={cfg.train.lam_reg}, reg={cfg.train.reg_type}) ...")
        history = train(can, zs, cfg)

        W = can.get_W().detach()

        n_vis = cfg.n_vis_steps
        box_w, box_h = cfg.box_width, cfg.box_height
        x_vals = np.linspace(0, box_w, n_vis)
        y_vals = np.linspace(0, box_h, n_vis)

        demo_training_curves(history)
        demo_eigenvalue_spectrum(can)
        x_vals, y_vals, E_grid = demo_energy_landscape(model, W, cfg)
        demo_alpha_sweep(model, W, decoder, x_vals, y_vals, E_grid, cfg)
        demo_reg_sweep(model, zs, x_vals, y_vals, cfg)

    print("\n" + "=" * 60)
    print("All demos complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
