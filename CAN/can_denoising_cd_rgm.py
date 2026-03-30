"""
Denoising Energy-Based CAN with Contrastive Divergence.

Combines the denoising energy from can_energy_volume_rgm with the
Persistent Contrastive Divergence training from can_ebm_rgm.

Energy function:
    E_θ(z) = ||z - g_θ(z)||²

where g_θ: R^D → R^D is a learned denoising map.  E_θ is zero exactly
on the fixed-point manifold M = {z : g_θ(z) = z} and positive elsewhere,
giving the continuous attractor structure by construction.

Training loss (CD + optional volume regularizer):
    L = mean E_θ(z⁺) - mean E_θ(z̃)
      + λ_vol · (-1/N) Σ log|det(I - J_{g_θ}(z⁺))|

where:
  - z⁺  positive samples — RGM latents of visited positions
  - z̃   negative samples from Langevin MCMC on E_θ
  - λ_vol controls basin geometry (0 to disable)

The CD term prevents g_θ from collapsing to the identity (g = id gives
E = 0 everywhere → zero gap → large CD gradient).  The volume regularizer
tightens the basin geometry around visited states.

Langevin dynamics:
    z̃' = z̃ - η_L ∇_z E_θ(z̃) + ε,   ε ~ N(0, σ²I)
    z̃' ← z̃' / ||z̃'||   (sphere projection)

∇_z E_θ(z) = 2(I - J_g^T)(z - g(z)) is computed via autograd.
GD inference dynamics are the same without noise.
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
    model._scan_linear_transforms = model._scan_linear_transforms_impl
    return model


# ============================================================================
# 2. Latent representations
# ============================================================================

def get_z(model: ActionableRGM, x: torch.Tensor) -> torch.Tensor:
    """Return L2-normalised latent z for 2D position x (shape [2])."""
    x_input = x.unsqueeze(0).unsqueeze(0)
    _, z = model(x_input, norm=False)
    z = z.squeeze(0)
    return z / (torch.linalg.norm(z) + 1e-5)


def get_z_batch(model: ActionableRGM, xs: torch.Tensor) -> torch.Tensor:
    """Return L2-normalised latents for a batch of 2D positions, shape [B, D]."""
    x_input = xs.unsqueeze(1)
    _, z = model(x_input, norm=False)
    norms = torch.linalg.norm(z, dim=1, keepdim=True)
    return z / (norms + 1e-5)


# ============================================================================
# 3. Denoising map g_θ and energy E_θ(z) = ||z - g_θ(z)||²
# ============================================================================

class DenoisingMLP(nn.Module):
    """Learned denoising map g_θ: R^D → R^D with Tanh activations."""

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


def denoising_energy(g: DenoisingMLP, z: torch.Tensor) -> torch.Tensor:
    """E_θ(z) = ||z - g_θ(z)||², shape [B] for z: [B, D]."""
    return ((z - g(z)) ** 2).sum(dim=-1)


# ============================================================================
# 4. Collect positive samples
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
# 5. Langevin MCMC on E_θ(z) = ||z - g_θ(z)||²
# ============================================================================

def langevin_step(
    z: torch.Tensor,
    g: DenoisingMLP,
    step_size: float,
    noise_std: float,
) -> torch.Tensor:
    """
    One Langevin step on E_θ(z) = ||z - g_θ(z)||²:
        z' = z - η_L ∇_z E_θ(z) + ε,   ε ~ N(0, σ²I)
        z' ← z' / ||z'||

    ∇_z E_θ is computed via autograd through g_θ.
    """
    z = z.detach().requires_grad_(True)
    E = denoising_energy(g, z).sum()
    (grad,) = torch.autograd.grad(E, z)
    grad = torch.clamp(grad, -1.0, 1.0)
    z_new = (z - step_size * grad + noise_std * torch.randn_like(z)).detach()
    z_new = z_new / (torch.linalg.norm(z_new, dim=-1, keepdim=True) + 1e-8)
    return z_new


def advance_buffer(
    buffer: torch.Tensor,
    g: DenoisingMLP,
    n_steps: int,
    step_size: float,
    noise_std: float,
) -> torch.Tensor:
    """Run n_steps of Langevin on a batch of particles."""
    z = buffer
    for _ in range(n_steps):
        z = langevin_step(z, g, step_size, noise_std)
    return z


# ============================================================================
# 6. Loss: CD + optional volume regularizer
# ============================================================================

def compute_loss(
    g: DenoisingMLP,
    pos_batch: torch.Tensor,
    neg_batch: torch.Tensor,
    lam_vol: float,
) -> tuple[torch.Tensor, dict]:
    """
    L = mean E_θ(z⁺) - mean E_θ(z̃)
      + λ_vol · (-1/N) Σ log|det(I - J_{g_θ}(z⁺))|

    CD term: pulls E down at positives, pushes E up at negatives.
    Volume term: tightens basins around visited states (optional, λ_vol=0 disables).
    """
    E_pos = denoising_energy(g, pos_batch)
    E_neg = denoising_energy(g, neg_batch.detach())
    cd_loss = E_pos.mean() - E_neg.mean()

    if lam_vol > 0:
        D = pos_batch.shape[1]
        J_batch = vmap(jacrev(g))(pos_batch)  # [B, D, D]
        I = torch.eye(D, device=pos_batch.device).unsqueeze(0)
        log_det = torch.linalg.slogdet(I - J_batch).logabsdet.mean()
        vol_reg = -lam_vol * log_det
    else:
        vol_reg = torch.zeros(1, device=pos_batch.device).squeeze()

    loss = cd_loss + vol_reg

    return loss, {
        "loss": loss.item(),
        "E_pos": E_pos.mean().item(),
        "E_neg": E_neg.mean().item(),
        "vol_reg": vol_reg.item(),
    }


# ============================================================================
# 7. Training
# ============================================================================

def train(
    g: DenoisingMLP,
    zs: torch.Tensor,
    cfg: DictConfig,
) -> list[dict]:
    """Train g_θ with CD on the denoising energy."""
    optimizer = torch.optim.Adam(g.parameters(), lr=cfg.train.lr)

    N = len(zs)
    batch_size = cfg.train.batch_size
    n_epochs = cfg.train.n_epochs
    grad_clip = cfg.train.grad_clip
    log_interval = cfg.train.log_interval

    buffer_size = min(cfg.cd.buffer_size, N)
    perm = torch.randperm(N)[:buffer_size]
    buffer = zs[perm].clone()

    noise_std_max = cfg.cd.noise_std_max
    noise_std_min = cfg.cd.noise_std_min
    decay = np.log(noise_std_max / noise_std_min) / max(n_epochs - 1, 1)

    history: list[dict] = []

    for epoch in range(n_epochs):
        noise_std = float(noise_std_max * np.exp(-decay * epoch))

        idx = torch.randperm(N)
        agg = {"loss": 0.0, "E_pos": 0.0, "E_neg": 0.0, "vol_reg": 0.0}
        n_batches = 0

        for start in range(0, N, batch_size):
            pos_batch = zs[idx[start : start + batch_size]]

            # Refresh buffer slot with positive samples, then advance via Langevin
            buf_idx = torch.randperm(buffer_size)[:batch_size]
            pos_idx = torch.randperm(N)[:batch_size]
            buffer[buf_idx] = zs[pos_idx].clone()
            neg_batch = advance_buffer(
                buffer[buf_idx], g,
                n_steps=cfg.cd.n_steps,
                step_size=cfg.cd.step_size,
                noise_std=noise_std,
            )
            buffer[buf_idx] = neg_batch

            optimizer.zero_grad()
            loss, metrics = compute_loss(g, pos_batch, neg_batch, cfg.cd.lam_vol)
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
                f"E_pos={avg['E_pos']:.5f}  "
                f"E_neg={avg['E_neg']:.5f}  "
                f"vol={avg['vol_reg']:.5f}  "
                f"σ={noise_std:.4f}"
            )
            mlflow.log_metrics(
                {
                    "train/loss": avg["loss"],
                    "train/E_pos": avg["E_pos"],
                    "train/E_neg": avg["E_neg"],
                    "train/vol_reg": avg["vol_reg"],
                    "train/noise_std": noise_std,
                },
                step=epoch,
            )

    return history


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

    print(f"  Decoder final val MSE: {val_losses[-1]:.6f}")
    mlflow.log_metric("decoder/final_val_mse", val_losses[-1])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, cfg.decoder.n_epochs + 1), train_losses, label="Train MSE")
    ax.plot(range(1, cfg.decoder.n_epochs + 1), val_losses, label="Val MSE")
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
# 9. Inference dynamics helpers
# ============================================================================

def step_z_gd(
    z: torch.Tensor,
    g: DenoisingMLP,
    alpha: float,
) -> tuple[torch.Tensor, float]:
    """
    One GD step on E_θ(z) = ||z - g_θ(z)||²:
        z' = z - α ∇_z E_θ(z)
        z' ← z' / ||z'||
    """
    z = z.detach().requires_grad_(True)
    E = denoising_energy(g, z.unsqueeze(0)).squeeze(0)
    (dE_dz,) = torch.autograd.grad(E, z)
    z_new = (z - alpha * dE_dz).detach()
    z_new = z_new / (torch.linalg.norm(z_new) + 1e-8)
    return z_new, E.item()


def compute_energy_grid(
    model: ActionableRGM,
    g: DenoisingMLP,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
) -> np.ndarray:
    """Evaluate E_θ(z(x)) on a 2D grid, shape [len(x_vals), len(y_vals)]."""
    E_grid = np.zeros((len(x_vals), len(y_vals)))
    with torch.no_grad():
        for i, xi in enumerate(x_vals):
            for j, yi in enumerate(y_vals):
                z = get_z(model, torch.tensor([xi, yi], dtype=torch.float32))
                E_grid[i, j] = denoising_energy(g, z.unsqueeze(0)).item()
    return E_grid


# ============================================================================
# 10. Demos
# ============================================================================

def demo_training_curves(history: list[dict]) -> None:
    print("=" * 60)
    print("DEMO 1: Training Curves")
    print("=" * 60)

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    axes[0].plot(epochs, [h["loss"] for h in history])
    axes[0].set_title("Total loss")
    axes[0].set_xlabel("Epoch")

    axes[1].plot(epochs, [h["E_pos"] for h in history], label="E_pos")
    axes[1].plot(epochs, [h["E_neg"] for h in history], label="E_neg")
    axes[1].set_title("Mean energy at positives / negatives")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    gap = [h["E_neg"] - h["E_pos"] for h in history]
    axes[2].plot(epochs, gap)
    axes[2].axhline(0, color="k", linestyle="--", linewidth=0.8)
    axes[2].set_title("Energy gap  E₋ − E₊  (should be > 0)")
    axes[2].set_xlabel("Epoch")

    axes[3].plot(epochs, [h["vol_reg"] for h in history])
    axes[3].set_title("Volume regularizer")
    axes[3].set_xlabel("Epoch")

    plt.tight_layout()
    mlflow.log_figure(fig, "training_curves.png")
    plt.close()
    print("  Saved training_curves.png")


def demo_energy_landscape(
    model: ActionableRGM,
    g: DenoisingMLP,
    cfg: DictConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    print("=" * 60)
    print("DEMO 2: Energy Landscape  E_θ(z(x)) = ‖z(x) − g_θ(z(x))‖²")
    print("=" * 60)

    n_vis = cfg.n_vis_steps
    box_w, box_h = cfg.box_width, cfg.box_height
    x_vals = np.linspace(0, box_w, n_vis)
    y_vals = np.linspace(0, box_h, n_vis)

    E_grid = compute_energy_grid(model, g, x_vals, y_vals)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(E_grid.T, origin="lower", extent=(0, box_w, 0, box_h), cmap="hot_r")
    ax.contour(x_vals, y_vals, E_grid.T, levels=20, colors="k", linewidths=0.5, alpha=0.4)
    ax.set_title("Energy  E_θ(z(x)) = ‖z(x) − g_θ(z(x))‖²")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    plt.colorbar(im, ax=ax, label="E_θ(z(x))")
    plt.tight_layout()
    mlflow.log_figure(fig, "energy_landscape.png")
    plt.close()
    print("  Saved energy_landscape.png")

    return x_vals, y_vals, E_grid


def demo_alpha_sweep(
    model: ActionableRGM,
    g: DenoisingMLP,
    decoder: PositionDecoder,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    E_grid: np.ndarray,
    cfg: DictConfig,
) -> None:
    print("=" * 60)
    print("DEMO 3: α Sweep — GD inference  z_{t+1} = z_t − α∇E_θ(z_t)")
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
        ax_traj.imshow(E_grid.T, origin="lower", extent=(0, box_w, 0, box_h), cmap="hot_r", alpha=0.5)

        for x0 in starts:
            with torch.no_grad():
                z = get_z(model, torch.tensor(x0, dtype=torch.float32))

            pos_traj = [list(x0)]
            energies = []

            for _ in range(n_steps):
                z, e = step_z_gd(z, g, alpha)
                energies.append(e)
                pos_traj.append(list(decode_z(z, decoder)))

            pos_traj_arr = np.array(pos_traj)
            ax_traj.plot(pos_traj_arr[:, 0], pos_traj_arr[:, 1], linewidth=0.8, alpha=0.7)
            ax_traj.plot(*pos_traj_arr[0], "go", markersize=4)
            ax_traj.plot(*pos_traj_arr[-1], "rs", markersize=4)
            ax_energy.plot(energies, alpha=0.6, linewidth=0.8)

        ax_traj.set_xlim(0, box_w)
        ax_traj.set_ylim(0, box_h)
        ax_traj.set_ylabel(f"α={alpha}\nx₂")
        ax_traj.set_xlabel("x₁")
        ax_energy.set_ylabel(f"α={alpha}\nE_θ(z)")
        ax_energy.set_xlabel("Step")
        if row == 0:
            ax_traj.set_title("GD trajectories (decoder z→x)")
            ax_energy.set_title("E_θ(z_t) — must decrease")

    fig.suptitle("GD inference on E_θ: varying α", fontsize=13)
    plt.tight_layout()
    mlflow.log_figure(fig, "alpha_sweep.png")
    plt.close()
    print("  Saved alpha_sweep.png")


def demo_langevin_samples(
    g: DenoisingMLP,
    zs: torch.Tensor,
    decoder: PositionDecoder,
    E_grid: np.ndarray,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    cfg: DictConfig,
) -> None:
    print("=" * 60)
    print("DEMO 4: Langevin Fantasy Particles")
    print("=" * 60)

    box_w, box_h = cfg.box_width, cfg.box_height
    n_particles = cfg.langevin_vis.n_particles
    n_steps = cfg.langevin_vis.n_steps

    idx = torch.randperm(len(zs))[:n_particles]
    z = zs[idx].clone()

    for _ in range(n_steps):
        z = langevin_step(z, g, cfg.cd.step_size, cfg.cd.noise_std_min)

    positions = np.array([list(decode_z(zi, decoder)) for zi in z])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(E_grid.T, origin="lower", extent=(0, box_w, 0, box_h), cmap="hot_r", alpha=0.5)
    ax.scatter(positions[:, 0], positions[:, 1], s=15, color="cyan", alpha=0.7,
               zorder=3, label=f"Langevin particles ({n_steps} steps)")
    ax.set_xlim(0, box_w)
    ax.set_ylim(0, box_h)
    ax.set_title(f"Langevin fantasy particles after {n_steps} steps")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.legend(fontsize=9)
    plt.tight_layout()
    mlflow.log_figure(fig, "langevin_samples.png")
    plt.close()
    print("  Saved langevin_samples.png")


# ============================================================================
# Main
# ============================================================================

@hydra.main(version_base=None, config_path="config", config_name="can_denoising_cd_rgm")
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri(_TRACKING_URI)
    mlflow.set_experiment("can_denoising_cd_rgm")

    print("Denoising Energy-Based CAN with Contrastive Divergence")
    print("=" * 60)
    print(f"Loading model from run {cfg.exp_id}, iteration k={cfg.k}")

    model = load_model(cfg.exp_id, cfg.k)
    D = model.latent_size
    print(f"Model loaded: latent_size={D}, M={model.M}")

    print(f"Sampling {cfg.n_samples} positive latents ...")
    zs = sample_visited_latents(model, cfg.n_samples, cfg.box_width, cfg.box_height)
    print(f"Latents shape: {zs.shape}")

    g = DenoisingMLP(dim=D, hidden_dim=cfg.model.hidden_dim, n_layers=cfg.model.n_layers)
    n_params = sum(p.numel() for p in g.parameters())
    print(f"DenoisingMLP: hidden_dim={cfg.model.hidden_dim}, n_layers={cfg.model.n_layers}, params={n_params}")

    with mlflow.start_run():
        mlflow.log_text(OmegaConf.to_yaml(cfg, resolve=True), "config.yaml")
        mlflow.log_params({
            "exp_id": cfg.exp_id,
            "k": cfg.k,
            "latent_size": D,
            "n_samples": cfg.n_samples,
            "hidden_dim": cfg.model.hidden_dim,
            "n_layers": cfg.model.n_layers,
            "n_epochs": cfg.train.n_epochs,
            "lr": cfg.train.lr,
            "cd_n_steps": cfg.cd.n_steps,
            "cd_step_size": cfg.cd.step_size,
            "cd_noise_std_max": cfg.cd.noise_std_max,
            "cd_noise_std_min": cfg.cd.noise_std_min,
            "cd_lam_vol": cfg.cd.lam_vol,
        })

        print("\nTraining position decoder ...")
        decoder = PositionDecoder(latent_size=D, hidden_dim=cfg.decoder.hidden_dim, n_layers=cfg.decoder.n_layers)
        train_decoder(decoder, model, cfg)

        print(f"\nTraining for {cfg.train.n_epochs} epochs (PCD-{cfg.cd.n_steps}) ...")
        history = train(g, zs, cfg)

        n_vis = cfg.n_vis_steps
        box_w, box_h = cfg.box_width, cfg.box_height
        x_vals = np.linspace(0, box_w, n_vis)
        y_vals = np.linspace(0, box_h, n_vis)

        demo_training_curves(history)
        x_vals, y_vals, E_grid = demo_energy_landscape(model, g, cfg)
        demo_alpha_sweep(model, g, decoder, x_vals, y_vals, E_grid, cfg)
        demo_langevin_samples(g, zs, decoder, E_grid, x_vals, y_vals, cfg)

    print("\n" + "=" * 60)
    print("All demos complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
