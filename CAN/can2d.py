"""
2D plane attractor CAN extension.

Cotteret et al. (2025) "High-resolution spatial memory requires grid-cell-like neural codes"

Implements the 2D compositional embedding via Local Circular Convolution (LCC):
    x(u, w) = x_u(u) *_LCC x_w(w)

For binary 1-hot block codes the LCC reduces to an analytic shift per block:
    i_result[m] = (i_u[m] + i_w[m]) mod L

The bound vector has the same dimensionality N and sparsity 1/L as each component.
The 2D weight matrix follows the same Hebbian covariance rule, integrated over the plane.
"""

import hydra
import torch
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from omegaconf import DictConfig, OmegaConf
from typing import Tuple

from can import SpatialEmbedding, block_wta, add_neural_noise, add_synaptic_noise


# ============================================================================
# 2D Position Embedding via LCC (Eq. 20)
# ============================================================================

class SpatialEmbedding2D:
    """
    2D position embedding via Local Circular Convolution (LCC) of two independent
    1D embeddings (supplement S2, Eq. 10):

        x(u, w) = x_u(u) *_LCC x_w(w)

    For binary 1-hot inputs (one active neuron per block), the LCC within
    block m is simply:
        i_result[m] = (i_u[m] + i_w[m]) mod L

    The bound vector preserves the block structure and sparsity (1/L).
    """

    def __init__(
        self,
        N: int = 4096,
        L: int = 8,
        omega_MA: float = 32.0,
        freq_dist: str = "gaussian",
        device: str = "cpu",
    ):
        assert N % L == 0
        self.N = N
        self.L = L
        self.M = N // L
        self.omega_MA = omega_MA
        self.device = device

        # Two independently sampled 1D embeddings
        self.emb_u = SpatialEmbedding(N, L, omega_MA, freq_dist, device)
        self.emb_w = SpatialEmbedding(N, L, omega_MA, freq_dist, device)

    def lcc(self, xu: torch.Tensor, xw: torch.Tensor) -> torch.Tensor:
        """
        Local Circular Convolution (analytical, for binary 1-hot block codes).

        Within each block m: i_result[m] = (i_u[m] + i_w[m]) mod L.

        Args:
            xu: (B, N) binary block vectors
            xw: (B, N) binary block vectors
        Returns:
            x:  (B, N) binary block vectors
        """
        B = xu.shape[0]
        xu_idx = xu.view(B, self.M, self.L).argmax(dim=-1)   # (B, M)
        xw_idx = xw.view(B, self.M, self.L).argmax(dim=-1)   # (B, M)
        result_idx = (xu_idx + xw_idx) % self.L               # (B, M)

        out = torch.zeros(B, self.M, self.L, device=xu.device)
        out[
            torch.arange(B, device=xu.device).unsqueeze(1),
            torch.arange(self.M, device=xu.device).unsqueeze(0),
            result_idx,
        ] = 1.0
        return out.view(B, self.N)

    def encode(self, u: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Encode 2D position(s) (u, w) into binary sparse block vectors.

        Args:
            u, w: scalar or (B,) batch of positions
        Returns:
            x: (N,) or (B, N) binary vectors
        """
        squeeze = u.dim() == 0
        if squeeze:
            u = u.unsqueeze(0)
            w = w.unsqueeze(0)

        xu = self.emb_u.encode(u)  # (B, N)
        xw = self.emb_w.encode(w)  # (B, N)
        x = self.lcc(xu, xw)       # (B, N)

        if squeeze:
            x = x.squeeze(0)
        return x

    def decode(
        self,
        z: torch.Tensor,
        u_range: Tuple[float, float] = (0.0, 1.0),
        w_range: Tuple[float, float] = (0.0, 1.0),
        n_candidates: int = 40,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode neural state z to 2D position by max similarity over a grid.
        Returns (u*, w*).
        """
        us = torch.linspace(u_range[0], u_range[1], n_candidates, device=self.device)
        ws = torch.linspace(w_range[0], w_range[1], n_candidates, device=self.device)
        ug, wg = torch.meshgrid(us, ws, indexing="ij")
        uf, wf = ug.reshape(-1), wg.reshape(-1)

        x_cands = self.encode(uf, wf)       # (n^2, N)
        sims = x_cands @ z.float()          # (n^2,)
        best = sims.argmax()
        return uf[best], wf[best]


# ============================================================================
# 2D Weight Matrix (Eq. 22)
# ============================================================================

def build_autoassociative_weights_2d(
    embedding: SpatialEmbedding2D,
    p_min: float = 0.0,
    p_max: float = 1.0,
    n_steps: int = 25,
) -> torch.Tensor:
    """
    2D Hebbian covariance weight matrix (Eq. 22):

        A = ∫∫ x̄(u,w) x̄(u,w)^T du dw

    where x̄ = x - 1/L. Computed via Riemann sum over an n_steps × n_steps grid.
    Within-block weights are zeroed (no self-connections within a module).
    """
    N = embedding.N
    L = embedding.L
    dp = (p_max - p_min) / n_steps

    A = torch.zeros(N, N, device=embedding.device)

    us = torch.linspace(p_min + dp / 2, p_max - dp / 2, n_steps, device=embedding.device)
    ws = torch.linspace(p_min + dp / 2, p_max - dp / 2, n_steps, device=embedding.device)

    for w_val in ws:
        x_batch = embedding.encode(us, w_val.expand(n_steps))  # (n_steps, N)
        x_bar = x_batch - 1.0 / L
        A += (x_bar.T @ x_bar) * (dp ** 2)

    for m in range(embedding.M):
        s = m * L
        A[s:s + L, s:s + L] = 0.0

    return A


# ============================================================================
# Demos
# ============================================================================

def demo_2d_similarity_kernel(emb: SpatialEmbedding2D, n_steps: int = 60):
    """
    2D similarity kernel K(Δu, Δw) as a heatmap.
    Fix reference at (0.5, 0.5), scan displacements on a grid.
    """
    print("=" * 60)
    print("DEMO 2D-1: Similarity Kernel K(Δu, Δw)")
    print("=" * 60)

    u0 = torch.tensor(0.5, device=emb.device)
    w0 = torch.tensor(0.5, device=emb.device)
    x_ref = emb.encode(u0, w0)  # (N,)

    deltas = torch.linspace(-0.2, 0.2, n_steps, device=emb.device)
    K = torch.zeros(n_steps, n_steps, device=emb.device)

    for i, du in enumerate(deltas):
        u_batch = (u0 + du).expand(n_steps)
        x_batch = emb.encode(u_batch, w0 + deltas)  # (n_steps, N)
        K[i] = (x_batch @ x_ref) / (emb.N / emb.L)

    fig, ax = plt.subplots(figsize=(6, 5))
    d = deltas.cpu().numpy()
    im = ax.imshow(
        K.cpu().numpy(),
        extent=(d[0], d[-1], d[-1], d[0]),
        cmap="viridis",
        aspect="equal",
    )
    plt.colorbar(im, ax=ax, label="Normalized similarity K")
    ax.set_xlabel("Δw")
    ax.set_ylabel("Δu")
    ax.set_title(f"2D Similarity Kernel  (ω_MA = {emb.omega_MA} m⁻¹)")
    plt.tight_layout()
    mlflow.log_figure(fig, "2d_similarity_kernel.png")
    plt.close()


def demo_2d_energy_landscape(emb: SpatialEmbedding2D, A: torch.Tensor, n_steps: int = 60):
    """
    2D energy landscape E(u,w) = -x(u,w)^T A x(u,w) as a heatmap.
    """
    print("=" * 60)
    print("DEMO 2D-2: Energy Landscape E(u, w)")
    print("=" * 60)

    ps = torch.linspace(0.05, 0.95, n_steps, device=emb.device)
    ug, wg = torch.meshgrid(ps, ps, indexing="ij")
    uf, wf = ug.reshape(-1), wg.reshape(-1)

    batch_size = 128
    energies = []
    for start in range(0, len(uf), batch_size):
        x_b = emb.encode(uf[start:start + batch_size], wf[start:start + batch_size])
        E_b = -((x_b @ A) * x_b).sum(dim=-1)
        energies.append(E_b)

    E = torch.cat(energies).view(n_steps, n_steps).cpu().numpy()
    E = (E - E.mean()) / E.std()

    fig, ax = plt.subplots(figsize=(6, 5))
    p = ps.cpu().numpy()
    im = ax.imshow(
        E,
        extent=(p[0], p[-1], p[-1], p[0]),
        cmap="RdBu_r",
        aspect="equal",
    )
    plt.colorbar(im, ax=ax, label="Energy (normalized)")
    ax.set_xlabel("w")
    ax.set_ylabel("u")
    ax.set_title(f"2D Energy Landscape  (ω_MA = {emb.omega_MA} m⁻¹)")
    plt.tight_layout()
    mlflow.log_figure(fig, "2d_energy_landscape.png")
    plt.close()


def demo_2d_ratemaps(emb: SpatialEmbedding2D, n_steps: int = 60, n_neurons: int = 16):
    """
    Plot individual neuron ratemaps: activity x_i(u, w) as a function of 2D position.
    One neuron selected from each of n_neurons evenly-spaced blocks.
    """
    print("=" * 60)
    print("DEMO 2D-3: Neuron Ratemaps")
    print("=" * 60)

    ps = torch.linspace(0.0, 1.0, n_steps, device=emb.device)
    ug, wg = torch.meshgrid(ps, ps, indexing="ij")
    uf, wf = ug.reshape(-1), wg.reshape(-1)

    batch_size = 256
    all_x = []
    for start in range(0, len(uf), batch_size):
        all_x.append(emb.encode(uf[start:start + batch_size], wf[start:start + batch_size]))
    all_x = torch.cat(all_x, dim=0)  # (n_steps^2, N)

    # One neuron per evenly-spaced block
    block_stride = max(1, emb.M // n_neurons)
    neuron_ids = [m * emb.L for m in range(0, emb.M, block_stride)][:n_neurons]

    ncols = 4
    nrows = (n_neurons + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))

    for idx, nid in enumerate(neuron_ids):
        ax = axes.flat[idx]
        ratemap = all_x[:, nid].view(n_steps, n_steps).cpu().numpy()
        ax.imshow(ratemap, cmap="hot", aspect="equal", origin="lower",
                  extent=[0, 1, 0, 1])
        ax.set_title(f"Neuron {nid}", fontsize=8)
        ax.axis("off")

    for idx in range(len(neuron_ids), nrows * ncols):
        axes.flat[idx].axis("off")

    plt.suptitle("2D Neuron Ratemaps  x_i(u, w)", fontsize=13)
    plt.tight_layout()
    mlflow.log_figure(fig, "2d_ratemaps.png")
    plt.close()


def demo_2d_stability(
    embeddings: list,
    weight_matrices: list,
    noise_levels: list,
    n_steps: int = 50,
    n_init: int = 16,
    synaptic_noise: float = 1.0,
):
    """
    2D attractor stability across ω_MA values (columns) and noise levels (rows).
    Each subplot is a 3D plot: x = w, y = u, z = time.
    The energy landscape is shown on the uw floor at 30% opacity.
    """
    print("=" * 60)
    print("DEMO 2D-4: Attractor Stability")
    print("=" * 60)

    n_record = n_steps // 5 + 1
    times = np.arange(n_record) * 5
    colors = plt.colormaps["tab20"](np.linspace(0, 1, n_init))

    # Shared random initial positions (same across all subplots)
    emb0 = embeddings[0]
    init_us = torch.rand(n_init, device=emb0.device) * 0.8 + 0.1
    init_ws = torch.rand(n_init, device=emb0.device) * 0.8 + 0.1

    n_cols = len(embeddings)
    n_rows = len(noise_levels)
    fig = plt.figure(figsize=(6 * n_cols, 6 * n_rows))

    for col, (emb, A) in enumerate(zip(embeddings, weight_matrices)):
        print(f"\n  ω_MA = {emb.omega_MA} ...")
        L = emb.L
        W = add_synaptic_noise(A, noise_scale=synaptic_noise)
        for m in range(emb.M):
            s = m * L
            W[s:s + L, s:s + L] = 0.0

        # Energy landscape for this embedding
        n_e = 40
        ps_e = torch.linspace(0.0, 1.0, n_e, device=emb.device)
        ug_e, wg_e = torch.meshgrid(ps_e, ps_e, indexing="ij")
        uf_e, wf_e = ug_e.reshape(-1), wg_e.reshape(-1)
        x_e = emb.encode(uf_e, wf_e)
        E_flat = -((x_e @ A) * x_e).sum(dim=-1)
        E_grid = E_flat.view(n_e, n_e).cpu().numpy()
        E_norm = (E_grid - E_grid.min()) / (E_grid.max() - E_grid.min())
        ww_e = wg_e.cpu().numpy()
        uu_e = ug_e.cpu().numpy()
        face_colors = plt.colormaps["RdBu_r"](E_norm)
        face_colors[..., 3] = 0.2 # opacity

        for row, noise in enumerate(noise_levels):
            print(f"    noise = {noise} ...")
            ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1, projection="3d")

            ax.plot_surface(ww_e, uu_e, np.zeros_like(E_grid),
                            facecolors=face_colors, shade=False, linewidth=0)

            for traj_idx, (u0, w0) in enumerate(zip(init_us, init_ws)):
                z = emb.encode(u0, w0)
                u_traj = [u0.item()]
                w_traj = [w0.item()]

                for t in range(n_steps):
                    h = W @ z
                    z = block_wta(h, L)
                    z = add_neural_noise(z, L, noise)
                    if t % 5 == 0:
                        u_dec, w_dec = emb.decode(z, n_candidates=30)
                        u_traj.append(u_dec.item())
                        w_traj.append(w_dec.item())

                ax.plot(w_traj, u_traj, times[:len(w_traj)],
                        color=colors[traj_idx], alpha=0.8, linewidth=1.0)

            ax.set_xlabel("w")
            ax.set_ylabel("u")
            ax.set_zlabel("time")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_zlim(0, times[-1])
            ax.view_init(elev=25, azim=-60)

            if row == 0:
                ax.set_title(f"ω_MA = {emb.omega_MA} m⁻¹", fontsize=11)

    # Row labels: use fig.text so they are always within the saved figure bounds
    for row, noise in enumerate(noise_levels):
        y = 1 - (row + 0.5) / n_rows
        fig.text(0.03, y, f"noise = {noise}", va="center", rotation=90, fontsize=10)

    # subplots_adjust instead of tight_layout: tight_layout interacts poorly with
    # 3D axes and ignores suptitle, causing the overlap
    plt.suptitle("2D Attractor Stability", fontsize=13)
    plt.subplots_adjust(left=0.07, top=0.93, wspace=0.05, hspace=0.05)
    mlflow.log_figure(fig, "2d_stability.png")
    plt.close()


# ============================================================================
# Main
# ============================================================================

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("can_2d")

    print("Cotteret et al. (2025) - 2D Plane Attractor CAN")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    c = cfg.can
    c2 = c.embedding2d

    # Single embedding for kernel / energy / ratemap demos
    emb = SpatialEmbedding2D(c.N, c.L, c2.omega_MA, freq_dist="gaussian")

    print(f"Building 2D weight matrix ({c2.n_weight_steps}×{c2.n_weight_steps} grid)...")
    A = build_autoassociative_weights_2d(emb, n_steps=c2.n_weight_steps)

    # Three embeddings across ω_MA values for the stability demo
    print("Building stability embeddings and weight matrices...")
    stability_embeddings = [
        SpatialEmbedding2D(c.N, c.L, omega_MA, freq_dist="gaussian")
        for omega_MA in c.stability.omega_MAs
    ]
    stability_weights = [
        build_autoassociative_weights_2d(e, n_steps=c2.n_weight_steps)
        for e in stability_embeddings
    ]

    with mlflow.start_run():
        mlflow.log_text(OmegaConf.to_yaml(cfg.can, resolve=True), "config.yaml")
        demo_2d_similarity_kernel(emb, n_steps=c2.n_vis_steps)
        demo_2d_energy_landscape(emb, A, n_steps=c2.n_vis_steps)
        demo_2d_ratemaps(emb, n_steps=c2.n_vis_steps, n_neurons=c2.n_ratemaps)
        demo_2d_stability(
            stability_embeddings,
            stability_weights,
            noise_levels=list(c.stability.noise_levels),
            n_steps=c.stability.n_steps,
            synaptic_noise=c.stability.synaptic_noise,
        )

    print("\n" + "=" * 60)
    print("All 2D demos complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
