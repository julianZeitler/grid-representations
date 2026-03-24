"""
1D line attractor CAN with real-valued (continuous) position embeddings.

Uses cosine tuning curves:
    x_i(p) = max(0, cos(omega_i * p + theta_i) - cos(pi / L))

where L controls the tuning width (not a block structure parameter).
Network dynamics use global softmax normalization instead of WTA.
The CAN (weight matrix + dynamics) does not require L.
"""

import hydra
import torch
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from omegaconf import DictConfig, OmegaConf
from typing import Tuple


# ============================================================================
# 1. Position Embedding
# ============================================================================

class SpatialEmbeddingReal:
    """
    Real-valued position embedding using cosine tuning curves.

    Each neuron i has a periodic receptive field:
        x_i(p) = max(0, cos(omega_i * p + theta_i) - cos(pi / L))

    L controls tuning width: neuron i is active when
        |omega_i * p + theta_i mod 2pi| < pi/L
    giving a soft cosine bump of angular half-width pi/L.

    Neurons are grouped into M = N/L blocks. All neurons in a block share
    the same frequency omega (one sample per block), but have independent
    phases theta_i ~ U[0, 2*pi). L also sets the cosine activation threshold.
    """

    def __init__(
        self,
        N: int = 4096,
        L: int = 8,
        omega_MA: float = 64.0,
        freq_dist: str = "gaussian",
        device: str = "cpu",
    ):
        """
        Args:
            N: Total number of neurons (must be divisible by L)
            L: Block size — neurons in a block share one frequency omega;
               also sets cosine threshold cos(pi/L)
            omega_MA: Mean absolute embedding frequency (m^-1)
            freq_dist: Distribution for sampling frequencies ("gaussian" or "rectangular")
            device: torch device
        """
        assert N % L == 0, "N must be divisible by L"
        self.N = N
        self.L = L
        self.M = N // L  # number of blocks
        self.omega_MA = omega_MA
        self.device = device
        self.threshold = np.cos(np.pi / L)

        # One frequency per block, expanded to per-neuron
        if freq_dist == "gaussian":
            sigma_omega = np.sqrt(np.pi / 2) * omega_MA
            omegas_per_block = torch.randn(self.M, device=device) * sigma_omega
        elif freq_dist == "rectangular":
            omegas_per_block = (torch.rand(self.M, device=device) * 2 - 1) * 2 * omega_MA
        else:
            raise ValueError(f"Unknown freq_dist: {freq_dist}")

        self.omega = omegas_per_block.repeat_interleave(L)

        # Phases sampled independently per neuron over [0, 2*pi)
        self.theta = torch.rand(N, device=device) * 2 * np.pi

    def encode(self, p: torch.Tensor) -> torch.Tensor:
        """
        Encode position(s) p into real-valued vectors.

        Args:
            p: Position scalar or batch of positions, shape () or (B,)
        Returns:
            x: Real-valued vector(s), shape (N,) or (B, N)
        """
        if p.dim() == 0:
            p = p.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        # phase: (B, N)
        phase = p.unsqueeze(-1) * self.omega.unsqueeze(0) + self.theta.unsqueeze(0)
        x = torch.clamp(torch.cos(phase) - self.threshold, min=0.0)

        if squeeze:
            x = x.squeeze(0)
        return x

    def decode(self, z: torch.Tensor, p_range: Tuple[float, float] = (0.0, 1.0),
               n_candidates: int = 1000) -> torch.Tensor:
        """
        Decode a neural state z by finding the most similar embedding.
        Uses cosine similarity to be robust to scale differences.
        """
        ps = torch.linspace(p_range[0], p_range[1], n_candidates, device=self.device)
        x_candidates = self.encode(ps)  # (n_candidates, N)
        # Cosine similarity
        z_norm = z / (z.norm() + 1e-8)
        x_norm = x_candidates / (x_candidates.norm(dim=1, keepdim=True) + 1e-8)
        similarities = x_norm @ z_norm  # (n_candidates,)
        best_idx = similarities.argmax()
        return ps[best_idx]


# ============================================================================
# 2. Weight Matrix Construction
# ============================================================================

def build_autoassociative_weights(
    embedding: SpatialEmbeddingReal,
    p_min: float = 0.0,
    p_max: float = 1.0,
    n_steps: int = 500,
    zero_within_block: bool = True,
) -> torch.Tensor:
    """
    Construct the autoassociative weight matrix:

        A = integral_{p_min}^{p_max} x_bar(p) x_bar(p)^T dp

    where x_bar(p) = x(p) - mean(x(p)) (empirically mean-centered).

    Args:
        zero_within_block: If True, zero out within-block weights (appropriate
            for block softmax dynamics). If False, only zero the diagonal.

    Returns:
        A: (N, N) symmetric weight matrix
    """
    N = embedding.N
    dp = (p_max - p_min) / n_steps

    A = torch.zeros(N, N, device=embedding.device)

    for i in range(n_steps):
        p = p_min + (i + 0.5) * dp
        p_t = torch.tensor(p, device=embedding.device)
        x = embedding.encode(p_t)  # (N,)
        x_bar = x - x.mean()
        A += torch.outer(x_bar, x_bar) * dp

    if zero_within_block:
        L = embedding.L
        for m in range(embedding.M):
            s = m * L
            e = s + L
            A[s:e, s:e] = 0.0
    else:
        A.fill_diagonal_(0.0)

    return A


# ============================================================================
# 3. Network Dynamics
# ============================================================================

def block_softmax(h: torch.Tensor, L: int, temperature: float = 1.0) -> torch.Tensor:
    """
    Per-block softmax normalization. Softmax is applied independently
    within each block of L neurons. Each block's outputs sum to 1.
    """
    N = h.shape[0]
    M = N // L
    return torch.softmax(h.view(M, L) / temperature, dim=1).view(N)


def softmax_normalize(h: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Global softmax normalization. Output sums to 1.
    Temperature controls sharpness: lower = more peaked.
    """
    return torch.softmax(h / temperature, dim=0)


def add_gaussian_noise(z: torch.Tensor, std: float = 0.01) -> torch.Tensor:
    """
    Additive Gaussian noise for real-valued neural states.
    """
    return z + torch.randn_like(z) * std


def add_synaptic_noise(W: torch.Tensor, noise_scale: float = 1.0) -> torch.Tensor:
    """
    Fixed synaptic nonidealities: w_ij += chi_ij * w_RMS, chi_ij ~ N(0,1).
    """
    w_rms = torch.sqrt(torch.mean(W ** 2))
    return W + torch.randn_like(W) * w_rms * noise_scale


# ============================================================================
# 4. Visualization & Demos
# ============================================================================

def demo_embedding(emb: SpatialEmbeddingReal):
    print("=" * 60)
    print("DEMO 1: Spatial Embedding and Similarity Kernel")
    print("=" * 60)

    ps = torch.linspace(0, 1, 200)
    xs = emb.encode(ps)  # (200, N)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Similarity kernel K(Δp)
    ax = axes[0]
    p_ref = torch.tensor(0.5)
    x_ref = emb.encode(p_ref)  # (N,)
    deltas = torch.linspace(-0.15, 0.15, 300)
    similarities = []
    for dp in deltas:
        x_dp = emb.encode(p_ref + dp)
        # Cosine similarity
        sim = (x_ref * x_dp).sum() / (x_ref.norm() * x_dp.norm() + 1e-8)
        similarities.append(sim.item())
    ax.plot(deltas.numpy(), similarities)
    ax.set_xlabel("Δp (m)")
    ax.set_ylabel("Cosine similarity K(Δp)")
    ax.set_title(f"Similarity kernel (ω_MA={emb.omega_MA})")

    # (b) Full state matrix (positions × neurons)
    ax = axes[1]
    ax.imshow(xs[:, :64].numpy().T, aspect='auto', cmap='hot',
              extent=[0, 1, 64, 0])
    ax.set_xlabel("Position p (m)")
    ax.set_ylabel("Neuron index (first 64)")
    ax.set_title("Real-valued state vectors x(p)")

    # (c) Mean activity
    ax = axes[2]
    mean_act = xs.mean(dim=1)
    ax.plot(ps.numpy(), mean_act.numpy())
    ax.set_xlabel("Position p (m)")
    ax.set_ylabel("Mean activity")
    ax.set_title("Mean activity per position")

    plt.tight_layout()
    mlflow.log_figure(fig, "demo_embedding_real.png")
    plt.close()


def demo_attractor_stability(
    embeddings: list,
    weight_matrices: list,
    noise_levels: list,
    n_steps: int,
    n_init_positions: int,
    synaptic_noise: float,
    normalize_fn=None,
    zero_within_block: bool = True,
    temperature: float = 1.0,
):
    """
    Attractor stability demo using softmax dynamics.

    Args:
        normalize_fn: Callable(h) -> z. Defaults to block_softmax using emb.L.
        zero_within_block: If True, re-zero within-block weights after synaptic noise.
    """
    print("=" * 60)
    print("DEMO 2: Attractor Stability")
    print("=" * 60)

    fig, axes = plt.subplots(len(noise_levels), len(embeddings),
                             figsize=(6 * len(embeddings), 4 * len(noise_levels)))

    for col, (emb, A) in enumerate(zip(embeddings, weight_matrices)):
        L = emb.L
        _normalize = normalize_fn if normalize_fn is not None else (lambda h: block_softmax(h, L, temperature))
        print(f"\n  omega_MA = {emb.omega_MA} ...")
        W = add_synaptic_noise(A, noise_scale=synaptic_noise)
        if zero_within_block:
            for m in range(emb.M):
                s, e = m * L, (m + 1) * L
                W[s:e, s:e] = 0.0
        else:
            W.fill_diagonal_(0.0)

        init_positions = torch.linspace(0.05, 0.95, n_init_positions)

        for row, noise in enumerate(noise_levels):
            print(f"    noise = {noise} ...")
            ax = axes[row, col]

            for p0 in init_positions:
                z = emb.encode(p0)
                traj = [emb.decode(z, (0, 1)).item()]

                for t in range(n_steps):
                    h = W @ z
                    z = _normalize(h)
                    z = add_gaussian_noise(z, std=noise)
                    z = torch.clamp(z, min=0.0)
                    if t % 5 == 0:
                        traj.append(emb.decode(z, (0, 1)).item())

                times = np.arange(len(traj)) * 5
                times[0] = 0
                ax.plot(times, traj, alpha=0.6, linewidth=0.8)

            ax.set_ylim(-0.05, 1.05)
            if row == 0:
                ax.set_title(f"ω_MA = {emb.omega_MA} m⁻¹", fontsize=12)
            if col == 0:
                ax.set_ylabel(f"noise = {noise}\nDecoded position (m)")
            if row == len(noise_levels) - 1:
                ax.set_xlabel("Time step")

    plt.suptitle("Attractor stability: softmax dynamics", fontsize=14)
    plt.tight_layout()
    mlflow.log_figure(fig, "demo_stability_real.png")
    plt.close()


def demo_energy_landscape(embeddings: list, weight_matrices: list, temperature: float = 1.0,
                          use_block_softmax: bool = True):
    print("=" * 60)
    print("DEMO 3: Energy Landscape")
    print("=" * 60)

    fig, axes = plt.subplots(2, len(embeddings), figsize=(7 * len(embeddings), 10))
    if len(embeddings) == 1:
        axes = axes.reshape(2, 1)

    for idx, (emb, A) in enumerate(zip(embeddings, weight_matrices)):
        omega_MA = emb.omega_MA

        ps = torch.linspace(0.05, 0.95, 500)
        energies = []
        free_energies = []
        for p in ps:
            x = emb.encode(p)
            E = -(((x @ A) * x).sum()).item()
            energies.append(E)

            if use_block_softmax:
                z = block_softmax(x, emb.L, temperature)
            else:
                z = softmax_normalize(x, temperature)
            entropy_term = (z * torch.log(z + 1e-12)).sum().item()
            free_energies.append(E + temperature * entropy_term)

        scale = 1 / (2 * omega_MA)

        for row, (vals, label, fname_tag) in enumerate([
            (energies, "E(p) = −x(p)ᵀ A x(p)", "E"),
            (free_energies, f"F(p) = −x(p)ᵀ A x(p) + T∑zᵢ log zᵢ  (T={temperature})", "F"),
        ]):
            arr = np.array(vals)
            arr = (arr - arr.mean()) / arr.std()

            ax = axes[row, idx]
            ax.plot(ps.numpy(), arr, linewidth=0.8)
            ax.set_xlabel("Position p (m)")
            ax.set_ylabel(f"{fname_tag}(p) (normalized)")
            if row == 0:
                ax.set_title(f"ω_MA = {omega_MA} m⁻¹\n"
                             f"Expected autocorr. length ≈ {1/(2*omega_MA)*1000:.1f} mm")
            ax.axvline(x=0.5, color='r', alpha=0.3, linestyle='--')
            ax.axvline(x=0.5 + scale, color='r', alpha=0.3, linestyle='--')
            ax.annotate('1/(2ω_MA)', xy=(0.5 + scale/2, ax.get_ylim()[1]*0.9),
                        ha='center', color='red', fontsize=9)
            ax.set_ylabel(label, fontsize=8)

    plt.suptitle("Energy landscape — real-valued cosine embedding", fontsize=14)
    plt.tight_layout()
    mlflow.log_figure(fig, "demo_energy_landscape_real.png")
    plt.close()


# ============================================================================
# Main
# ============================================================================

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("can_real")

    print("CAN with Real-Valued Cosine Embeddings")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    c = cfg.can

    emb_demo1 = SpatialEmbeddingReal(c.N, c.L, c.embedding.omega_MA, freq_dist="gaussian")
    embeddings = [
        SpatialEmbeddingReal(c.N, c.L, omega_MA, freq_dist="gaussian")
        for omega_MA in c.stability.omega_MAs
    ]

    use_block_softmax = OmegaConf.select(c, "wta", default="block") == "block"
    temperature = OmegaConf.select(c, "temperature", default=1.0)
    zero_within_block = OmegaConf.select(c, "zero_within_block", default=True)

    print("Building autoassociative weight matrices...")
    weight_matrices = [
        build_autoassociative_weights(emb, n_steps=300, zero_within_block=zero_within_block)
        for emb in embeddings
    ]

    if use_block_softmax:
        normalize_fn = None  # demo defaults to block_softmax per embedding
    else:
        normalize_fn = lambda h: softmax_normalize(h, temperature=temperature)  # noqa: E731

    with mlflow.start_run():
        mlflow.log_text(OmegaConf.to_yaml(cfg.can, resolve=True), "config.yaml")
        demo_embedding(emb_demo1)
        demo_attractor_stability(
            embeddings,
            weight_matrices,
            list(c.stability.noise_levels),
            c.stability.n_steps,
            c.stability.n_init_positions,
            c.stability.synaptic_noise,
            normalize_fn=normalize_fn,
            zero_within_block=use_block_softmax,
            temperature=temperature,
        )
        demo_energy_landscape(embeddings, weight_matrices, temperature=temperature,
                              use_block_softmax=use_block_softmax)

    print("\n" + "=" * 60)
    print("All demos complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
