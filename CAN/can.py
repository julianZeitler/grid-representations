"""
Reference implementation of:
  Cotteret et al. (2025) "High-resolution spatial memory requires grid-cell-like neural codes"
  arXiv:2507.00598

Implements a 1D line attractor CAN with:
  - Sparse block-code position embeddings with periodic (grid-cell-like) receptive fields
  - Autoassociative weight matrix via Hebbian covariance rule
  - Synaptic and neural nonidealities
  - Winner-take-all (WTA) block dynamics

Author: Claude (reference implementation from paper equations)
"""

import hydra
import torch
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from omegaconf import DictConfig, OmegaConf
from typing import Tuple


# ============================================================================
# 1. Position Embedding (Eq. 1)
# ============================================================================

class SpatialEmbedding:
    """
    Binary sparse block-code embedding of a scalar position p.

    Each neuron i has a periodic receptive field:
        x_i(p) = 1[mod_L(omega_i * p + theta_i) < 1]

    Neurons are grouped into N/L blocks of size L. Within each block,
    exactly one neuron is active for any position p (like a grid cell module).
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
            L: Block size (1/L is the sparsity)
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

        # Sample one frequency per block from P(omega)
        # For Gaussian: sigma_omega = sqrt(pi/2) * omega_MA  (Eq. 6 / Methods)
        if freq_dist == "gaussian":
            sigma_omega = np.sqrt(np.pi / 2) * omega_MA
            omegas_per_block = torch.randn(self.M, device=device) * sigma_omega
        elif freq_dist == "rectangular":
            # Rectangular distribution with mean absolute = omega_MA
            # P(omega) ~ rect(omega / (4*omega_MA)), so uniform on [-2*omega_MA, 2*omega_MA]
            omegas_per_block = (torch.rand(self.M, device=device) * 2 - 1) * 2 * omega_MA
        else:
            raise ValueError(f"Unknown freq_dist: {freq_dist}")

        # Expand to per-neuron frequencies: all neurons in a block share the same omega
        # omega shape: (N,)
        self.omega = omegas_per_block.repeat_interleave(L)

        # Generate offsets theta such that exactly one neuron per block is active
        # theta_{m,n} = pi^(m)_n + theta_m  (Methods: Scalar position encoding)
        # pi^(m) is a random permutation of 0..L-1, theta_m ~ U[0, L]
        thetas = torch.zeros(N, device=device)
        for m in range(self.M):
            perm = torch.randperm(L, device=device).float()
            offset = torch.rand(1, device=device) * L
            thetas[m * L : (m + 1) * L] = perm + offset
        self.theta = thetas

    def encode(self, p: torch.Tensor) -> torch.Tensor:
        """
        Encode position(s) p into binary sparse block vectors.

        Args:
            p: Position scalar or batch of positions, shape () or (B,)
        Returns:
            x: Binary vector(s), shape (N,) or (B, N)
        """
        if p.dim() == 0:
            p = p.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        # p: (B,1), omega: (N,), theta: (N,)
        # phase: (B, N)
        phase = p.unsqueeze(-1) * self.omega.unsqueeze(0) + self.theta.unsqueeze(0)
        phase_mod = torch.remainder(phase, self.L)  # mod_L
        x = (phase_mod < 1.0).float()

        if squeeze:
            x = x.squeeze(0)
        return x

    def decode(self, z: torch.Tensor, p_range: Tuple[float, float] = (0.0, 1.0),
               n_candidates: int = 1000) -> torch.Tensor:
        """
        Decode a neural state z back to the represented position by finding
        the position whose embedding is most similar to z.
        """
        ps = torch.linspace(p_range[0], p_range[1], n_candidates, device=self.device)
        x_candidates = self.encode(ps)  # (n_candidates, N)
        similarities = x_candidates @ z.float()  # (n_candidates,)
        best_idx = similarities.argmax()
        return ps[best_idx]


# ============================================================================
# 2. Weight Matrix Construction
# ============================================================================

def build_autoassociative_weights(
    embedding: SpatialEmbedding,
    p_min: float = 0.0,
    p_max: float = 1.0,
    n_steps: int = 500,
    zero_within_block: bool = True,
) -> torch.Tensor:
    """
    Construct the autoassociative weight matrix A (Eq. 3):

        A = integral_{p_min}^{p_max} x_bar(p) x_bar(p)^T dp

    where x_bar(p) = x(p) - mean(x(p)) (empirically mean-centered).
    Computed via Riemann sum.

    Args:
        zero_within_block: If True, zero out within-block weights (appropriate
            for block WTA dynamics). Set to False for global kWTA.

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
        # Zero out within-block weights (no self-connections within a module)
        L = embedding.L
        for m in range(embedding.M):
            s = m * L
            e = s + L
            A[s:e, s:e] = 0.0
    else:
        # Always zero self-connections (diagonal)
        A.fill_diagonal_(0.0)

    return A


# ============================================================================
# 3. Network Dynamics
# ============================================================================

def block_wta(z: torch.Tensor, L: int) -> torch.Tensor:
    """
    Per-block winner-take-all activation function (Eq. 4).
    Within each block of L neurons, only the neuron with the
    greatest input is set to 1; all others are set to 0.
    """
    N = z.shape[0]
    M = N // L
    z_blocks = z.view(M, L)
    winners = z_blocks.argmax(dim=1)  # (M,)
    out = torch.zeros_like(z_blocks)
    out[torch.arange(M, device=z.device), winners] = 1.0
    return out.view(N)


def global_kwta(z: torch.Tensor, k: int) -> torch.Tensor:
    """
    Global k-winner-take-all activation function.
    The k neurons with the highest input are set to 1; all others to 0.
    """
    topk_indices = torch.topk(z, k).indices
    out = torch.zeros_like(z)
    out[topk_indices] = 1.0
    return out


def add_neural_noise(z: torch.Tensor, L: int, bit_error_rate: float = 0.1) -> torch.Tensor:
    """
    Stochastic neural nonidealities (Methods):
    On each time step, each neuron's state is set to 0 or 1 with
    equal probability (1/2)*b, or left unaltered with probability 1-b.

    After perturbation, we do NOT re-apply WTA here (that happens in the dynamics step).
    """
    N = z.shape[0]
    noise_mask = torch.rand(N, device=z.device) < bit_error_rate
    random_bits = (torch.rand(N, device=z.device) < 0.5).float()
    z_noisy = z.clone()
    z_noisy[noise_mask] = random_bits[noise_mask]
    return z_noisy


def add_synaptic_noise(W: torch.Tensor, noise_scale: float = 1.0) -> torch.Tensor:
    """
    Fixed synaptic nonidealities (Eq. 9):
    w_ij^nonideal = w_ij^ideal + chi_ij * w_RMS
    where chi_ij ~ N(0,1) and w_RMS is the RMS of ideal weights.
    """
    w_rms = torch.sqrt(torch.mean(W ** 2))
    noise = torch.randn_like(W) * w_rms * noise_scale
    return W + noise


# ============================================================================
# 4. Visualization & Demos
# ============================================================================

def demo_embedding(emb: SpatialEmbedding):
    """Demonstrate the spatial embedding and similarity kernel."""
    print("=" * 60)
    print("DEMO 1: Spatial Embedding and Similarity Kernel")
    print("=" * 60)

    N, L, omega_MA = emb.N, emb.L, emb.omega_MA

    # Show embedding for a few positions
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
        sim = (x_ref * x_dp).sum() / (N / L)
        similarities.append(sim.item())
    ax.plot(deltas.numpy(), similarities)
    ax.set_xlabel("Δp (m)")
    ax.set_ylabel("Normalized similarity K(Δp)")
    ax.set_title(f"Similarity kernel (ω_MA={omega_MA})")
    ax.axhline(y=1/L, color='r', linestyle='--', alpha=0.5, label='1/L baseline')
    ax.legend()

    # (b) Full state matrix (positions × neurons)
    ax = axes[1]
    ax.imshow(xs[:, :64].numpy().T, aspect='auto', cmap='binary',
              extent=[0, 1, 64, 0])
    ax.set_xlabel("Position p (m)")
    ax.set_ylabel("Neuron index (first 64)")
    ax.set_title("Binary state vectors x(p)")

    # (c) Sparsity verification
    ax = axes[2]
    sparsities = xs.mean(dim=1)
    ax.plot(ps.numpy(), sparsities.numpy())
    ax.axhline(y=1/L, color='r', linestyle='--', label=f'Expected 1/L = {1/L:.3f}')
    ax.set_xlabel("Position p (m)")
    ax.set_ylabel("Fraction of active neurons")
    ax.set_title("Sparsity verification")
    ax.legend()

    plt.tight_layout()
    mlflow.log_figure(fig, "demo_embedding.png")
    plt.close()


def demo_attractor_stability(
    embeddings: list,
    weight_matrices: list,
    noise_levels: list,
    n_steps: int,
    n_init_positions: int,
    synaptic_noise: float,
    wta_fn=None,
):
    """
    Demonstrate the resolution-stability advantage of multimodal codes.
    Initialize from multiple positions and observe drift behavior.
    Corresponds to Fig. 1f-i in the paper.

    Args:
        wta_fn: Callable(z) -> binary vector. Defaults to block_wta using emb.L.
    """
    print("=" * 60)
    print("DEMO 2: Attractor Stability (Fig. 1f-i analog)")
    print("=" * 60)

    fig, axes = plt.subplots(len(noise_levels), len(embeddings), figsize=(6 * len(embeddings), 4 * len(noise_levels)))

    for col, (emb, A) in enumerate(zip(embeddings, weight_matrices)):
        L = emb.L
        _wta = wta_fn if wta_fn is not None else (lambda h: block_wta(h, L))
        print(f"\n  omega_MA = {emb.omega_MA} ...")
        W = add_synaptic_noise(A, noise_scale=synaptic_noise)

        # Zero within-block weights
        for m in range(emb.M):
            s = m * L
            e = s + L
            W[s:e, s:e] = 0.0

        init_positions = torch.linspace(0.05, 0.95, n_init_positions)

        for row, noise in enumerate(noise_levels):
            print(f"    noise = {noise} ...")
            ax = axes[row, col]

            for p0 in init_positions:
                z = emb.encode(p0)
                traj = [emb.decode(z, (0, 1)).item()]

                for t in range(n_steps):
                    h = W @ z
                    z = _wta(h)
                    z = add_neural_noise(z, L, noise)
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

    plt.suptitle("Attractor stability: ω_MA (columns) × neural noise (rows)", fontsize=14)
    plt.tight_layout()
    mlflow.log_figure(fig, "demo_stability.png")
    plt.close()


def demo_energy_landscape(embeddings: list, weight_matrices: list):
    """
    Visualize the energy landscape E(p) along the line attractor.
    Shows the rough landscape with autocorrelation length ~1/(2*omega_MA).
    """
    print("=" * 60)
    print("DEMO 3: Energy Landscape")
    print("=" * 60)

    fig, axes = plt.subplots(1, len(embeddings), figsize=(7 * len(embeddings), 5))
    if len(embeddings) == 1:
        axes = [axes]

    for idx, (emb, A) in enumerate(zip(embeddings, weight_matrices)):
        omega_MA = emb.omega_MA

        # Compute E(p) = -x(p)^T A x(p) for many positions
        ps = torch.linspace(0.05, 0.95, 500)
        energies = []
        for p in ps:
            x = emb.encode(p)
            E = -(x @ A @ x).item()
            energies.append(E)

        energies = np.array(energies)
        # Normalize for visualization
        energies = (energies - energies.mean()) / energies.std()

        ax = axes[idx]
        ax.plot(ps.numpy(), energies, linewidth=0.8)
        ax.set_xlabel("Position p (m)")
        ax.set_ylabel("Energy E(p) (normalized)")
        ax.set_title(f"ω_MA = {omega_MA} m⁻¹\n"
                     f"Expected autocorr. length ≈ {1/(2*omega_MA)*1000:.1f} mm")

        # Mark expected autocorrelation scale
        scale = 1 / (2 * omega_MA)
        ax.axvline(x=0.5, color='r', alpha=0.3, linestyle='--')
        ax.axvline(x=0.5 + scale, color='r', alpha=0.3, linestyle='--')
        ax.annotate(f'1/(2ω_MA)', xy=(0.5 + scale/2, ax.get_ylim()[1]*0.9),
                   ha='center', color='red', fontsize=9)

    plt.suptitle("Energy landscape along line attractor (Eq. 12)", fontsize=14)
    plt.tight_layout()
    mlflow.log_figure(fig, "demo_energy_landscape.png")
    plt.close()

def demo_energy_trajectories(
    embeddings: list,
    weight_matrices: list,
    start_positions: list,
    noise_levels: list,
    n_steps: int,
    synaptic_noise: float,
    wta_fn=None,
):
    """
    Visualize how network state evolves on the energy landscape from different
    starting positions. For each omega_MA a separate figure is created with
    rows = noise levels and columns = starting positions.

    Each decoded position at every time step is overlaid as an opaque scatter
    point on the energy curve.
    """
    print("=" * 60)
    print("DEMO 4: Trajectories on Energy Landscape")
    print("=" * 60)

    ps_landscape = torch.linspace(0.05, 0.95, 500)

    for emb, A in zip(embeddings, weight_matrices):
        omega_MA = emb.omega_MA
        L = emb.L
        _wta = wta_fn if wta_fn is not None else (lambda h: block_wta(h, L))

        W = add_synaptic_noise(A, noise_scale=synaptic_noise)
        for m in range(emb.M):
            s, e = m * L, (m + 1) * L
            W[s:e, s:e] = 0.0

        # Compute energy landscape once
        energies = []
        for p in ps_landscape:
            x = emb.encode(p)
            energies.append(-(x @ A @ x).item())
        energies = np.array(energies)
        energies = (energies - energies.mean()) / energies.std()

        n_rows = len(noise_levels)
        n_cols = len(start_positions)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(4 * n_cols, 3 * n_rows),
            sharex=True, sharey=True,
        )
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes[np.newaxis, :]
        elif n_cols == 1:
            axes = axes[:, np.newaxis]

        print(f"\n  omega_MA = {omega_MA} ...")
        for row, noise in enumerate(noise_levels):
            print(f"    noise = {noise} ...")
            for col, p0 in enumerate(start_positions):
                ax = axes[row, col]

                ax.plot(ps_landscape.numpy(), energies, color="steelblue",
                        linewidth=1.2, zorder=1)

                z = emb.encode(torch.tensor(float(p0)))
                decoded = [emb.decode(z, (0, 1)).item()]
                for _ in range(n_steps):
                    h = W @ z
                    z = _wta(h)
                    z = add_neural_noise(z, L, noise)
                    decoded.append(emb.decode(z, (0, 1)).item())

                p_arr = np.array(decoded)
                e_arr = np.interp(p_arr, ps_landscape.numpy(), energies)

                ax.scatter(p_arr, e_arr, s=18, color="tomato", alpha=0.3, zorder=2)
                ax.axvline(x=float(p0), color="green", linewidth=0.8,
                           linestyle="--", alpha=0.6)

                if row == 0:
                    ax.set_title(f"p₀ = {p0:.2f}", fontsize=10)
                if col == 0:
                    ax.set_ylabel(f"noise = {noise}\nE(p) (norm.)", fontsize=9)
                if row == n_rows - 1:
                    ax.set_xlabel("Position p (m)")

        fig.suptitle(
            f"Trajectories on energy landscape — ω_MA = {omega_MA} m⁻¹",
            fontsize=12,
        )
        plt.tight_layout()
        mlflow.log_figure(fig, f"demo_energy_trajectories_omega{omega_MA}.png")
        plt.close()

# ============================================================================
# Main
# ============================================================================

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("can")

    print("Cotteret et al. (2025) - Grid Cell CAN Reference Implementation")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    c = cfg.can

    # Build embeddings and weight matrices once, shared across demos
    emb_demo1 = SpatialEmbedding(c.N, c.L, c.embedding.omega_MA, freq_dist="gaussian")
    embeddings = [
        SpatialEmbedding(c.N, c.L, omega_MA, freq_dist="gaussian")
        for omega_MA in c.stability.omega_MAs
    ]
    use_block_wta = OmegaConf.select(c, "wta", default="block") == "block"
    zero_within_block = OmegaConf.select(c, "zero_within_block", default=True)

    print("Building autoassociative weight matrices...")
    weight_matrices = [
        build_autoassociative_weights(emb, n_steps=300, zero_within_block=zero_within_block)
        for emb in embeddings
    ]

    with mlflow.start_run():
        mlflow.log_text(OmegaConf.to_yaml(cfg.can, resolve=True), "config.yaml")
        demo_embedding(emb_demo1)
        if use_block_wta:
            wta_fn = None  # demo will default to block_wta per embedding
        else:
            k = OmegaConf.select(c, "k", default=None)
            if k is None:
                k = c.N // c.L
            wta_fn = lambda h: global_kwta(h, k)  # noqa: E731

        demo_attractor_stability(
            embeddings,
            weight_matrices,
            list(c.stability.noise_levels),
            c.stability.n_steps,
            c.stability.n_init_positions,
            c.stability.synaptic_noise,
            wta_fn=wta_fn,
        )
        demo_energy_landscape(embeddings, weight_matrices)

        start_positions = list(OmegaConf.select(
            c, "trajectories.start_positions",
            default=[0.1, 0.3, 0.5, 0.7, 0.9],
        ))
        traj_omega_MAs = set(OmegaConf.select(
            c, "trajectories.omega_MAs",
            default=list(c.stability.omega_MAs),
        ))
        traj_embeddings = [emb for emb in embeddings if emb.omega_MA in traj_omega_MAs]
        traj_weight_matrices = [A for emb, A in zip(embeddings, weight_matrices)
                                 if emb.omega_MA in traj_omega_MAs]
        demo_energy_trajectories(
            traj_embeddings,
            traj_weight_matrices,
            start_positions=start_positions,
            noise_levels=list(OmegaConf.select(
                c, "trajectories.noise_levels",
                default=list(c.stability.noise_levels),
            )),
            n_steps=c.stability.n_steps,
            synaptic_noise=c.stability.synaptic_noise,
            wta_fn=wta_fn,
        )

    print("\n" + "=" * 60)
    print("All demos complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
