"""
CAN with correct on-manifold energy gradient descent.

The on-manifold energy is:
    E(p) = -1/2 x(p)^T A x(p)

Its gradient w.r.t. position p is:
    dE/dp = -x(p)^T A (dx/dp)

So the correct gradient descent dynamics are:
    dp/dt = -dE/dp = x(p)^T A (dx/dp)

This is guaranteed to decrease E(p) at every step (for small enough alpha),
and fixed points are exactly the local minima of E(p).
"""

import hydra
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from omegaconf import DictConfig, OmegaConf


# ============================================================================
# 1. Embedding
# ============================================================================

def make_embedding(N=2048, L=8, omega_MA=48.0, seed=42):
    rng = np.random.default_rng(seed)
    M = N // L
    sigma_omega = np.sqrt(np.pi / 2) * omega_MA
    omegas_per_block = rng.normal(0, sigma_omega, size=M)
    omega = np.repeat(omegas_per_block, L)
    theta = rng.uniform(0, 2 * np.pi, size=N)
    thresh = np.cos(np.pi / L)
    return omega, theta, thresh, N, L, M


def encode(p, omega, theta, thresh):
    p = np.atleast_1d(p)
    phase = p[:, None] * omega[None, :] + theta[None, :]
    return np.maximum(0.0, np.cos(phase) - thresh)


def encode_with_grad(p, omega, theta, thresh):
    p = float(p)
    phase = omega * p + theta
    cos_val = np.cos(phase)
    active = cos_val > thresh
    x = np.maximum(0.0, cos_val - thresh)
    dx_dp = np.zeros_like(x)
    dx_dp[active] = -omega[active] * np.sin(phase[active])
    return x, dx_dp


def decode(z, omega, theta, thresh, n_candidates=1000):
    ps = np.linspace(0.0, 1.0, n_candidates)
    xs = encode(ps, omega, theta, thresh)
    z_norm = z / (np.linalg.norm(z) + 1e-8)
    x_norms = xs / (np.linalg.norm(xs, axis=1, keepdims=True) + 1e-8)
    return ps[np.argmax(x_norms @ z_norm)]


# ============================================================================
# 2. Weight matrix
# ============================================================================

def build_A(omega, theta, thresh, N, n_steps=400):
    dp = 1.0 / n_steps
    A = np.zeros((N, N))
    for i in range(n_steps):
        p = (i + 0.5) * dp
        x = encode(np.array([p]), omega, theta, thresh)[0]
        x_bar = x - x.mean()
        A += np.outer(x_bar, x_bar) * dp
    np.fill_diagonal(A, 0)
    return A


# ============================================================================
# 3. On-manifold gradient dynamics
# ============================================================================

def energy(p, A, omega, theta, thresh):
    x = encode(np.array([p]), omega, theta, thresh)[0]
    return -0.5 * x @ A @ x


def energy_gradient(p, A, omega, theta, thresh):
    x, dx_dp = encode_with_grad(p, omega, theta, thresh)
    return -(x @ A @ dx_dp)


def step_gradient(p, A, omega, theta, thresh, alpha=1.0, noise_std=0.0, rng=None):
    rng = rng or np.random.default_rng()
    dE_dp = energy_gradient(p, A, omega, theta, thresh)
    dp = -alpha * dE_dp
    if noise_std > 0:
        dp += rng.normal(0, noise_std)
    return np.clip(p + dp, 0.001, 0.999)


# ============================================================================
# 4. Demos
# ============================================================================

def demo_embedding(omega, theta, thresh, omega_MA):
    print("=" * 60)
    print("DEMO 1: Spatial Embedding and Similarity Kernel")
    print("=" * 60)

    ps = np.linspace(0, 1, 200)
    xs = encode(ps, omega, theta, thresh)  # (200, N)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Similarity kernel K(Δp)
    p_ref = 0.5
    x_ref = encode(np.array([p_ref]), omega, theta, thresh)[0]
    deltas = np.linspace(-0.15, 0.15, 300)
    similarities = []
    for dp in deltas:
        x_dp = encode(np.array([p_ref + dp]), omega, theta, thresh)[0]
        sim = (x_ref * x_dp).sum() / (np.linalg.norm(x_ref) * np.linalg.norm(x_dp) + 1e-8)
        similarities.append(sim)
    axes[0].plot(deltas, similarities)
    axes[0].set_xlabel("Δp (m)")
    axes[0].set_ylabel("Cosine similarity K(Δp)")
    axes[0].set_title(f"Similarity kernel (ω_MA={omega_MA})")

    # (b) State matrix (positions × neurons, first 64)
    axes[1].imshow(xs[:, :64].T, aspect='auto', cmap='hot', extent=[0, 1, 64, 0])
    axes[1].set_xlabel("Position p (m)")
    axes[1].set_ylabel("Neuron index (first 64)")
    axes[1].set_title("Real-valued state vectors x(p)")

    # (c) Mean activity
    axes[2].plot(ps, xs.mean(axis=1))
    axes[2].set_xlabel("Position p (m)")
    axes[2].set_ylabel("Mean activity")
    axes[2].set_title("Mean activity per position")

    plt.tight_layout()
    mlflow.log_figure(fig, "embedding.png")
    plt.close()


def demo_energy_landscape(omega, theta, thresh, A):
    print("=" * 60)
    print("DEMO 2: Energy Landscape and Gradient")
    print("=" * 60)

    ps_dense = np.linspace(0.01, 0.99, 1000)
    E_landscape = np.array([energy(p, A, omega, theta, thresh) for p in ps_dense])
    dE_landscape = np.array([energy_gradient(p, A, omega, theta, thresh) for p in ps_dense])

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    axes[0].plot(ps_dense, E_landscape, linewidth=1)
    axes[0].set_ylabel("E(p) = -½ x(p)ᵀ A x(p)")
    axes[0].set_title("On-manifold energy landscape")
    axes[1].plot(ps_dense, dE_landscape, linewidth=1, color='tab:orange')
    axes[1].axhline(0, color='gray', linestyle='--', alpha=0.3)
    axes[1].set_ylabel("dE/dp")
    axes[1].set_xlabel("Position p (m)")
    axes[1].set_title("Energy gradient (zeros with positive-to-negative crossing = stable minima)")
    plt.tight_layout()
    mlflow.log_figure(fig, "energy_and_gradient.png")
    plt.close()

    return ps_dense, E_landscape


def demo_alpha_sweep(omega, theta, thresh, A, alphas, n_steps, n_init):
    print("=" * 60)
    print("DEMO 3: Alpha Sweep")
    print("=" * 60)

    init_positions = np.linspace(0.05, 0.95, n_init)

    fig, axes = plt.subplots(len(alphas), 3, figsize=(18, 3.5 * len(alphas)))

    for row, alpha in enumerate(alphas):
        print(f"  alpha = {alpha} ...")
        for p0 in init_positions:
            p = p0
            positions = [p]
            energies = [energy(p, A, omega, theta, thresh)]
            grads = []
            for _ in range(n_steps):
                grads.append(abs(energy_gradient(p, A, omega, theta, thresh)))
                p = step_gradient(p, A, omega, theta, thresh, alpha=alpha)
                positions.append(p)
                energies.append(energy(p, A, omega, theta, thresh))
            axes[row, 0].plot(positions, alpha=0.5, linewidth=0.8)
            axes[row, 1].plot(energies, alpha=0.5, linewidth=0.8)
            axes[row, 2].plot(grads, alpha=0.5, linewidth=0.8)

        axes[row, 0].set_ylabel(f"α={alpha}")
        axes[row, 0].set_ylim(-0.05, 1.05)
        if row == 0:
            axes[row, 0].set_title("Position p(t)")
            axes[row, 1].set_title("Energy E(p) — must decrease")
            axes[row, 2].set_title("|dE/dp| — must → 0")

    for col in range(3):
        axes[-1, col].set_xlabel("Time step")

    fig.suptitle("On-manifold gradient descent: varying α", fontsize=13)
    plt.tight_layout()
    mlflow.log_figure(fig, "alpha_sweep.png")
    plt.close()


def demo_noise(omega, theta, thresh, A, alpha, noise_levels, n_steps, n_init):
    print("=" * 60)
    print("DEMO 4: Noise Robustness")
    print("=" * 60)

    init_positions = np.linspace(0.05, 0.95, n_init)

    fig, axes = plt.subplots(len(noise_levels), 2, figsize=(14, 3.5 * len(noise_levels)))

    for row, noise in enumerate(noise_levels):
        print(f"  noise = {noise} ...")
        rng = np.random.default_rng(42)
        for p0 in init_positions:
            p = p0
            positions = [p]
            energies = [energy(p, A, omega, theta, thresh)]
            for _ in range(n_steps):
                p = step_gradient(p, A, omega, theta, thresh, alpha=alpha,
                                  noise_std=noise, rng=rng)
                positions.append(p)
                energies.append(energy(p, A, omega, theta, thresh))
            axes[row, 0].plot(positions, alpha=0.5, linewidth=0.8)
            axes[row, 1].plot(energies, alpha=0.5, linewidth=0.8)

        axes[row, 0].set_ylabel(f"noise={noise}")
        axes[row, 0].set_ylim(-0.05, 1.05)
        if row == 0:
            axes[row, 0].set_title("Position p(t)")
            axes[row, 1].set_title("Energy E(p)")

    for col in range(2):
        axes[-1, col].set_xlabel("Time step")

    fig.suptitle(f"Noise robustness (α={alpha})", fontsize=13)
    plt.tight_layout()
    mlflow.log_figure(fig, "noise_robustness.png")
    plt.close()


def demo_trajectories(omega, theta, thresh, A, ps_dense, E_landscape,
                      start_positions, noise_levels, alpha, n_steps, omega_MA=None):
    """
    Show individual trajectories overlaid on the energy landscape.
    Rows = noise levels, columns = starting positions.
    Each decoded position at every step is scattered onto the energy curve.
    """
    print("=" * 60)
    print("DEMO 5: Trajectories on Energy Landscape")
    print("=" * 60)

    if omega_MA is not None:
        omega, theta, thresh, N, _, _ = make_embedding(omega_MA=omega_MA)
        A = build_A(omega, theta, thresh, N, n_steps=len(ps_dense))
        E_landscape = np.array([energy(p, A, omega, theta, thresh) for p in ps_dense])

    E_norm = (E_landscape - E_landscape.mean()) / E_landscape.std()

    n_rows = len(noise_levels)
    n_cols = len(start_positions)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 3 * n_rows),
                             sharex=True, sharey=True)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for row, noise in enumerate(noise_levels):
        print(f"  noise = {noise} ...")
        for col, p0 in enumerate(start_positions):
            ax = axes[row, col]
            ax.plot(ps_dense, E_norm, color='steelblue', linewidth=1.2, zorder=1)

            rng = np.random.default_rng(42)
            p = float(p0)
            trajectory = [p]
            for _ in range(n_steps):
                p = step_gradient(p, A, omega, theta, thresh, alpha=alpha,
                                  noise_std=noise, rng=rng)
                trajectory.append(p)

            p_arr = np.array(trajectory)
            e_arr = np.interp(p_arr, ps_dense, E_norm)
            ax.scatter(p_arr, e_arr, s=18, color='tomato', alpha=0.3, zorder=2)
            ax.axvline(x=p0, color='green', linewidth=0.8, linestyle='--', alpha=0.6)

            if row == 0:
                ax.set_title(f"p₀ = {p0:.2f}", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"noise = {noise}\nE(p) (norm.)", fontsize=9)
            if row == n_rows - 1:
                ax.set_xlabel("Position p (m)")

    title_om = f", ω_MA={omega_MA}" if omega_MA is not None else ""
    fig.suptitle(f"Trajectories on energy landscape (α={alpha}{title_om})", fontsize=12)
    plt.tight_layout()
    mlflow.log_figure(fig, "trajectories.png")
    plt.close()


def demo_omega_sweep(omega_MAs, alpha, noise_std, n_steps, n_init, n_weight_steps):
    print("=" * 60)
    print("DEMO 6: omega_MA Sweep")
    print("=" * 60)

    ps_dense = np.linspace(0.01, 0.99, 1000)
    init_positions = np.linspace(0.05, 0.95, n_init)

    fig, axes = plt.subplots(len(omega_MAs), 2, figsize=(16, 4 * len(omega_MAs)),
                             gridspec_kw={"width_ratios": [1, 5]})

    for row, om in enumerate(omega_MAs):
        print(f"  omega_MA = {om} ...")
        o, t, th, oN, _, _ = make_embedding(omega_MA=om)
        oA = build_A(o, t, th, oN, n_steps=n_weight_steps)

        E_land = np.array([energy(p, oA, o, t, th) for p in ps_dense])
        E_norm = (E_land - E_land.mean()) / (E_land.std() + 1e-10)
        axes[row, 0].plot(E_norm, ps_dense, linewidth=0.8)
        axes[row, 0].set_ylim(-0.05, 1.05)
        axes[row, 0].set_ylabel(f"ω_MA={om}\nPosition p (m)")
        for tick in axes[row, 0].get_yticks():
            axes[row, 0].axhline(tick, color='gray', linewidth=0.4, alpha=0.5, linestyle=(0, (5, 10)), zorder=0)
        if row == 0:
            axes[row, 0].set_title("Energy landscape (normalized)")

        rng = np.random.default_rng(42)
        for p0 in init_positions:
            p = p0
            positions = [p]
            for _ in range(n_steps):
                p = step_gradient(p, oA, o, t, th, alpha=alpha,
                                  noise_std=noise_std, rng=rng)
                positions.append(p)
            axes[row, 1].plot(positions, alpha=0.5, linewidth=0.8)
        axes[row, 1].set_ylim(-0.05, 1.05)
        for tick in axes[row, 1].get_yticks():
            axes[row, 1].axhline(tick, color='gray', linewidth=0.4, alpha=0.5, linestyle=(0, (5, 10)), zorder=0)
        if row == 0:
            axes[row, 1].set_title(f"Position trajectories (noise={noise_std})")

    axes[-1, 0].set_xlabel("E(p) (normalized)")
    axes[-1, 1].set_xlabel("Time step")

    fig.suptitle("Resolution vs ω_MA: higher frequency → more minima → tighter pinning",
                 fontsize=13)
    plt.tight_layout()
    mlflow.log_figure(fig, "omega_sweep.png")
    plt.close()


# ============================================================================
# Main
# ============================================================================

@hydra.main(version_base=None, config_path="config", config_name="can_real_manifold")
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("can_real_manifold")

    print("On-Manifold Gradient Descent CAN")
    print("=" * 60)

    np.random.seed(42)

    print("Building embedding and weight matrix...")
    omega, theta, thresh, N, _, _ = make_embedding(
        N=cfg.N, L=cfg.L, omega_MA=cfg.omega_MA
    )
    A = build_A(omega, theta, thresh, N, n_steps=cfg.n_weight_steps)

    with mlflow.start_run():
        mlflow.log_text(OmegaConf.to_yaml(cfg, resolve=True), "config.yaml")

        demo_embedding(omega, theta, thresh, omega_MA=cfg.omega_MA)
        ps_dense, E_landscape = demo_energy_landscape(omega, theta, thresh, A)

        demo_alpha_sweep(
            omega, theta, thresh, A,
            alphas=list(cfg.alpha_sweep.alphas),
            n_steps=cfg.alpha_sweep.n_steps,
            n_init=cfg.alpha_sweep.n_init,
        )
        demo_noise(
            omega, theta, thresh, A,
            alpha=cfg.noise.alpha,
            noise_levels=list(cfg.noise.noise_levels),
            n_steps=cfg.noise.n_steps,
            n_init=cfg.noise.n_init,
        )
        demo_trajectories(
            omega, theta, thresh, A,
            ps_dense, E_landscape,
            start_positions=list(cfg.trajectories.start_positions),
            noise_levels=list(cfg.trajectories.noise_levels),
            alpha=cfg.trajectories.alpha,
            n_steps=cfg.trajectories.n_steps,
            omega_MA=cfg.trajectories.omega_MA,
        )
        demo_omega_sweep(
            omega_MAs=list(cfg.omega_sweep.omega_MAs),
            alpha=cfg.omega_sweep.alpha,
            noise_std=cfg.omega_sweep.noise_std,
            n_steps=cfg.omega_sweep.n_steps,
            n_init=cfg.omega_sweep.n_init,
            n_weight_steps=cfg.n_weight_steps,
        )

    print("\n" + "=" * 60)
    print("All demos complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
