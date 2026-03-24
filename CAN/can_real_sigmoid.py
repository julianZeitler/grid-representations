"""
1D line attractor CAN with sigmoid activation (Hopfield 1984).

Replaces softmax with element-wise sigmoid, using the correct energy
function from Hopfield (1984) "Neurons with graded response":

    E = -1/2 * sum_{i,j} T_ij V_i V_j
        + sum_i (1/R_i) * integral_0^{V_i} g^{-1}(V) dV
        - sum_i I_i V_i

For g(u) = sigmoid(lambda * u), the integral term becomes:
    (1/lambda) * [V log V + (1-V) log(1-V)]
which is (1/lambda) * negative binary entropy per neuron.

The dynamics follow the RC charging equation (Eq. 5):
    C du/dt = sum_j T_ij V_j - u/R + I
    V = g(u) = sigmoid(lambda * u)

Key insight from Hopfield 1984: lambda (gain) plays the role analogous
to inverse temperature. High lambda -> binary-like states (corners of
hypercube). Low lambda -> smooth states near V=0.5 (center of hypercube).
"""

import hydra
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from omegaconf import DictConfig, OmegaConf


# ============================================================================
# 1. Embedding (cosine tuning curves)
# ============================================================================

def make_embedding(N=2048, L=8, omega_MA=48.0, seed=42):
    """Real-valued cosine tuning curve embedding."""
    rng = np.random.default_rng(seed)
    M = N // L
    sigma_omega = np.sqrt(np.pi / 2) * omega_MA
    omegas_per_block = rng.normal(0, sigma_omega, size=M)
    omega = np.repeat(omegas_per_block, L)
    theta = rng.uniform(0, 2 * np.pi, size=N)
    threshold = np.cos(np.pi / L)
    return omega, theta, threshold, N, L, M


def encode(p, omega, theta, threshold):
    """Encode position(s) with cosine tuning curves: max(0, cos(w*p+t) - threshold)."""
    p = np.atleast_1d(p)
    phase = p[:, None] * omega[None, :] + theta[None, :]
    x = np.maximum(0.0, np.cos(phase) - threshold)
    return x


def decode(z, omega, theta, threshold, p_range=(0.0, 1.0), n_candidates=1000):
    """Decode by maximum cosine similarity."""
    ps = np.linspace(p_range[0], p_range[1], n_candidates)
    xs = encode(ps, omega, theta, threshold)
    z_norm = z / (np.linalg.norm(z) + 1e-8)
    x_norms = xs / (np.linalg.norm(xs, axis=1, keepdims=True) + 1e-8)
    sims = x_norms @ z_norm
    return ps[np.argmax(sims)]


# ============================================================================
# 2. Weight matrix (Hebbian covariance)
# ============================================================================

def build_A(omega, theta, threshold, N, p_min=0.0, p_max=1.0, n_steps=400):
    """Autoassociative weights, diagonal zeroed (no self-connections)."""
    dp = (p_max - p_min) / n_steps
    A = np.zeros((N, N))
    for i in range(n_steps):
        p = p_min + (i + 0.5) * dp
        x = encode(np.array([p]), omega, theta, threshold)[0]
        x_bar = x - x.mean()
        A += np.outer(x_bar, x_bar) * dp
    np.fill_diagonal(A, 0)
    return A


# ============================================================================
# 3. Sigmoid activation and Hopfield 1984 energy
# ============================================================================

def sigmoid(u, lam=1.0):
    """Sigmoid activation g(u) = 1 / (1 + exp(-lambda*u))."""
    x = lam * u
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def sigmoid_inv(V, lam=1.0):
    """Inverse sigmoid: g^{-1}(V) = (1/lambda) * log(V / (1-V))."""
    V_safe = np.clip(V, 1e-10, 1 - 1e-10)
    return (1.0 / lam) * np.log(V_safe / (1 - V_safe))


def binary_entropy(V):
    """
    Per-neuron binary entropy: V log V + (1-V) log(1-V).
    Always <= 0; minimum at V=0.5, maximum (0) at V=0 or V=1.
    """
    V_safe = np.clip(V, 1e-10, 1 - 1e-10)
    return V_safe * np.log(V_safe) + (1 - V_safe) * np.log(1 - V_safe)


def energy_hopfield84(V, A, lam, R=1.0, I_ext=None):
    """
    Hopfield 1984 energy (Eq. 7/11):

        E = -1/2 V^T A V + (1/(R*lambda)) * sum_i binary_entropy(V_i) - I^T V

    For sigmoid g(u) = sigmoid(lambda*u), the regularizer integral evaluates to
    (1/lambda) * binary_entropy, up to a constant.
    """
    E_interact = -0.5 * V @ A @ V
    E_reg = (1.0 / (R * lam)) * np.sum(binary_entropy(V))
    E_input = 0.0 if I_ext is None else -np.sum(I_ext * V)
    return E_interact + E_reg + E_input


# ============================================================================
# 4. RC dynamics (Eq. 5 from Hopfield 1984)
# ============================================================================

def step_rc_dynamics(V, A, lam, R=1.0, C=1.0, dt=0.1, I_ext=None,
                     noise_std=0.0, rng=None):
    """
    One Euler step of the Hopfield 1984 RC dynamics:

        C du/dt = A V - u/R + I
        V = sigmoid(lambda * u)
    """
    rng = rng or np.random.default_rng()
    u = sigmoid_inv(V, lam)
    h = A @ V
    I = I_ext if I_ext is not None else 0.0
    du_dt = (h - u / R + I) / C
    u_new = u + dt * du_dt
    if noise_std > 0:
        u_new += rng.normal(0, noise_std, size=u_new.shape)
    return sigmoid(u_new, lam), u_new


# ============================================================================
# 5. Demos
# ============================================================================

def demo_energy_comparison(omega, theta, threshold, N, A, lambdas, R):
    print("=" * 60)
    print("DEMO 1: Energy Decomposition")
    print("=" * 60)

    ps = np.linspace(0.05, 0.95, 500)
    xs = encode(ps, omega, theta, threshold)

    # --- Figure 1: per-lambda decomposition rows ---
    fig, axes = plt.subplots(3, len(lambdas), figsize=(4 * len(lambdas), 10),
                             sharey='row')

    for col, lam in enumerate(lambdas):
        E_interact = np.array([-0.5 * x @ A @ x for x in xs])
        E_reg = np.array([(1.0 / (R * lam)) * np.sum(binary_entropy(x)) for x in xs])
        E_total = E_interact + E_reg

        axes[0, col].plot(ps, E_interact, linewidth=1)
        axes[0, col].set_title(f"λ = {lam}", fontsize=10)
        if col == 0:
            axes[0, col].set_ylabel("−½ VᵀAV\n(interaction)")

        axes[1, col].plot(ps, E_reg, linewidth=1, color='tab:orange')
        if col == 0:
            axes[1, col].set_ylabel("(1/Rλ) Σ H(Vᵢ)\n(regularizer)")

        axes[2, col].plot(ps, E_total, linewidth=1, color='tab:green')
        axes[2, col].set_xlabel("Position p (m)")
        if col == 0:
            axes[2, col].set_ylabel("Total E\n(Hopfield 1984)")

    fig.suptitle(
        "Hopfield 1984 energy decomposition along attractor manifold\n"
        "E = −½VᵀAV + (1/Rλ)·Σ[Vᵢ log Vᵢ + (1−Vᵢ) log(1−Vᵢ)]",
        fontsize=12)
    plt.tight_layout()
    mlflow.log_figure(fig, "energy_decomposition.png")
    plt.close()

    # --- Figure 2: overlay all lambdas ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for lam in lambdas:
        E_interact = np.array([-0.5 * x @ A @ x for x in xs])
        E_reg = np.array([(1.0 / (R * lam)) * np.sum(binary_entropy(x)) for x in xs])
        E_total = E_interact + E_reg
        axes[0].plot(ps, E_interact, linewidth=1, label=f"λ={lam}")
        axes[1].plot(ps, E_reg, linewidth=1, label=f"λ={lam}")
        axes[2].plot(ps, E_total, linewidth=1, label=f"λ={lam}")

    axes[0].set_title("Interaction −½VᵀAV")
    axes[1].set_title("Regularizer (1/Rλ)·Σ H(Vᵢ)")
    axes[2].set_title("Total energy")
    for ax in axes:
        ax.set_xlabel("Position p (m)")
        ax.set_ylabel("Energy")
        ax.legend(fontsize=7)

    plt.tight_layout()
    mlflow.log_figure(fig, "energy_overlay.png")
    plt.close()

    # --- Figure 3: barrier height vs lambda ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    lam_range = np.linspace(0.3, 50, 100)

    barrier_interact, barrier_reg, barrier_total = [], [], []
    for lam in lam_range:
        E_i = np.array([-0.5 * x @ A @ x for x in xs])
        E_r = np.array([(1.0 / (R * lam)) * np.sum(binary_entropy(x)) for x in xs])
        E_t = E_i + E_r
        barrier_interact.append(E_i.std())
        barrier_reg.append(E_r.std())
        barrier_total.append(E_t.std())

    axes[0].plot(lam_range, barrier_interact, linewidth=2)
    axes[0].set_title("Interaction barrier height")
    axes[1].plot(lam_range, barrier_reg, linewidth=2, color='tab:orange')
    axes[1].set_title("Regularizer barrier height")
    axes[2].plot(lam_range, barrier_total, linewidth=2, color='tab:green')
    axes[2].set_title("Total energy barrier height")
    for ax in axes:
        ax.set_xlabel("Gain λ")
        ax.set_ylabel("std(E) along manifold")

    plt.suptitle("Energy barrier height vs sigmoid gain λ", fontsize=13)
    plt.tight_layout()
    mlflow.log_figure(fig, "energy_barriers.png")
    plt.close()


def demo_dynamics(omega, theta, threshold, N, A, lambdas, R, C, dt,
                  n_steps, noise_std, n_init):
    print("=" * 60)
    print("DEMO 2: RC Dynamics")
    print("=" * 60)

    rng = np.random.default_rng(42)
    init_positions = np.linspace(0.1, 0.9, n_init)

    fig, axes = plt.subplots(len(lambdas), 3, figsize=(18, 4 * len(lambdas)))

    for row, lam in enumerate(lambdas):
        print(f"  λ = {lam}...")

        for p0 in init_positions:
            x0 = encode(np.array([p0]), omega, theta, threshold)[0]
            V = np.clip(x0 / (x0.max() + 1e-8) * 0.8 + 0.1, 0.01, 0.99)

            positions = [decode(V, omega, theta, threshold)]
            energies = [energy_hopfield84(V, A, lam, R)]
            sparsities = [np.mean(V > 0.1)]

            for t in range(n_steps):
                V, _ = step_rc_dynamics(V, A, lam, R=R, C=C, dt=dt,
                                        noise_std=noise_std, rng=rng)
                if t % 5 == 0:
                    positions.append(decode(V, omega, theta, threshold))
                    energies.append(energy_hopfield84(V, A, lam, R))
                    sparsities.append(np.mean(V > 0.1))

            times = np.arange(len(positions)) * 5
            axes[row, 0].plot(times, positions, alpha=0.6, linewidth=0.8)
            axes[row, 1].plot(times, energies, alpha=0.6, linewidth=0.8)
            axes[row, 2].plot(times, sparsities, alpha=0.6, linewidth=0.8)

        axes[row, 0].set_ylabel(f"λ={lam}\nDecoded pos")
        axes[row, 0].set_ylim(-0.05, 1.05)
        axes[row, 1].set_ylabel("E (Hopfield 84)")
        axes[row, 2].set_ylabel("Frac active (>0.1)")
        axes[row, 2].set_ylim(0, 1)

        if row == 0:
            axes[row, 0].set_title("Position trajectory")
            axes[row, 1].set_title("Energy (should decrease)")
            axes[row, 2].set_title("Activity level")

    for col in range(3):
        axes[-1, col].set_xlabel("Time step")

    plt.suptitle("Sigmoid RC dynamics (Hopfield 1984) — attractor stability", fontsize=14)
    plt.tight_layout()
    mlflow.log_figure(fig, "dynamics.png")
    plt.close()


def demo_sigmoid_vs_softmax():
    print("=" * 60)
    print("DEMO 3: Sigmoid vs Softmax Regularization")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    V = np.linspace(0.01, 0.99, 500)
    softmax_term = V * np.log(V)
    sigmoid_term = V * np.log(V) + (1 - V) * np.log(1 - V)

    axes[0].plot(V, softmax_term, linewidth=2, label="Softmax: V log V")
    axes[0].plot(V, sigmoid_term, linewidth=2,
                 label="Sigmoid: V log V + (1−V) log(1−V)")
    axes[0].set_xlabel("Neuron activation V")
    axes[0].set_ylabel("Regularization penalty")
    axes[0].set_title("Per-neuron regularization term")
    axes[0].legend()
    axes[0].axhline(0, color='gray', linestyle='--', alpha=0.3)

    d_softmax = np.log(V) + 1
    d_sigmoid = np.log(V) - np.log(1 - V)
    axes[1].plot(V, d_softmax, linewidth=2, label="d/dV softmax term = log V + 1")
    axes[1].plot(V, d_sigmoid, linewidth=2,
                 label="d/dV sigmoid term = log(V/(1−V))")
    axes[1].set_xlabel("Neuron activation V")
    axes[1].set_ylabel("Gradient (force on V)")
    axes[1].set_title("Gradient of regularization\n(positive = pushes V higher)")
    axes[1].legend(fontsize=8)
    axes[1].axhline(0, color='gray', linestyle='--', alpha=0.3)
    axes[1].axvline(0.5, color='gray', linestyle=':', alpha=0.3)
    axes[1].set_ylim(-5, 5)

    plt.tight_layout()
    mlflow.log_figure(fig, "sigmoid_vs_softmax.png")
    plt.close()


# ============================================================================
# Main
# ============================================================================

@hydra.main(version_base=None, config_path="config", config_name="can_sigmoid")
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("can_real_sigmoid")

    print("CAN with Sigmoid Activation (Hopfield 1984)")
    print("=" * 60)

    np.random.seed(42)

    print("Building embedding and weight matrix...")
    omega, theta, threshold, N, L, M = make_embedding(
        N=cfg.N, L=cfg.L, omega_MA=cfg.omega_MA
    )
    A = build_A(omega, theta, threshold, N, n_steps=cfg.n_weight_steps)

    with mlflow.start_run():
        mlflow.log_text(OmegaConf.to_yaml(cfg, resolve=True), "config.yaml")

        demo_sigmoid_vs_softmax()
        demo_energy_comparison(
            omega, theta, threshold, N, A,
            lambdas=list(cfg.energy.lambdas),
            R=cfg.energy.R,
        )
        demo_dynamics(
            omega, theta, threshold, N, A,
            lambdas=list(cfg.dynamics.lambdas),
            R=cfg.dynamics.R,
            C=cfg.dynamics.C,
            dt=cfg.dynamics.dt,
            n_steps=cfg.dynamics.n_steps,
            noise_std=cfg.dynamics.noise_std,
            n_init=cfg.dynamics.n_init,
        )

    print("\n" + "=" * 60)
    print("All demos complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
