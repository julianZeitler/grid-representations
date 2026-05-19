"""
CAN energy landscape: grid cells vs place cells (no RGM, no dynamics).

Workflow for each population type:
  1. Construct artificial cell representations z(x)
  2. Build auto-associative weight matrix
       A = (1/area) ∫ z̄(x) z̄(x)ᵀ dx   (diagonal zeroed)
  3. Compute energy landscape E(x) = -½ z(x)ᵀ A z(x)
  4. Compute 2-D Fourier transform of E(x)
  5. Compare grid vs place side-by-side

Grid cells: hexagonally-symmetric periodic tuning via sum-of-cosines,
            one or more modules with distinct spacings / orientations.
Place cells: Gaussian tuning curves centred at random locations.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from omegaconf import DictConfig, OmegaConf


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRACKING_URI = f"sqlite:///{os.path.join(_PROJECT_ROOT, 'mlruns.db')}"


# ============================================================================
# 1. Cell population factories
# ============================================================================

def make_grid_cell_fn(
    modules: list[dict],
    n_cells_total: int,
    rng: np.random.Generator,
) -> tuple[callable, int]:
    """
    Build a grid-cell population response function.

    Each cell fires as a sum of Gaussian bumps placed on a hexagonal lattice.
    Cells within a module share the same lattice geometry but are offset by a
    random vector drawn uniformly from the lattice unit cell.

    Module dict keys:
        lam         : lattice spacing (box units)
        orientation : lattice rotation in degrees
        sigma       : Gaussian bump width (default lam / 4)

    Returns:
        z_fn : callable x [2] -> rates [D]
        D    : number of cells
    """
    n_mod = len(modules)
    cells_per_mod = n_cells_total // n_mod
    D = cells_per_mod * n_mod

    module_data = []
    for mod in modules:
        lam = float(mod["lam"])
        ori = np.deg2rad(float(mod["orientation"]))
        sigma = float(mod.get("sigma", lam / 4.0))

        # Hexagonal lattice basis vectors
        a1 = lam * np.array([np.cos(ori),              np.sin(ori)])
        a2 = lam * np.array([np.cos(ori + np.pi / 3),  np.sin(ori + np.pi / 3)])

        # Precompute lattice points large enough to cover any query position
        # (we query inside [0, box] so ±n_range lattice steps always covers it)
        n_range = 12
        pts = np.array([
            i * a1 + j * a2
            for i in range(-n_range, n_range + 1)
            for j in range(-n_range, n_range + 1)
        ])  # [N_pts, 2]

        # Random offset per cell: uniform sample from the unit cell
        uv = rng.uniform(0.0, 1.0, size=(cells_per_mod, 2))
        offsets = uv[:, 0:1] * a1[None, :] + uv[:, 1:2] * a2[None, :]  # [C, 2]

        inv_2s2 = 1.0 / (2.0 * sigma ** 2)
        module_data.append((pts, offsets, inv_2s2))

    def z_fn(x: np.ndarray) -> np.ndarray:
        parts = []
        for pts, offsets, inv_2s2 in module_data:
            # shifted lattice centres for every cell: [C, N_pts, 2]
            centres = pts[None, :, :] + offsets[:, None, :]
            diff = centres - x[None, None, :]          # [C, N_pts, 2]
            d2 = np.sum(diff ** 2, axis=2)             # [C, N_pts]
            parts.append(np.sum(np.exp(-d2 * inv_2s2), axis=1))  # [C]
        return np.concatenate(parts)

    return z_fn, D


def make_place_cell_fn(
    n_cells: int,
    box_width: float,
    box_height: float,
    sigma: float,
    rng: np.random.Generator,
) -> tuple[callable, int]:
    """
    Build a place-cell population response function.

    Each cell fires as a Gaussian bump centred at a random location.

    Returns:
        z_fn : callable x [2] -> rates [D]
        D    : number of cells
    """
    centers = rng.uniform([0.0, 0.0], [box_width, box_height], size=(n_cells, 2))
    inv_2s2 = 1.0 / (2 * sigma ** 2)

    def z_fn(x: np.ndarray) -> np.ndarray:
        diff = centers - x[None, :]           # [D, 2]
        return np.exp(-np.sum(diff ** 2, axis=1) * inv_2s2)  # [D]

    return z_fn, n_cells


# ============================================================================
# 2. Weight matrix
# ============================================================================

def build_A(
    z_fn: callable,
    D: int,
    n_steps: int,
    box_width: float,
    box_height: float,
) -> np.ndarray:
    """
    Build autoassociative weight matrix via Riemann sum.

    A = (1/area) ∫ z̄(x) z̄(x)ᵀ dx   where  z̄(x) = z(x) - mean(z(x))
    Diagonal is zeroed.
    """
    dx = box_width / n_steps
    dy = box_height / n_steps
    x_vals = np.linspace(dx / 2, box_width - dx / 2, n_steps)
    y_vals = np.linspace(dy / 2, box_height - dy / 2, n_steps)

    A = np.zeros((D, D))
    for xi in x_vals:
        for yi in y_vals:
            z = z_fn(np.array([xi, yi]))
            z_bar = z - z.mean()
            A += np.outer(z_bar, z_bar) * dx * dy

    np.fill_diagonal(A, 0.0)
    return A


# ============================================================================
# 3. Energy landscape
# ============================================================================

def compute_energy_landscape(
    z_fn: callable,
    A: np.ndarray,
    n_vis: int,
    box_width: float,
    box_height: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate E(x) = -½ z(x)ᵀ A z(x) on a regular grid.

    Returns x_vals, y_vals, E_grid  (shapes [n_vis], [n_vis], [n_vis, n_vis]).
    """
    x_vals = np.linspace(0, box_width, n_vis)
    y_vals = np.linspace(0, box_height, n_vis)
    E_grid = np.zeros((n_vis, n_vis))
    for i, xi in enumerate(x_vals):
        for j, yi in enumerate(y_vals):
            z = z_fn(np.array([xi, yi]))
            E_grid[i, j] = -0.5 * z @ A @ z
    return x_vals, y_vals, E_grid


# ============================================================================
# 4. Fourier analysis
# ============================================================================

def fourier_power(
    E_grid: np.ndarray,
    box_width: float,
    box_height: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    2-D power spectrum of the energy landscape.

    Returns:
        fx, fy : frequency axes (cycles per box unit), each shape [n_vis]
        power  : log10 power spectrum, shape [n_vis, n_vis]
    """
    n_vis = E_grid.shape[0]
    dx = box_width / n_vis
    dy = box_height / n_vis

    F = np.fft.fft2(E_grid - E_grid.mean())   # remove DC
    F_shifted = np.fft.fftshift(F)
    power = np.log10(np.abs(F_shifted) ** 2 + 1e-12)

    fx = np.fft.fftshift(np.fft.fftfreq(n_vis, d=dx))
    fy = np.fft.fftshift(np.fft.fftfreq(n_vis, d=dy))

    return fx, fy, power


def compute_coverage(
    z_fn: callable,
    n_vis: int,
    box_width: float,
    box_height: float,
) -> np.ndarray:
    """
    Evaluate sum of all cell firing rates on a grid.

    Returns coverage map of shape [n_vis, n_vis].
    """
    x_vals = np.linspace(0, box_width, n_vis)
    y_vals = np.linspace(0, box_height, n_vis)
    coverage = np.zeros((n_vis, n_vis))
    for i, xi in enumerate(x_vals):
        for j, yi in enumerate(y_vals):
            coverage[i, j] = z_fn(np.array([xi, yi])).sum()
    return coverage


def compute_rate_maps(
    z_fn: callable,
    n_vis: int,
    box_width: float,
    box_height: float,
    cell_indices: list[int],
) -> list[np.ndarray]:
    """
    Evaluate individual cell firing rates on a grid.

    Returns a list of 2-D rate maps (one per index in cell_indices),
    each of shape [n_vis, n_vis].
    """
    x_vals = np.linspace(0, box_width, n_vis)
    y_vals = np.linspace(0, box_height, n_vis)
    maps = [np.zeros((n_vis, n_vis)) for _ in cell_indices]
    for i, xi in enumerate(x_vals):
        for j, yi in enumerate(y_vals):
            z = z_fn(np.array([xi, yi]))
            for k, idx in enumerate(cell_indices):
                maps[k][i, j] = z[idx]
    return maps


# ============================================================================
# 5. Plotting helpers
# ============================================================================

def plot_energy_and_fourier(
    label: str,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    E_grid: np.ndarray,
    fx: np.ndarray,
    fy: np.ndarray,
    power: np.ndarray,
    box_width: float,
    box_height: float,
) -> plt.Figure:
    """Two-panel figure: energy | 2-D spectrum."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # --- Energy landscape ---
    ax = axes[0]
    im = ax.imshow(
        E_grid.T, origin="lower",
        extent=(0, box_width, 0, box_height),
        cmap="RdBu_r",
    )
    ax.contour(
        x_vals, y_vals, E_grid.T, levels=20, colors="k", linewidths=0.4, alpha=0.35,
    )
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    plt.colorbar(im, ax=ax, label="E(x)")

    # --- 2-D power spectrum ---
    ax = axes[1]
    extent_f = (fx.min(), fx.max(), fy.min(), fy.max())
    vlim = np.abs(power).max()
    im2 = ax.imshow(power.T, origin="lower", extent=extent_f, cmap="RdBu_r", vmin=-vlim, vmax=vlim)
    ax.set_xlabel("spatial frequency f₁ (1/unit)", fontsize=20, labelpad=2)
    ax.set_ylabel("spatial frequency f₂ (1/unit)", fontsize=20, labelpad=2)
    plt.colorbar(im2, ax=ax, label="")
    ax.axhline(0, color="w", linewidth=0.4, alpha=0.5)
    ax.axvline(0, color="w", linewidth=0.4, alpha=0.5)

    plt.tight_layout(pad=0.3)
    return fig


def plot_comparison(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    E_grid_grid: np.ndarray,
    E_grid_place: np.ndarray,
    fx: np.ndarray,
    fy: np.ndarray,
    power_grid: np.ndarray,
    power_place: np.ndarray,
    rate_maps_grid: list[np.ndarray],
    rate_maps_place: list[np.ndarray],
    coverage_grid: np.ndarray,
    coverage_place: np.ndarray,
    box_width: float,
    box_height: float,
) -> plt.Figure:
    """
    2×4 comparison figure.

    Columns: [2×2 rate maps] | [coverage] | [energy landscape] | (spacer) | [2-D Fourier spectrum]
    Rows:    grid cells      | place cells
    """
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    # 5-col GridSpec with col 3 as a spacer to add extra gap before frequency plot
    fig = plt.figure(figsize=(19, 9))
    gs = GridSpec(2, 5, figure=fig, hspace=0.18, wspace=0.15,
                  left=0.04, right=0.96, top=0.94, bottom=0.08,
                  width_ratios=[0.91, 1, 1, 0.08, 1])

    rows = [
        ("Grid cells",  E_grid_grid,  power_grid,  rate_maps_grid,  coverage_grid),
        ("Place cells", E_grid_place, power_place, rate_maps_place, coverage_place),
    ]

    for row_idx, (label, E_grid, power, rate_maps, coverage) in enumerate(rows):

        # --- Col 0: 2×2 rate map mini-grid ---
        gs_inner = GridSpecFromSubplotSpec(
            2, 2, subplot_spec=gs[row_idx, 0], hspace=0.0, wspace=0.0
        )
        for k, rmap in enumerate(rate_maps):
            ax = fig.add_subplot(gs_inner[k // 2, k % 2])
            ax.imshow(rmap.T, origin="lower", cmap="viridis", aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("white")
                spine.set_linewidth(1.5)

        # --- Col 1: Coverage map ---
        ax_c = fig.add_subplot(gs[row_idx, 1])
        im_c = ax_c.imshow(
            coverage.T, origin="lower",
            extent=(0, box_width, 0, box_height),
            cmap="viridis",
        )
        plt.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.04).ax.tick_params(labelsize=18)
        ax_c.set_xticks([])
        ax_c.set_yticks([])
        if row_idx == 0:
            ax_c.set_title("Coverage", fontsize=20)

        # --- Col 2: Energy landscape ---
        ax_e = fig.add_subplot(gs[row_idx, 2])
        E_z = (E_grid - E_grid.mean()) / E_grid.std()
        im = ax_e.imshow(
            E_z.T, origin="lower",
            extent=(0, box_width, 0, box_height),
            cmap="viridis",
        )
        plt.colorbar(im, ax=ax_e, fraction=0.046, pad=0.04).ax.tick_params(labelsize=18)
        ax_e.set_xticks([])
        ax_e.set_yticks([])
        if row_idx == 0:
            ax_e.set_title("Energy landscape", fontsize=20)

        # --- Col 4: 2-D Fourier spectrum (col 3 is spacer) ---
        ax_f = fig.add_subplot(gs[row_idx, 4])
        extent_f = (fx.min(), fx.max(), fy.min(), fy.max())
        vlim = np.abs(power).max()
        im2 = ax_f.imshow(power.T, origin="lower", extent=extent_f, cmap="RdBu_r", vmin=-vlim, vmax=vlim)
        ax_f.axhline(0, color="w", linewidth=0.4, alpha=0.5)
        ax_f.axvline(0, color="w", linewidth=0.4, alpha=0.5)
        cb = plt.colorbar(im2, ax=ax_f, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=18)
        ax_f.tick_params(labelsize=18)
        ax_f.set_xlabel("f₁", fontsize=20, labelpad=2)
        ax_f.set_ylabel("f₂", fontsize=20, labelpad=-15)
        if row_idx == 0:
            ax_f.set_title("Power spectrum", fontsize=20)

    return fig


# ============================================================================
# 6. Main
# ============================================================================

@hydra.main(version_base=None, config_path="config", config_name="can_artificial_cells")
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri(_TRACKING_URI)
    mlflow.set_experiment("can_artificial_cells")

    box_w, box_h = cfg.box_width, cfg.box_height
    n_weight = cfg.n_weight_steps
    n_vis = cfg.n_vis_steps

    print("=" * 60)
    print("CAN: Artificial cells — grid vs place")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("=" * 60)

    # ---- Grid cells ----
    print("Building grid-cell population ...")
    gz_fn, D_grid = make_grid_cell_fn(
        modules=OmegaConf.to_container(cfg.grid.modules, resolve=True),
        n_cells_total=cfg.grid.n_cells,
        rng=np.random.default_rng(cfg.seed),
    )
    print(f"  D = {D_grid} cells")

    print("Building A (grid) ...")
    A_grid = build_A(gz_fn, D_grid, n_weight, box_w, box_h)
    print(f"  ||A||_F = {np.linalg.norm(A_grid):.4f}")

    print("Computing energy landscape (grid) ...")
    x_vals, y_vals, E_grid_g = compute_energy_landscape(gz_fn, A_grid, n_vis, box_w, box_h)

    # ---- Place cells ----
    print("Building place-cell population ...")
    pz_fn, D_place = make_place_cell_fn(
        n_cells=cfg.place.n_cells,
        box_width=box_w,
        box_height=box_h,
        sigma=cfg.place.sigma,
        rng=np.random.default_rng(cfg.seed),
    )
    print(f"  D = {D_place} cells")

    print("Building A (place) ...")
    A_place = build_A(pz_fn, D_place, n_weight, box_w, box_h)
    print(f"  ||A||_F = {np.linalg.norm(A_place):.4f}")

    print("Computing energy landscape (place) ...")
    _, _, E_grid_p = compute_energy_landscape(pz_fn, A_place, n_vis, box_w, box_h)

    # ---- Fourier analysis ----
    print("Fourier analysis ...")
    fx_g, fy_g, pow_g = fourier_power(E_grid_g, box_w, box_h)
    fx_p, fy_p, pow_p = fourier_power(E_grid_p, box_w, box_h)

    # ---- Rate maps (4 example cells each) ----
    print("Computing example rate maps ...")
    def _pick_indices(D: int) -> list[int]:
        return [int(D * t) for t in [0.05, 0.25, 0.55, 0.80]]

    rm_grid = compute_rate_maps(gz_fn, n_vis, box_w, box_h, _pick_indices(D_grid))
    rm_place = compute_rate_maps(pz_fn, n_vis, box_w, box_h, _pick_indices(D_place))

    # ---- Coverage maps ----
    print("Computing coverage maps ...")
    cov_grid = compute_coverage(gz_fn, n_vis, box_w, box_h)
    cov_place = compute_coverage(pz_fn, n_vis, box_w, box_h)

    # ---- Log to MLflow ----
    with mlflow.start_run():
        mlflow.log_text(OmegaConf.to_yaml(cfg, resolve=True), "config.yaml")
        mlflow.log_params({
            "grid_n_cells": D_grid,
            "place_n_cells": D_place,
            "place_sigma": cfg.place.sigma,
            "n_weight_steps": n_weight,
            "n_vis_steps": n_vis,
        })

        # Individual figures
        fig_g = plot_energy_and_fourier(
            "Grid cells", x_vals, y_vals, E_grid_g,
            fx_g, fy_g, pow_g, box_w, box_h,
        )
        mlflow.log_figure(fig_g, "grid_cells.png")
        plt.close(fig_g)

        fig_p = plot_energy_and_fourier(
            "Place cells", x_vals, y_vals, E_grid_p,
            fx_p, fy_p, pow_p, box_w, box_h,
        )
        mlflow.log_figure(fig_p, "place_cells.png")
        plt.close(fig_p)

        # Side-by-side comparison
        fig_cmp = plot_comparison(
            x_vals, y_vals,
            E_grid_g, E_grid_p,
            fx_g, fy_g, pow_g, pow_p,
            rm_grid, rm_place,
            cov_grid, cov_place,
            box_w, box_h,
        )
        mlflow.log_figure(fig_cmp, "comparison.png")
        plt.close(fig_cmp)

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
