"""
Flat torus visualisation for arbitrary lattice angle.
  Figure 1 – Lattice in R^2
  Figure 2 – Fundamental parallelogram with edge identifications
  Figure 3 – Flat torus embedded in R^3 via sheared map

Usage:
  python torus_vis.py --angle 60   # hexagonal (default)
  python torus_vis.py --angle 90   # square

Embedding
─────────
  x(θ, φ) = (R + r·cos φ) · cos(θ + s·φ)
  y(θ, φ) = (R + r·cos φ) · sin(θ + s·φ)
  z(θ, φ) = r · sin φ

where s = cos(α) encodes the lattice angle α.
For α=60°, s=0.5 (hexagonal). For α=90°, s=0 (standard torus).

Constant-θ rings vary φ over [0, 4π] so that lon = θ + s·φ completes a
full 2π revolution and the ring closes in R^3 (winds twice around the torus).
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from matplotlib.colors import Normalize
import mlflow

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Flat torus visualisation")
parser.add_argument("--angle", type=float, default=60.0,
                    help="Lattice angle in degrees (default: 60)")
args = parser.parse_args()

alpha_deg = args.angle
alpha_rad = np.deg2rad(alpha_deg)

# ── lattice / metric parameters ──────────────────────────────────────────────
lam  = 1.0
r_c  = lam / (2 * np.pi)
a1   = lam * np.array([1.0, 0.0])
a2   = lam * np.array([np.cos(alpha_rad), np.sin(alpha_rad)])

shear = np.cos(alpha_rad)

R_emb = 3.0 * r_c
r_emb = r_c

def embed(theta, phi):
    lon = theta + shear * phi
    x = (R_emb + r_emb * np.cos(phi)) * np.cos(lon)
    y = (R_emb + r_emb * np.cos(phi)) * np.sin(lon)
    z = r_emb * np.sin(phi)
    return x, y, z

# ── colour palette ────────────────────────────────────────────────────────────
C_BG    = "#ffffff"
C_PANEL = "#f5f5f5"
C_GRID  = "#cccccc"
C_DOT   = "#888888"
C_A1    = "#E8593C"
C_A2    = "#378ADD"
C_TEXT  = "#111111"
C_DIM   = "#555555"

THETA_COLORS = ["#E8593C", "#F2A623", "#FCDE5A"]
PHI_COLORS   = ["#378ADD", "#1D9E75", "#9F77DD"]
THETA_VALS   = [0.0, 2*np.pi/3, 4*np.pi/3]
PHI_VALS     = [0.0, 2*np.pi/3, 4*np.pi/3]
N = 500

mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("torus_vis")
mlflow.start_run()
mlflow.log_param("angle_deg", alpha_deg)

# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 – Lattice
# ═══════════════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(6, 6), facecolor=C_BG)
ax1.set_facecolor(C_PANEL)
ax1.set_aspect("equal")

rng = range(-3, 4)
for m in rng:
    for n in rng:
        p = m * a1 + n * a2
        ax1.plot(*p, "o", color=C_DOT, markersize=3, zorder=3)
        for da, db in [(1, 0), (0, 1)]:
            q = (m + da) * a1 + (n + db) * a2
            ax1.plot([p[0], q[0]], [p[1], q[1]], color=C_GRID, lw=0.6, zorder=1)

ax1.plot(0, 0, "o", color=C_TEXT, markersize=5, zorder=5)
kw = dict(arrowstyle="-|>", mutation_scale=14, lw=1.8, zorder=6)
ax1.add_patch(FancyArrowPatch((0, 0), tuple(a1), color=C_A1, **kw))
ax1.add_patch(FancyArrowPatch((0, 0), tuple(a2), color=C_A2, **kw))

ax1.text(*(a1 * 0.5 + [0, -0.14]), r"$\mathbf{a}_1$",
         color=C_A1, fontsize=14, ha="center", va="top")
ax1.text(*(a2 * 0.6 + [-0.10, 0.06]), r"$\mathbf{a}_2$",
         color=C_A2, fontsize=14, ha="right", va="center")

t_arc = np.linspace(0, alpha_rad, 60)
ax1.plot(0.28*np.cos(t_arc), 0.28*np.sin(t_arc), color=C_DIM, lw=1.0)
ax1.text(0.32, 0.08, f"{alpha_deg:.4g}°", color=C_DIM, fontsize=13)

ax1.set_xlim(-1.8, 2.8); ax1.set_ylim(-1.5, 2.5); ax1.axis("off")

fig1.tight_layout()
mlflow.log_figure(fig1, f"torus_{alpha_deg:.4g}deg_lattice.png")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 – Fundamental domain
# ═══════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(6, 6), facecolor=C_BG)
ax2.set_facecolor(C_PANEL)
ax2.set_aspect("equal")

O = np.zeros(2); A = a1.copy(); B = a2.copy(); AB = a1 + a2

ax2.add_patch(plt.Polygon([O, A, AB, B], closed=True,
              facecolor="#ddeeff", edgecolor="none", alpha=0.7, zorder=1))

for start, end, col in [(O, A, C_A1), (B, AB, C_A1),
                         (O, B, C_A2), (A, AB, C_A2)]:
    ax2.annotate("", xy=end*0.9 + start*0.1, xytext=start*0.9 + end*0.1,
                 arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6,
                                 mutation_scale=12), zorder=5)
    perp = np.array([-(end-start)[1], (end-start)[0]])
    perp /= np.linalg.norm(perp)
    n_ticks = 2 if col == C_A1 else 1
    for frac in np.linspace(0.4, 0.6, n_ticks):
        pt = start + frac*(end-start)
        ax2.plot([pt[0]-perp[0]*0.04, pt[0]+perp[0]*0.04],
                 [pt[1]-perp[1]*0.04, pt[1]+perp[1]*0.04],
                 color=col, lw=1.4, zorder=5)

for pt in [O, A, B, AB]:
    ax2.plot(*pt, "o", color=C_TEXT, markersize=5, zorder=6)

ax2.text(*(A*0.5 + [0,-0.11]), r"$\theta$", color=C_A1, fontsize=14,
         ha="center", va="top", fontweight="bold")
ax2.text(*(B*0.5 + [-0.13,0]), r"$\phi$", color=C_A2, fontsize=14,
         ha="right", va="center", fontweight="bold")
ax2.text(*(B + (AB-B)*0.5 + [0,0.09]), r"$\theta$", color=C_A1, fontsize=14,
         ha="center", va="bottom", fontweight="bold")
ax2.text(*(A + (AB-A)*0.5 + [0.13,0]), r"$\phi$", color=C_A2, fontsize=14,
         ha="left", va="center", fontweight="bold")

t_arc = np.linspace(0, alpha_rad, 60)
ax2.plot(0.22*np.cos(t_arc), 0.22*np.sin(t_arc), color=C_DIM, lw=1.0)
ax2.text(0.24, 0.07, f"{alpha_deg:.4g}°", color=C_DIM, fontsize=13)

# tight bounds around the parallelogram with small padding
all_pts = np.array([O, A, B, AB])
pad = 0.18
ax2.set_xlim(all_pts[:,0].min() - pad, all_pts[:,0].max() + pad)
ax2.set_ylim(all_pts[:,1].min() - pad, all_pts[:,1].max() + pad)
ax2.axis("off")

fig2.subplots_adjust(left=0, right=1, top=1, bottom=0)
mlflow.log_figure(fig2, f"torus_{alpha_deg:.4g}deg_fundamental.png")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 – Torus embedding with rings
# ═══════════════════════════════════════════════════════════════════════════
fig3 = plt.figure(figsize=(6, 6), facecolor=C_BG)
ax3 = fig3.add_axes([0, 0, 1, 1], projection="3d", facecolor=C_PANEL)

th_s = np.linspace(0, 2*np.pi, 160)
ph_s = np.linspace(0, 2*np.pi, 160)
TH, PH = np.meshgrid(th_s, ph_s)
XS, YS, ZS = embed(TH, PH)

face_c = plt.cm.twilight_shifted(Normalize(0, 2*np.pi)(PH))
ax3.plot_surface(XS, YS, ZS, facecolors=face_c,
                 rstride=2, cstride=2, linewidth=0,
                 antialiased=True, alpha=0.28, shade=False)

# constant-theta rings (solid): vary phi over 4pi so the ring closes in R^3
ph_r = np.linspace(0, 4*np.pi, N)
for th_val, col in zip(THETA_VALS, THETA_COLORS):
    xr, yr, zr = embed(th_val, ph_r)
    ax3.plot(xr, yr, zr, color=col, lw=2.2, zorder=9)

th_r = np.linspace(0, 2*np.pi, N)
for ph_val, col in zip(PHI_VALS, PHI_COLORS):
    xr, yr, zr = embed(th_r, ph_val)
    ax3.plot(xr, yr, zr, color=col, lw=2.2, zorder=9,
             linestyle="--", dashes=(6, 3))

ax3.set_box_aspect([1, 1, 0.52], zoom=1.5)
ax3.set_axis_off()
ax3.set_facecolor(C_PANEL)
ax3.view_init(elev=22, azim=-48)


handles = []
for col, tv in zip(THETA_COLORS, THETA_VALS):
    handles.append(mpatches.Patch(color=col,
        label=r"$\theta=$" + f"{tv/np.pi:.2g}" + r"$\pi$"))
for col, pv in zip(PHI_COLORS, PHI_VALS):
    handles.append(mpatches.Patch(color=col,
        label=r"$\phi=$" + f"{pv/np.pi:.2g}" + r"$\pi$"))

fig3.legend(handles=handles, loc="upper left", fontsize=12,
            labelcolor=C_DIM, frameon=True, framealpha=0.8,
            facecolor=C_BG, edgecolor=C_GRID,
            bbox_to_anchor=(0.02, 0.98), ncol=2)

mlflow.log_figure(fig3, f"torus_{alpha_deg:.4g}deg_embedding.png")


# ═══════════════════════════════════════════════════════════════════════════
# Shared: Gaussian activity map on the lattice
# ═══════════════════════════════════════════════════════════════════════════
sigma = 0.18 * lam

# 2D physical grid covering a few unit cells
res = 500
x_range = np.linspace(-1.5*lam, 2.5*lam, res)
y_range = np.linspace(-1.5*lam, 2.5*lam, res)
XX, YY  = np.meshgrid(x_range, y_range)

activity_2d = np.zeros((res, res))
for m in range(-5, 6):
    for n in range(-5, 6):
        cx, cy = m * a1 + n * a2
        activity_2d += np.exp(-((XX - cx)**2 + (YY - cy)**2) / (2 * sigma**2))

# Activity on torus: (θ,φ) → physical position → evaluate Gaussians
res_t = 300
th_t = np.linspace(0, 2*np.pi, res_t)
ph_t = np.linspace(0, 2*np.pi, res_t)
TH_T, PH_T = np.meshgrid(th_t, ph_t)

u = TH_T / (2*np.pi)
v = PH_T / (2*np.pi)
Xp = u * a1[0] + v * a2[0]
Yp = u * a1[1] + v * a2[1]

activity_torus = np.zeros_like(TH_T)
for m in range(-3, 4):
    for n in range(-3, 4):
        cx, cy = m * a1 + n * a2
        activity_torus += np.exp(-((Xp - cx)**2 + (Yp - cy)**2) / (2 * sigma**2))

cmap_act = plt.cm.inferno

# ═══════════════════════════════════════════════════════════════════════════
# Figure 4 – Lattice with Gaussian grid-cell activity
# ═══════════════════════════════════════════════════════════════════════════
fig4, ax4 = plt.subplots(figsize=(6, 6), facecolor=C_BG)
ax4.set_facecolor(C_BG)
ax4.set_aspect("equal")

ax4.pcolormesh(XX, YY, activity_2d, cmap=cmap_act,
               shading="gouraud", rasterized=True)

# lattice dots
rng = range(-3, 4)
for m in rng:
    for n in rng:
        p = m * a1 + n * a2
        if (x_range[0] <= p[0] <= x_range[-1] and
                y_range[0] <= p[1] <= y_range[-1]):
            ax4.plot(*p, "o", color="white", markersize=4,
                     markeredgecolor="#aaaaaa", markeredgewidth=0.5, zorder=3)

pad = 0.1
ax4.set_xlim(x_range[0] - pad, x_range[-1] + pad)
ax4.set_ylim(y_range[0] - pad, y_range[-1] + pad)
ax4.axis("off")

fig4.tight_layout(pad=0)
mlflow.log_figure(fig4, f"torus_{alpha_deg:.4g}deg_activity2d.png")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 5 – Activity map on the torus surface
# ═══════════════════════════════════════════════════════════════════════════
fig5 = plt.figure(figsize=(6, 6), facecolor=C_BG)
ax5 = fig5.add_axes([0, 0, 1, 1], projection="3d", facecolor=C_BG)

XS_t, YS_t, ZS_t = embed(TH_T, PH_T)

norm_act = Normalize(vmin=activity_torus.min(), vmax=activity_torus.max())
face_act  = cmap_act(norm_act(activity_torus))

ax5.plot_surface(XS_t, YS_t, ZS_t, facecolors=face_act,
                 rstride=1, cstride=1, linewidth=0,
                 antialiased=True, shade=False)

ax5.set_box_aspect([1, 1, 0.52], zoom=1.8)
ax5.set_axis_off()
ax5.set_facecolor(C_BG)
ax5.view_init(elev=22, azim=-48)

mlflow.log_figure(fig5, f"torus_{alpha_deg:.4g}deg_activity_torus.png")

mlflow.end_run()
plt.show()
