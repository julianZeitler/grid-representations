import tempfile
import os

import mlflow
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Optional

from scores import GridScorer


@torch.no_grad()
def get_ratemaps(model: nn.Module, res: int, widths: tuple) -> list[np.ndarray]:
    """Compute ratemaps at different spatial scales using model forward pass.

    Args:
        model: Trained ActionableRGM model
        res: Resolution of the ratemap grid
        widths: Tuple of (small, medium, large) box widths

    Returns:
        List of ratemaps [V_small, V_medium, V_large], each of shape (latent_size, res*res)
    """
    model.eval()
    device = next(model.parameters()).device

    ratemaps = []
    for width in widths:
        # Create position grid
        phi = np.linspace(-0.5, 0.5, res) * width
        phi_x, phi_y = np.meshgrid(phi, phi)
        positions = np.stack([phi_x.flatten(), phi_y.flatten()], axis=1)

        positions_shifted = np.roll(positions, 1, axis=0)
        positions_shifted[0] = 0
        d_pos = positions - positions_shifted

        # Convert to tensor and compute representations
        displacements = torch.tensor(d_pos, dtype=torch.float32, device=device).unsqueeze(0)
        # positions_tensor = torch.tensor(positions, dtype=torch.float32, device=device).unsqueeze(1)

        # Use step function to get representation at each position from z0
        z, _ = model(displacements, norm=True)

        # Shape: (res*res, latent_size) -> (latent_size, res*res)
        V = z.cpu().numpy().T

        # Shape: (Batch=4900, 2)
        # pos_tensor = torch.tensor(positions, dtype=torch.float32, device=device)

        # # get_T is vectorized and O(1) parallel, not sequential!
        # T = model.get_T(pos_tensor) 

        # # Apply transform to z0 for all positions in one parallel operation
        # z0 = model.z0.unsqueeze(0) # (1, D)
        # V = torch.einsum('bij,kj->bi', T, z0).cpu().numpy().T # (4900, D)
        
        ratemaps.append(V)

    return ratemaps


def quantitative_analysis(Vs: list[np.ndarray], widths: tuple, res: int = 70) -> tuple[plt.Figure, dict]:
    """Compute grid scores for ratemaps at different scales.

    Args:
        Vs: List of ratemaps [V_small, V_medium, V_large]
        widths: Tuple of box widths corresponding to each ratemap
        res: Resolution of ratemaps

    Returns:
        Tuple of (figure, scores_dict)
    """
    scores = {}
    fig_score, axes = plt.subplots(1, 3, figsize=(12, 5))
    scale_names = ["sm", "md", "lg"]
    scale_titles = ["Small Ratemaps", "Medium Ratemaps", "Large Ratemaps"]

    for idx, V in enumerate(Vs):
        maps = [V[i, :] for i in range(V.shape[0])]

        starts = [0.2] * 10
        ends = np.linspace(0.4, 1.0, num=10)
        box_width = widths[idx]
        box_height = widths[idx]
        coord_range = ((-box_width / 2, box_width / 2), (-box_height / 2, box_height / 2))
        masks_parameters = zip(starts, ends.tolist())
        scorer = GridScorer(res, coord_range, masks_parameters)

        score_60, score_90, max_60_mask, max_90_mask, sac, max_60_ind = zip(
            *[scorer.get_scores(rm.reshape(res, res)) for rm in tqdm(maps, desc=f"Scoring {scale_titles[idx]}")])
        score_60 = np.nan_to_num(score_60)
        score_90 = np.nan_to_num(score_90)

        axes[idx].hist(score_60, range=(-1, 2.5), bins=15)
        axes[idx].set_xlabel('Grid score')
        axes[idx].set_ylabel('Count')
        axes[idx].set_title(scale_titles[idx])

        max_score = np.max(score_60)
        mean_score = np.mean(score_60)
        axes[idx].text(0.05, 0.95, f'Max: {max_score:.3f}\nMean: {mean_score:.3f}',
                       transform=axes[idx].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        prefix = scale_names[idx]
        scores[f"{prefix}_60"] = score_60
        scores[f"{prefix}_90"] = score_90
        scores[f"{prefix}_60_max"] = max_score
        scores[f"{prefix}_60_mean"] = mean_score
        scores[f"{prefix}_90_max"] = np.max(score_90)
        scores[f"{prefix}_90_mean"] = np.mean(score_90)

    fig_score.tight_layout()
    return fig_score, scores


def loss_plots(L: np.ndarray, min_L: tuple, lambda_pos: Optional[np.ndarray] = None,
               lambda_norm: Optional[np.ndarray] = None, val_L: Optional[np.ndarray] = None) -> plt.Figure:
    """Generate loss evolution plots.

    Args:
        L: Loss array of shape [4, n_iters] containing [total, separation, positivity, norm]
        min_L: Minimum loss information
        lambda_pos: Optional array of lambda_pos values over iterations
        lambda_norm: Optional array of lambda_norm values over iterations
        val_L: Optional validation loss array of shape [4, n_val_iters]
    """
    titles = ['Loss', 'Separation', 'Positivity', 'Norm']
    fig = plt.figure(figsize=(20, 8))

    if val_L is not None:
        n_train = L.shape[1]
        n_val = val_L.shape[1]
        val_x = np.linspace(0, n_train - 1, n_val)

    for counter in range(4):
        ax1 = plt.subplot(1, 4, counter + 1)
        color1 = 'tab:blue'
        ax1.plot(L[counter, :], color=color1, label='Train')

        if val_L is not None:
            ax1.plot(val_x, val_L[counter, :], color='tab:green', linestyle='--',
                     alpha=0.8, label='Validation')

        ax1.set_xlabel('Epoch')
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.set_ylabel(titles[counter], color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_title(titles[counter])

        if val_L is not None and counter == 0:
            ax1.legend(loc='upper right')

        if counter == 2 and lambda_pos is not None:
            ax2 = ax1.twinx()
            color2 = 'tab:orange'
            ax2.plot(lambda_pos, color=color2, alpha=0.7, linestyle='--')
            ax2.set_ylabel('lambda_pos', color=color2)
            ax2.tick_params(axis='y', labelcolor=color2)
            ax2.set_zorder(ax1.get_zorder() + 1)
            ax1.set_frame_on(False)

        elif counter == 3 and lambda_norm is not None:
            ax2 = ax1.twinx()
            color2 = 'tab:red'
            ax2.plot(lambda_norm, color=color2, alpha=0.7, linestyle='--')
            ax2.set_ylabel('lambda_norm', color=color2)
            ax2.tick_params(axis='y', labelcolor=color2)
            ax2.set_zorder(ax1.get_zorder() + 1)
            ax1.set_frame_on(False)

    plt.suptitle(f'Min Loss: {min_L[1]:.3f}')
    plt.tight_layout()

    return fig


def neuron_plotter_2d(V: np.ndarray, res: int, scores: np.ndarray = None) -> plt.Figure:
    """Plot individual neuron ratemaps.

    Args:
        V: Ratemap array of shape (n_neurons, res*res)
        res: Resolution of ratemaps
        scores: Optional array of scores for each neuron
    """
    D = V.shape[0]
    RowsD = int(np.ceil(np.sqrt(D)))
    ColumnsD = int(np.ceil(D / RowsD))

    fig = plt.figure(figsize=(20, 16))
    for neuron in range(D):
        plt.subplot(RowsD, ColumnsD, neuron + 1)
        plt.axis('off')
        plt.imshow(np.reshape(V[neuron, :], [res, res]), vmin=V.min(), vmax=V.max())
        plt.colorbar()
        if scores is not None:
            plt.title(f'{scores[neuron]:.3f}')
    fig.tight_layout()
    return fig


def log_figure(fig: plt.Figure, name: str) -> None:
    """Log a matplotlib figure to MLflow."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, f"{name}.png")
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        mlflow.log_artifact(filepath, artifact_path="figures")
    plt.close(fig)


def create_loss_plots_from_mlflow(k: int) -> None:
    """Create loss plots from MLflow logged metrics.

    Args:
        k: Run index to plot
    """
    client = mlflow.tracking.MlflowClient()
    run_id = mlflow.active_run().info.run_id

    # Metric names logged during training
    metric_names = {
        "train_loss": f"k{k}/train_loss",
        "val_loss": f"k{k}/val_loss",
        "separation": f"k{k}/separation",
        "positivity_geco": f"k{k}/positivity_geco",
        "norm_geco": f"k{k}/norm_geco",
        "lambda_pos": f"k{k}/lambda_pos",
        "lambda_norm": f"k{k}/lambda_norm",
    }

    # Fetch metrics from MLflow
    metrics_data = {}
    for name, metric_key in metric_names.items():
        history = client.get_metric_history(run_id, metric_key)
        if history:
            # Sort by step and extract values
            history = sorted(history, key=lambda x: x.step)
            metrics_data[name] = np.array([m.value for m in history])

    if not metrics_data:
        return

    # Build arrays for loss_plots function
    n_epochs = len(metrics_data.get("train_loss", []))
    if n_epochs == 0:
        return

    train_L = np.array([
        metrics_data.get("train_loss", np.zeros(n_epochs)),
        metrics_data.get("separation", np.zeros(n_epochs)),
        metrics_data.get("positivity_geco", np.zeros(n_epochs)),
        metrics_data.get("norm_geco", np.zeros(n_epochs)),
    ])

    val_L = np.array([
        metrics_data.get("val_loss", np.zeros(n_epochs)),
        np.zeros(n_epochs),  # val separation not logged separately
        np.zeros(n_epochs),  # val positivity not logged separately
        np.zeros(n_epochs),  # val norm not logged separately
    ])

    lambda_pos = metrics_data.get("lambda_pos")
    lambda_norm = metrics_data.get("lambda_norm")

    min_idx = np.argmin(train_L[0])
    min_L = (min_idx, train_L[0][min_idx])

    fig = loss_plots(train_L, min_L, lambda_pos=lambda_pos, lambda_norm=lambda_norm, val_L=val_L)
    log_figure(fig, f"loss_curves_k{k}")


def generate_2d_plots(model: nn.Module, k: int = 0):# -> dict:
    """Generate analysis plots for a trained 2D model.

    Args:
        model: Trained ActionableRGM model
        k: Run index for labeling

    Returns:
        Dictionary of scores
    """
    model.eval()

    # Extract model parameters for visualization
    om = model.om.detach().cpu().numpy()
    S = model.S.detach().cpu().numpy()

    # Frequency plot
    fig_freq, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(om[:, 0], om[:, 1], s=100, alpha=0.6)
    ax.set_xlabel('$\\omega_x$', fontsize=12)
    ax.set_ylabel('$\\omega_y$', fontsize=12)
    ax.set_title('Frequency vectors', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_aspect('equal')
    fig_freq.tight_layout()
    log_figure(fig_freq, f"freq_plot_k{k}")

    # S analysis
    fig_S, axes = plt.subplots(1, 3, figsize=(12, 3))

    S_inv = np.linalg.inv(S)

    im1 = axes[0].imshow(S, cmap='RdBu_r', aspect='equal')
    axes[0].set_title('S')
    axes[0].set_xlabel('Column')
    axes[0].set_ylabel('Row')
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(S_inv, cmap='RdBu_r', aspect='equal')
    axes[1].set_title('S^-1')
    axes[1].set_xlabel('Column')
    axes[1].set_ylabel('Row')
    plt.colorbar(im2, ax=axes[1])

    im3 = axes[2].imshow(S @ S_inv, cmap='RdBu_r', aspect='equal')
    axes[2].set_title('S @ S^-1')
    axes[2].set_xlabel('Column')
    axes[2].set_ylabel('Row')
    plt.colorbar(im3, ax=axes[2])
    fig_S.tight_layout()
    log_figure(fig_S, f"S_analysis_k{k}")

    # Grid scores
    res = 70
    widths = (1, 2, 4)
    Vs = get_ratemaps(model, res, widths)
    V_small, V_medium, V_large = Vs

    fig_score, scores = quantitative_analysis(Vs, widths, res)
    log_figure(fig_score, f"grid_scores_k{k}")

    # Log scalar scores as metrics
    for key, value in scores.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(f"k{k}/{key}", value)

    # Normalize by large room norms for visualization
    large_norms = np.linalg.norm(V_large, axis=1, keepdims=True)
    V_small_norm = V_small / large_norms
    V_medium_norm = V_medium / large_norms
    V_large_norm = V_large / large_norms

    # Neuron plots
    neuron_sm_fig = neuron_plotter_2d(V_small_norm, res, scores["sm_60"])
    log_figure(neuron_sm_fig, f"neurons_small_k{k}")

    neuron_md_fig = neuron_plotter_2d(V_medium_norm, res, scores["md_60"])
    log_figure(neuron_md_fig, f"neurons_medium_k{k}")

    neuron_lg_fig = neuron_plotter_2d(V_large_norm, res, scores["lg_60"])
    log_figure(neuron_lg_fig, f"neurons_large_k{k}")

    return scores, neuron_lg_fig
