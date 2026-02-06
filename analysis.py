import mlflow
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

        positions_tensor = torch.tensor(positions, dtype=torch.float32, device=device).unsqueeze(1)

        # Use step function to get representation at each position from z0
        z, _ = model(positions_tensor, norm=False)
        z = z / (z.norm(dim=0) + 1e-5)

        # Shape: (res*res, latent_size) -> (latent_size, res*res)
        V = z.cpu().numpy().T

        ratemaps.append(V)

    return ratemaps


def quantitative_analysis(Vs: list[np.ndarray], widths: tuple, res: int = 70) -> tuple[go.Figure, dict]:
    """Compute grid scores for ratemaps at different scales.

    Args:
        Vs: List of ratemaps [V_small, V_medium, V_large]
        widths: Tuple of box widths corresponding to each ratemap
        res: Resolution of ratemaps

    Returns:
        Tuple of (figure, scores_dict)
    """
    scores = {}
    fig_score = make_subplots(rows=1, cols=3, subplot_titles=["Small Ratemaps", "Medium Ratemaps", "Large Ratemaps"])
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

        max_score = np.max(score_60)
        mean_score = np.mean(score_60)

        fig_score.add_trace(
            go.Histogram(x=score_60, xbins=dict(start=-2, end=2, size=4/15),
                         name=scale_titles[idx], marker_color='steelblue'),
            row=1, col=idx + 1
        )

        axis_suffix = "" if idx == 0 else str(idx + 1)
        fig_score.add_annotation(
            x=0.05, y=0.95, xref=f"x{axis_suffix} domain", yref=f"y{axis_suffix} domain",
            text=f"Max: {max_score:.3f}<br>Mean: {mean_score:.3f}",
            showarrow=False, bgcolor="wheat", opacity=0.8,
            xanchor="left", yanchor="top", align="left"
        )

        prefix = scale_names[idx]
        scores[f"{prefix}_60"] = score_60
        scores[f"{prefix}_90"] = score_90
        scores[f"{prefix}_60_max"] = max_score
        scores[f"{prefix}_60_mean"] = mean_score
        scores[f"{prefix}_90_max"] = np.max(score_90)
        scores[f"{prefix}_90_mean"] = np.mean(score_90)

    fig_score.update_xaxes(title_text="Grid score", range=[-2, 2])
    fig_score.update_yaxes(title_text="Count")
    fig_score.update_layout(height=500, width=1200, showlegend=False)
    return fig_score, scores


def loss_plots(
    train_losses: dict[str, np.ndarray],
    val_losses: Optional[dict[str, np.ndarray]] = None,
    lambda_pos: Optional[np.ndarray] = None,
    lambda_norm: Optional[np.ndarray] = None,
) -> go.Figure:
    """Generate loss evolution plots.

    Args:
        train_losses: Dict with keys 'loss', 'separation', 'positivity', 'norm'
        val_losses: Optional dict with same keys for validation losses
        lambda_pos: Optional array of lambda_pos values over iterations
        lambda_norm: Optional array of lambda_norm values over iterations
    """
    keys = ['loss', 'separation', 'positivity', 'norm']
    titles = ['Loss', 'Separation', 'Positivity', 'Norm']

    # Determine which subplots need secondary y-axis (2x2 grid)
    specs = [[], []]
    for i, key in enumerate(keys):
        has_secondary = (key == 'positivity' and lambda_pos is not None) or (key == 'norm' and lambda_norm is not None)
        specs[i // 2].append({"secondary_y": has_secondary})

    fig = make_subplots(rows=2, cols=2, subplot_titles=titles, specs=specs,
                        horizontal_spacing=0.12, vertical_spacing=0.16)

    n_train = len(train_losses.get('loss', []))

    for counter, key in enumerate(keys):
        if key not in train_losses:
            continue

        train_data = train_losses[key]
        x_train = list(range(len(train_data)))

        has_val = val_losses is not None and key in val_losses
        show_in_legend = counter == 0 and has_val

        row = counter // 2 + 1
        col = counter % 2 + 1

        fig.add_trace(
            go.Scatter(x=x_train, y=train_data, mode='lines', name='Train',
                       line=dict(color='blue'), showlegend=show_in_legend),
            row=row, col=col
        )

        if has_val:
            val_data = val_losses[key]
            n_val = len(val_data)
            val_x = np.linspace(0, n_train - 1, n_val).tolist()
            fig.add_trace(
                go.Scatter(x=val_x, y=val_data, mode='lines', name='Validation',
                           line=dict(color='green', dash='dash'), opacity=0.8,
                           showlegend=show_in_legend),
                row=row, col=col
            )

        if key == 'positivity' and lambda_pos is not None:
            fig.add_trace(
                go.Scatter(x=list(range(len(lambda_pos))), y=lambda_pos, mode='lines',
                           name='λ<sub>pos</sub>', line=dict(color='red', dash='dash'),
                           opacity=0.8, showlegend=False),
                row=row, col=col, secondary_y=True
            )
            fig.update_yaxes(title_text="λ<sub>pos</sub>", secondary_y=True, row=row, col=col,
                             tickfont=dict(color='red'), title_font=dict(color='red'), title_standoff=5)

        elif key == 'norm' and lambda_norm is not None:
            fig.add_trace(
                go.Scatter(x=list(range(len(lambda_norm))), y=lambda_norm, mode='lines',
                           name='λ<sub>norm</sub>', line=dict(color='red', dash='dash'),
                           opacity=0.8, showlegend=False),
                row=row, col=col, secondary_y=True
            )
            fig.update_yaxes(title_text="λ<sub>norm</sub>", secondary_y=True, row=row, col=col,
                             tickfont=dict(color='red'), title_font=dict(color='red'), title_standoff=5)

    fig.update_xaxes(title_text="Epoch", title_standoff=5)
    y_titles = {
        'Loss': 'Loss',
        'Separation': 'Separation',
        'Positivity': 'Positivity = log(L<sub>2</sub>) - k<sub>pos</sub>',
        'Norm': 'Norm = log(L<sub>3</sub>) - k<sub>norm</sub>',
    }
    for counter, title in enumerate(titles):
        row = counter // 2 + 1
        col = counter % 2 + 1
        fig.update_yaxes(title_text=y_titles[title], row=row, col=col, secondary_y=False,
                         tickfont=dict(color='blue'), title_font=dict(color='blue'))

    fig.update_layout(
        height=600, width=1000,
        legend=dict(
            x=0.4, y=0.99,
            xanchor='right', yanchor='top',
            bgcolor='rgba(255,255,255,0.7)',
        )
    )

    return fig


def neuron_plotter_2d(V: np.ndarray, res: int, scores: np.ndarray = None) -> go.Figure:
    """Plot individual neuron ratemaps.

    Args:
        V: Ratemap array of shape (n_neurons, res*res)
        res: Resolution of ratemaps
        scores: Optional array of scores for each neuron
    """
    D = V.shape[0]
    RowsD = int(np.ceil(np.sqrt(D)))
    ColumnsD = int(np.ceil(D / RowsD))

    subplot_titles = [f'{scores[i]:.3f}' if scores is not None else '' for i in range(D)]
    fig = make_subplots(rows=RowsD, cols=ColumnsD, subplot_titles=subplot_titles,
                        horizontal_spacing=0.002, vertical_spacing=0.03)

    vmin, vmax = V.min(), V.max()

    for neuron in range(D):
        row = neuron // ColumnsD + 1
        col = neuron % ColumnsD + 1
        heatmap = go.Heatmap(
            z=np.reshape(V[neuron, :], [res, res]),
            zmin=vmin, zmax=vmax, colorscale='Viridis',
            showscale=(neuron == 0)
        )
        fig.add_trace(heatmap, row=row, col=col)

        axis_idx = neuron + 1
        axis_suffix = "" if axis_idx == 1 else str(axis_idx)
        fig.update_xaxes(showticklabels=False, showgrid=False, constrain='domain', row=row, col=col)
        fig.update_yaxes(showticklabels=False, showgrid=False, scaleanchor=f'x{axis_suffix}',
                         scaleratio=1, constrain='domain', row=row, col=col)
    fig.update_layout(height=800, width=800)
    return fig


def log_figure(fig: go.Figure, name: str) -> None:
    """Log a plotly figure to MLflow in both PNG and HTML formats."""
    mlflow.log_figure(fig, f"figures/{name}.png")
    mlflow.log_figure(fig, f"figures/{name}.html")


def create_loss_plots_from_mlflow(k: int) -> None:
    """Create loss plots from MLflow logged metrics.

    Args:
        k: Run index to plot
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    active_run = mlflow.active_run()
    if active_run is None:
        return
    run_id = active_run.info.run_id

    # Metric name mappings: internal key -> mlflow metric name
    train_metrics = {
        "loss": f"k{k}/train_loss",
        "separation": f"k{k}/separation",
        "positivity": f"k{k}/positivity_geco",
        "norm": f"k{k}/norm_geco",
    }
    val_metrics = {
        "loss": f"k{k}/val_loss",
    }
    lambda_metrics = {
        "lambda_pos": f"k{k}/lambda_pos",
        "lambda_norm": f"k{k}/lambda_norm",
    }

    def fetch_metric(metric_key: str) -> Optional[np.ndarray]:
        history = client.get_metric_history(run_id, metric_key)
        if history:
            history = sorted(history, key=lambda x: x.step)
            return np.array([m.value for m in history])
        return None

    # Build train losses dict
    train_losses = {}
    for key, metric_name in train_metrics.items():
        data = fetch_metric(metric_name)
        if data is not None:
            train_losses[key] = data

    if not train_losses:
        return

    # Build val losses dict (only include metrics that exist)
    val_losses = {}
    for key, metric_name in val_metrics.items():
        data = fetch_metric(metric_name)
        if data is not None:
            val_losses[key] = data

    # Fetch lambda values
    lambda_pos = fetch_metric(lambda_metrics["lambda_pos"])
    lambda_norm = fetch_metric(lambda_metrics["lambda_norm"])

    fig = loss_plots(
        train_losses,
        val_losses=val_losses if val_losses else None,
        lambda_pos=lambda_pos,
        lambda_norm=lambda_norm,
    )
    log_figure(fig, f"loss_curves_k{k}")


def frequency_plot(om: np.ndarray) -> go.Figure:
    """Plot frequency vectors.

    Args:
        om: Frequency array of shape (n_neurons, 2)

    Returns:
        Plotly figure
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=om[:, 0], y=om[:, 1], mode='markers',
        marker=dict(size=12, opacity=0.6)
    ))
    fig.update_layout(
        title='Frequency vectors',
        xaxis_title='ω_x', yaxis_title='ω_y',
        xaxis=dict(scaleanchor='y', scaleratio=1, zeroline=True, zerolinewidth=1, zerolinecolor='black'),
        yaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor='black'),
        width=600, height=600
    )
    return fig


def s_matrix_plot(S: np.ndarray) -> go.Figure:
    """Plot S matrix analysis (S, S^-1, and their product).

    Args:
        S: Square matrix

    Returns:
        Plotly figure with 3 subplots
    """
    S_inv = np.linalg.inv(S)
    matrices = [S, S_inv, S @ S_inv]
    titles = ['S', 'S⁻¹', 'S @ S⁻¹']

    fig = make_subplots(rows=1, cols=3, subplot_titles=titles, horizontal_spacing=0.01)
    for i, mat in enumerate(matrices):
        fig.add_trace(
            go.Heatmap(z=mat, colorscale='RdBu_r', showscale=(i == 2)),
            row=1, col=i + 1
        )
        axis_suffix = "" if i == 0 else str(i + 1)
        fig.update_xaxes(constrain='domain', row=1, col=i + 1)
        fig.update_yaxes(scaleanchor=f'x{axis_suffix}', scaleratio=1,
                         constrain='domain', row=1, col=i + 1)
    fig.update_layout(height=350, width=800)
    return fig


def generate_2d_plots(model: nn.Module, k: int = 0) -> dict:
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
    fig_freq = frequency_plot(om)
    log_figure(fig_freq, f"freq_plot_k{k}")

    # S analysis
    fig_S = s_matrix_plot(S)
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

    # Neuron plots
    neuron_sm_fig = neuron_plotter_2d(V_small, res, scores["sm_60"])
    log_figure(neuron_sm_fig, f"neurons_small_k{k}")

    neuron_md_fig = neuron_plotter_2d(V_medium, res, scores["md_60"])
    log_figure(neuron_md_fig, f"neurons_medium_k{k}")

    neuron_lg_fig = neuron_plotter_2d(V_large, res, scores["lg_60"])
    log_figure(neuron_lg_fig, f"neurons_large_k{k}")

    return scores
