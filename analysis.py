import tempfile

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
from omegaconf import OmegaConf
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


def sweep_boxplot(
    scores: dict[str, list[list[float]]],
    x_values: list[float],
    x_param: str,
    log_x: bool,
    base_key: str,
    base_title: str,
    n_neurons_str: str,
    n_samples_str: str,
) -> go.Figure:
    """Create a boxplot figure for sweep score distributions.

    Args:
        scores: Dictionary mapping metric keys to list of score lists per x-value.
        x_values: Sorted unique x-axis values.
        x_param: Name of the x-axis parameter for labeling.
        log_x: Whether to use logarithmic x-axis.
        base_key: Base key identifier (e.g., 'sm_60').
        base_title: Human-readable title for the scale/angle.
        n_neurons_str: Formatted string for neuron count (e.g., " over 64 neurons,") or empty.
        n_samples_str: Formatted string for sample count (e.g., " K=5") or empty.

    Returns:
        Plotly figure with boxplots.
    """
    mean_key = f"{base_key}_mean"
    max_key = f"{base_key}_max"

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"Mean ({n_neurons_str}{n_samples_str})",
            f"Max ({n_neurons_str}{n_samples_str})",
        ]
    )
    fig.update_layout(title=f"Distribution of grid scores over runs<br>{base_title}")

    for col, key in enumerate([mean_key, max_key], start=1):
        for i, x_val in enumerate(x_values):
            data = scores[key][i]
            fig.add_trace(
                go.Box(
                    y=data,
                    x=[x_val] * len(data),
                    name=str(x_val),
                    showlegend=False,
                    marker_color="steelblue",
                    boxpoints="outliers",
                ),
                row=1, col=col
            )

    if log_x:
        fig.update_xaxes(
            title_text=x_param,
            type="log",
            dtick=1,
            minor=dict(dtick="D1", ticks="outside", showgrid=True, gridcolor="white"),
            showgrid=True,
            gridcolor="white",
            exponentformat="power",
        )
    else:
        fig.update_xaxes(title_text=x_param)
    fig.update_yaxes(title_text="Grid score")
    fig.update_layout(height=500, width=1000)

    return fig


def sweep_violin_plot(
    scores: dict[str, list[list[float]]],
    x_values: list[float],
    x_param: str,
    log_x: bool,
    base_key: str,
    base_title: str,
    n_neurons_str: str,
    n_samples_str: str,
) -> go.Figure:
    """Create a violin plot figure for sweep score distributions.

    Args:
        scores: Dictionary mapping metric keys to list of score lists per x-value.
        x_values: Sorted unique x-axis values.
        x_param: Name of the x-axis parameter for labeling.
        log_x: Whether to use logarithmic x-axis.
        base_key: Base key identifier (e.g., 'sm_60').
        base_title: Human-readable title for the scale/angle.
        n_neurons_str: Formatted string for neuron count (e.g., " over 64 neurons,") or empty.
        n_samples_str: Formatted string for sample count (e.g., " K=5") or empty.

    Returns:
        Plotly figure with violin plots.
    """
    mean_key = f"{base_key}_mean"
    max_key = f"{base_key}_max"

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"Mean ({n_neurons_str}{n_samples_str})",
            f"Max ({n_neurons_str}{n_samples_str})",
        ]
    )
    fig.update_layout(title=f"Distribution of grid scores over runs<br>{base_title}")

    for col, key in enumerate([mean_key, max_key], start=1):
        for i, x_val in enumerate(x_values):
            data = scores[key][i]
            fig.add_trace(
                go.Violin(
                    y=data,
                    x=[x_val] * len(data),
                    name=str(x_val),
                    showlegend=False,
                    fillcolor="steelblue",
                    line_color="steelblue",
                    box_visible=True,
                    meanline_visible=True,
                ),
                row=1, col=col
            )

    if log_x:
        fig.update_xaxes(
            title_text=x_param,
            type="log",
            dtick=1,
            minor=dict(dtick="D1", ticks="outside", showgrid=True, gridcolor="white"),
            showgrid=True,
            gridcolor="white",
            exponentformat="power",
        )
    else:
        fig.update_xaxes(title_text=x_param)
    fig.update_yaxes(title_text="Grid score")
    fig.update_layout(height=500, width=1000)

    return fig


def sweep_iqr_plot(
    scores: dict[str, list[list[float]]],
    x_values: list[float],
    x_param: str,
    log_x: bool,
    base_key: str,
    base_title: str,
    n_neurons_str: str,
    n_samples_str: str,
) -> go.Figure:
    """Create a quantile (IQR) plot figure for sweep score distributions.

    Args:
        scores: Dictionary mapping metric keys to list of score lists per x-value.
        x_values: Sorted unique x-axis values.
        x_param: Name of the x-axis parameter for labeling.
        log_x: Whether to use logarithmic x-axis.
        base_key: Base key identifier (e.g., 'sm_60').
        base_title: Human-readable title for the scale/angle.
        n_neurons_str: Formatted string for neuron count (e.g., " over 64 neurons,") or empty.
        n_samples_str: Formatted string for sample count (e.g., " K=5") or empty.

    Returns:
        Plotly figure with median line and IQR shading.
    """
    mean_key = f"{base_key}_mean"
    max_key = f"{base_key}_max"

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"Mean ({n_neurons_str}{n_samples_str})",
            f"Max ({n_neurons_str}{n_samples_str})",
        ]
    )
    fig.update_layout(title=f"Distribution of grid scores over runs<br>{base_title}")

    for col, key in enumerate([mean_key, max_key], start=1):
        data_array = np.array([scores[key][i] for i in range(len(x_values))])
        medians = np.median(data_array, axis=1)
        q1 = np.quantile(data_array, 0.25, axis=1)
        q3 = np.quantile(data_array, 0.75, axis=1)

        # IQR fill (upper bound, then lower bound reversed for fill)
        fig.add_trace(
            go.Scatter(
                x=list(x_values) + list(x_values)[::-1],
                y=list(q3) + list(q1)[::-1],
                fill="toself",
                fillcolor="rgba(70, 130, 180, 0.3)",
                line=dict(color="rgba(255,255,255,0)"),
                name="IQR (Q1-Q3)",
                showlegend=(col == 1),
            ),
            row=1, col=col
        )

        # Median line
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=medians,
                mode="lines",
                line=dict(color="steelblue", width=2),
                name="Median",
                showlegend=(col == 1),
            ),
            row=1, col=col
        )

    if log_x:
        fig.update_xaxes(
            title_text=x_param,
            type="log",
            dtick=1,
            minor=dict(dtick="D1", ticks="outside", showgrid=True, gridcolor="white"),
            showgrid=True,
            gridcolor="white",
            exponentformat="power",
        )
    else:
        fig.update_xaxes(title_text=x_param)
    fig.update_yaxes(title_text="Grid score")
    fig.update_layout(height=500, width=1000)

    return fig


def sweep_score_distributions_mlflow(
    parent_run_id: str,
    x_param: str = "data.seq_len",
    log_x: bool = True,
    tracking_uri: Optional[str] = "sqlite:///mlruns.db",
) -> dict[str, go.Figure]:
    """Generate distribution plots of grid scores across all child runs in a sweep.

    Creates boxplots and median/IQR plots showing the distribution of mean and max
    grid scores over all child runs for each scale/angle combination. Reads metrics
    from MLflow and logs figures to the parent run.

    Args:
        parent_run_id: MLflow run ID of the parent sweep run.
        x_param: Config parameter path for x-axis (e.g., 'data.seq_len').
        log_x: Whether to use logarithmic x-axis scaling.
        tracking_uri: MLflow tracking URI. If None, uses mlflow.get_tracking_uri().

    Returns:
        Dictionary mapping plot names to figure objects.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    figures: dict[str, go.Figure] = {}

    # Get child runs
    parent_run = client.get_run(parent_run_id)
    experiment_id = parent_run.info.experiment_id

    child_runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
    )

    if not child_runs:
        print(f"No child runs found for parent {parent_run_id}")
        return figures

    # Collect data from child runs
    # Structure: {x_value: {metric_key: [values across k]}}
    data_by_x: dict[float, dict[str, list[float]]] = {}
    n_neurons_str: str = ""
    n_samples_str: str = ""

    # Determine which metrics exist by finding a child run that has grid score metrics
    metric_keys = []
    for run in child_runs:
        sample_metrics = run.data.metrics
        metric_keys = [k for k in sample_metrics.keys() if "/" in k and any(
            k.endswith(suffix) for suffix in ["_mean", "_max"]
        )]
        if metric_keys:
            break

    if not metric_keys:
        print("No child runs have grid score metrics (_mean/_max)")
        return figures

    # Extract base metric names (without k prefix)
    base_metric_names = set()
    for mk in metric_keys:
        # e.g., "k0/sm_60_mean" -> "sm_60_mean"
        parts = mk.split("/")
        if len(parts) == 2:
            base_metric_names.add(parts[1])

    for run in child_runs:
        # Load config to get x-axis parameter value
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                config_path = client.download_artifacts(run.info.run_id, "config.yaml", tmpdir)
                cfg = OmegaConf.load(config_path)
                # Navigate nested config path (e.g., "data.seq_len")
                x_value = float(OmegaConf.select(cfg, x_param))
                # Get K and number of neurons from first successful config load
                if not n_samples_str:
                    k_val = OmegaConf.select(cfg, "training.K")
                    if k_val is not None:
                        n_samples_str = f" K={int(k_val)}"
                    latent_size = OmegaConf.select(cfg, "model.latent_size")
                    if latent_size is not None:
                        n_neurons_str = f" over {int(latent_size)} neurons,"
        except Exception as e:
            print(f"Could not load config for run {run.info.run_id}: {e}")
            continue

        # Skip runs without grid score metrics
        metrics = run.data.metrics
        if not any(k.endswith("_mean") or k.endswith("_max") for k in metrics.keys()):
            continue

        if x_value not in data_by_x:
            data_by_x[x_value] = {name: [] for name in base_metric_names}

        # Collect metrics across all k values
        for base_name in base_metric_names:
            # Find all k values for this metric
            k = 0
            while True:
                metric_key = f"k{k}/{base_name}"
                if metric_key in metrics:
                    data_by_x[x_value][base_name].append(metrics[metric_key])
                    k += 1
                else:
                    break

    if not data_by_x:
        print("No valid data collected from child runs")
        return figures

    # Sort x values and restructure data for plotting
    x_values = sorted(data_by_x.keys())

    # Restructure: {metric_key: [list_of_values_at_x0, list_of_values_at_x1, ...]}
    scores: dict[str, list[list[float]]] = {}
    for base_name in base_metric_names:
        scores[base_name] = [data_by_x[x][base_name] for x in x_values]

    # Find unique base keys for grouping (e.g., 'sm_60', 'md_60')
    base_keys = sorted(set(
        "_".join(name.split("_")[:2]) for name in base_metric_names
        if name.endswith("_mean") or name.endswith("_max")
    ))

    size_map = {"sm": "Small", "md": "Medium", "lg": "Large"}

    # Generate plots for each base key
    with mlflow.start_run(run_id=parent_run_id):
        for base_key in base_keys:
            mean_key = f"{base_key}_mean"
            max_key = f"{base_key}_max"

            if mean_key not in scores or max_key not in scores:
                continue

            size, angle = base_key.split("_")
            base_title = f"{size_map.get(size, size)} ratemap, {angle}° angle"

            # Boxplot
            fig_box = sweep_boxplot(scores, x_values, x_param, log_x, base_key, base_title, n_neurons_str, n_samples_str)
            log_figure(fig_box, f"sweep_boxplot_{base_key}")
            figures[f"boxplot_{base_key}"] = fig_box

            # Violin plot
            fig_violin = sweep_violin_plot(scores, x_values, x_param, log_x, base_key, base_title, n_neurons_str, n_samples_str)
            log_figure(fig_violin, f"sweep_violin_{base_key}")
            figures[f"violin_{base_key}"] = fig_violin

            # Quantile plot
            fig_iqr = sweep_iqr_plot(scores, x_values, x_param, log_x, base_key, base_title, n_neurons_str, n_samples_str)
            log_figure(fig_iqr, f"sweep_quantile_{base_key}")
            figures[f"quantile_{base_key}"] = fig_iqr

    print(f"Sweep score distribution plots logged to MLflow for run {parent_run_id}")
    return figures