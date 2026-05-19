import os
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
from numpy.typing import NDArray
from sklearn.decomposition import PCA
import umap
import logging

import matplotlib.cm as mpl_cm
from matplotlib.figure import Figure as MplFigure


def _mpl_colorscale(name: str, n: int = 16) -> list:
    """Convert a matplotlib colormap to a Plotly colorscale."""
    cmap = mpl_cm.get_cmap(name)
    return [
        [i / (n - 1), 'rgb({},{},{})'.format(int(r*255), int(g*255), int(b*255))]
        for i, (r, g, b, _) in enumerate(cmap(np.linspace(0, 1, n)))
    ]

from scores import GridScorer
from data import TrajectoryGenerator


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


def quantitative_analysis(Vs: list[np.ndarray], widths: tuple, res: int = 70) -> tuple[list[go.Figure], dict]:
    """Compute grid scores and produce one joint distribution plot per spatial scale.

    Each figure: scatter (score_60 vs score_90, colored by module) with top histogram
    (score_60 projection) and right histogram (score_90 projection). Scatter is square
    with equal axes.

    Args:
        Vs: List of ratemaps [V_small, V_medium, V_large]
        widths: Tuple of box widths corresponding to each ratemap
        res: Resolution of ratemaps

    Returns:
        Tuple of (list of figures [fig_sm, fig_md, fig_lg], scores_dict)
    """
    import plotly.colors as pc

    scores = {}
    scale_names = ["sm", "md", "lg"]
    scale_titles = ["Small", "Medium", "Large"]
    all_score_60 = []
    all_score_90 = []
    all_labels = []

    MODULE_COLORS = pc.qualitative.Plotly

    for idx, V in enumerate(Vs):
        maps = [V[i, :] for i in range(V.shape[0])]
        starts = [0.2] * 10
        ends = np.linspace(0.4, 1.0, num=10)
        box_width = widths[idx]
        coord_range = ((-box_width / 2, box_width / 2), (-box_width / 2, box_width / 2))
        scorer = GridScorer(res, coord_range, zip(starts, ends.tolist()))

        results = [scorer.get_scores(rm.reshape(res, res)) for rm in tqdm(maps, desc=f"Scoring {scale_titles[idx]}")]
        score_60, score_90, *_ = zip(*results)
        score_60 = np.nan_to_num(score_60)
        score_90 = np.nan_to_num(score_90)
        all_score_60.append(score_60)
        all_score_90.append(score_90)

        sacs = [scorer.calculate_sac(rm.reshape(res, res)) for rm in maps]
        labels, _ = scorer.get_modules(sacs, max_m=15)
        all_labels.append(labels)

        prefix = scale_names[idx]
        scores[f"{prefix}_60"] = score_60
        scores[f"{prefix}_90"] = score_90
        scores[f"{prefix}_60_max"] = np.max(score_60)
        scores[f"{prefix}_60_mean"] = np.mean(score_60)
        scores[f"{prefix}_90_max"] = np.max(score_90)
        scores[f"{prefix}_90_mean"] = np.mean(score_90)
        scores[f"{prefix}_labels"] = labels

    scores_60_all = np.stack([scores["sm_60"], scores["md_60"], scores["lg_60"]], axis=0)
    scores_90_all = np.stack([scores["sm_90"], scores["md_90"], scores["lg_90"]], axis=0)
    max_60 = np.max(scores_60_all, axis=0)
    max_90 = np.max(scores_90_all, axis=0)
    scores["pattern_type"] = np.where(max_90 > max_60, 90, 60)
    scores["count_60"] = np.sum(scores["pattern_type"] == 60)
    scores["count_90"] = np.sum(scores["pattern_type"] == 90)

    SCORE_RANGE = [-1, 1.5]
    BIN_SIZE = 4 / 15

    # Per-figure layout: 2 rows × 2 cols, None at (1,2)
    # Non-None subplot order: (1,1)=1, (2,1)=2, (2,2)=3
    # top_hist=1 → xaxis; scatter=2 → xaxis2/yaxis2; right_hist=3 → xaxis3/yaxis3
    specs = [
        [{"type": "histogram"}, None],
        [{"type": "scatter"}, {"type": "histogram"}],
    ]

    figs = []
    for i in range(3):
        s60, s90 = all_score_60[i], all_score_90[i]
        labels = all_labels[i]
        n_modules = int(labels.max()) + 1

        fig = make_subplots(
            rows=2, cols=2,
            specs=specs,
            column_widths=[3, 1],
            row_heights=[1, 3],
            horizontal_spacing=0.03,
            vertical_spacing=0.04,
        )

        # scatter colored by module
        for m in range(n_modules):
            mask = labels == m
            color = MODULE_COLORS[m % len(MODULE_COLORS)]
            fig.add_trace(
                go.Scatter(x=s60[mask], y=s90[mask], mode='markers',
                           marker=dict(size=8, color=color, opacity=0.6),
                           name=f"{m}", showlegend=True),
                row=2, col=1,
            )

        # top histogram (score_60, single)
        fig.add_trace(
            go.Histogram(x=s60, xbins=dict(start=SCORE_RANGE[0], end=SCORE_RANGE[1], size=BIN_SIZE),
                         marker_color='steelblue', showlegend=False),
            row=1, col=1,
        )

        # right histogram (score_90, horizontal, single)
        fig.add_trace(
            go.Histogram(y=s90, ybins=dict(start=SCORE_RANGE[0], end=SCORE_RANGE[1], size=BIN_SIZE),
                         marker_color='steelblue', showlegend=False, orientation='h'),
            row=2, col=2,
        )

        # link axes: top-hist x → scatter x; right-hist y → scatter y
        fig.layout.xaxis.matches = 'x2'
        fig.layout.yaxis3.matches = 'y2'

        # diagonal x=y line
        fig.add_trace(
            go.Scatter(x=SCORE_RANGE, y=SCORE_RANGE, mode='lines',
                       line=dict(color='gray', dash='dash', width=1),
                       showlegend=False),
            row=2, col=1,
        )

        ticks = [-1, -0.5, 0, 0.5, 1.0, 1.5]
        fig.update_xaxes(range=SCORE_RANGE, tickvals=ticks, title_text="60° score",
                         title_font=dict(size=24), tickfont=dict(size=20), row=2, col=1)
        fig.update_yaxes(range=SCORE_RANGE, tickvals=ticks, title_text="90° score",
                         title_font=dict(size=24), tickfont=dict(size=20),
                         scaleanchor='x2', scaleratio=1, row=2, col=1)
        fig.update_xaxes(showticklabels=False, row=1, col=1)
        fig.update_yaxes(showticklabels=False, row=2, col=2)
        # count axis tick fonts on histograms
        fig.update_yaxes(tickfont=dict(size=20), row=1, col=1)
        fig.update_xaxes(tickfont=dict(size=20), row=2, col=2)

        # row_heights=[1,3], spacing=0.04 → scatter top ≈ 0.72 in paper coords
        fig.update_layout(
            height=600, width=650,
            barmode='overlay',
            legend=dict(x=0.02, y=0.70, xanchor='left', yanchor='top',
                        font=dict(size=20), bgcolor='rgba(255,255,255,0.7)'),
            margin=dict(l=10, r=10, t=10, b=10),
        )

        figs.append(fig)

    return figs, scores

def manifold_cloud(
    embedding: NDArray[np.floating],
    positions: NDArray[np.floating],
    m: int,
    n_components: int,
    traj_idx: int = 0,
) -> go.Figure:
    seq_len = positions.shape[1] if positions.ndim == 3 else None

    positions = positions.reshape(-1, 2)
    pos_normalized = (positions - positions.min(axis=0)) / (positions.max(axis=0) - positions.min(axis=0))
    # 2D teal-green colormap via bilinear interpolation across four corners
    c00 = np.array([1.00, 1.00, 1.00])
    c10 = np.array([0.00, 1.00, 0.00])
    c01 = np.array([0.10, 0.00, 1.00])
    c11 = np.array([0.00, 1.00, 1.00])
    x = pos_normalized[:, 0:1]
    y = pos_normalized[:, 1:2]
    pos_colors = (
        (1 - x) * (1 - y) * c00
        + x       * (1 - y) * c10
        + (1 - x) * y       * c01
        + x       * y       * c11
    )

    centered = embedding - embedding.mean(axis=0)

    fig = go.Figure(
        data=[go.Scatter3d(
            x=centered[:, 0],
            y=centered[:, 1],
            z=centered[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=['rgb({},{},{})'.format(int(r*255), int(g*255), int(b*255))
                    for r, g, b in pos_colors],
                opacity=0.6
            ),
            name='All points',
        )]
    )

    if seq_len is not None:
        start = traj_idx * seq_len
        traj_emb = centered[start : start + seq_len]
        fig.add_trace(go.Scatter3d(
            x=traj_emb[:, 0],
            y=traj_emb[:, 1],
            z=traj_emb[:, 2],
            mode='lines',
            # marker=dict(
            #     size=3,
            #     color=np.arange(seq_len),
            #     colorscale=_mpl_colorscale('autumn'),
            #     colorbar=dict(title='Time step'),
            #     opacity=1.0,
            # ),
            line=dict(
                color=np.arange(seq_len),
                colorscale=_mpl_colorscale('autumn'),
                width=6,
            ),
            name=f'Trajectory {traj_idx}',
        ))

    fig.update_layout(
        showlegend=False,
        scene=dict(
            xaxis=dict(title=dict(text="x", font=dict(size=20)), showticklabels=False),
            yaxis=dict(title=dict(text="y", font=dict(size=20)), showticklabels=False),
            zaxis=dict(title=dict(text="z", font=dict(size=20)), showticklabels=False),
        ),
    )
    return fig

def manifold_slice(
    embedding: NDArray[np.floating],
    axes: tuple[int, int],
    m: int,
    n_components: int,
    epsilon: float = 0.1,
) -> go.Figure:
    n_dims = embedding.shape[1]
    assert all(0 <= ax < n_dims for ax in axes), f"axes {axes} out of range for embedding with {n_dims} dimensions"
    assert axes[0] != axes[1], "axes must be distinct"

    # Translate to center of mass
    centered = embedding - embedding.mean(axis=0)

    ax0, ax1 = axes
    other_ax = next(i for i in range(n_dims) if i not in axes)

    # Keep only points within epsilon of the slice plane
    dist_to_plane = np.abs(centered[:, other_ax])
    mask = dist_to_plane <= epsilon
    sliced = centered[mask]
    sliced_dist = dist_to_plane[mask]

    # Monochrome: dark (0) at plane, light (255) at epsilon
    brightness = (sliced_dist / epsilon * 220).astype(int)
    colors = ['rgb({v},{v},{v})'.format(v=int(b)) for b in brightness]

    fig = go.Figure(
        data=[go.Scatter(
            x=sliced[:, ax0],
            y=sliced[:, ax1],
            mode='markers',
            marker=dict(
                size=4,
                color=colors,
            ),
        )]
    )
    axis_labels = ['x', 'y', 'z']
    fig.update_layout(
        xaxis_title=f"{axis_labels[ax0]}",
        yaxis_title=f"{axis_labels[ax1]}",
        xaxis=dict(scaleanchor='y', scaleratio=1, showticklabels=False, title=axis_labels[ax0], title_font=dict(size=20)),
        yaxis=dict(showticklabels=False, title=axis_labels[ax1], title_font=dict(size=20)),
        height=600,
        width=600,
    )
    return fig

def manifold(
    positions: NDArray[np.floating],
    representations: NDArray[np.floating],
    m: int,
    traj_idx: int = 0,
) -> tuple[go.Figure, go.Figure, go.Figure, go.Figure, go.Figure, MplFigure, NDArray[np.floating], NDArray[np.floating]] | None:
    """Compute PCA and PaCMAP embedding of neural representations.

    Args:
        positions: 2D positions in the environment, shape (N, 2).
        representations: Neural activations at each position, shape (N, Dm)
            where D is the latent dimension.
        m: Module index

    Returns:
        Tuple of (manifold_fig, scree_fig, slice_xy, slice_xz, slice_yz, positions, embedding)
        where manifold_fig is a 3D scatter plot colored by 2D position, scree_fig shows
        PCA explained variance, slice_xy/xz/yz are 2D slice plots for each plane,
        positions are the input positions (N, 2), and embedding is the UMAP output (N, 3).
        Returns None if embedding fails.
    """
    pca = PCA(n_components=0.95)
    pcs = pca.fit_transform(representations)

    # Scree plot
    n_components = len(pca.explained_variance_ratio_)
    pc_numbers = list(range(1, n_components + 1))
    scree_fig = go.Figure()
    scree_fig.add_trace(go.Bar(
        x=pc_numbers,
        y=pca.explained_variance_ratio_ * 100,
        name='Individual',
        opacity=0.7,
    ))
    scree_fig.add_trace(go.Scatter(
        x=pc_numbers,
        y=np.cumsum(pca.explained_variance_ratio_) * 100,
        mode='lines+markers',
        name='Cumulative',
        marker=dict(color='red'),
        line=dict(color='red'),
    ))
    scree_fig.update_layout(
        title=f'Scree Plot - Module {m} ({n_components} components)',
        xaxis_title='Principal Component',
        yaxis_title='Variance Explained (%)',
        height=500,
        width=800,
    )

    reducer = umap.UMAP(
        n_components=3,
        metric='cosine',
        n_neighbors=500,
        min_dist=0.8,
        init='spectral',
        n_jobs=24
    )

    embedding: NDArray[np.floating] = reducer.fit_transform(pcs)

    manifold_fig = manifold_cloud(embedding, positions, m, n_components, traj_idx)
    slice_xy = manifold_slice(embedding, (0, 1), m, n_components)
    slice_xz = manifold_slice(embedding, (0, 2), m, n_components)
    slice_yz = manifold_slice(embedding, (1, 2), m, n_components)

    traj_fig, _ = TrajectoryGenerator().visualize_trajectory_time(
        positions[traj_idx] if positions.ndim == 3 else positions,
        background='colormap', cmap='Wistia'
    )

    return manifold_fig, scree_fig, slice_xy, slice_xz, slice_yz, traj_fig, positions, embedding


def loss_plots(
    train_losses: dict[str, np.ndarray],
    lambda_pos: Optional[np.ndarray] = None,
    lambda_norm: Optional[np.ndarray] = None,
) -> go.Figure:
    """Generate loss evolution plots.

    Args:
        train_losses: Dict with keys 'loss', 'separation', 'positivity', 'norm'
        lambda_pos: Optional array of lambda_pos values over iterations
        lambda_norm: Optional array of lambda_norm values over iterations
    """
    keys = ['loss', 'separation', 'positivity', 'norm']
    titles = ['Loss', 'Separation', 'Positivity', 'Norm']

    specs = [[]]
    for key in keys:
        has_secondary = (key == 'positivity' and lambda_pos is not None) or (key == 'norm' and lambda_norm is not None)
        specs[0].append({"secondary_y": has_secondary})

    fig = make_subplots(rows=1, cols=4, subplot_titles=titles, specs=specs,
                        horizontal_spacing=0.05)

    for counter, key in enumerate(keys):
        if key not in train_losses:
            continue

        train_data = train_losses[key]
        x_train = list(range(len(train_data)))
        col = counter + 1

        fig.add_trace(
            go.Scatter(x=x_train, y=train_data, mode='lines', name='Train',
                       line=dict(color='blue'), showlegend=False),
            row=1, col=col
        )

        if key == 'positivity' and lambda_pos is not None:
            fig.add_trace(
                go.Scatter(x=list(range(len(lambda_pos))), y=lambda_pos, mode='lines',
                           name='λ<sub>pos</sub>', line=dict(color='red', dash='dash'),
                           opacity=0.8, showlegend=False),
                row=1, col=col, secondary_y=True
            )
            fig.update_yaxes(title_text="λ<sub>pos</sub>", secondary_y=True, row=1, col=col,
                             tickfont=dict(color='red', size=20), title_font=dict(color='red', size=20), title_standoff=5)

        elif key == 'norm' and lambda_norm is not None:
            fig.add_trace(
                go.Scatter(x=list(range(len(lambda_norm))), y=lambda_norm, mode='lines',
                           name='λ<sub>norm</sub>', line=dict(color='red', dash='dash'),
                           opacity=0.8, showlegend=False),
                row=1, col=col, secondary_y=True
            )
            fig.update_yaxes(title_text="λ<sub>norm</sub>", secondary_y=True, row=1, col=col,
                             tickfont=dict(color='red', size=20), title_font=dict(color='red', size=20), title_standoff=5)

    fig.update_xaxes(title_text="Epoch", title_standoff=5, title_font=dict(size=20), tickfont=dict(size=20))
    y_titles = {
        'Loss': 'Loss',
        'Separation': 'Separation',
        'Positivity': 'Positivity',
        'Norm': 'Norm',
    }
    for counter, title in enumerate(titles):
        col = counter + 1
        fig.update_yaxes(row=1, col=col, secondary_y=False,
                         tickfont=dict(color='blue', size=20))

    fig.update_layout(height=250, width=1400, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))

    for ann in fig.layout.annotations:
        ann.font.size = 22

    return fig


def neuron_plotter_2d(
    V: np.ndarray,
    res: int,
    scores: Optional[np.ndarray] = None,
    show_colorbar: bool = True,
) -> go.Figure:
    """Plot individual neuron ratemaps.

    Args:
        V: Ratemap array of shape (n_neurons, res*res)
        res: Resolution of ratemaps
        scores: Optional array of scores for each neuron; if None, no titles shown
        show_colorbar: Whether to show the colorbar on the first heatmap
    """
    D = V.shape[0]
    RowsD = int(np.ceil(np.sqrt(D)))
    ColumnsD = int(np.ceil(D / RowsD))

    vertical_spacing = 0.03 if scores is not None else 0.01
    subplot_titles = [f'{scores[i]:.3f}' for i in range(D)] if scores is not None else None
    fig = make_subplots(rows=RowsD, cols=ColumnsD, subplot_titles=subplot_titles,
                        horizontal_spacing=0.001, vertical_spacing=vertical_spacing)

    vmin, vmax = V.min(), V.max()

    for neuron in range(D):
        row = neuron // ColumnsD + 1
        col = neuron % ColumnsD + 1
        heatmap = go.Heatmap(
            z=np.reshape(V[neuron, :], [res, res]),
            zmin=vmin, zmax=vmax, colorscale='Viridis',
            showscale=(neuron == 0 and show_colorbar)
        )
        fig.add_trace(heatmap, row=row, col=col)

        axis_idx = neuron + 1
        axis_suffix = "" if axis_idx == 1 else str(axis_idx)
        fig.update_xaxes(showticklabels=False, showgrid=False, constrain='domain', row=row, col=col)
        fig.update_yaxes(showticklabels=False, showgrid=False, scaleanchor=f'x{axis_suffix}',
                         scaleratio=1, constrain='domain', row=row, col=col)
    t_margin = 40 if scores is not None else 10
    fig.update_layout(height=800, width=800, margin=dict(l=10, r=10, t=t_margin, b=10))
    return fig


def module_ratemaps_plot(
    V: np.ndarray,
    res: int,
    scores: np.ndarray,
    labels: NDArray[np.intp],
    max_cols: int = 8,
) -> go.Figure:
    """Plot ratemaps organized by module with grid scores.

    Args:
        V: Ratemap array of shape (n_neurons, res*res).
        res: Resolution of ratemaps.
        scores: Grid scores for each neuron, shape (n_neurons,).
        labels: Module assignment for each neuron, shape (n_neurons,).
        max_cols: Maximum columns per module before wrapping.

    Returns:
        Figure with ratemaps arranged by module, each module having its own
        2D grid that wraps after max_cols columns.
    """
    n_modules = int(max(labels)) + 1
    module_indices = [np.where(labels == m)[0] for m in range(n_modules)]

    # Calculate rows needed per module and total rows
    rows_per_module = [int(np.ceil(len(idx) / max_cols)) for idx in module_indices]
    total_rows = sum(rows_per_module)
    n_cols = min(max_cols, max(len(idx) for idx in module_indices))

    # Build subplot titles
    subplot_titles = []
    for m in range(n_modules):
        n_neurons = len(module_indices[m])
        module_rows = rows_per_module[m]
        for i in range(module_rows * n_cols):
            if i < n_neurons:
                neuron_idx = module_indices[m][i]
                subplot_titles.append(f'{scores[neuron_idx]:.2f}')
            else:
                subplot_titles.append('')

    # Calculate row heights with spacing between modules
    row_heights = []
    for m in range(n_modules):
        for _ in range(rows_per_module[m]):
            row_heights.append(1)

    fig = make_subplots(
        rows=total_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.005,
        vertical_spacing=0.04,
        row_heights=row_heights,
    )

    vmin, vmax = V.min(), V.max()

    current_row = 1
    for m in range(n_modules):
        module_neuron_indices = module_indices[m]
        module_scores = scores[module_neuron_indices]
        avg_score = np.mean(module_scores)

        for i, neuron_idx in enumerate(module_neuron_indices):
            row = current_row + i // n_cols
            col = i % n_cols + 1

            heatmap = go.Heatmap(
                z=np.reshape(V[neuron_idx, :], [res, res]),
                zmin=vmin, zmax=vmax,
                colorscale='Viridis',
                showscale=False,
            )
            fig.add_trace(heatmap, row=row, col=col)

            axis_idx = (row - 1) * n_cols + col
            axis_suffix = "" if axis_idx == 1 else str(axis_idx)
            fig.update_xaxes(showticklabels=False, showgrid=False, constrain='domain', row=row, col=col)
            fig.update_yaxes(showticklabels=False, showgrid=False, scaleanchor=f'x{axis_suffix}',
                             scaleratio=1, constrain='domain', row=row, col=col)

        # Add module label with average score on the left
        module_y = 1 - (current_row - 0.5) / total_rows
        fig.add_annotation(
            x=-0.02, y=module_y,
            xref='paper', yref='paper',
            text=f'<b>Module {m}</b><br>Avg: {avg_score:.3f}',
            showarrow=False,
            font=dict(size=14),
            xanchor='right',
            align='right',
        )

        current_row += rows_per_module[m]

    fig.update_layout(
        height=120 * total_rows,
        width=100 * n_cols + 100,
        showlegend=False,
        margin=dict(l=120),
    )
    return fig


def log_figure(fig: go.Figure | MplFigure, name: str) -> None:
    """Log a figure to MLflow. Plotly: PNG + HTML. Matplotlib: PNG only."""
    for logger_name in ('kaleido', 'choreographer'):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    mlflow.log_figure(fig, f"figures/{name}.png")
    if isinstance(fig, go.Figure):
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

    fig = loss_plots(
        train_losses,
        lambda_pos=fetch_metric(lambda_metrics["lambda_pos"]),
        lambda_norm=fetch_metric(lambda_metrics["lambda_norm"]),
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
        xaxis_title='ω<sub>x</sub>', yaxis_title='ω<sub>y</sub>',
        xaxis=dict(scaleanchor='y', scaleratio=1, zeroline=True, zerolinewidth=1, zerolinecolor='black',
                   title_font=dict(size=28), tickfont=dict(size=28)),
        yaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor='black',
                   title_font=dict(size=28), tickfont=dict(size=28)),
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
            go.Heatmap(z=mat, colorscale='Viridis', showscale=(i == 2),
                       colorbar=dict(tickfont=dict(size=20))),
            row=1, col=i + 1
        )
        axis_suffix = "" if i == 0 else str(i + 1)
        fig.update_xaxes(constrain='domain', tickfont=dict(size=20), row=1, col=i + 1)
        fig.update_yaxes(scaleanchor=f'x{axis_suffix}', scaleratio=1,
                         constrain='domain', autorange='reversed', tickfont=dict(size=20), row=1, col=i + 1)
    for ann in fig.layout.annotations:
        ann.font.size = 24
    fig.update_layout(height=350, width=800)
    return fig

def grid_type(
    grid_scores_60: NDArray[np.floating],
    grid_scores_90: NDArray[np.floating],
    modules: NDArray[np.intp],
) -> MplFigure:
    import matplotlib.pyplot as plt

    n_modules = int(max(modules)) + 1
    colors = plt.cm.hsv(np.linspace(0, 1, n_modules, endpoint=False))

    raw_lo = min(grid_scores_90.min(), grid_scores_60.min())
    raw_hi = max(grid_scores_90.max(), grid_scores_60.max())
    pad = 0.05 * (raw_hi - raw_lo)
    lo = raw_lo - pad
    hi = raw_hi + pad

    fig, ax = plt.subplots(figsize=(6, 6))

    for m in range(n_modules):
        mask = modules == m
        ax.scatter(grid_scores_90[mask], grid_scores_60[mask],
                   color=colors[m], s=15, alpha=0.7, label=f'Module {m}')

    ax.plot([lo, hi], [lo, hi], '--', color='grey', linewidth=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Grid score (90°)', fontsize=14)
    ax.set_ylabel('Grid score (60°)', fontsize=14)
    ax.legend(fontsize=12, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0, frameon=False)
    fig.tight_layout(pad=0)

    return fig


def _barcode_figure(
    dgms: list,
    module_idx: int,
    k: int,
    n_landmarks: int,
    n_pcs: int,
) -> go.Figure:
    dim_colors = ['steelblue', 'tomato', 'forestgreen']
    dim_names = ['H\u2080', 'H\u2081', 'H\u2082']

    all_finite_deaths = [
        d for dgm in dgms for d in dgm[np.isfinite(dgm[:, 1]), 1].tolist()
    ]
    x_max = max(all_finite_deaths) * 1.05 if all_finite_deaths else 1.0

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=dim_names,
        vertical_spacing=0.06,
        shared_xaxes=True,
    )

    for dim, (dgm, color) in enumerate(zip(dgms, dim_colors)):
        if len(dgm) == 0:
            continue
        lifetimes = dgm[:, 1] - dgm[:, 0]
        top30_idx = np.argsort(lifetimes)[::-1][:30]
        dgm_top = dgm[top30_idx]

        for i, (birth, death) in enumerate(dgm_top):
            death_plot = x_max if not np.isfinite(death) else death
            y = i * 1.5
            fig.add_trace(
                go.Scatter(
                    x=[birth, death_plot],
                    y=[y, y],
                    mode='lines',
                    line=dict(color=color, width=2),
                    showlegend=False,
                ),
                row=dim + 1, col=1,
            )

    fig.update_xaxes(
        title_text='Filtration value', range=[0, x_max],
        row=3, col=1,
        title_font=dict(size=16), tickfont=dict(size=14),
    )
    for dim in range(3):
        fig.update_yaxes(
            showticklabels=False, showgrid=False, zeroline=False,
            row=dim + 1, col=1,
        )
    for ann in fig.layout.annotations:
        ann.font.size = 18
    fig.update_layout(
        height=500, width=400,
        showlegend=False,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def persistent_homology_analysis(
    pcs: NDArray[np.floating],
    module_idx: int,
    k: int,
    n_landmarks: int = 500,
    thresh: float = np.inf,
) -> tuple[go.Figure, dict]:
    """Verify torus topology via persistent cohomology on PCA space.

    Follows Gardner et al.: apply Ripser to the PCA intermediate representation
    (not the UMAP output), using landmark-based (maxmin) downsampling and
    Z_47 coefficients. Expected torus signature: \u03b2=(1,2,1).

    Args:
        pcs: PCA-projected activations for one module, shape (N, n_components).
        module_idx: Module index for labeling.
        k: Run index for labeling.
        n_landmarks: Target number of maxmin landmark points (default 500).
        thresh: Max filtration radius passed to ripser; caps simplex explosion at maxdim=2.

    Returns:
        Tuple of (barcode_figure, metrics_dict).
    """
    try:
        from ripser import Rips
    except ImportError:
        logging.warning('ripser not installed — skipping persistent homology analysis')
        return go.Figure(), {}

    n_pc = pcs.shape[1]

    # --- Maxmin landmark subsampling + Ripser with Z_47 coefficients ---
    # Rips(n_perm=n_lm) runs a greedy furthest-point permutation internally,
    # selecting n_lm landmarks from the point cloud before building the filtration.
    N = pcs.shape[0]
    n_lm = min(n_landmarks, N)

    rips = Rips(maxdim=2, coeff=47, n_perm=n_lm, thresh=thresh, verbose=False)
    dgms = rips.fit_transform(pcs, metric='cosine')

    # --- 5. Parse barcodes and compute torus metrics ---
    metrics: dict[str, float] = {}

    def _sorted_lifetimes(dgm: NDArray) -> NDArray:
        lt = dgm[:, 1] - dgm[:, 0]
        return np.sort(lt[np.isfinite(lt)])[::-1]

    h0_lt = _sorted_lifetimes(dgms[0])
    h1_lt = _sorted_lifetimes(dgms[1])
    h2_lt = _sorted_lifetimes(dgms[2]) if len(dgms) > 2 else np.array([])

    # Torus ratio: second-longest H1 / third-longest H1 — large = clean torus signal
    if len(h1_lt) >= 3 and h1_lt[2] > 0:
        h1_torus_ratio = float(h1_lt[1] / h1_lt[2])
    elif len(h1_lt) >= 2:
        h1_torus_ratio = float('inf')
    else:
        h1_torus_ratio = 0.0

    prefix = f'k{k}/topo_m{module_idx}'
    metrics[f'{prefix}_h0_bars']         = float(len(dgms[0]))
    metrics[f'{prefix}_h1_bars']         = float(len(dgms[1]))
    metrics[f'{prefix}_h2_bars']         = float(len(dgms[2])) if len(dgms) > 2 else 0.0
    metrics[f'{prefix}_h1_bar1']         = float(h1_lt[0]) if len(h1_lt) > 0 else 0.0
    metrics[f'{prefix}_h1_bar2']         = float(h1_lt[1]) if len(h1_lt) > 1 else 0.0
    metrics[f'{prefix}_h1_bar3']         = float(h1_lt[2]) if len(h1_lt) > 2 else 0.0
    metrics[f'{prefix}_h1_torus_ratio']  = h1_torus_ratio if np.isfinite(h1_torus_ratio) else -1.0
    metrics[f'{prefix}_h2_bar1']         = float(h2_lt[0]) if len(h2_lt) > 0 else 0.0
    metrics[f'{prefix}_n_pcs']       = float(n_pc)
    metrics[f'{prefix}_n_landmarks'] = float(n_lm)

    fig = _barcode_figure(dgms, module_idx, k, n_lm, n_pc)
    return fig, metrics


def generate_2d_plots(model: nn.Module, k: int = 0, generate_manifold = False) -> dict:
    """Generate analysis plots for a trained 2D model.

    Args:
        model: Trained ActionableRGM model
        k: Run index for labeling
        generate_manifold: Whether to run manifold analysis or not

    Returns:
        Dictionary of scores
    """
    model.eval()

    # Frequency plot (ActionableRGM only)
    if hasattr(model, 'om'):
        om: np.ndarray = model.om.detach().cpu().numpy()
        fig_freq = frequency_plot(om)
        log_figure(fig_freq, f"freq_plot_k{k}")

    # S matrix analysis (ActionableRGM only)
    if hasattr(model, 'S'):
        S: np.ndarray = model.S.detach().cpu().numpy()
        fig_S = s_matrix_plot(S)
        log_figure(fig_S, f"S_analysis_k{k}")

    # Grid scores
    res = 70
    widths = (1, 2, 4)
    Vs = get_ratemaps(model, res, widths)
    V_small, V_medium, V_large = Vs

    figs_scores, scores = quantitative_analysis(Vs, widths, res)
    for scale_name, fig_scores in zip(["sm", "md", "lg"], figs_scores):
        log_figure(fig_scores, f"grid_scores_{scale_name}_k{k}")

    # Log scalar scores as metrics
    for key, value in scores.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(f"k{k}/{key}", value)

    # Neuron plots
    neuron_sm_fig_60 = neuron_plotter_2d(V_small, res, scores["sm_60"])
    log_figure(neuron_sm_fig_60, f"neurons_small_k{k}_scores_60")

    neuron_md_fig_60 = neuron_plotter_2d(V_medium, res, scores["md_60"])
    log_figure(neuron_md_fig_60, f"neurons_medium_k{k}_scores_60")

    neuron_lg_fig_60 = neuron_plotter_2d(V_large, res, scores["lg_60"])
    log_figure(neuron_lg_fig_60, f"neurons_large_k{k}_scores_60")

    neuron_sm_fig_90 = neuron_plotter_2d(V_small, res, scores["sm_90"])
    log_figure(neuron_sm_fig_90, f"neurons_small_k{k}_scores_90")

    neuron_md_fig_90 = neuron_plotter_2d(V_medium, res, scores["md_90"])
    log_figure(neuron_md_fig_90, f"neurons_medium_k{k}_scores_90")

    neuron_lg_fig_90 = neuron_plotter_2d(V_large, res, scores["lg_90"])
    log_figure(neuron_lg_fig_90, f"neurons_large_k{k}_scores_90")

    # Neuron plots without scores/colorbar
    log_figure(neuron_plotter_2d(V_small, res, show_colorbar=False), f"neurons_small_k{k}_plain")
    log_figure(neuron_plotter_2d(V_medium, res, show_colorbar=False), f"neurons_medium_k{k}_plain")
    log_figure(neuron_plotter_2d(V_large, res, show_colorbar=False), f"neurons_large_k{k}_plain")

    # Module extraction and manifold analysis using large ratemaps
    large_width = widths[2]
    coord_range = ((-large_width / 2, large_width / 2), (-large_width / 2, large_width / 2))
    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    masks_parameters = zip(starts, ends.tolist())
    scorer = GridScorer(res, coord_range, masks_parameters)

    # Compute SACs from large ratemaps
    sacs = [scorer.calculate_sac(V_large[i, :].reshape(res, res)) for i in range(V_large.shape[0])]
    labels, _ = scorer.get_modules(sacs, max_m=15)
    n_modules = int(max(labels)) + 1
    mlflow.log_metric(f"k{k}/n_modules", n_modules)

    # Plot ratemaps organized by module
    module_fig = module_ratemaps_plot(V_large, res, scores["lg_60"], labels)
    log_figure(module_fig, f"module_ratemaps_k{k}")

    for prefix in ["sm", "md", "lg"]:
        grid_type_fig = grid_type(scores[f"{prefix}_60"], scores[f"{prefix}_90"], np.array(labels))
        log_figure(grid_type_fig, f"grid_type_{prefix}_k{k}")

    if not generate_manifold:
        return scores

    # Check which modules already have precomputed manifold artifacts
    client = MlflowClient()
    active_run = mlflow.active_run()
    run_id = active_run.info.run_id if active_run is not None else None

    def _artifact_exists(artifact_path: str) -> bool:
        if run_id is None:
            return False
        try:
            parent = "/".join(artifact_path.rsplit("/", 1)[:-1])
            filename = artifact_path.rsplit("/", 1)[-1]
            artifacts = client.list_artifacts(run_id, parent)
            return any(a.path == artifact_path or a.path.endswith("/" + filename) for a in artifacts)
        except Exception:
            return False

    def _load_artifact_npy(artifact_path: str) -> NDArray[np.floating] | None:
        if run_id is None:
            return None
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                local_path = client.download_artifacts(run_id, artifact_path, tmpdir)
                return np.load(local_path)
        except Exception:
            return None

    # Determine which modules have cached embeddings (only UMAP is skipped, not model forward pass)
    modules_need_umap = []
    for module_idx in range(n_modules):
        emb_artifact = f"manifold/k{k}/embedding_module{module_idx}.npy"
        if _artifact_exists(emb_artifact):
            logging.info(f"Manifold embedding found for module {module_idx}, k={k} — skipping UMAP")
        else:
            modules_need_umap.append(module_idx)

    # Always run model forward pass (needed for PCA / scree plot)
    generator = TrajectoryGenerator()
    positions = generator.generate_trajectory(2, 2, 1000, 100)  # (B, L, 2), absolute
    positions_shifted = np.roll(positions, 1, axis=1)
    positions_shifted[:, 0, :] = 0
    data = torch.tensor(positions - positions_shifted, dtype=torch.float32, device=next(model.parameters()).device)
    representations, _ = model(data)
    representations = representations.detach().cpu().numpy()
    B, L, D = representations.shape
    representations = representations.reshape(B * L, D)

    torus_ratios: list[float] = []

    # Manifold analysis per module
    for module_idx in range(n_modules):
        emb_artifact = f"manifold/k{k}/embedding_module{module_idx}.npy"
        module_mask = np.array(labels) == module_idx
        module_representations = representations[:, module_mask]

        # Always compute PCA for scree plot
        pca = PCA(n_components=0.95)
        pcs = pca.fit_transform(module_representations)
        n_components = len(pca.explained_variance_ratio_)
        pc_numbers = list(range(1, n_components + 1))
        scree_fig = go.Figure()
        scree_fig.add_trace(go.Bar(x=pc_numbers, y=pca.explained_variance_ratio_ * 100, name='Individual', opacity=0.7))
        scree_fig.add_trace(go.Scatter(
            x=pc_numbers, y=np.cumsum(pca.explained_variance_ratio_) * 100,
            mode='lines+markers', name='Cumulative',
            marker=dict(color='red'), line=dict(color='red'),
        ))
        scree_fig.update_layout(
            title=f'Scree Plot - Module {module_idx} ({n_components} components)',
            xaxis_title='Principal Component', yaxis_title='Variance Explained (%)',
            height=500, width=800,
        )
        log_figure(scree_fig, f"scree_module{module_idx}_k{k}")

        # Persistent homology on PCA space (before UMAP)
        topo_fig, topo_metrics = persistent_homology_analysis(pcs, module_idx, k)
        log_figure(topo_fig, f"persistence_barcodes_module{module_idx}_k{k}")
        for metric_key, metric_val in topo_metrics.items():
            mlflow.log_metric(metric_key, metric_val)
        torus_ratios.append(topo_metrics.get(f'k{k}/topo_m{module_idx}_h1_torus_ratio', 0.0))

        if module_idx not in modules_need_umap:
            # Load cached embedding, regenerate figures
            module_embedding = _load_artifact_npy(emb_artifact)
            if module_embedding is None:
                logging.warning(f"Failed to load embedding for module {module_idx}, k={k} — skipping")
                continue
            module_positions = positions
        else:
            # Run full UMAP
            reducer = umap.UMAP(n_components=3, metric='cosine', n_neighbors=500, min_dist=0.8, init='spectral', n_jobs=24)
            module_embedding: NDArray[np.floating] = reducer.fit_transform(pcs)
            module_positions = positions
            with tempfile.TemporaryDirectory() as tmpdir:
                pos_path = f"{tmpdir}/positions_module{module_idx}.npy"
                emb_path = f"{tmpdir}/embedding_module{module_idx}.npy"
                np.save(pos_path, module_positions)
                np.save(emb_path, module_embedding)
                mlflow.log_artifact(pos_path, artifact_path=f"manifold/k{k}")
                mlflow.log_artifact(emb_path, artifact_path=f"manifold/k{k}")

        manifold_fig = manifold_cloud(module_embedding, module_positions, module_idx, n_components)
        slice_xy = manifold_slice(module_embedding, (0, 1), module_idx, n_components)
        slice_xz = manifold_slice(module_embedding, (0, 2), module_idx, n_components)
        slice_yz = manifold_slice(module_embedding, (1, 2), module_idx, n_components)
        traj_fig, _ = TrajectoryGenerator().visualize_trajectory_time(
            module_positions[0] if module_positions.ndim == 3 else module_positions,
            background='colormap', cmap='Wistia'
        )
        log_figure(manifold_fig, f"manifold_module{module_idx}_k{k}")
        log_figure(slice_xy, f"manifold_slice_xy_module{module_idx}_k{k}")
        log_figure(slice_xz, f"manifold_slice_xz_module{module_idx}_k{k}")
        log_figure(slice_yz, f"manifold_slice_yz_module{module_idx}_k{k}")
        mlflow.log_figure(traj_fig, f"figures/trajectory_module{module_idx}_k{k}.png")

    # Torus ratio summary across modules
    if torus_ratios:
        ratio_fig = go.Figure(go.Bar(
            x=list(range(len(torus_ratios))),
            y=[r if r >= 0 else float('nan') for r in torus_ratios],
            marker_color='steelblue',
            text=['∞' if r < 0 else f'{r:.2f}' for r in torus_ratios],
            textposition='auto',
        ))
        ratio_fig.update_layout(
            yaxis_title='Torus ratio ξ',
            xaxis_title='Module',
            yaxis=dict(title_font=dict(size=18), tickfont=dict(size=16)),
            xaxis=dict(
                title_font=dict(size=18), tickfont=dict(size=16),
                tickmode='array',
                tickvals=list(range(len(torus_ratios))),
                ticktext=[str(i) for i in range(len(torus_ratios))],
            ),
            height=300, width=400,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        log_figure(ratio_fig, f"torus_ratio_k{k}")

    return scores


def sweep_boxplot(
    scores: dict[str, list[list[float]]],
    x_values: list[float],
    x_param: str,
    base_key: str,
    base_title: str,
    n_neurons_str: str,
    n_samples_str: str,
    log_x: bool = False,
    x_label: Optional[str] = None,
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

    x_label = x_label or x_param
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"Mean ({n_neurons_str.strip()})",
            f"Max ({n_neurons_str.strip()})",
        ],
        horizontal_spacing=0.04,
    )

    for col, key in enumerate([mean_key, max_key], start=1):
        for i, x_val in enumerate(x_values):
            data = scores[key][i]
            fig.add_trace(
                go.Box(
                    y=data,
                    x=[x_val] * len(data),
                    name=str(x_val),
                    showlegend=False,
                    marker_color="black",
                    fillcolor="white",
                    line=dict(color="black", width=1.5),
                    boxpoints="outliers",
                ),
                row=1, col=col
            )

    if log_x:
        fig.update_xaxes(
            title_text=f"{x_label} (log scale)",
            type="log",
            dtick=1,
            ticks="outside",
            minor=dict(dtick="D1", ticks="outside", showgrid=True, gridcolor="white"),
            showgrid=True,
            gridcolor="white",
            exponentformat="power",
            title_font=dict(size=20), tickfont=dict(size=16),
        )
    else:
        fig.update_xaxes(title_text=x_label, title_font=dict(size=20), tickfont=dict(size=16))
    fig.update_xaxes(title_font=dict(size=24), tickfont=dict(size=20), title_standoff=5)
    fig.update_yaxes(title_text="Grid score", title_font=dict(size=24), tickfont=dict(size=20), range=[-0.2, 1.6], title_standoff=5)
    fig.update_yaxes(showticklabels=False, title_text="", row=1, col=2)
    for ann in fig.layout.annotations:
        ann.font.size = 26
    fig.update_layout(height=300, width=700, margin=dict(l=10, r=10, t=40, b=10))

    return fig


def sweep_violin_plot(
    scores: dict[str, list[list[float]]],
    x_values: list[float],
    x_param: str,
    base_key: str,
    base_title: str,
    n_neurons_str: str,
    n_samples_str: str,
    log_x: bool = False,
    x_label: Optional[str] = None,
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

    x_label = x_label or x_param
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"Mean ({n_neurons_str.strip()})",
            f"Max ({n_neurons_str.strip()})",
        ],
        horizontal_spacing=0.04,
    )

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
            title_text=x_label,
            type="log",
            dtick=1,
            ticks="outside",
            minor=dict(dtick="D1", ticks="outside", showgrid=True, gridcolor="white"),
            showgrid=True,
            gridcolor="white",
            exponentformat="power",
            title_font=dict(size=20), tickfont=dict(size=16),
        )
    else:
        fig.update_xaxes(title_text=x_label, title_font=dict(size=20), tickfont=dict(size=16))
    fig.update_xaxes(title_font=dict(size=24), tickfont=dict(size=20), title_standoff=5)
    fig.update_yaxes(title_text="Grid score", title_font=dict(size=24), tickfont=dict(size=20), range=[-0.2, 1.6], title_standoff=5)
    fig.update_yaxes(showticklabels=False, title_text="", row=1, col=2)
    for ann in fig.layout.annotations:
        ann.font.size = 26
    fig.update_layout(height=300, width=700, margin=dict(l=10, r=10, t=40, b=10))

    return fig


def sweep_iqr_plot(
    scores: dict[str, list[list[float]]],
    x_values: list[float],
    x_param: str,
    base_key: str,
    base_title: str,
    n_neurons_str: str,
    n_samples_str: str,
    log_x: bool = False,
    x_label: Optional[str] = None,
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

    x_label = x_label or x_param
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"Mean ({n_neurons_str.strip()})",
            f"Max ({n_neurons_str.strip()})",
        ],
        horizontal_spacing=0.04,
    )

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
            title_text=x_label,
            type="log",
            dtick=1,
            ticks="outside",
            minor=dict(dtick="D1", ticks="outside", showgrid=True, gridcolor="white"),
            showgrid=True,
            gridcolor="white",
            exponentformat="power",
            title_font=dict(size=20), tickfont=dict(size=16),
        )
    else:
        fig.update_xaxes(title_text=x_label, title_font=dict(size=20), tickfont=dict(size=16))
    fig.update_xaxes(title_font=dict(size=24), tickfont=dict(size=20), title_standoff=5)
    fig.update_yaxes(title_text="Grid score", title_font=dict(size=24), tickfont=dict(size=20), range=[-0.2, 1.6], title_standoff=5)
    fig.update_yaxes(showticklabels=False, title_text="", row=1, col=2)
    for ann in fig.layout.annotations:
        ann.font.size = 26
    fig.update_layout(height=320, width=850, margin=dict(l=10, r=10, t=40, b=10))

    return fig


def sweep_score_distributions_mlflow(
    parent_run_id: str,
    x_param: str = "data.seq_len",
    log_x: bool = False,
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
                        n_neurons_str = f" over {int(latent_size)} neurons"
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

            x_label = "Sequence length" if x_param == "data.seq_len" else x_param

            # Boxplot
            fig_box = sweep_boxplot(scores, x_values, x_param, base_key, base_title, n_neurons_str, n_samples_str, log_x=log_x, x_label=x_label)
            log_figure(fig_box, f"sweep_boxplot_{base_key}")
            figures[f"boxplot_{base_key}"] = fig_box

            # Violin plot
            fig_violin = sweep_violin_plot(scores, x_values, x_param, base_key, base_title, n_neurons_str, n_samples_str, log_x=log_x, x_label=x_label)
            log_figure(fig_violin, f"sweep_violin_{base_key}")
            figures[f"violin_{base_key}"] = fig_violin

            # Quantile plot
            fig_iqr = sweep_iqr_plot(scores, x_values, x_param, base_key, base_title, n_neurons_str, n_samples_str, log_x=log_x, x_label=x_label)
            log_figure(fig_iqr, f"sweep_quantile_{base_key}")
            figures[f"quantile_{base_key}"] = fig_iqr

    print(f"Sweep score distribution plots logged to MLflow for run {parent_run_id}")
    return figures


def sweep_heatmap_2d(
    mean_grid: dict[tuple[float, float], list[float]],
    max_grid: dict[tuple[float, float], list[float]],
    x_label: str,
    y_label: str,
) -> go.Figure:
    """Create a 2D heatmap for a two-parameter sweep.

    Args:
        mean_grid: Dict mapping (x_val, y_val) -> list of mean-scores across runs.
        max_grid: Dict mapping (x_val, y_val) -> list of max-scores across runs.
        x_label: Label for x-axis parameter.
        y_label: Label for y-axis parameter.

    Returns:
        Plotly figure with two heatmap subplots (mean of means | mean of maxes).
    """
    all_keys = set(mean_grid) | set(max_grid)
    xs = sorted(set(x for x, y in all_keys))
    ys = sorted(set(y for x, y in all_keys))
    # Use string labels so Plotly treats axes as categorical → equal cell sizes
    xs_str = [str(x) for x in xs]
    ys_str = [str(y) for y in ys]

    def build_matrix(grid: dict[tuple[float, float], list[float]]) -> list[list[float]]:
        return [
            [float(np.mean(grid[(x, y)])) if (x, y) in grid and grid[(x, y)] else float("nan") for x in xs]
            for y in ys
        ]

    def fmt(v: float) -> str:
        return "" if np.isnan(v) else f"{v:.3f}"

    mean_matrix = build_matrix(mean_grid)
    max_matrix = build_matrix(max_grid)
    mean_text = [[fmt(v) for v in row] for row in mean_matrix]
    max_text = [[fmt(v) for v in row] for row in max_matrix]

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Mean Scores", "Max Scores"],
                        horizontal_spacing=0.12)

    for col, (matrix, text) in enumerate([(mean_matrix, mean_text), (max_matrix, max_text)], start=1):
        fig.add_trace(
            go.Heatmap(
                x=xs_str, y=ys_str, z=matrix,
                text=text, texttemplate="%{text}",
                colorscale="RdBu_r", zmin=-2, zmax=2,
                showscale=(col == 2),
                colorbar=dict(tickfont=dict(size=16)),
            ),
            row=1, col=col,
        )

    fig.update_xaxes(title_text=x_label, title_font=dict(size=20), tickfont=dict(size=16))
    fig.update_yaxes(title_text=y_label, title_font=dict(size=20), tickfont=dict(size=16))
    for ann in fig.layout.annotations:
        ann.font.size = 24
    fig.update_layout(height=400, width=1000, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def sweep_score_distributions_mlflow_2d(
    parent_run_id: str,
    x_param: str = "model.sigma_1",
    y_param: str = "model.sigma_2",
    tracking_uri: Optional[str] = "sqlite:///mlruns.db",
) -> dict[str, go.Figure]:
    """Generate 2D heatmap plots of grid scores for a two-parameter sweep.

    Args:
        parent_run_id: MLflow run ID of the parent sweep run.
        x_param: Config parameter path for x-axis (e.g., 'model.sigma_1').
        y_param: Config parameter path for y-axis (e.g., 'model.sigma_2').
        tracking_uri: MLflow tracking URI.

    Returns:
        Dictionary mapping plot names to figure objects.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    figures: dict[str, go.Figure] = {}

    parent_run = client.get_run(parent_run_id)
    experiment_id = parent_run.info.experiment_id

    child_runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
    )

    if not child_runs:
        print(f"No child runs found for parent {parent_run_id}")
        return figures

    # Determine available metric keys
    metric_keys = []
    for run in child_runs:
        metric_keys = [k for k in run.data.metrics if "/" in k and any(
            k.endswith(s) for s in ["_mean", "_max"]
        )]
        if metric_keys:
            break

    if not metric_keys:
        print("No child runs have grid score metrics")
        return figures

    base_metric_names = {mk.split("/")[1] for mk in metric_keys if len(mk.split("/")) == 2}

    # grid[(x_val, y_val)][base_name] = [scores across k]
    grid: dict[tuple[float, float], dict[str, list[float]]] = {}

    for run in child_runs:
        try:
            params = run.data.params
            tags = run.data.tags
            # Try params first, then tags
            x_raw = params.get(x_param) or tags.get(f"param.{x_param}")
            y_raw = params.get(y_param) or tags.get(f"param.{y_param}")
            if x_raw is None or y_raw is None:
                continue
            x_val = float(x_raw)
            y_val = float(y_raw)
        except Exception as e:
            print(f"Could not read tags for run {run.info.run_id}: {e}")
            continue

        metrics = run.data.metrics
        if not any(k.endswith("_mean") or k.endswith("_max") for k in metrics):
            continue

        key = (x_val, y_val)
        if key not in grid:
            grid[key] = {name: [] for name in base_metric_names}
            grid[key]["n_modules"] = []

        for base_name in base_metric_names:
            k = 0
            while True:
                metric_key = f"k{k}/{base_name}"
                if metric_key in metrics:
                    grid[key][base_name].append(metrics[metric_key])
                    k += 1
                else:
                    break

        k = 0
        while True:
            if f"k{k}/n_modules" in metrics:
                grid[key]["n_modules"].append(metrics[f"k{k}/n_modules"])
                k += 1
            else:
                break

    if not grid:
        print("No valid data collected from child runs")
        return figures

    base_keys = sorted(set(
        "_".join(name.split("_")[:2]) for name in base_metric_names
        if name.endswith("_mean") or name.endswith("_max")
    ))
    size_map = {"sm": "Small", "md": "Medium", "lg": "Large"}

    _param_labels = {
        "loss.sigma_sq": "σ²",
        "loss.sigma_theta": "σ<sub>θ</sub>",
        "model.sigma_sq": "σ²",
        "model.sigma_theta": "σ<sub>θ</sub>",
    }
    x_label = _param_labels.get(x_param, x_param.split(".")[-1])
    y_label = _param_labels.get(y_param, y_param.split(".")[-1])

    with mlflow.start_run(run_id=parent_run_id):
        for base_key in base_keys:
            mean_key = f"{base_key}_mean"
            max_key = f"{base_key}_max"
            if mean_key not in base_metric_names and max_key not in base_metric_names:
                continue

            size, angle = base_key.split("_")
            base_title = f"{size_map.get(size, size)} ratemap, {angle}° angle"

            mean_grid: dict[tuple[float, float], list[float]] = {
                k: v[mean_key] for k, v in grid.items() if mean_key in v and v[mean_key]
            }
            max_grid: dict[tuple[float, float], list[float]] = {
                k: v[max_key] for k, v in grid.items() if max_key in v and v[max_key]
            }

            fig = sweep_heatmap_2d(mean_grid, max_grid, x_label, y_label)
            log_figure(fig, f"sweep_heatmap2d_{base_key}")
            figures[f"heatmap2d_{base_key}"] = fig

        # n_modules heatmap (once, not per base_key)
        if base_keys:
            n_modules_grid: dict[tuple[float, float], list[float]] = {
                k: v["n_modules"] for k, v in grid.items() if v.get("n_modules")
            }
            if n_modules_grid:
                fig_nm = sweep_n_modules_heatmap_2d(n_modules_grid, x_label, y_label)
                log_figure(fig_nm, "sweep_heatmap2d_n_modules")
                figures["heatmap2d_n_modules"] = fig_nm

    print(f"2D sweep heatmap plots logged to MLflow for run {parent_run_id}")
    return figures


def sweep_n_modules_heatmap_2d(
    grid: dict[tuple[float, float], list[float]],
    x_label: str,
    y_label: str,
) -> go.Figure:
    """2D heatmap of mean number of modules across k iterations per parameter cell."""
    xs = sorted(set(x for x, y in grid))
    ys = sorted(set(y for x, y in grid))
    xs_str = [str(x) for x in xs]
    ys_str = [str(y) for y in ys]

    matrix = [
        [float(np.mean(grid[(x, y)])) if (x, y) in grid and grid[(x, y)] else float("nan") for x in xs]
        for y in ys
    ]
    text = [["" if np.isnan(v) else f"{v:.1f}" for v in row] for row in matrix]

    fig = go.Figure(go.Heatmap(
        x=xs_str, y=ys_str, z=matrix,
        text=text, texttemplate="%{text}",
        colorscale="Reds",
        colorbar=dict(tickfont=dict(size=16)),
    ))
    fig.update_xaxes(title_text=x_label, title_font=dict(size=20), tickfont=dict(size=16))
    fig.update_yaxes(title_text=y_label, title_font=dict(size=20), tickfont=dict(size=16))
    fig.update_layout(height=400, width=600, margin=dict(l=10, r=10, t=10, b=10))
    return fig