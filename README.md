# Grid Representations

Code for my master's thesis in computer science. The project investigates the emergence of grid-like representations in recurrent grid models (RGMs) trained on 2D navigation trajectories, and studies continuous attractor network (CAN) dynamics in the learned latent space.

## Setup

Install dependencies with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
# GPU support:
uv sync --extra gpu
```

## Training

Training is configured with [Hydra](https://hydra.cc) and tracked with [MLflow](https://mlflow.org).

```bash
uv run python train.py
```

Override any config value on the command line:

```bash
uv run python train.py training.epochs=200 model.latent_size=128
```

Use a named experiment config from `config/experiment/`:

```bash
uv run python train.py +experiment=seq_100
```

Available experiment presets: `seq_2`, `seq_5`, `seq_10`, `seq_50`, `seq_100`, `seq_200`, `seq_500`, `seq_1000`. Each overrides `data.seq_len` and `training.batch_size` for the corresponding trajectory length.

To run parameter sweeps across multiple configurations, use `sweep.py`:

```bash
uv run python sweep.py
```

Edit `sweep.py` directly to define the experiment name, the list of experiment presets, and the parameter grid to sweep over.

### MLflow

Runs are stored locally in `mlruns/`. Launch the UI with:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlruns.db
```

Each run logs hyperparameters, loss curves, and analysis plots. The run ID is used to load a trained model for downstream experiments.

### Config reference (`config/config.yaml`)

**`data`**

| Field | Default | Description |
|---|---|---|
| `seq_len` | `200` | Number of timesteps per trajectory |
| `dim` | `2` | Dimensionality of the navigation space |
| `box_width` | `2` | Width of the environment box |
| `box_height` | `2` | Height of the environment box |
| `n_shift` | `15` | Number of shifted trajectory copies per sample |
| `sigma_shift` | `3` | Std. dev. of the shift distribution |
| `save_path` | `"data/"` | Directory for cached datasets |
| `sigma` | `11.52` | Std. dev. of rotation velocity (rad/s) |
| `b` | `0.8168` | Rayleigh scale for forward velocity (m/s) |
| `dt` | `0.02` | Simulation timestep (s) |
| `mu` | `0.0` | Turn angle bias (rad) |

**`model`**

| Field | Default | Description |
|---|---|---|
| `latent_size` | `65` | Dimension `D` of the RGM latent space |
| `om_init_scale` | `2` | Scale for initialising the omega matrix |

**`training`**

| Field | Default | Description |
|---|---|---|
| `epochs` | `400` | Total training epochs |
| `train_iterations` | `10` | Gradient steps per training epoch |
| `val_iterations` | `1` | Evaluation steps per epoch |
| `K` | `1` | Number of total training runs |
| `batch_size` | `5` | Trajectories per batch |
| `vary_seed` | `false` | Randomly initialize for each k |
| `vary_data` | `false` | Resample dataset for each k |

**`loss`**

| Field | Default | Description |
|---|---|---|
| `sigma_sq` | `0.04` | Kernel bandwidth (variance) for the separation loss |
| `sigma_theta` | `0.5` | Bandwidth for the target similarity matrix (chi) |
| `f` | `1` | Scaling factor for chi |
| `sep_scale` | `100` | Weight of the separation loss term |
| `causal` | `true` | Mask future timesteps in the similarity matrix |
| `decay` | `0.95` | Exponential decay factor for temporal weighting (`null` to disable) |

**`regularization`** (GECO schedulers for positivity and norm constraints)

| Field | Default | Description |
|---|---|---|
| `positivity.lambda_init` | `1` | Initial Lagrange multiplier for positivity |
| `positivity.k` | `-9` | Log-scale target constraint level |
| `positivity.alpha` | `0.9` | EMA smoothing for constraint tracking |
| `positivity.gamma` | `0.001` | Multiplier update rate |
| `norm.lambda_init` | `0.005` | Initial Lagrange multiplier for norm |
| `norm.k` | `-3` | Log-scale target constraint level |
| `norm.alpha` | `0.9` | EMA smoothing for constraint tracking |
| `norm.gamma` | `0.001` | Multiplier update rate |

**`optimizer`**

| Field | Default | Description |
|---|---|---|
| `lr` | `0.1` | Learning rate |
| `betas` | `[0.9, 0.9]` | Adam beta coefficients |
| `eps` | `1e-8` | Adam epsilon for numerical stability |

**`convergence`**

| Field | Default | Description |
|---|---|---|
| `enabled` | `false` | Enable early stopping |
| `window` | `1000` | Loss history window size |
| `patience` | `100` | Steps without improvement before stopping |
| `threshold` | `0.1` | Minimum improvement threshold |
| `smoothing` | `20` | Smoothing window for loss curve |

## CAN Experiments

The CAN experiments analyse the continuous attractor dynamics of a trained `ActionableRGM`. The primary script is `can_path_integration.py`; the shared config is `config/can_path_integration.yaml`.

### `can_path_integration.py` (main)

Implements the `CANPathIntegrator`: combined CAN attractor relaxation and RGM path integration in a unified state `(z, x)`. Per step:

1. CAN correction: `Δx_CAN = -alpha_can * dE/dx` where `E(x) = -½ z(x)ᵀ A z(x)`
2. Path integration step: `z_{t+1} = T(Δx_total) @ z_t`, `x_{t+1} = x_t + Δx_total`

Run with an MLflow run ID from a completed training run:

```bash
uv run python can_path_integration.py exp_id=<mlflow_run_id>
```

Override CAN parameters:

```bash
uv run python can_path_integration.py exp_id=<run_id> can.path_integration.alpha_can=0.1 can.path_integration.n_steps=200
```

Results (plots, metrics) are logged back to MLflow under the same experiment.

### Config reference (`config/can_path_integration.yaml`)

**Top-level**

| Field | Default | Description |
|---|---|---|
| `exp_id` | — | MLflow run ID of the trained model to load |
| `k` | `0` | Model checkpoint iteration to load |

**`can`**

| Field | Default | Description |
|---|---|---|
| `alpha_can` | `0.01` | Default CAN step size |
| `n_weight_steps` | `20` | Grid resolution per dimension for building the attractor weight matrix `A` (`n_weight_steps²` total samples) |
| `n_vis_steps` | `40` | Grid resolution per dimension for visualisations (`n_vis_steps²` evaluations) |

**`can.alpha_sweep`**

| Field | Default | Description |
|---|---|---|
| `alphas` | `[0.01, 0.1, 1.0]` | CAN step sizes to sweep over |
| `n_steps` | `100` | Integration steps per run |
| `n_init` | `5` | Number of random initialisations per alpha |

**`can.noise`**

| Field | Default | Description |
|---|---|---|
| `alpha_can` | `0.01` | CAN step size |
| `noise_levels` | `[0.0, 0.001, 0.01, 0.1]` | Input noise levels to evaluate |
| `n_steps` | `100` | Integration steps per run |
| `n_init` | `5` | Number of random initialisations per noise level |

**`can.trajectories`**

| Field | Default | Description |
|---|---|---|
| `start_positions` | (5 positions) | List of `[x, y]` start positions for trajectory evaluation |
| `noise_levels` | `[0.0, 0.01, 0.1]` | Noise levels to evaluate |
| `alpha_can` | `0.1` | CAN step size |
| `n_steps` | `50` | Trajectory length |

**`can.path_integration`**

| Field | Default | Description |
|---|---|---|
| `alpha_can` | `0.0` | CAN step size (set to `0` for pure path integration) |
| `n_steps` | `100` | Number of integration steps |
| `n_trajectories` | `5` | Number of test trajectories |

**`can.combined`**

| Field | Default | Description |
|---|---|---|
| `alphas` | `[0.0, 0.01, 0.1]` | CAN step sizes to evaluate |
| `n_can_steps` | `1` | CAN relaxation steps per PI step |
| `n_steps` | `100` | Number of integration steps |
| `n_trajectories` | `5` | Number of test trajectories |
