import argparse
import itertools
import os

import mlflow
from mlflow.tracking import MlflowClient
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from train import train

mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("grid-representations")

config_dir = os.path.abspath("config")

EXPERIMENT_NAME = "seq_len_sweep"
experiments = None
param_grid = None

# Define experiments (optional, use None or [] to skip)
experiments = [
    "seq_2",
    "seq_5",
    "seq_10",
    "seq_50",
    "seq_100",
    "seq_200",
    "seq_500",
    "seq_1000",
]

# Define parameter grid for outer product sweeps (optional)
# param_grid = {
#     "regularization.norm.k": [-4, -2, -1, 1, 2],
# }


def make_overrides(experiment: str | None, params: dict) -> list[str]:
    """Create override list from experiment name and parameter dict."""
    overrides = []
    if experiment:
        overrides.append(f"+experiment={experiment}")
    for key, value in params.items():
        overrides.append(f"{key}={value}")
    return overrides


def make_run_name(experiment: str | None, params: dict) -> str:
    """Create descriptive run name."""
    parts = []
    if experiment:
        parts.append(experiment)
    for key, value in params.items():
        short_key = key.split(".")[-1]
        parts.append(f"{short_key}={value}")
    return "_".join(parts) if parts else "default"


def generate_sweep_configs():
    """Generate all sweep configurations as (experiment, params) tuples."""
    # Get all parameter combinations
    if param_grid:
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        param_combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    else:
        param_combos = [{}]

    # Combine with experiments
    if experiments:
        for exp in experiments:
            for params in param_combos:
                yield exp, params
    else:
        for params in param_combos:
            yield None, params


def get_child_run_ids(parent_run_id: str) -> dict[str, str]:
    """Fetch child runs and return a mapping of run_name to run_id."""
    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name("grid-representations")
    if experiment is None:
        return {}
    child_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'"
    )
    return {
        run.info.run_name: run.info.run_id
        for run in child_runs
        if run.info.run_name is not None
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--continue-id", type=str, default=None, help="Parent run ID to continue from")
    args = parser.parse_args()

    child_run_ids: dict[str, str] = {}
    if args.continue_id:
        child_run_ids = get_child_run_ids(args.continue_id)

    with mlflow.start_run(run_id=args.continue_id, run_name=EXPERIMENT_NAME) as parent:
        for experiment, params in generate_sweep_configs():
            GlobalHydra.instance().clear()

            with initialize_config_dir(config_dir=config_dir, version_base=None):
                overrides = make_overrides(experiment, params)
                cfg = compose(config_name="config", overrides=overrides)
                run_name = make_run_name(experiment, params)

                continue_id = child_run_ids.get(run_name)

                with mlflow.start_run(run_id=continue_id, run_name=run_name, nested=True):
                    mlflow.set_tag("experiment", experiment or "default")
                    for key, value in params.items():
                        mlflow.set_tag(f"param.{key}", value)

                    train(cfg, continue_id=continue_id)
