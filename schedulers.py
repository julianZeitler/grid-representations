from __future__ import annotations

import math
import torch


class GECO:
    """GECO (Growing Exponentially via Constraint Optimization) for adaptive lambda weighting.

    Dynamically adjusts regularization weights based on constraint satisfaction.
    If loss > target, lambda increases; if loss < target, lambda decreases.

    Args:
        lambda_init: Initial lambda value.
        k: Target constraint in log space (lambda grows when log(loss) > k).
        alpha: EMA smoothing factor for the constraint.
        gamma: Learning rate for lambda updates.
        min_lambda: Minimum lambda value (default 1e-10).
        max_lambda: Maximum lambda value (default 1e10).
    """

    def __init__(
        self,
        lambda_init: float = 0.1,
        k: float = -9.0,
        alpha: float = 0.9,
        gamma: float = 0.001,
        min_lambda: float = 1e-10,
        max_lambda: float = 1e10,
    ):
        self.lambda_val = lambda_init
        self.k = k
        self.alpha = alpha
        self.gamma = gamma
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda

        self.L = 0.0  # EMA of constraint violation

    def step(self, loss: float) -> float:
        """Update lambda based on current loss.

        Args:
            loss: Current loss value (must be positive).

        Returns:
            Updated lambda value.
        """
        if loss > 0:
            L_current = math.log(loss) - self.k
        else:
            L_current = -5.0  # Fallback for zero/negative loss

        # Exponential moving average
        self.L = self.L * self.alpha + (1 - self.alpha) * L_current

        # Update lambda
        self.lambda_val = self.lambda_val * math.exp(self.L * self.gamma)
        self.lambda_val = max(self.min_lambda, min(self.max_lambda, self.lambda_val))

        return self.lambda_val

    def reset(self, lambda_init: float | None = None) -> None:
        """Reset the GECO state."""
        if lambda_init is not None:
            self.lambda_val = lambda_init
        self.L = 0.0

    @property
    def constraint_violation(self) -> float:
        """Current EMA of constraint violation."""
        return self.L


class ConvergenceChecker:
    """Check convergence based on multiple loss terms.

    Monitors multiple loss terms and signals convergence when none have
    improved by more than a threshold over a sliding window.

    Args:
        window: Size of the history window to consider.
        patience: Number of checks without improvement before converging.
        threshold: Minimum relative improvement to count as progress.
        smoothing: Kernel size for smoothing loss values.
        loss_names: Names of the loss terms to track.
    """

    def __init__(
        self,
        window: int = 1000,
        patience: int = 100,
        threshold: float = 0.1,
        smoothing: int = 20,
        loss_names: list[str] = ["L1", "L2", "L3"],
    ):
        self.window = window
        self.patience = patience
        self.threshold = threshold
        self.smoothing = smoothing
        self.loss_names = loss_names

        self.loss_history: dict[str, list[float]] = {name: [] for name in loss_names}
        self.no_improvement_count = 0

    def step(self, **losses: float) -> bool:
        """Record losses and check for convergence.

        Args:
            **losses: Loss values keyed by name (e.g., L1=0.5, L2=0.3, L3=0.1).

        Returns:
            True if converged, False otherwise.
        """
        for name in self.loss_names:
            if name in losses:
                self.loss_history[name].append(float(losses[name]))

        min_length = min(len(self.loss_history[name]) for name in self.loss_names)
        if min_length < self.window + self.smoothing:
            return False

        kernel = torch.ones(self.smoothing) / self.smoothing
        any_improved = False

        for name in self.loss_names:
            window_start = max(0, len(self.loss_history[name]) - self.window)
            values = torch.tensor(self.loss_history[name][window_start:])

            # Convolve for smoothing
            smoothed = torch.conv1d(
                values.view(1, 1, -1),
                kernel.view(1, 1, -1),
            ).squeeze()

            if len(smoothed) < 2:
                continue

            current = smoothed[-1].item()
            window_min = smoothed[:-1].min().item()

            # Check if improved by threshold
            improved = (window_min - current) / (abs(window_min) + 1e-8) > self.threshold
            if improved:
                any_improved = True

        if not any_improved:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                return True
        else:
            self.no_improvement_count = 0

        return False

    def reset(self) -> None:
        """Reset the checker state."""
        self.loss_history = {name: [] for name in self.loss_names}
        self.no_improvement_count = 0

    @property
    def current_smoothed(self) -> dict[str, float | None]:
        """Get current smoothed values for each loss."""
        result = {}
        kernel = torch.ones(self.smoothing) / self.smoothing

        for name in self.loss_names:
            if len(self.loss_history[name]) < self.smoothing:
                result[name] = None
                continue

            values = torch.tensor(self.loss_history[name][-self.window:])
            smoothed = torch.conv1d(
                values.view(1, 1, -1),
                kernel.view(1, 1, -1),
            ).squeeze()
            result[name] = smoothed[-1].item() if len(smoothed) > 0 else None

        return result
