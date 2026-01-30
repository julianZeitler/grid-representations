from __future__ import annotations

import json
import os
import pickle
from typing import Any, Generator

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from numpy.typing import NDArray
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

def make_collate_fn(device: torch.device):
    """Create a collate function that reshapes shifted trajectories and moves to device.

    Args:
        device: Target device for the tensors.

    Returns:
        Collate function for use with DataLoader.
    """
    def collate_fn(batch):
        normal, shift = default_collate(batch)
        B, n_shift, L, d = shift.shape
        shift = shift.reshape(B * n_shift, L, d)
        return normal.to(device), shift.to(device)
    return collate_fn 


class TrajectoryDataset(Dataset):
    """PyTorch Dataset for loading pre-generated trajectory data.

    Loads trajectory batches from disk, where each batch contains a trajectory
    and its shifted versions for contrastive learning.

    Attributes:
        base_path: Path to the dataset directory.
    """

    def __init__(self, base_path: str) -> None:
        """Initialize the dataset.

        Args:
            base_path: Path to the dataset directory containing batch files
                and metadata.json.
        """
        self.base_path = base_path

        with open(os.path.join(base_path, 'metadata.json'), 'r') as f:
            self.metadata: dict[str, Any] = json.load(f)

        self.num_sequences: int = self.metadata['num_sequences']
        self.n_shift: int = self.metadata['n_shift']
        self.sequence_length: int = self.metadata['sequence_length']
        self.box_width: float = self.metadata['box_width']
        self.box_height: float = self.metadata['box_height']
        self.sigma_shift: float = self.metadata['sigma_shift']

    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return self.num_sequences

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load and return a trajectory batch.

        Args:
            idx: Index of the batch to load.

        Returns:
            Tuple of (normal, shift) tensors where:
                - normal: Original trajectory of shape [L, 2]
                - shift: Shifted trajectories of shape [n_shift, L, 2]
        """
        batch_path = os.path.join(self.base_path, f'batch_{idx:05d}.pkl')
        with open(batch_path, 'rb') as f:
            batch = pickle.load(f)

        return torch.from_numpy(batch["normal"]), torch.from_numpy(batch["shift"])

class TrajectoryGenerator:
    """Generator for random walk trajectories in a rectangular environment.

    Simulates an agent performing a random walk with realistic motion dynamics,
    including velocity sampling from a Rayleigh distribution and heading changes
    from a normal distribution. Supports both bounded and periodic environments.

    Attributes:
        periodic: Whether to use periodic boundary conditions.
        border_region: Distance from wall at which avoidance behavior activates.
    """

    def __init__(self, periodic: bool = False) -> None:
        """Initialize the trajectory generator.

        Args:
            periodic: If True, use periodic boundary conditions. If False,
                the agent avoids walls by turning away when near boundaries.
        """
        self.periodic = periodic
        self.border_region: float = 0.03

    def avoid_wall(
        self,
        position: NDArray[np.floating],
        hd: NDArray[np.floating],
        box_width: float,
        box_height: float,
    ) -> tuple[NDArray[np.bool_], NDArray[np.floating]]:
        """Compute wall avoidance turn angles for agents near boundaries.

        Args:
            position: Agent positions of shape [batch_size, 2].
            hd: Agent heading directions in radians of shape [batch_size].
            box_width: Width of the environment box.
            box_height: Height of the environment box.

        Returns:
            Tuple of:
                - is_near_wall: Boolean array of shape [batch_size] indicating
                    which agents are near a wall and facing it.
                - turn_angle: Turn angles in radians of shape [batch_size] to
                    apply for wall avoidance.
        """
        x = position[:, 0]
        y = position[:, 1]
        dists = [box_width / 2 - x, box_height / 2 - y, box_width / 2 + x, box_height / 2 + y]
        d_wall = np.min(dists, axis=0)
        angles = np.arange(4) * np.pi / 2
        theta = angles[np.argmin(dists, axis=0)]
        hd = np.mod(hd, 2 * np.pi)
        a_wall = hd - theta
        a_wall = np.mod(a_wall + np.pi, 2 * np.pi) - np.pi

        is_near_wall = (d_wall < self.border_region) * (np.abs(a_wall) < np.pi / 2)
        turn_angle = np.zeros_like(hd)
        turn_angle[is_near_wall] = np.sign(a_wall[is_near_wall]) * (np.pi / 2 - np.abs(a_wall[is_near_wall]))

        return is_near_wall, turn_angle

    def generate_trajectory(
        self,
        box_width: float,
        box_height: float,
        sequence_length: int,
        batch_size: int | None = None,
    ) -> NDArray[np.floating]:
        """Generate random walk trajectories in a rectangular box.

        Simulates agents performing random walks with velocity sampled from a
        Rayleigh distribution and heading changes from a normal distribution.

        Args:
            box_width: Width of the environment box in meters.
            box_height: Height of the environment box in meters.
            sequence_length: Number of time steps in each trajectory.
            batch_size: Number of trajectories to generate. If None, returns
                a single trajectory without batch dimension.

        Returns:
            Array of positions with shape [batch_size, sequence_length, 2] if
            batch_size is given, else [sequence_length, 2]. Contains (x, y)
            coordinates in meters.
        """
        dt = 0.02  # time step increment (seconds)
        sigma = 5.76 * 2  # stdev rotation velocity (rads/sec)
        b = 0.13 * 2 * np.pi  # forward velocity rayleigh dist scale (m/sec)
        mu = 0  # turn angle bias

        if batch_size is None:
            batch_size_internal = 1
            squeeze_batch = True
        else:
            batch_size_internal = batch_size
            squeeze_batch = False

        position = np.zeros([batch_size_internal, sequence_length, 2])
        head_dir = np.zeros([batch_size_internal, sequence_length])

        position[:, 0, 0] = np.random.uniform(-box_width / 2, box_width / 2, batch_size_internal)
        position[:, 0, 1] = np.random.uniform(-box_height / 2, box_height / 2, batch_size_internal)
        head_dir[:, 0] = np.random.uniform(0, 2 * np.pi, batch_size_internal)

        random_turn = np.random.normal(mu, sigma, [batch_size_internal, sequence_length - 1])
        random_vel = np.random.rayleigh(b, [batch_size_internal, sequence_length - 1])

        for t in range(sequence_length - 1):
            v = random_vel[:, t].copy()
            turn_angle = np.zeros(batch_size_internal)

            if not self.periodic:
                is_near_wall, turn_angle = self.avoid_wall(position[:, t], head_dir[:, t], box_width, box_height)
                v[is_near_wall] *= 0.25

            turn_angle += dt * random_turn[:, t]

            velocity = v * dt
            update = velocity[:, None] * np.stack([np.cos(head_dir[:, t]), np.sin(head_dir[:, t])], axis=-1)
            position[:, t + 1] = position[:, t] + update

            head_dir[:, t + 1] = head_dir[:, t] + turn_angle

        if self.periodic:
            position[:, :, 0] = np.mod(position[:, :, 0] + box_width / 2, box_width) - box_width / 2
            position[:, :, 1] = np.mod(position[:, :, 1] + box_height / 2, box_height) - box_height / 2

        if squeeze_batch:
            return position[0]  # Shape: [sequence_length, 2]
        return position

    def get_generator(
        self,
        box_width: float = 2,
        box_height: float = 2,
        sequence_length: int = 50,
        batch_size: int = 32,
    ) -> Generator[NDArray[np.floating], None, None]:
        """Return an infinite generator that yields batches of trajectories.

        Args:
            box_width: Width of the environment box in meters.
            box_height: Height of the environment box in meters.
            sequence_length: Number of time steps in each trajectory.
            batch_size: Number of trajectories per batch.

        Yields:
            Array of positions with shape [sequence_length, batch_size, 2].
        """
        while True:
            positions = self.generate_trajectory(box_width, box_height, sequence_length, batch_size)
            positions = positions.transpose(1, 0, 2)
            yield positions

    def get_test_batch(
        self,
        box_width: float = 2,
        box_height: float = 2,
        sequence_length: int = 50,
        batch_size: int = 32,
    ) -> NDArray[np.floating]:
        """Generate a single batch of trajectories for testing.

        Args:
            box_width: Width of the environment box in meters.
            box_height: Height of the environment box in meters.
            sequence_length: Number of time steps in each trajectory.
            batch_size: Number of trajectories in the batch.

        Returns:
            Array of positions with shape [sequence_length, batch_size, 2].
        """
        positions = self.generate_trajectory(box_width, box_height, sequence_length, batch_size)
        positions = positions.transpose(1, 0, 2)
        return positions

    def generate_dataset(
        self,
        savepath: str,
        num_sequences: int,
        sequence_length: int,
        box_width: float = 2,
        box_height: float = 2,
        n_shift: int = 15,
        sigma_shift: float = 0.03,
    ) -> None:
        """Generate a dataset of trajectory batches and save to disk.

        Each batch contains one trajectory with n_shift shifted versions.
        Shifts are sampled from N(0, sigma_shift) and applied as constant offsets.

        Args:
            savepath: Directory to save the dataset.
            num_sequences: Number of trajectories to generate.
            sequence_length: Number of time steps in each trajectory.
            box_width: Width of the environment box in meters.
            box_height: Height of the environment box in meters.
            n_shift: Number of shifted versions per trajectory.
            sigma_shift: Standard deviation for shift sampling in meters.
        """
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        print(f"Generating {num_sequences} sequences with {n_shift} shifts each...")

        for idx in range(num_sequences):
            # Generate single trajectory [L, 2]
            positions = self.generate_trajectory(box_width, box_height, sequence_length)

            # Sample shifts [n_shift, 2]
            phi_shift = np.random.normal(0, sigma_shift, [n_shift, 2])

            # Create shifted trajectories [n_shift, L, 2]
            #                  [n_shift, L, 2]             [n_shift, L,    2]
            shifted = positions[None,    :, :] + phi_shift[:,        None, :]

            batch = {
                "normal": positions.astype(np.float32),
                "shift": shifted.astype(np.float32),
            }
            batch_path = os.path.join(savepath, f'batch_{idx:05d}.pkl')
            with open(batch_path, 'wb') as f:
                pickle.dump(batch, f)

            if (idx + 1) % 100 == 0:
                print(f"Generated {idx + 1}/{num_sequences} sequences")

        # Save metadata
        metadata = {
            'num_sequences': num_sequences,
            'n_shift': n_shift,
            'sequence_length': sequence_length,
            'box_width': box_width,
            'box_height': box_height,
            'sigma_shift': sigma_shift,
        }
        with open(os.path.join(savepath, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)

        print(f"Dataset generation complete! Saved to {savepath}")

    def visualize_trajectories(
        self,
        positions: NDArray[np.floating],
        box_width: float = 2,
        box_height: float = 2,
        num_trajectories: int | None = None,
        show_start: bool = True,
        show_end: bool = True,
        title: str | None = None,
        figsize: tuple[int, int] = (10, 10),
        save_path: str | None = None,
    ) -> tuple[Figure, Axes]:
        """Visualize multiple trajectories on a single plot.

        Args:
            positions: Array of positions with shape [batch_size, sequence_length, 2].
            box_width: Width of environment box for drawing boundaries.
            box_height: Height of environment box for drawing boundaries.
            num_trajectories: Number of trajectories to plot. If None, plots up to 50.
            show_start: Whether to mark starting points with circles.
            show_end: Whether to mark ending points with squares.
            title: Plot title. If None, auto-generated.
            figsize: Figure size as (width, height) tuple.
            save_path: If provided, save figure to this path.

        Returns:
            Tuple of (figure, axes) matplotlib objects.
        """
        fig, ax = plt.subplots(figsize=figsize)

        batch_size, sequence_length, _ = positions.shape

        # Determine how many trajectories to plot
        if num_trajectories is None:
            n_traj = min(batch_size, 50)  # Cap at 50 for visibility
        else:
            n_traj = min(num_trajectories, batch_size)

        # Color map for trajectories
        cmap = plt.get_cmap('viridis')
        colors = cmap(np.linspace(0, 1, n_traj))

        for i in range(n_traj):
            traj = positions[i]  # Shape: [sequence_length, 2]

            # Plot trajectory
            ax.plot(traj[:, 0], traj[:, 1], '-', alpha=0.6, color=colors[i], linewidth=1.5)

            # Mark start point
            if show_start:
                ax.plot(traj[0, 0], traj[0, 1], 'o', color=colors[i], markersize=8,
                       markeredgecolor='black', markeredgewidth=1.5, label='Start' if i == 0 else '')

            # Mark end point
            if show_end and sequence_length > 1:
                ax.plot(traj[-1, 0], traj[-1, 1], 's', color=colors[i], markersize=8,
                       markeredgecolor='black', markeredgewidth=1.5, label='End' if i == 0 else '')


        # Draw box boundaries
        rect = Rectangle((-box_width / 2, -box_height / 2), box_width, box_height,
                        linewidth=2, edgecolor='dimgrey', facecolor='none')
        ax.add_patch(rect)

        # Set equal aspect ratio and labels
        ax.set_aspect('equal')
        ax.set_xlabel('X position (m)', fontsize=12)
        ax.set_ylabel('Y position (m)', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Set title
        if title is None:
            title = f'Trajectories (showing {n_traj}/{batch_size})'
        ax.set_title(title, fontsize=14)

        # Add legend
        if show_start or show_end or True:
            ax.legend(loc='upper right')

        plt.tight_layout()

        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        return fig, ax

    def visualize_trajectory_time(
        self,
        positions: NDArray[np.floating],
        box_width: float = 2,
        box_height: float = 2,
        title: str | None = None,
        figsize: tuple[int, int] = (8, 8),
        cmap: str = 'viridis',
        save_path: str | None = None,
    ) -> tuple[Figure, Axes]:
        """Visualize a single trajectory with color encoding time progression.

        Args:
            positions: Array of positions with shape [sequence_length, 2].
            box_width: Width of environment box for drawing boundaries.
            box_height: Height of environment box for drawing boundaries.
            title: Plot title.
            figsize: Figure size as (width, height) tuple.
            cmap: Matplotlib colormap name for time encoding.
            save_path: If provided, save figure to this path.

        Returns:
            Tuple of (figure, axes) matplotlib objects.
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Create line segments for LineCollection
        points = positions.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create time-based colors
        t = np.linspace(0, 1, len(segments))
        lc = LineCollection(segments, cmap=cmap, norm=Normalize(0, 1))
        lc.set_array(t)
        lc.set_linewidth(2)
        ax.add_collection(lc)

        # Mark start and end points
        ax.scatter(positions[0, 0], positions[0, 1], c='green', s=100, zorder=5, label='Start')
        ax.scatter(positions[-1, 0], positions[-1, 1], c='red', s=100, zorder=5, label='End')

        # Draw environment box
        box = Rectangle((-box_width/2, -box_height/2), box_width, box_height,
                        fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(box)

        ax.set_xlim(-box_width/2 - 0.1, box_width/2 + 0.1)
        ax.set_ylim(-box_height/2 - 0.1, box_height/2 + 0.1)
        ax.set_aspect('equal')
        ax.legend()

        # Add colorbar
        cbar = plt.colorbar(lc, ax=ax)
        cbar.set_label('Time')

        if title:
            ax.set_title(title)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        return fig, ax

    def visualize_trajectory_grid(
        self,
        positions: NDArray[np.floating],
        box_width: float = 2,
        box_height: float = 2,
        grid_size: tuple[int, int] = (4, 4),
        title: str | None = None,
        figsize: tuple[int, int] = (16, 16),
        save_path: str | None = None,
    ) -> tuple[Figure, NDArray[np.object_]]:
        """Visualize multiple trajectories in a grid layout.

        Each trajectory is shown in its own subplot with time-based coloring.

        Args:
            positions: Array of positions with shape [batch_size, sequence_length, 2].
            box_width: Width of environment box for drawing boundaries.
            box_height: Height of environment box for drawing boundaries.
            grid_size: Tuple of (rows, cols) for subplot grid.
            title: Overall figure title.
            figsize: Figure size as (width, height) tuple.
            save_path: If provided, save figure to this path.

        Returns:
            Tuple of (figure, axes_array) matplotlib objects.
        """
        batch_size, sequence_length, _ = positions.shape
        rows, cols = grid_size
        num_plots = min(rows * cols, batch_size)

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes_flat: list[Axes] = list(np.asarray(axes).flatten())

        for idx in range(num_plots):
            ax = axes_flat[idx]
            traj = positions[idx]  # Shape: [sequence_length, 2]

            # Plot trajectory with time-based color gradient
            if sequence_length > 1:
                points = traj.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                lc = LineCollection(segments, cmap='viridis', linewidth=2)
                lc.set_array(np.linspace(0, 1, sequence_length - 1))
                ax.add_collection(lc)

            # Mark start and end
            ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=10,
                   markeredgecolor='black', markeredgewidth=1.5, label='Start', zorder=10)
            if sequence_length > 1:
                ax.plot(traj[-1, 0], traj[-1, 1], 'rs', markersize=10,
                       markeredgecolor='black', markeredgewidth=1.5, label='End', zorder=10)

            # Draw box boundaries
            rect = Rectangle((-box_width / 2, -box_height / 2), box_width, box_height,
                            linewidth=1.5, edgecolor='red', facecolor='none', linestyle='--')
            ax.add_patch(rect)

            # Set limits with some padding
            padding = 0.2
            ax.set_xlim(-box_width / 2 - padding, box_width / 2 + padding)
            ax.set_ylim(-box_height / 2 - padding, box_height / 2 + padding)

            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'Trajectory {idx}', fontsize=10)

            if idx == 0:
                ax.legend(loc='upper right', fontsize=8)

        # Hide unused subplots
        for idx in range(num_plots, rows * cols):
            axes_flat[idx].axis('off')

        if title:
            fig.suptitle(title, fontsize=16, y=0.995)

        plt.tight_layout()

        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        return fig, np.asarray(axes)