from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch.nn import init
from abc import abstractmethod


class RGM(nn.Module):
    """
    Recurrent Grid Module - base class for recurrent sequence models.

    Has the same API as torch.nn.RNN for input_size and latent_size.
    The actual transition model is defined by subclasses implementing the step method.

    Args:
        input_size: The number of expected features in the input x
        latent_size: The number of features in the latent state z
        batch_first: If True, input and output tensors are provided as
            (batch, seq_len, feature). Default: True
        learn_z0: If True, learn the initial latent state as a parameter.
            Default: False
    """

    def __init__(
        self,
        input_size: int,
        latent_size: int,
        batch_first: bool = True,
        device: str = 'cpu'
    ):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.batch_first = batch_first
        self.device = device

        self.reset_parameters()
        
    @abstractmethod
    def step(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single step transition function. Must be implemented by subclasses.

        Args:
            x: Input tensor of shape (batch, input_size)
            z: Latent state tensor of shape (batch, latent_size)

        Returns:
            New latent state of shape (batch, latent_size)
        """
        raise NotImplementedError
    
    @abstractmethod
    def init_latent(self) -> torch.Tensor:
        """
        Initialize latent state. Must be implemented by subclasses.

        Args:
            device: Device to create tensor on (ignored if learn_z0=True)

        Returns:
            Initial latent state of shape (latent_size,)
        """
        raise NotImplementedError
    
    @abstractmethod
    def reset_parameters(self) -> None:
        """
        Resets parameters based on their initialization used in ``__init__``.
        """
        raise NotImplementedError

    def forward(
        self,
        input: torch.Tensor,
        z0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass over the entire sequence.

        Args:
            input: Input tensor of shape (batch, seq_len, input_size) if batch_first=True,
                   or (seq_len, batch, input_size) if batch_first=False
            z0: Initial latent state of shape (batch, latent_size).
                Defaults to zeros if not provided.

        Returns:
            output: Tensor of shape (batch, seq_len, latent_size) if batch_first=True,
                    or (seq_len, batch, latent_size) if batch_first=False
            z_n: Final latent state of shape (batch, latent_size)
        """
        if self.batch_first:
            batch_size, seq_len, _ = input.shape
        else:
            seq_len, batch_size, _ = input.shape
            input = input.transpose(0, 1)  # -> (batch, seq_len, input_size)

        if z0 is None:
            z0 = self.init_latent()

        z = z0.unsqueeze(0).expand(batch_size, -1)
        outputs = []

        for t in range(seq_len):
            x_t = input[:, t, :]  # (batch, input_size)
            z = self.step(x_t, z)
            outputs.append(z)

        output = torch.stack(outputs, dim=1)  # (batch, seq_len, latent_size)

        if not self.batch_first:
            output = output.transpose(0, 1)  # -> (seq_len, batch, latent_size)

        return output, z


class ElmanRGM(RGM):
    """
    RGM implementing an Elman-style RNN transition.

    Transition: z' = tanh(W_x @ x + W_z @ z + b)
    """

    def __init__(self, input_size: int, latent_size: int, **kwargs):
        super().__init__(input_size, latent_size, **kwargs)
        self.i2z = nn.Linear(input_size, latent_size)
        self.z2z = nn.Linear(latent_size, latent_size)

    def step(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.i2z(x) + self.z2z(z))

class ActionableRGM(RGM):
    """
    Transitions with irrep basis
    """

    def __init__(self, input_size: int, latent_size: int, **kwargs):
        super().__init__(input_size, latent_size, **kwargs)
        self.M = int(math.floor((latent_size - 1) / 2))

        self.z0 = nn.Parameter(torch.empty(latent_size))
        self.om = nn.Parameter(torch.empty((M, 2)))
        self.S = nn.Parameter(torch.empty((2*self.M+1, 2*self.M+1)))
    
    def init_latent(self) -> torch.Tensor:
        """
        Initialize latent state.

        Returns:
            Initial latent state of shape (latent_size,)
        """
        init.normal_(self.z0)
        return self.z0
    
    def reset_parameters(self) -> None:
        """
        Resets parameters based on their initialization used in ``__init__``.
        """
        init.normal_(self.S)
        init.normal_(self.z0)
        init.uniform_(self.om)
    
    def get_T(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute transformation matrix T(x) = S @ T_irrep(x) @ S^(-1)

        Args:
            x: displacement vector, shape [Batch, 2] or [Batch, Sequence, 2]

        Returns:
            T: transformation matrix, shape [Batch, D, D] or [Batch, Sequence, D, D]
        """
        has_sequence_dim = x.ndim == 3

        if has_sequence_dim:
            B, L = x.shape[0], x.shape[1]
            x_flat = x.reshape(B * L, 2)
            N = B * L
        else:
            B = x.shape[0]
            x_flat = x
            N = B

        # Compute k · x for all frequencies: [N, M]
        k_dot_x = torch.sum(self.om[None, :, :] * x_flat[:, None, :], dim=2)

        # Build T_irrep as block-diagonal matrix [N, D, D]
        T_irrep = torch.zeros(N, self.latent_size, self.latent_size, device=self.device, dtype=x.dtype)

        # First element is always 1 (constant term)
        T_irrep[:, 0, 0] = 1.0

        # Fill in 2x2 rotation blocks for each frequency
        cos_vals = torch.cos(k_dot_x)  # [N, M]
        sin_vals = torch.sin(k_dot_x)  # [N, M]

        for m in range(self.M):
            idx1 = 2 * m + 1
            idx2 = 2 * m + 2
            T_irrep[:, idx1, idx1] = cos_vals[:, m]
            T_irrep[:, idx1, idx2] = -sin_vals[:, m]
            T_irrep[:, idx2, idx1] = sin_vals[:, m]
            T_irrep[:, idx2, idx2] = cos_vals[:, m]

        # T = S @ T_irrep @ S^(-1)
        S_inv = torch.linalg.inv(self.S)
        T = torch.einsum('ij,njk,kl->nil', self.S, T_irrep, S_inv)

        if has_sequence_dim:
            T = T.reshape(B, x.shape[1], self.latent_size, self.latent_size)

        return T

    def step(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single step transition function.

        Args:
            x: Input tensor of shape (batch, input_size)
            z: Latent state tensor of shape (batch, latent_size)

        Returns:
            New latent state of shape (batch, latent_size)
        """
        T = self.get_T(x)

        return torch.einsum('bij,bj->bi', T, z)