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

        self.z0 = torch.zeros(latent_size)

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
    def reset_parameters(self) -> None:
        """
        Resets parameters based on their initialization used in ``__init__``.
        """
        raise NotImplementedError

    def forward(
        self,
        input: torch.Tensor,
        z0: torch.Tensor | None = None,
        norm: bool = False
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
            z0 = self.z0

        z = z0.unsqueeze(0).expand(batch_size, -1)
        outputs = []

        for t in range(seq_len):
            x_t = input[:, t, :]  # (batch, input_size)
            z = self.step(x_t, z)
            outputs.append(z)

        output = torch.stack(outputs, dim=1)  # (batch, seq_len, latent_size)

        if norm:
            norms = torch.linalg.norm(g, axis=1, keepdim=True)
            g = g / (norms + 1e-5)

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

        # Compile the scan function for better performance
        self._scan_elman = torch.compile(self._scan_elman_impl)

    def reset_parameters(self) -> None:
        """Reset parameters for linear layers."""
        self.i2z.reset_parameters()
        self.z2z.reset_parameters()

    def step(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.i2z(x) + self.z2z(z))

    def forward(
        self,
        input: torch.Tensor,
        z0: torch.Tensor | None = None,
        norm: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Optimized forward pass for ElmanRGM.

        Pre-computes all input transformations then applies recurrence.
        """
        if self.batch_first:
            batch_size, seq_len, _ = input.shape
        else:
            seq_len, batch_size, _ = input.shape
            input = input.transpose(0, 1)  # -> (batch, seq_len, input_size)

        if z0 is None:
            z0 = self.z0

        z = z0.unsqueeze(0).expand(batch_size, -1)

        # Pre-compute all input transformations: (batch, seq_len, latent_size)
        # This batches all the input linear transformations together
        input_transformed = self.i2z(input)

        # Now do the recurrent scan using compiled version
        outputs = self._scan_elman(input_transformed, z)

        if norm:
            norms = torch.linalg.norm(outputs, dim=1, keepdim=True)
            outputs = outputs / (norms + 1e-5)

        if not self.batch_first:
            outputs = outputs.transpose(0, 1)  # -> (seq_len, batch, latent_size)

        z_final = outputs[:, -1, :]
        return outputs, z_final

    def _scan_elman_impl(
        self,
        x_transformed: torch.Tensor,
        z0: torch.Tensor
    ) -> torch.Tensor:
        """
        Scan through sequence applying Elman recurrence.

        This is the implementation that gets compiled with torch.compile.

        Args:
            x_transformed: Pre-computed i2z(x) of shape (batch, seq_len, latent_size)
            z0: Initial state (batch, latent_size)

        Returns:
            All hidden states (batch, seq_len, latent_size)
        """
        batch_size, seq_len, _ = x_transformed.shape

        outputs = torch.zeros(batch_size, seq_len, self.latent_size,
                             device=x_transformed.device, dtype=x_transformed.dtype)

        z = z0
        for t in range(seq_len):
            z = torch.tanh(x_transformed[:, t] + self.z2z(z))
            outputs[:, t] = z

        return outputs

class ActionableRGM(RGM):
    """
    Transitions with irrep basis
    """

    def __init__(self, input_size: int, latent_size: int, **kwargs):
        super().__init__(input_size, latent_size, **kwargs)
        self.M = int(math.floor((latent_size - 1) / 2))

        # Learnable parameters
        self.z0 = nn.Parameter(torch.empty(latent_size))
        self.om = nn.Parameter(torch.empty((self.M, 2)))
        self.S = nn.Parameter(torch.empty((2*self.M+1, 2*self.M+1)))

        # Compile the scan function for better performance
        self._scan_linear_transforms = torch.compile(self._scan_linear_transforms_impl)
    
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

    def forward(
        self,
        input: torch.Tensor,
        z0: torch.Tensor | None = None,
        norm: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized forward pass for ActionableRGM.

        Since transformations are linear, we can compute all T matrices at once
        and apply them sequentially using associative scan.
        """
        if self.batch_first:
            batch_size, seq_len, _ = input.shape
        else:
            seq_len, batch_size, _ = input.shape
            input = input.transpose(0, 1)  # -> (batch, seq_len, input_size)

        if z0 is None:
            z0 = self.z0

        z = z0.unsqueeze(0).expand(batch_size, -1)  # (batch, latent_size)

        # Compute all transformation matrices at once: (batch, seq_len, latent_size, latent_size)
        T_all = self.get_T(input)

        # Efficient sequential application using compiled scan
        outputs = self._scan_linear_transforms(T_all, z)

        if norm:
            norms = torch.linalg.norm(outputs, dim=1, keepdim=True)
            outputs = outputs / (norms + 1e-5)

        if not self.batch_first:
            outputs = outputs.transpose(0, 1)  # -> (seq_len, batch, latent_size)

        # Final state is the last output
        z_final = outputs[:, -1, :]

        return outputs, z_final

    def _scan_linear_transforms_impl(
        self,
        T: torch.Tensor,
        z0: torch.Tensor
    ) -> torch.Tensor:
        """
        Efficiently apply sequence of linear transformations using scan pattern.

        Similar to JAX's lax.scan but in PyTorch. This computes the cumulative
        application of transformations: z_t = T_t @ T_{t-1} @ ... @ T_1 @ z0

        This is the implementation that gets compiled with torch.compile.

        Args:
            T: Transformation matrices (batch, seq_len, latent_size, latent_size)
            z0: Initial state (batch, latent_size)

        Returns:
            All intermediate states (batch, seq_len, latent_size)
        """
        batch_size, seq_len, _, _ = T.shape

        # Preallocate output tensor
        outputs = torch.zeros(batch_size, seq_len, self.latent_size,
                             device=T.device, dtype=T.dtype)

        # First step
        outputs[:, 0] = torch.einsum('bij,bj->bi', T[:, 0], z0)

        # Scan through remaining steps
        # torch.compile will optimize this loop
        for t in range(1, seq_len):
            outputs[:, t] = torch.einsum('bij,bj->bi', T[:, t], outputs[:, t-1])

        return outputs