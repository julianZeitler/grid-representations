from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch.nn import init

class ActionableRGM(nn.Module):
    """
    Transitions with irrep basis
    """

    input_size: int
    latent_size: int
    batch_first: bool
    device: str

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
        self.M = int(math.floor((latent_size - 1) / 2))

        # Learnable parameters
        self.z0 = nn.Parameter(torch.empty(latent_size, device=self.device))
        self.om = nn.Parameter(torch.empty((self.M, 2), device=self.device))
        self.S = nn.Parameter(torch.empty((2*self.M+1, 2*self.M+1), device=self.device))

        # Compile the scan function for better performance
        self._scan_linear_transforms = torch.compile(self._scan_linear_transforms_impl)

        self.reset_parameters()

    def reset_parameters(self, seed: int | None = None) -> None:
        """
        Resets parameters based on their initialization used in ``__init__``.
        """
        if seed:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(seed)
        else:
            gen = None
        init.normal_(self.S, generator=gen)
        init.normal_(self.z0, generator=gen)
        init.uniform_(self.om, generator=gen)
    
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
            batch_size, _, _ = input.shape
        else:
            _, batch_size, _ = input.shape
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
        outputs[:, 0] = torch.einsum('bij,bj->bi', T[:, 0, :, :], z0)

        # Scan through remaining steps
        # torch.compile will optimize this loop
        for t in range(1, seq_len):
            outputs[:, t] = torch.einsum('bij,bj->bi', T[:, t, :, :], outputs[:, t-1, :])

        return outputs


class ElmanRGM(nn.Module):
    """
    RGM implementing an Elman-style RNN transition using PyTorch's nn.RNN.

    Transition: z' = tanh(W_x @ x + W_z @ z + b)
    """

    input_size: int
    latent_size: int
    batch_first: bool
    device: str

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

        self.rnn = nn.RNN(input_size, latent_size, batch_first=batch_first)

    def forward(
        self,
        input: torch.Tensor,
        z0: torch.Tensor | None = None,
        norm: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # z0 shape: (batch, latent_size) -> (1, batch, latent_size) for nn.RNN
        if z0 is not None:
            z0 = z0.unsqueeze(0)

        outputs, z_final = self.rnn(input, z0)

        if norm:
            norms = torch.linalg.norm(outputs, dim=-1, keepdim=True)
            outputs = outputs / (norms + 1e-5)

        # z_final shape: (1, batch, latent_size) -> (batch, latent_size)
        z_final = z_final.squeeze(0)

        return outputs, z_final
