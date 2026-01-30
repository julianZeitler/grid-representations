import torch

def reconstruction(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean squared error reconstruction loss.

    Args:
        x: Predicted tensor.
        y: Target tensor.

    Returns:
        Scalar MSE loss.
    """
    return torch.mean((x - y) ** 2)

def separation(z: torch.Tensor, chi: torch.Tensor, sigma_sq: float, causal: bool = False, decay: float | None = None) -> torch.Tensor:
    """Separation loss formulated directly on latent representations.

    Computes kernel similarity between latent states and weights with similarity matrix chi.

    Args:
        z: Latent states of shape (batch, seq_len, latent_size).
        chi: Target similarity matrix of shape (batch, seq_len, seq_len).
        sigma_sq: Kernel bandwidth (variance).
        causal: If True, mask future timesteps with lower triangular.
        decay: Exponential decay factor for temporal weighting.

    Returns:
        Scalar separation loss.
    """
    _, L, _ = z.shape

    # Compute pairwise similarity matrix [batch, seq_len, seq_len]
    Xi = torch.exp(-torch.sum(torch.pow(z[:,None,:,:] - z[:,:,None,:],2)/(2*sigma_sq),dim=3))

    if causal:
        Xi = torch.tril(Xi)

    if decay is not None:
        decay_arr = torch.pow(decay, torch.arange(L, device=z.device, dtype=z.dtype))
        row_idx, col_idx = torch.meshgrid(torch.arange(L, device=z.device), torch.arange(L, device=z.device), indexing='ij')
        decay_matrix = torch.where(row_idx >= col_idx, decay_arr[row_idx - col_idx], 0.0)
        Xi = Xi * decay_matrix[None, :, :]

    return torch.mean(Xi * chi)

def positivity(z: torch.Tensor) -> torch.Tensor:
    """Soft positivity loss penalizing negative values.

    Args:
        z: Tensor to constrain to be positive.

    Returns:
        Scalar loss (zero when all values are non-negative).
    """
    z_neg = (z - torch.abs(z))/2
    return -torch.mean(z_neg)

def norm(z: torch.Tensor, z_shift: torch.Tensor) -> torch.Tensor:
    """Soft norm loss penalizing deviations from unit norm per room.

    Normalizes z_shift using feature-wise norms from z, then penalizes
    deviations from unit norm in each shifted room.

    Args:
        z: Reference latent states of shape (batch, seq_len, latent_size).
        z_shift: Shifted latent states of shape (batch * n_shift, seq_len, latent_size).

    Returns:
        Scalar norm loss.
    """
    B, L, D = z.shape
    B_out, _, _ = z_shift.shape
    N_shift = B_out // B

    norms = z.reshape(B * L, D).norm(dim=0).detach()
    z_normalized = z_shift / norms

    norms_out = z_normalized.pow(2).reshape(B, N_shift, L, D).sum(dim=(0, 2))  # (N_shift, D)
    return (norms_out - 1).norm() / (N_shift * D)