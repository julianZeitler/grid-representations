import torch
from omegaconf import DictConfig

from schedulers import GECO

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

def chi(x: torch.Tensor, sigma_theta: float, f: float, causal: bool = True) -> torch.Tensor:
    """Compute target similarity matrix from positions.

    Args:
        x: Position tensor of shape (seq_len, dim) or (batch, seq_len, dim).
        sigma_theta: Bandwidth for kernel.
        f: Scaling factor for similarity.
        causal: If True, mask future timesteps with lower triangular.

    Returns:
        Chi matrix of shape (seq_len, seq_len) or (batch, seq_len, seq_len).
    """
    if x.ndim == 2:
        dist = torch.sum((x[:, None, :] - x[None, :, :]) ** 2, dim=2)
    else:
        dist = torch.sum((x[:, :, None, :] - x[:, None, :, :]) ** 2, dim=3)

    if causal:
        dist = torch.tril(dist)

    return 1 - f * torch.exp(-dist / (2 * sigma_theta ** 2))

def total_sep_loss(
    z: torch.Tensor,
    z_shift: torch.Tensor,
    x: torch.Tensor,
    lambda_pos: float,
    lambda_norm: float,
    cfg: DictConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute combined loss with GECO-weighted regularization.

    Args:
        z: Latent states of shape (batch, seq_len, latent_size).
        z_shift: Shifted latent states for norm loss, or None to skip.
        x: Position tensor of shape (batch, seq_len, dim).
        geco_pos: GECO instance for positivity regularization.
        geco_norm: GECO instance for norm regularization.
        cfg: Loss config with sigma_sq, sigma_theta, f, decay.

    Returns:
        Tuple of (total_loss, components_dict).
    """
    target_chi = chi(x, cfg.sigma_theta, cfg.f, causal=True)
    loss_sep = cfg.sep_scale * separation(z, target_chi, cfg.sigma_sq, causal=True, decay=cfg.get("decay"))
    loss_pos = positivity(z)
    loss_norm = norm(z, z_shift)

    # Total loss
    total = loss_sep + lambda_pos * loss_pos + lambda_norm * loss_norm

    components = {
        "separation": loss_sep.item(),
        "positivity": loss_pos.item(),
        "norm": loss_norm.item()
    }

    return total, components
