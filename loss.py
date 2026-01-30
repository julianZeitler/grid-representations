import torch

def reconstruction(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    pass

# Formulate objective directly on z
def separation(z: torch.Tensor, causal: bool = False, eligibility: float | None = None) -> torch.Tensor:
    pass

# soft positivity loss
def positivity(z: torch.Tensor) -> torch.Tensor:
    pass

# soft norm loss
def norm(z: torch.Tensor, causal: bool = False) -> torch.Tensor:
    pass