import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random.seed, np.random.seed, torch.manual_seed, and torch.mps.manual_seed (if MPS available)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
