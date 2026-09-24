"""
medforget/utils/seed.py
-----------------------
Reproducibility utility.
"""

import random
import numpy as np
import torch

def set_seed(seed: int) -> None:
    """Fix random state for Python, NumPy, and PyTorch (CPU + GPU).

    Args:
        seed: Integer seed value.  Any positive integer works;
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
