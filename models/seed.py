import os
import random
from typing import Optional

import numpy as np
import torch


def set_global_seed(seed: Optional[int]) -> None:
    """Seed the random number generators.

    Args:
        seed: Seed for global random state.
    """
    if seed is None:
        return
    
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONASSEED"] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

