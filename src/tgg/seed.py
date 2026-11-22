"""One seed for every random draw in the project."""
import os
import random

import numpy as np
import torch

GLOBAL_SEED = 651


def seed_everything(seed: int = GLOBAL_SEED) -> None:
    """Seed Python, NumPy, PyTorch and CUDA, and make cuDNN deterministic."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
