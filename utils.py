import random

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(use_cpu: bool = False):
    if torch is None:
        raise RuntimeError("PyTorch is required for neural model training or fallback inference.")
    if torch.cuda.is_available() and not use_cpu:
        return torch.device("cuda")
    return torch.device("cpu")
