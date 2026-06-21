"""CarDD · utils/seed.py — reproducibilidad (Fase 0, Equipo P).

Fija la semilla en random, numpy y torch (CPU + GPU) para que la ablación
mida señal y no ruido. La higiene de P: misma semilla entre filas.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Fija la semilla global.

    Args:
        seed: valor de la semilla.
        deterministic: si True, fuerza algoritmos deterministas en cuDNN
            (más reproducible, algo más lento). Para auditar la ablación.
            Si False, deja cuDNN en modo benchmark (más rápido).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
