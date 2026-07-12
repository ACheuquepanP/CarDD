"""src/cardd/data/build.py

Dueno: Equipo D - Fase: 1 - Contrato: C1

Arma Dataset y DataLoader desde la config (C4). El DataLoader respeta C1:
  - collate_fn que NO apila imagenes (cada una tiene su H,W): viajan como listas
    (images, targets), que es lo que espera MaskRCNN.forward.
  - reproducibilidad de la augmentation: generator + worker_init_fn sembrados,
    para que comparar filas de ablacion mida senal y no ruido de aug (higiene P).
  - transforms: se construyen por nombre desde el registry (C3), nunca a mano.

Nada se hardcodea: root, batch, workers y semilla salen de la config. Claves que
lee: data.root, data.batch_size, data.num_workers, seed,
data.transforms.{train,eval} (lista de specs {name, ...} para el registry).
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cardd.data.coco_dataset import CarDDCocoDataset
from cardd.utils.config import get
from cardd.utils.logging import get_logger
from cardd.utils.registry import TRANSFORMS

log = get_logger("cardd.data.build")
_MISSING = object()


def _cfg_get(cfg, key: str, default=_MISSING):
    try:
        return get(cfg, key)
    except Exception:
        if default is _MISSING:
            raise
        return default


def collate_fn(batch):
    """(image, target)* -> (tuple(images), tuple(targets)). MaskRCNN espera listas."""
    return tuple(zip(*batch))


def _worker_init_fn(worker_id: int) -> None:
    """Siembra numpy/random por worker desde la semilla de torch (aug determinista)."""
    seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(seed)
    random.seed(seed)


def build_transforms(cfg, split: str):
    """Compone el pipeline de transforms del split desde el registry (C3)."""
    key = "data.transforms.train" if split == "train" else "data.transforms.eval"
    specs = _cfg_get(cfg, key, [])
    if not specs:
        return None
    # import perezoso: registra las clases en TRANSFORMS (transforms.py, paso 3b)
    from torchvision.transforms import v2

    from cardd.data import transforms as _transforms  # noqa: F401
    from cardd.data import transforms_paperms as _transforms_paperms  # noqa: F401

    return v2.Compose([TRANSFORMS.build(spec) for spec in specs])


def build_dataset(cfg, split: str) -> CarDDCocoDataset:
    root = Path(_cfg_get(cfg, "data.root"))
    transforms = build_transforms(cfg, split)
    ds = CarDDCocoDataset(root=root, split=split, transforms=transforms)
    log.info("dataset '%s': %d imagenes, %d clases", split, len(ds), ds.num_classes)
    return ds


def build_dataloader(cfg, split: str) -> DataLoader:
    ds = build_dataset(cfg, split)
    is_train = split == "train"

    batch_size = int(_cfg_get(cfg, "data.batch_size", 2))
    num_workers = int(_cfg_get(cfg, "data.num_workers", 4))
    seed = int(_cfg_get(cfg, "seed", 0))

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        collate_fn=collate_fn,
        worker_init_fn=_worker_init_fn,
        generator=generator,
        pin_memory=True,
        drop_last=is_train,
    )
