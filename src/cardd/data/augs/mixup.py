"""CarDD · src/cardd/data/augs/mixup.py

MIXUP: operador aparte (no compite en el sorteo). Mezcla alpha de N imágenes del mismo
shape, uniendo sus anotaciones. En el orquestador, las N imágenes provienen de N
sorteos independientes entre {normal + augs activas} (sin volver a sacar mixup).

Parámetros: `alpha_beta` (distribución del lambda de mezcla), `n_images`.
"""
from __future__ import annotations

from typing import Sequence

import torch

from cardd.data.augs._core import as_t, rebuild_target


def compose(items: Sequence[tuple[torch.Tensor, dict]], *, alpha_beta: float = 1.5,
            n_images: int = 2, min_area: int = 16) -> tuple[torch.Tensor, dict]:
    """Mezcla N imágenes (mismo shape). Pesos ~ Dirichlet(alpha_beta) para N>2, Beta
    para N=2. Une todas las anotaciones (el modelo detecta objetos de todas)."""
    assert len(items) >= 2, "mixup requiere >=2 items"
    H, W = items[0][0].shape[-2], items[0][0].shape[-1]
    for img, _ in items:
        assert img.shape[-2:] == (H, W), "mixup: shapes distintos"

    n = len(items)
    if n == 2:
        lam = float(torch.distributions.Beta(alpha_beta, alpha_beta).sample())
        weights = [lam, 1.0 - lam]
    else:
        d = torch.distributions.Dirichlet(torch.full((n,), float(alpha_beta)))
        weights = d.sample().tolist()

    img = torch.zeros_like(items[0][0])
    for wgt, (im, _) in zip(weights, items):
        img = img + wgt * im
    img = img.clamp_(0.0, 1.0)

    mask_dtype = as_t(items[0][1]["masks"]).dtype
    masks_all, labels_all, iscrowd_all = [], [], []
    for _, tgt in items:
        m = as_t(tgt["masks"]).bool()
        if m.shape[0] == 0:
            continue
        for k in range(m.shape[0]):
            masks_all.append(m[k])
        labels_all += as_t(tgt["labels"]).tolist()
        ic = as_t(tgt.get("iscrowd", torch.zeros(m.shape[0])))
        iscrowd_all += ic.tolist()

    masks_t = (torch.stack(masks_all) if masks_all
               else torch.zeros((0, H, W), dtype=torch.bool))
    tgt = rebuild_target(masks_t, labels_all, iscrowd_all, H, W, mask_dtype,
                         items[0][1], min_area=min_area)
    return img, tgt
