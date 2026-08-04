"""CarDD · src/cardd/data/augs/mosaic_crop.py

MOSAIC_CROP: rejilla NxN donde cada celda recibe un PARCHE recortado de la fuente
(fracción [crop_frac_min,crop_frac_max] independiente por eje), reescalado a la celda
(estilo YOLO: recorte+reescalado). El recorte puede GUIARSE hacia una instancia de
clase difícil (contenerla sin centrarla) — el guiado lo pasa el orquestador vía la
función crop_sample.

Parámetros de intensidad (enhance): `grid`.
"""
from __future__ import annotations

import random
from typing import Callable, Sequence

import torch

from cardd.data.augs._core import as_t, rebuild_target
from cardd.data.augs import mosaic as _mosaic


def crop_sample(img: torch.Tensor, tgt: dict, *, crop_frac_min: float, crop_frac_max: float,
                min_area: int, guide_fn: Callable | None = None) -> tuple[torch.Tensor, dict]:
    """Recorta una ventana [fmin,fmax] (frac independiente por eje) de (img,tgt).

    `guide_fn(masks, labels, H, W, cw, ch) -> (x0,y0) | None` decide la posición de la
    ventana (guiado a clase difícil). Si None o devuelve None, posición aleatoria.
    Instancias recortadas a la ventana (parte visible, descartar bajo min_area).
    """
    _, H, W = img.shape
    cw = max(1, min(W, int(round(W * random.uniform(crop_frac_min, crop_frac_max)))))
    ch = max(1, min(H, int(round(H * random.uniform(crop_frac_min, crop_frac_max)))))

    masks = as_t(tgt["masks"])
    labels = as_t(tgt["labels"])

    x0 = y0 = None
    if guide_fn is not None and masks.numel() and masks.shape[0] > 0:
        pos = guide_fn(masks, labels, H, W, cw, ch)
        if pos is not None:
            x0, y0 = pos
    if x0 is None:
        x0 = random.randint(0, W - cw)
        y0 = random.randint(0, H - ch)

    img_c = img[:, y0:y0 + ch, x0:x0 + cw].contiguous()
    mask_dtype = masks.dtype if masks.numel() else torch.uint8
    if masks.numel() and masks.shape[0] > 0:
        m_c = masks[:, y0:y0 + ch, x0:x0 + cw].bool()
        iscrowd = as_t(tgt.get("iscrowd", torch.zeros(masks.shape[0], dtype=torch.uint8)))
        km, kl, ki = [], [], []
        for k in range(m_c.shape[0]):
            if bool(m_c[k].any()):
                km.append(m_c[k]); kl.append(int(labels[k]))
                ki.append(int(iscrowd[k]) if iscrowd.numel() else 0)
        mstack = torch.stack(km) if km else torch.zeros((0, ch, cw), dtype=torch.bool)
        tgt_c = rebuild_target(mstack, kl, ki, ch, cw, mask_dtype, tgt, min_area=min_area)
    else:
        tgt_c = rebuild_target(torch.zeros((0, ch, cw), dtype=torch.bool), [], [],
                               ch, cw, mask_dtype, tgt, min_area=min_area)
    return img_c, tgt_c


def compose(samples: Sequence[tuple[torch.Tensor, dict]], *, out_h: int, out_w: int,
            grid: int = 2, center_jitter: float = 0.25, center_shape: str = "square",
            min_area: int = 16) -> tuple[torch.Tensor, dict]:
    """Las `samples` ya vienen recortadas (el orquestador aplicó crop_sample). Aquí solo
    se ensamblan en la rejilla, reusando mosaic.compose."""
    return _mosaic.compose(samples, out_h=out_h, out_w=out_w, grid=grid,
                           center_jitter=center_jitter, center_shape=center_shape,
                           min_area=min_area)


def num_sources(grid: int = 2, **_) -> int:
    g = max(2, int(grid))
    return g * g
