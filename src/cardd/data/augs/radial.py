"""CarDD · src/cardd/data/augs/radial.py

RADIAL: parte el lienzo en `m_wedges` cuñas angulares que convergen en un punto
(centro ± jitter) y se extienden hasta el borde (warp total, sin zonas vacías). Cada
cuña se rellena con una de `n_sources` fuentes (muestreo con reemplazo). m=4 -> aspa.

Parámetros de intensidad (enhance): `n_sources`, `m_wedges`.
"""
from __future__ import annotations

import math
import random
from typing import Sequence

import torch

from cardd.data.augs._core import as_t, rebuild_target, resize_image, resize_masks


def compose(samples: Sequence[tuple[torch.Tensor, dict]], *, out_h: int, out_w: int,
            m_wedges: int = 6, center_jitter: float = 0.15, center_shape: str = "circle",
            min_area: int = 16) -> tuple[torch.Tensor, dict]:
    m = max(3, int(m_wedges))
    n = len(samples)
    assert n >= 1
    dtype = samples[0][0].dtype
    mask_dtype = as_t(samples[0][1]["masks"]).dtype

    # centro con jitter (circle o square)
    if center_shape == "circle":
        r = center_jitter * math.sqrt(random.random())
        a = random.uniform(0, 2 * math.pi)
        fx, fy = r * math.cos(a), r * math.sin(a)
    else:
        fx = random.uniform(-center_jitter, center_jitter)
        fy = random.uniform(-center_jitter, center_jitter)
    cx = out_w * (0.5 + fx)
    cy = out_h * (0.5 + fy)

    ys, xs = torch.meshgrid(torch.arange(out_h, dtype=torch.float32),
                            torch.arange(out_w, dtype=torch.float32), indexing="ij")
    ang = torch.atan2(ys - cy, xs - cx)
    ang = (ang + 2 * math.pi) % (2 * math.pi)
    rot = random.uniform(0.0, 2 * math.pi / m)
    wedge_of_pixel = (((ang + rot) % (2 * math.pi)) / (2 * math.pi) * m).floor().long()
    wedge_of_pixel = wedge_of_pixel.clamp_(0, m - 1)

    canvas = torch.zeros(3, out_h, out_w, dtype=dtype)
    all_masks, all_labels, all_iscrowd = [], [], []

    for w in range(m):
        img, tgt = samples[random.randrange(n)]
        img_r = resize_image(img, out_h, out_w)
        sector = wedge_of_pixel == w
        canvas[:, sector] = img_r[:, sector]
        msk = as_t(tgt["masks"])
        if msk.numel() == 0 or msk.shape[0] == 0:
            continue
        m_r = resize_masks(msk, out_h, out_w)
        labels = as_t(tgt["labels"]).tolist()
        iscrowd = as_t(tgt.get("iscrowd", torch.zeros(msk.shape[0]))).tolist()
        for k in range(m_r.shape[0]):
            inst = m_r[k] & sector
            if not bool(inst.any()):
                continue
            all_masks.append(inst)
            all_labels.append(int(labels[k]))
            all_iscrowd.append(int(iscrowd[k]) if k < len(iscrowd) else 0)

    masks_t = (torch.stack(all_masks) if all_masks
               else torch.zeros((0, out_h, out_w), dtype=torch.bool))
    tgt = rebuild_target(masks_t, all_labels, all_iscrowd, out_h, out_w, mask_dtype,
                         samples[0][1], min_area=min_area)
    return canvas, tgt


def num_sources(n_sources: int = 4, **_) -> int:
    return max(1, int(n_sources))
