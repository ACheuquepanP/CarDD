"""CarDD · src/cardd/data/augs/mosaic.py

MOSAIC: rejilla NxN de imágenes ENTERAS reescaladas. Cada celda recibe una imagen
completa deformada para llenarla. El punto de corte de la rejilla se jitterea alrededor
del centro (cuadrado o círculo). `grid` controla NxN (2=2x2, 3=3x3...).

Parámetro de intensidad (enhance): `grid`.
"""
from __future__ import annotations

import random
from typing import Sequence

import torch

from cardd.data.augs._core import (as_t, rebuild_target, resize_image, resize_masks,
                                    sample_center)


def compose(samples: Sequence[tuple[torch.Tensor, dict]], *, out_h: int, out_w: int,
            grid: int = 2, center_jitter: float = 0.25, center_shape: str = "square",
            min_area: int = 16) -> tuple[torch.Tensor, dict]:
    """Compone grid*grid imágenes enteras en una rejilla recortada a (out_h,out_w).

    Requiere len(samples) == grid*grid. Las líneas de corte se derivan de un punto
    central jittereado: para grid=2 es el clásico 2x2 con centro móvil; para grid>2 las
    líneas se reparten proporcionalmente alrededor de ese centro.
    """
    g = max(2, int(grid))
    need = g * g
    assert len(samples) == need, f"mosaic grid={g} requiere {need} muestras"

    cx, cy = sample_center(out_h, out_w, center_jitter, center_shape)
    # líneas de corte verticales (x) y horizontales (y): g+1 bordes, con la interior
    # central desplazada al punto jittereado. Para g=2 hay 1 línea interior (=centro).
    def cuts(total, c, n):
        # bordes 0..total; interiores repartidos, con la más cercana al centro en c
        if n == 2:
            return [0, int(round(c)), total]
        # g>2: repartir uniforme pero escalando para que una interior caiga en c
        base = [round(total * i / n) for i in range(n + 1)]
        return [max(0, min(total, b)) for b in base]

    xs = cuts(out_w, cx, g)
    ys = cuts(out_h, cy, g)

    canvas = torch.zeros(3, out_h, out_w, dtype=samples[0][0].dtype)
    mask_dtype = as_t(samples[0][1]["masks"]).dtype
    all_masks, all_labels, all_iscrowd = [], [], []

    k = 0
    for gy in range(g):
        for gx in range(g):
            img, tgt = samples[k]; k += 1
            x0, x1 = xs[gx], xs[gx + 1]
            y0, y1 = ys[gy], ys[gy + 1]
            cw, ch = x1 - x0, y1 - y0
            if cw <= 0 or ch <= 0:
                continue
            canvas[:, y0:y1, x0:x1] = resize_image(img, ch, cw)
            m = as_t(tgt["masks"])
            if m.numel() == 0 or m.shape[0] == 0:
                continue
            m_r = resize_masks(m, ch, cw)
            labels = as_t(tgt["labels"]).tolist()
            iscrowd = as_t(tgt.get("iscrowd", torch.zeros(m.shape[0]))).tolist()
            for j in range(m_r.shape[0]):
                full = torch.zeros(out_h, out_w, dtype=torch.bool)
                full[y0:y1, x0:x1] = m_r[j]
                if not bool(full.any()):
                    continue
                all_masks.append(full)
                all_labels.append(int(labels[j]))
                all_iscrowd.append(int(iscrowd[j]) if j < len(iscrowd) else 0)

    masks_t = (torch.stack(all_masks) if all_masks
               else torch.zeros((0, out_h, out_w), dtype=torch.bool))
    tgt = rebuild_target(masks_t, all_labels, all_iscrowd, out_h, out_w, mask_dtype,
                         samples[0][1], min_area=min_area)
    return canvas, tgt


def num_sources(grid: int = 2, **_) -> int:
    """Cuántas muestras necesita esta aug (para que el orquestador las muestree)."""
    g = max(2, int(grid))
    return g * g
