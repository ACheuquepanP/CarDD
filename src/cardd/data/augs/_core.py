"""CarDD · src/cardd/data/augs/_core.py

Núcleo compartido por todas las augmentaciones de composición. No depende de
ninguna aug concreta: solo utilidades de target C1 (tv_tensors, boxes desde
máscara, ensamblado, blur, target vacío) reusadas por mosaic/radial/crop/copy_paste.

Convención C1 (idéntica al dataset base):
    image:  Tensor[3,H,W] float32 en [0,1]
    target: {boxes: BoundingBoxes[N,4] xyxy, labels: int64[N], masks: Mask[N,H,W] uint8,
             image_id: int, area: float32[N], iscrowd: uint8[N]}
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torchvision import tv_tensors


def as_t(x) -> torch.Tensor:
    return torch.as_tensor(x)


def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """[N,H,W] bool/uint8 -> [N,4] xyxy float32. Filas vacías -> ceros (no revienta
    con máscaras vacías tras oclusión; el filtro por área ocurre después)."""
    n = masks.shape[0]
    if n == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    m = masks.bool()
    boxes = torch.zeros((n, 4), dtype=torch.float32)
    rows = m.any(dim=2)
    cols = m.any(dim=1)
    for i in range(n):
        r = torch.where(rows[i])[0]
        c = torch.where(cols[i])[0]
        if r.numel() == 0 or c.numel() == 0:
            continue
        boxes[i] = torch.tensor(
            [c[0].item(), r[0].item(), c[-1].item() + 1.0, r[-1].item() + 1.0],
            dtype=torch.float32,
        )
    return boxes


def empty_target_like(template: dict, h: int, w: int) -> dict:
    """Target C1 vacío (0 instancias) con el canvas (h,w)."""
    out = dict(template)
    out["boxes"] = tv_tensors.BoundingBoxes(
        torch.zeros((0, 4), dtype=torch.float32),
        format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=(h, w))
    out["labels"] = torch.zeros((0,), dtype=torch.int64)
    out["masks"] = tv_tensors.Mask(torch.zeros((0, h, w), dtype=torch.uint8))
    out["area"] = torch.zeros((0,), dtype=torch.float32)
    out["iscrowd"] = torch.zeros((0,), dtype=torch.uint8)
    return out


def rebuild_target(masks: torch.Tensor, labels: Sequence[int], iscrowd: Sequence[int],
                   h: int, w: int, mask_dtype: torch.dtype, template: dict,
                   *, min_area: int = 16) -> dict:
    """Ensambla target C1 desde máscaras+labels, recalculando cajas desde la máscara
    y descartando instancias degeneradas (caja nula o área < min_area). tv_tensors
    reconstruidos con canvas correcto."""
    if masks.shape[0] == 0:
        return empty_target_like(template, h, w)
    masks = masks.to(mask_dtype)
    boxes = masks_to_boxes(masks)
    wh = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    areas = masks.bool().flatten(1).sum(1)
    keep = (wh > 0) & (areas >= min_area)
    if not bool(keep.all()):
        idx = torch.where(keep)[0].tolist()
        masks = masks[keep]
        boxes = boxes[keep]
        labels = [labels[i] for i in idx]
        iscrowd = [iscrowd[i] for i in idx]
    if masks.shape[0] == 0:
        return empty_target_like(template, h, w)
    out = dict(template)
    out["boxes"] = tv_tensors.BoundingBoxes(
        boxes, format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=(h, w))
    out["labels"] = torch.tensor(list(labels), dtype=torch.int64)
    out["masks"] = tv_tensors.Mask(masks)
    out["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    out["iscrowd"] = torch.tensor(list(iscrowd), dtype=torch.uint8)
    return out


def gaussian_blur(x: torch.Tensor, sigma: float, ksize: int) -> torch.Tensor:
    """Blur gaussiano separable sobre [1,1,H,W]. Sin dependencia de cv2."""
    if ksize % 2 == 0:
        ksize += 1
    xk = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2.0
    k = torch.exp(-(xk ** 2) / (2.0 * sigma ** 2))
    k = (k / k.sum()).to(x.device, x.dtype)
    pad = ksize // 2
    x = F.pad(x, (pad, pad, 0, 0), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1))
    x = F.pad(x, (0, 0, pad, pad), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, -1, 1))
    return x


def resize_image(img: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """[3,H,W] -> [3,h,w] bilineal, clamp [0,1]."""
    return F.interpolate(img.unsqueeze(0), size=(h, w), mode="bilinear",
                         align_corners=False).squeeze(0).clamp_(0.0, 1.0)


def resize_masks(masks: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """[N,H,W] -> [N,h,w] nearest, bool."""
    if masks.numel() == 0 or masks.shape[0] == 0:
        return torch.zeros((masks.shape[0], h, w), dtype=torch.bool)
    m = F.interpolate(masks.unsqueeze(1).float(), size=(h, w), mode="nearest").squeeze(1)
    return m > 0.5


def sample_center(out_h: int, out_w: int, jitter: float, shape: str = "square"):
    """Punto de convergencia/corte con jitter alrededor del centro.

    shape='square': offset independiente por eje en [-jitter, jitter] (región cuadrada).
    shape='circle': radio*jitter en dirección angular aleatoria (región circular).
    Devuelve (cx, cy) en píxeles.
    """
    import math
    import random
    if shape == "circle":
        r = jitter * math.sqrt(random.random())      # uniforme en área del disco
        a = random.uniform(0, 2 * math.pi)
        fx, fy = r * math.cos(a), r * math.sin(a)
    else:
        fx = random.uniform(-jitter, jitter)
        fy = random.uniform(-jitter, jitter)
    cx = out_w * (0.5 + fx)
    cy = out_h * (0.5 + fy)
    return cx, cy
