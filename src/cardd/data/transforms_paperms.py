"""CarDD · src/cardd/data/transforms_paperms.py

Dueño: Equipo C · Reevaluación · Contrato: C3 (registro TRANSFORMS) · C1 (I/O)

Multiscale estilo PAPER CarDD (DCN+): altura aleatoria en [h_min, h_max], ancho
FIJO en `width` -> DEFORMA (no preserva aspect ratio), a diferencia del multiscale
por ratios (isotrópico, en el modelo).

Neutralización del doble-resize: usar con min_size=max_size=width en la config
(el GeneralizedRCNNTransform queda ~identidad). Opera imagen+boxes+masks consistente.
I/O según C1: image Tensor [C,H,W] en [0,1]; target con boxes xyxy px, masks [N,H,W].

Debug: imprime ~0.5% de las imágenes con su tamaño REAL medido en el instante del
forward (shape del tensor que entra y del que sale), no lo que diga ninguna config.
"""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from cardd.utils.registry import TRANSFORMS


@TRANSFORMS.register("paper_multiscale")
class PaperMultiScale:
    """Resize del paper: alto ~ U[h_min,h_max], ancho fijo. Deforma a propósito."""

    def __init__(self, h_min: int = 640, h_max: int = 1200, width: int = 1333,
                 debug_sample: float = 0.005) -> None:
        assert 0 < h_min <= h_max
        self.h_min, self.h_max, self.width = int(h_min), int(h_max), int(width)
        self.debug_sample = float(debug_sample)

    def __call__(self, image, target=None):
        _, H, W = image.shape                      # tamaño REAL del tensor que entra
        new_h = random.randint(self.h_min, self.h_max)
        new_w = self.width
        sy, sx = new_h / H, new_w / W

        image = F.interpolate(image.unsqueeze(0), size=(new_h, new_w),
                              mode="bilinear", align_corners=False
                              ).squeeze(0).clamp_(0.0, 1.0)

        # debug: lectura en el instante (shapes medidos, no config)
        if self.debug_sample > 0 and random.random() < self.debug_sample:
            print(f"[paper_ms] entra {H}x{W} -> sale {image.shape[-2]}x{image.shape[-1]}",
                  flush=True)

        if target is None:
            return image, target
        if "boxes" in target and target["boxes"].numel():
            b = target["boxes"].clone().float()
            b[:, [0, 2]] *= sx
            b[:, [1, 3]] *= sy
            target["boxes"] = b
        if "masks" in target and target["masks"].numel():
            m = target["masks"].unsqueeze(1).float()
            m = F.interpolate(m, size=(new_h, new_w), mode="nearest")
            target["masks"] = (m.squeeze(1) > 0.5).to(target["masks"].dtype)
        if "area" in target and target["boxes"].numel():
            b = target["boxes"]
            target["area"] = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        return image, target


@TRANSFORMS.register("paper_hflip")
class PaperHFlip:
    """Flip horizontal p=0.5 (default de MMDetection/torchvision para detección).

    Mismo contrato C1 que PaperMultiScale: image [C,H,W] en [0,1]; boxes xyxy px;
    masks [N,H,W]. Voltea imagen+boxes+masks de forma consistente. 'area' no cambia.
    """

    def __init__(self, p: float = 0.5) -> None:
        assert 0.0 <= p <= 1.0
        self.p = float(p)

    def __call__(self, image, target=None):
        if random.random() >= self.p:
            return image, target

        W = image.shape[-1]
        image = torch.flip(image, dims=[-1])

        if target is None:
            return image, target
        if "boxes" in target and target["boxes"].numel():
            b = target["boxes"].clone().float()
            x1 = W - b[:, 2]
            x2 = W - b[:, 0]
            b[:, 0], b[:, 2] = x1, x2
            target["boxes"] = b
        if "masks" in target and target["masks"].numel():
            target["masks"] = torch.flip(target["masks"], dims=[-1])
        return image, target


__all__ = ["PaperMultiScale", "PaperHFlip"]
