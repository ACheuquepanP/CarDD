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

CAMBIOS (implementación Copy-Paste):
  1. FIX tv_tensors: boxes/masks se reenvuelven como BoundingBoxes/Mask al salir.
     Antes, `.clone()` preservaba la subclase pero NO actualizaba `canvas_size`, y
     `F.interpolate` sobre masks devolvía un Tensor plano (se perdía la subclase
     `Mask`). Inofensivo para MaskRCNN de torchvision -que no lee `canvas_size` ni
     las subclases- pero un transform v2 posterior habría recortado boxes contra un
     canvas obsoleto y habría ignorado las masks silenciosamente.
  2. `force_h` en PaperMultiScale.__call__: permite imponer la altura desde fuera en
     lugar de sortearla. Lo usa CopyPasteDataset para escalar la muestra fuente y la
     destino a la MISMA altura y componerlas sin un segundo resize.
  3. `size_divisible`: redondea alto y ancho al múltiplo de stride más cercano. El
     backbone con DCN+FPN exige dimensiones divisibles por el stride total (32 en
     FPN estándar); una altura arbitraria en [640,1200] rara vez lo es, y el residual
     del DCN reventaba con un mismatch de ±1 píxel. Antes lo salvaba el padding del
     GeneralizedRCNNTransform/ImageList; copy-paste cambió el camino y lo destapó.
     `1` desactiva el redondeo (comportamiento previo).
"""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from torchvision import tv_tensors

from cardd.utils.registry import TRANSFORMS


@TRANSFORMS.register("paper_multiscale")
class PaperMultiScale:
    """Resize del paper: alto ~ U[h_min,h_max], ancho fijo. Deforma a propósito."""

    def __init__(self, h_min: int = 640, h_max: int = 1200, width: int = 1333,
                 size_divisible: int = 32, debug_sample: float = 0.005) -> None:
        assert 0 < h_min <= h_max
        assert size_divisible >= 1
        self.h_min, self.h_max, self.width = int(h_min), int(h_max), int(width)
        self.size_divisible = int(size_divisible)
        self.debug_sample = float(debug_sample)

    def _round(self, v: int) -> int:
        """Redondea al múltiplo de `size_divisible` más cercano (mínimo un múltiplo)."""
        m = self.size_divisible
        if m <= 1:
            return int(v)
        return max(m, int(round(v / m)) * m)

    def sample_h(self) -> int:
        """Sortea una altura del rango, ya redondeada al stride. Público para que
        CopyPasteDataset sortee UNA vez y la imponga a las dos muestras vía `force_h`.
        Devolver la altura ya redondeada garantiza que fuente y destino compartan
        exactamente el mismo alto (si redondeara dentro de __call__, un `force_h`
        crudo podría redondearse igual, pero así el contrato es explícito)."""
        return self._round(random.randint(self.h_min, self.h_max))

    def __call__(self, image, target=None, force_h: int | None = None):
        _, H, W = image.shape                      # tamaño REAL del tensor que entra
        new_h = self._round(int(force_h)) if force_h is not None else self.sample_h()
        new_w = self._round(self.width)
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
            b = torch.as_tensor(target["boxes"]).clone().float()
            b[:, [0, 2]] *= sx
            b[:, [1, 3]] *= sy
            target["boxes"] = tv_tensors.BoundingBoxes(
                b, format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=(new_h, new_w)
            )
        elif "boxes" in target:
            # sin instancias: igualmente hay que refrescar canvas_size
            target["boxes"] = tv_tensors.BoundingBoxes(
                torch.zeros((0, 4), dtype=torch.float32),
                format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=(new_h, new_w)
            )
        if "masks" in target and target["masks"].numel():
            dtype = target["masks"].dtype
            m = torch.as_tensor(target["masks"]).unsqueeze(1).float()
            m = F.interpolate(m, size=(new_h, new_w), mode="nearest")
            target["masks"] = tv_tensors.Mask((m.squeeze(1) > 0.5).to(dtype))
        elif "masks" in target:
            target["masks"] = tv_tensors.Mask(
                torch.zeros((0, new_h, new_w), dtype=target["masks"].dtype)
            )
        if "area" in target and target["boxes"].numel():
            b = torch.as_tensor(target["boxes"])
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
            b = torch.as_tensor(target["boxes"]).clone().float()
            x1 = W - b[:, 2]
            x2 = W - b[:, 0]
            b[:, 0], b[:, 2] = x1, x2
            canvas = getattr(target["boxes"], "canvas_size", (image.shape[-2], W))
            target["boxes"] = tv_tensors.BoundingBoxes(
                b, format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=canvas
            )
        if "masks" in target and target["masks"].numel():
            target["masks"] = tv_tensors.Mask(
                torch.flip(torch.as_tensor(target["masks"]), dims=[-1])
            )
        return image, target


__all__ = ["PaperMultiScale", "PaperHFlip"]
