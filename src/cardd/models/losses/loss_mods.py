"""CarDD · src/cardd/models/losses/loss_mods.py

Dueño: Equipo C · Fase: B (combinables) · Contrato: C3 (registro propio LOSS_MODS)

Unidades componibles de modificación de loss, para combinar piezas por YAML sin
un companion por par. Cada mod declara su SITIO (dónde pelea en el loss) y reusa
la matemática ya testeada (focal / cldice / skeleton). El que las orquesta es
`mask_rcnn_composed` (un solo wrap instance-scoped que despacha por sitio).

Dos sitios, porque los mods NO atacan el mismo punto:
  - "classification": reemplaza el loss de clasificación (dentro de fastrcnn_loss).
                      Solo puede haber UNO por combinada (reemplazo, no suma).
  - "mask":           SUMA un término sobre la máscara, con clave propia
                      (loss_mask_*), tantos como se quieran.

El composer extrae UNA sola vez (pred, tgt) de máscara y se lo pasa a cada mask-mod
vía `.term(pred, tgt)`, para no rasterizar el GT N veces.
"""
from __future__ import annotations

import torch

from cardd.utils.registry import Registry

LOSS_MODS = Registry("loss_mod")


class LossMod:
    """Base. `site` ∈ {'classification','mask'}."""
    site: str = ""


# --------------------------------------------------------------------------- #
# Sitio: classification (reemplaza el loss de clasificación)                   #
# --------------------------------------------------------------------------- #
@LOSS_MODS.register("focal")
class FocalMod(LossMod):
    site = "classification"

    def __init__(self, alpha=0.5, gamma: float = 2.0):
        self.alpha = alpha
        self.gamma = gamma

    def classification_loss(self, class_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        from cardd.models.losses.focal import softmax_focal_loss
        return softmax_focal_loss(class_logits, labels, self.alpha, self.gamma, reduction="mean")


# --------------------------------------------------------------------------- #
# Sitio: mask (suma un término con clave propia)                              #
# --------------------------------------------------------------------------- #
class _MaskTermMod(LossMod):
    site = "mask"
    key: str = ""

    def term(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


@LOSS_MODS.register("cldice")
class ClDiceMod(_MaskTermMod):
    key = "loss_mask_cldice"

    def __init__(self, weight: float = 1.0, iters: int = 3, smooth: float = 1.0):
        self.weight = weight
        self.iters = iters
        self.smooth = smooth

    def term(self, pred, tgt):
        from cardd.models.losses.cldice import soft_cldice
        return self.weight * soft_cldice(pred, tgt, iters=self.iters, smooth=self.smooth)


@LOSS_MODS.register("skeleton")
class SkeletonMod(_MaskTermMod):
    key = "loss_mask_skeleton"

    def __init__(self, weight: float = 1.0, iters: int = 3, tube: bool = True, smooth: float = 1.0):
        self.weight = weight
        self.iters = iters
        self.tube = tube
        self.smooth = smooth

    def term(self, pred, tgt):
        from cardd.models.losses.skeleton_recall import skeleton_recall
        return self.weight * skeleton_recall(pred, tgt, iters=self.iters, tube=self.tube, smooth=self.smooth)


def build_loss_mod(entry):
    """Acepta 'focal' (str, defaults) o {'name':'focal','gamma':2.0} (dict)."""
    if isinstance(entry, str):
        return LOSS_MODS.get(entry)()
    return LOSS_MODS.build(entry)


__all__ = ["LOSS_MODS", "LossMod", "FocalMod", "ClDiceMod", "SkeletonMod", "build_loss_mod"]