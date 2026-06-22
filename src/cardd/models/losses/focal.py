"""CarDD · src/cardd/models/losses/focal.py

Dueño: Equipo C · Fase: 5 (fila a1_focal) · Contrato: C3 (registro LOSSES)

Focal loss MULTICLASE (softmax) para el head de clasificación de Mask R-CNN.

El clasificador de `RoIHeads` decide entre `num_classes = 7` (fondo=0 + 6 daños)
por softmax, no por sigmoide independiente. Por eso la variante correcta es el
focal multiclase de Lin et al. sobre la probabilidad de la clase verdadera:

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

con `p_t` = softmax(logits)[clase_verdadera]. Cuando `gamma=0` y `alpha=1`
colapsa exactamente a cross-entropy (invariante testeado en test_focal.py).

Hiperparámetros congelados para la fila a1 (matriz de ablación): alpha=0.5, gamma=2.0.
El módulo es autodescriptivo por config: lee alpha/gamma del YAML, no de M.

Este archivo entrega SOLO la matemática del loss (torch puro, sin tocar a M).
El cableado dentro de `RoIHeads` lo hace la subclase de detector
`mask_rcnn_focal` (ver detectors/mask_rcnn_focal.py), que es la vía recomendada
por M (handoff Fase 3/4 §4.4): el loss de clasificación se computa DENTRO de
`RoIHeads`, no es un kwarg suelto de `MaskRCNN`.
"""
from __future__ import annotations

from typing import Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from cardd.utils.registry import LOSSES

AlphaLike = Union[float, Sequence[float], torch.Tensor, None]


def softmax_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: AlphaLike = 0.5,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Focal loss multiclase sobre logits softmax.

    Args:
        logits:  [N, C] logits sin normalizar (C = num_classes, incluye fondo).
        targets: [N] int64 en 0..C-1 (0 = fondo, mismo espacio que C1/C2).
        alpha:   factor de balanceo. Escalar (a todas las clases), secuencia/
                 tensor [C] (alpha por clase, indexado por target), o None (sin
                 reponderado). a1 usa el escalar 0.5.
        gamma:   factor de enfoque (down-weight de ejemplos fáciles). a1 usa 2.0.
        reduction: 'mean' | 'sum' | 'none'.

    Returns:
        Escalar (mean/sum) o [N] (none). Con N=0 devuelve 0.0 (no NaN), para no
        romper la suma de losses cuando un batch no aporta propuestas.
    """
    if logits.numel() == 0:
        if reduction == "none":
            return logits.new_zeros((0,))
        return logits.new_zeros(())

    # ce_i = -log p_t  (cross-entropy por ejemplo, sin reducir)
    ce = F.cross_entropy(logits, targets, reduction="none")
    p_t = torch.exp(-ce)                       # p_t = softmax(logits)[target]
    loss = (1.0 - p_t) ** gamma * ce           # (1 - p_t)^gamma * (-log p_t)

    if alpha is not None:
        if isinstance(alpha, (int, float)):
            loss = float(alpha) * loss
        else:
            alpha_t = torch.as_tensor(alpha, dtype=loss.dtype, device=loss.device)
            loss = alpha_t.gather(0, targets) * loss

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"reduction desconocido: {reduction!r}")


@LOSSES.register("focal")
class FocalLoss(nn.Module):
    """Wrapper registrado en LOSSES (C3): construible desde config por nombre.

        LOSSES.build({"name": "focal", "alpha": 0.5, "gamma": 2.0})

    Mantiene la misma firma de llamada que `F.cross_entropy(logits, targets)`,
    de modo que es un reemplazo directo allí donde `RoIHeads` calcula el loss
    de clasificación.
    """

    def __init__(
        self,
        alpha: AlphaLike = 0.5,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return softmax_focal_loss(
            logits, targets, self.alpha, self.gamma, self.reduction
        )

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, gamma={self.gamma}, reduction={self.reduction!r}"


__all__ = ["softmax_focal_loss", "FocalLoss"]