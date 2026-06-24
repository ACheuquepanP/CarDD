"""CarDD · src/cardd/models/losses/skeleton_recall.py

Dueño: Equipo C · Fase: B (§5.2 custom) · Contrato: C3 (registro LOSSES)

Skeleton Recall (Kirchhoff et al., 2024): loss topológica barata y multi-clase.
A diferencia de clDice (que esqueletoniza pred y target iterativamente), aquí se
esqueletoniza SOLO el GT —una vez, sin gradiente— y se mide el *recall* de la
predicción sobre ese esqueleto (opcionalmente "entubado" con una dilatación):

    skel = tube(skeleton(target))            # detached, barato
    recall = |pred ∩ skel| / |skel|
    loss   = 1 - recall

Penaliza roturas de continuidad (la pred que no cubre el esqueleto del GT) sin el
costo de la doble skeletonización de clDice. Reusa las primitivas morfológicas de
cldice.py. El cableado aditivo en RoIHeads lo hace el companion mask_rcnn_skeleton
(clave loss_mask_skeleton).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from cardd.utils.registry import LOSSES
from cardd.models.losses.cldice import soft_skel, soft_dilate


def skeleton_recall(pred: torch.Tensor, target: torch.Tensor, iters: int = 3,
                    tube: bool = True, smooth: float = 1.0) -> torch.Tensor:
    """Loss Skeleton Recall. pred/target: [N,1,H,W] en [0,1]. Devuelve escalar.

    El esqueleto del GT se calcula sin gradiente (target es fijo): el gradiente
    fluye solo por `pred` -> barato.
    """
    with torch.no_grad():
        skel = soft_skel(target, iters)
        if tube:
            skel = soft_dilate(skel).clamp(max=1.0) * target
    recall = (torch.sum(pred * skel) + smooth) / (torch.sum(skel) + smooth)
    return 1.0 - recall


@LOSSES.register("skeleton_recall")
class SkeletonRecallLoss(nn.Module):
    """Construible por nombre: LOSSES.build({"name":"skeleton_recall","iters":3})."""

    def __init__(self, iters: int = 3, tube: bool = True, smooth: float = 1.0) -> None:
        super().__init__()
        self.iters = iters
        self.tube = tube
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return skeleton_recall(pred, target, self.iters, self.tube, self.smooth)


__all__ = ["skeleton_recall", "SkeletonRecallLoss"]