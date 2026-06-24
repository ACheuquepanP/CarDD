"""CarDD · src/cardd/models/losses/cldice.py

Dueño: Equipo C · Fase: B (§5.2 custom) · Contrato: C3 (registro LOSSES)

soft-clDice (Shit et al., CVPR'21): loss topológica para estructuras tubulares
(scratch, crack). Mide solape entre el esqueleto de una máscara y la otra, así
penaliza roturas de continuidad que el BCE/Dice estándar no ve.

Skeletonización soft = aperturas morfológicas iterativas vía min/max-pool
(diferenciable). clDice combina:
    Tprec = |skel_pred ∩ target| / |skel_pred|
    Tsens = |skel_true ∩ pred|   / |skel_true|
    clDice = 2·Tprec·Tsens / (Tprec+Tsens) ;  loss = 1 - clDice

Entrega SOLO la matemática (torch puro). El cableado como término aditivo en
RoIHeads lo hace el companion mask_rcnn_cldice (clave loss_mask_cldice).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cardd.utils.registry import LOSSES


def soft_erode(img: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
    return torch.min(p1, p2)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))


def soft_open(img: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(img))


def soft_skel(img: torch.Tensor, iters: int = 3) -> torch.Tensor:
    """Esqueleto soft por aperturas iterativas. img: [N,1,H,W] en [0,1]."""
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice(pred: torch.Tensor, target: torch.Tensor, iters: int = 3, smooth: float = 1.0) -> torch.Tensor:
    """Loss soft-clDice. pred/target: [N,1,H,W] en [0,1]. Devuelve escalar."""
    skel_pred = soft_skel(pred, iters)
    skel_true = soft_skel(target, iters)
    tprec = (torch.sum(skel_pred * target) + smooth) / (torch.sum(skel_pred) + smooth)
    tsens = (torch.sum(skel_true * pred) + smooth) / (torch.sum(skel_true) + smooth)
    return 1.0 - 2.0 * tprec * tsens / (tprec + tsens)


@LOSSES.register("cldice")
class SoftClDiceLoss(nn.Module):
    """Construible por nombre: LOSSES.build({"name":"cldice","iters":3})."""

    def __init__(self, iters: int = 3, smooth: float = 1.0) -> None:
        super().__init__()
        self.iters = iters
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return soft_cldice(pred, target, self.iters, self.smooth)


__all__ = ["soft_erode", "soft_dilate", "soft_open", "soft_skel", "soft_cldice", "SoftClDiceLoss"]