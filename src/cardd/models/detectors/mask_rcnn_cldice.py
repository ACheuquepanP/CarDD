"""CarDD · src/cardd/models/detectors/mask_rcnn_cldice.py

Dueño: Equipo C · Fase: B (§5.2 custom) · Contrato: C3 (registro DETECTORS)

Companion de clDice (Gate ④, aprobado por M). Suma el término topológico
soft-clDice al loss de máscara, SIN editar mask_rcnn.py. Mismo patrón que el
companion de PointRend: instance-scoped y reversible.

- Reusa el coarse `maskrcnn_loss` VERBATIM (riesgo cero en esa mitad).
- Calcula clDice sobre los logits de máscara (clase GT, sigmoid) vs el GT
  rasterizado a la caja (project_masks_on_boxes a la resolución de la máscara).
- Lo expone como CLAVE PROPIA `loss_mask_cldice` (no lo pliega en loss_mask),
  para que quede aislado en TensorBoard. El trainer hace sum(loss_dict.values()).
- No monkeypatch global: intercambia maskrcnn_loss solo durante el forward de la
  instancia y restaura en finally.

Suposición sobre M: `self.roi_heads` estándar (aislado en _resolve_roi_heads).
"""
from __future__ import annotations

import torch

from cardd.utils.config import get
from cardd.utils.registry import DETECTORS
from cardd.models.losses.cldice import soft_cldice


def make_cldice_maskrcnn_loss(orig_loss, holder, iters, smooth):
    """Drop-in de maskrcnn_loss: coarse verbatim + stash del término clDice."""
    from torchvision.models.detection.roi_heads import project_masks_on_boxes

    def cldice_maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
        coarse = orig_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs)
        if mask_logits.numel() == 0:
            holder["term"] = None
            return coarse
        R = mask_logits.shape[-1]
        labels = torch.cat(
            [gl[idx] for gl, idx in zip(gt_labels, mask_matched_idxs)], dim=0
        )
        targets = torch.cat(
            [project_masks_on_boxes(m, p, i, R)
             for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)],
            dim=0,
        )  # [N, R, R]
        idx = torch.arange(labels.shape[0], device=labels.device)
        pred = mask_logits[idx, labels].sigmoid().unsqueeze(1)   # [N,1,R,R]
        tgt = targets.unsqueeze(1).float()                       # [N,1,R,R]
        holder["term"] = soft_cldice(pred, tgt, iters=iters, smooth=smooth)
        return coarse

    return cldice_maskrcnn_loss


def install_cldice(roi_heads, weight: float = 1.0, iters: int = 3, smooth: float = 1.0):
    """Activa clDice en roi_heads. Idempotente, reversible por-forward."""
    import torchvision.models.detection.roi_heads as _rh

    if getattr(roi_heads, "_cldice_installed", False):
        return roi_heads

    holder = {"term": None}
    orig_forward = roi_heads.forward
    loss_fn = make_cldice_maskrcnn_loss(_rh.maskrcnn_loss, holder, iters, smooth)

    def _forward(*args, **kwargs):
        prev = _rh.maskrcnn_loss
        _rh.maskrcnn_loss = loss_fn
        holder["term"] = None
        try:
            out = orig_forward(*args, **kwargs)
        finally:
            _rh.maskrcnn_loss = prev
        if isinstance(out, tuple) and len(out) == 2 and holder["term"] is not None:
            _, losses = out
            losses["loss_mask_cldice"] = weight * holder["term"]
        return out

    roi_heads.forward = _forward
    roi_heads._cldice_installed = True
    roi_heads._cldice_params = {"weight": weight, "iters": iters}
    return roi_heads


def _resolve_roi_heads(detector):
    if not hasattr(detector, "roi_heads"):
        raise AttributeError(
            "El detector base no expone `.roi_heads` (se asume torchvision "
            "GeneralizedRCNN). Confirmar con M, handoff §4.2."
        )
    return detector.roi_heads


try:
    from cardd.models.detectors.mask_rcnn import CarddMaskRCNN  # provisto por M
    _HAS_BASE = True
except Exception:
    _HAS_BASE = False


if _HAS_BASE:

    @DETECTORS.register("mask_rcnn_cldice")
    class MaskRCNNClDice(CarddMaskRCNN):
        """a0 + término soft-clDice en el loss de máscara. Firma (cfg, **overrides)."""

        def __init__(self, cfg, **overrides):
            super().__init__(cfg, **overrides)
            weight = get(cfg, "model.detector.cldice.weight", 1.0)
            iters = get(cfg, "model.detector.cldice.iters", 3)
            install_cldice(_resolve_roi_heads(self), weight=weight, iters=iters)


__all__ = ["make_cldice_maskrcnn_loss", "install_cldice"]