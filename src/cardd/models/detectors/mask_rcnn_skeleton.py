"""CarDD · src/cardd/models/detectors/mask_rcnn_skeleton.py

Dueño: Equipo C · Fase: B (§5.2 custom) · Contrato: C3 (registro DETECTORS)

Companion de Skeleton Recall (Gate ④, aprobado por M). Gemelo estructural del
companion de clDice: mismo patrón instance-scoped y reversible, reusa el coarse
maskrcnn_loss verbatim, y suma el término como CLAVE PROPIA `loss_mask_skeleton`.
Cambia solo la matemática del término (skeleton_recall en vez de soft_cldice).

Suposición sobre M: `self.roi_heads` estándar (aislado en _resolve_roi_heads).
"""
from __future__ import annotations

import torch

from cardd.utils.config import get
from cardd.utils.registry import DETECTORS
from cardd.models.losses.skeleton_recall import skeleton_recall


def make_skeleton_maskrcnn_loss(orig_loss, holder, iters, tube, smooth):
    from torchvision.models.detection.roi_heads import project_masks_on_boxes

    def skeleton_maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
        coarse = orig_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs)
        if mask_logits.numel() == 0:
            holder["term"] = None
            return coarse
        R = mask_logits.shape[-1]
        labels = torch.cat([gl[idx] for gl, idx in zip(gt_labels, mask_matched_idxs)], dim=0)
        targets = torch.cat(
            [project_masks_on_boxes(m, p, i, R)
             for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)],
            dim=0,
        )
        idx = torch.arange(labels.shape[0], device=labels.device)
        pred = mask_logits[idx, labels].sigmoid().unsqueeze(1)
        tgt = targets.unsqueeze(1).float()
        holder["term"] = skeleton_recall(pred, tgt, iters=iters, tube=tube, smooth=smooth)
        return coarse

    return skeleton_maskrcnn_loss


def install_skeleton(roi_heads, weight: float = 1.0, iters: int = 3, tube: bool = True, smooth: float = 1.0):
    import torchvision.models.detection.roi_heads as _rh

    if getattr(roi_heads, "_skeleton_installed", False):
        return roi_heads

    holder = {"term": None}
    orig_forward = roi_heads.forward
    loss_fn = make_skeleton_maskrcnn_loss(_rh.maskrcnn_loss, holder, iters, tube, smooth)

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
            losses["loss_mask_skeleton"] = weight * holder["term"]
        return out

    roi_heads.forward = _forward
    roi_heads._skeleton_installed = True
    roi_heads._skeleton_params = {"weight": weight, "iters": iters, "tube": tube}
    return roi_heads


def _resolve_roi_heads(detector):
    if not hasattr(detector, "roi_heads"):
        raise AttributeError(
            "El detector base no expone `.roi_heads` (se asume torchvision "
            "GeneralizedRCNN). Confirmar con M, handoff §4.2."
        )
    return detector.roi_heads


try:
    from cardd.models.detectors.mask_rcnn import CarddMaskRCNN
    _HAS_BASE = True
except Exception:
    _HAS_BASE = False


if _HAS_BASE:

    @DETECTORS.register("mask_rcnn_skeleton")
    class MaskRCNNSkeleton(CarddMaskRCNN):
        """a0 + término Skeleton Recall en el loss de máscara. Firma (cfg, **overrides)."""

        def __init__(self, cfg, **overrides):
            super().__init__(cfg, **overrides)
            weight = get(cfg, "model.detector.skeleton.weight", 1.0)
            iters = get(cfg, "model.detector.skeleton.iters", 3)
            tube = get(cfg, "model.detector.skeleton.tube", True)
            install_skeleton(_resolve_roi_heads(self), weight=weight, iters=iters, tube=tube)


__all__ = ["make_skeleton_maskrcnn_loss", "install_skeleton"]