"""CarDD · src/cardd/models/detectors/mask_rcnn_composed.py

Dueño: Equipo C · Fase: B (combinables) · Contrato: C3 (registro DETECTORS)

Companion ÚNICO para combinar loss-mods por YAML. Lee la lista `loss_mods` del
config, la separa por sitio y hace UN solo wrap instance-scoped que intercepta
los dos free-functions de torchvision una vez:

  - `fastrcnn_loss`  -> sitio "classification": si hay un mod, reemplaza el loss
                        de clasificación; el box loss se reproduce verbatim.
  - `maskrcnn_loss`  -> sitio "mask": coarse verbatim + cada mask-mod suma su
                        término con clave propia (loss_mask_*).

Reversible (restaura en finally). C2 intacto (solo vive en el forward de loss).
Extrae (pred, tgt) de máscara UNA vez y se lo pasa a todos los mask-mods.

Combinar = agregar a la lista del YAML:
    model:
      detector: {name: mask_rcnn_composed, loss_mods: [focal, cldice]}

Caveat: `cascade` NO es un loss-mod (reemplaza RoIHeads). Combinar algo con
cascade es integración aparte (handoff de coordinación §4).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from cardd.utils.config import get
from cardd.utils.registry import DETECTORS
from cardd.models.losses.loss_mods import build_loss_mod


def _make_fastrcnn_loss(cls_mod):
    """Drop-in de fastrcnn_loss: clasificación del mod (o CE) + box loss verbatim."""
    def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        if cls_mod is not None:
            classification_loss = cls_mod.classification_loss(class_logits, labels)
        else:
            classification_loss = F.cross_entropy(class_logits, labels)

        sampled_pos_inds_subset = torch.where(labels > 0)[0]
        labels_pos = labels[sampled_pos_inds_subset]
        N, num_classes = class_logits.shape
        box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)
        box_loss = F.smooth_l1_loss(
            box_regression[sampled_pos_inds_subset, labels_pos],
            regression_targets[sampled_pos_inds_subset],
            beta=1 / 9, reduction="sum",
        )
        box_loss = box_loss / labels.numel()
        return classification_loss, box_loss

    return fastrcnn_loss


def _make_maskrcnn_loss(orig_loss, mask_mods, holder):
    """Drop-in de maskrcnn_loss: coarse verbatim + stash de cada término de máscara."""
    from torchvision.models.detection.roi_heads import project_masks_on_boxes

    def maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
        coarse = orig_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs)
        if not mask_mods or mask_logits.numel() == 0:
            return coarse
        R = mask_logits.shape[-1]
        labels = torch.cat([gl[i] for gl, i in zip(gt_labels, mask_matched_idxs)], dim=0)
        targets = torch.cat(
            [project_masks_on_boxes(m, p, i, R)
             for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)],
            dim=0,
        )
        idx = torch.arange(labels.shape[0], device=labels.device)
        pred = mask_logits[idx, labels].sigmoid().unsqueeze(1)
        tgt = targets.unsqueeze(1).float()
        for m in mask_mods:
            holder[m.key] = m.term(pred, tgt)
        return coarse

    return maskrcnn_loss


def install_composed(roi_heads, mods):
    """Instala una lista de loss-mods, instance-scoped y reversible. Idempotente."""
    import torchvision.models.detection.roi_heads as _rh

    if getattr(roi_heads, "_composed_installed", False):
        return roi_heads

    cls_mods = [m for m in mods if m.site == "classification"]
    mask_mods = [m for m in mods if m.site == "mask"]
    if len(cls_mods) > 1:
        raise ValueError(
            "Solo un loss-mod de sitio 'classification' por combinada (es reemplazo, "
            f"no suma). Recibidos: {[type(m).__name__ for m in cls_mods]}."
        )
    cls_mod = cls_mods[0] if cls_mods else None

    holder: dict = {}
    orig_forward = roi_heads.forward
    fastrcnn_fn = _make_fastrcnn_loss(cls_mod)
    maskrcnn_fn = _make_maskrcnn_loss(_rh.maskrcnn_loss, mask_mods, holder)

    def _forward(*args, **kwargs):
        pf, pm = _rh.fastrcnn_loss, _rh.maskrcnn_loss
        _rh.fastrcnn_loss = fastrcnn_fn
        _rh.maskrcnn_loss = maskrcnn_fn
        holder.clear()
        try:
            out = orig_forward(*args, **kwargs)
        finally:
            _rh.fastrcnn_loss, _rh.maskrcnn_loss = pf, pm
        if isinstance(out, tuple) and len(out) == 2 and holder:
            _, losses = out
            for key, term in holder.items():
                losses[key] = term
        return out

    roi_heads.forward = _forward
    roi_heads._composed_installed = True
    roi_heads._composed_mods = [type(m).__name__ for m in mods]
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

    @DETECTORS.register("mask_rcnn_composed")
    class MaskRCNNComposed(CarddMaskRCNN):
        """a0 + lista componible de loss-mods. Firma (cfg, **overrides)."""

        def __init__(self, cfg, **overrides):
            super().__init__(cfg, **overrides)
            entries = get(cfg, "model.detector.loss_mods", []) or []
            mods = [build_loss_mod(e) for e in entries]
            install_composed(_resolve_roi_heads(self), mods)


__all__ = ["install_composed", "_make_fastrcnn_loss", "_make_maskrcnn_loss"]