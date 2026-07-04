"""CarDD · src/cardd/models/detectors/cascade_mask_rcnn_composed.py

Dueño: Equipo C · Fase: B (reevaluación) · Contrato: C3 (registro DETECTORS)

Cascade Mask R-CNN + apilado de mejoras de MÁSCARA por config, en UN solo detector:
  · loss-mods topológicos (clDice / Skeleton Recall)  -> model.detector.loss_mods
  · PointRend (point-loss en train + subdivisión en inference) -> model.detector.pointrend: true

Resuelve el choque de slot: cascade, loss-mods y pointrend querían todos el slot
detector. Acá conviven porque son mejoras ORTOGONALES de la rama de máscara y se
integran en UN único wrap del forward (no installs anidados, que no encadenan).

Encadenamiento del loss de máscara (clave para que no se pisen):
  maskrcnn_loss efectivo = pointrend( mods( coarse ) )
  - `mods(coarse)`: calcula el coarse verbatim y STASHEA los términos topológicos
    en un holder -> se emiten como claves propias (loss_mask_cldice/skeleton).
  - `pointrend(...)`: usa ESE resultado como su `orig_loss` (lo llama y le suma el
    point-loss DENTRO de loss_mask). Reusa make_pointrend_maskrcnn_loss tal cual.
  Un solo swap de _rh.maskrcnn_loss (+ _rh.maskrcnn_inference si pointrend) por
  forward, reversible en finally. Los mods viven en el holder; pointrend en loss_mask.
  No se pisan.

focal (clasificación) queda FUERA por diseño: redundante con los stages de cascade.
"""
from __future__ import annotations

import torch

from cardd.utils.config import get
from cardd.utils.registry import DETECTORS
from cardd.models.losses.loss_mods import build_loss_mod


def _make_mask_loss_with_mods(orig_loss, mask_mods, holder):
    """Drop-in de maskrcnn_loss: coarse verbatim + stash de cada término de máscara.
    Sirve como `orig_loss` para pointrend (encadenamiento)."""
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


def install_mask_stack_on_cascade(roi_heads, mods, pointrend=False,
                                  point_loss_weight=1.0, target_resolution=112,
                                  pr_cfg=None):
    pr_cfg = pr_cfg or {}
    """Instala loss-mods de máscara (clDice/skeleton) y/o PointRend sobre un
    CascadeRoIHeads, en UN solo wrap encadenado. Instance-scoped, reversible,
    idempotente. Solo loss-mods de sitio 'mask'."""
    import torchvision.models.detection.roi_heads as _rh

    if getattr(roi_heads, "_cascade_stack_installed", False):
        return roi_heads

    bad = [type(m).__name__ for m in mods if m.site != "mask"]
    if bad:
        raise ValueError(
            f"cascade_mask_rcnn_composed solo acepta loss-mods de sitio 'mask' "
            f"(clDice/skeleton). Recibido sitio distinto: {bad}. focal no se apila "
            f"sobre cascade (redundante con los stages)."
        )

    holder: dict = {}
    orig_forward = roi_heads.forward

    # base del encadenamiento: coarse + términos de mods (holder)
    mods_loss = _make_mask_loss_with_mods(_rh.maskrcnn_loss, mods, holder)

    # si pointrend, lo envolvemos ENCIMA: usa mods_loss como su orig_loss.
    infer_fn = None
    if pointrend:
        from cardd.models.detectors.mask_rcnn_pointrend import (
            make_pointrend_maskrcnn_loss, make_pointrend_maskrcnn_inference,
        )
        from cardd.models.heads.pointrend import (
            PointRendMaskHead, PointRendMaskPredictor,
        )
        # Construimos pointrend AQUÍ (no vía inyección de M por el slot mask_predictor).
        # Motivo: inyectar mask_predictor obliga a num_classes=None en MaskRCNN, lo que
        # rompe cascade (necesita num_classes para sus box_predictors). Al construirlo
        # después, M arma el modelo normal y montamos pointrend sobre el roi_heads ya
        # hecho. num_classes/in_channels se derivan del roi_heads (como cascade con
        # base.box_predictor). pr_cfg trae los hiperparámetros del yaml.
        predictor = roi_heads.mask_predictor
        if not hasattr(predictor, "point_head"):
            in_ch = pr_cfg.get("in_channels", 256)
            num_classes = pr_cfg.get("num_classes")
            if num_classes is None:
                num_classes = roi_heads.box_predictor.cls_score.out_features
            roi_heads.mask_head = PointRendMaskHead(
                in_channels=in_ch,
                conv_dim=pr_cfg.get("conv_dim", 256),
                num_convs=pr_cfg.get("num_convs", 2),
            )
            roi_heads.mask_predictor = PointRendMaskPredictor(
                in_channels=roi_heads.mask_head.out_channels,
                num_classes=num_classes,
                dim_reduced=pr_cfg.get("dim_reduced", 256),
                train_num_points=pr_cfg.get("train_num_points", 196),
                oversample_ratio=pr_cfg.get("oversample_ratio", 3.0),
                importance_sample_ratio=pr_cfg.get("importance_sample_ratio", 0.75),
                subdivision_steps=pr_cfg.get("subdivision_steps", 3),
                subdivision_num_points=pr_cfg.get("subdivision_num_points", 196),
            )
            predictor = roi_heads.mask_predictor
        # stashear features finas (input del predictor) — igual que install_pointrend
        if not getattr(predictor, "_pr_wrapped", False):
            _orig_pred_forward = predictor.forward

            def _pred_forward(x, _f=_orig_pred_forward):
                predictor._last_fine = x
                return _f(x)

            predictor.forward = _pred_forward
            predictor._pr_wrapped = True

        mask_loss_fn = make_pointrend_maskrcnn_loss(
            mods_loss, predictor, point_loss_weight, target_resolution
        )
        infer_fn = make_pointrend_maskrcnn_inference(predictor)
    else:
        mask_loss_fn = mods_loss

    def _forward(*args, **kwargs):
        prev_loss = _rh.maskrcnn_loss
        prev_infer = _rh.maskrcnn_inference
        _rh.maskrcnn_loss = mask_loss_fn
        if infer_fn is not None:
            _rh.maskrcnn_inference = infer_fn
        holder.clear()
        try:
            out = orig_forward(*args, **kwargs)
        finally:
            _rh.maskrcnn_loss = prev_loss
            _rh.maskrcnn_inference = prev_infer
        if isinstance(out, tuple) and len(out) == 2 and holder:
            _, losses = out
            for key, term in holder.items():
                losses[key] = term
        return out

    roi_heads.forward = _forward
    roi_heads._cascade_stack_installed = True
    roi_heads._cascade_stack = {
        "mods": [type(m).__name__ for m in mods], "pointrend": bool(pointrend),
    }
    return roi_heads


try:
    from cardd.models.detectors.cascade_mask_rcnn import CascadeMaskRCNN
    _HAS_CASCADE = True
except Exception:
    _HAS_CASCADE = False


if _HAS_CASCADE:

    @DETECTORS.register("cascade_mask_rcnn_composed")
    class CascadeMaskRCNNComposed(CascadeMaskRCNN):
        """Cascade (3 stages de caja) + apilado de máscara (clDice/skeleton/pointrend).
        Firma (cfg, **overrides). Lee model.detector.loss_mods (lista) y
        model.detector.pointrend (bool)."""

        def __init__(self, cfg, **overrides):
            super().__init__(cfg, **overrides)
            entries = get(cfg, "model.detector.loss_mods", []) or []
            mods = [build_loss_mod(e) for e in entries]
            pointrend = bool(get(cfg, "model.detector.pointrend", False))
            plw = get(cfg, "model.detector.point_loss_weight", 1.0)
            res = get(cfg, "model.detector.point_target_resolution", 112)
            # hiperparámetros pointrend: del bloque model.mask_predictor si existe,
            # o defaults. Se construyen dentro del install (no vía inyección de M).
            pr_cfg = dict(get(cfg, "model.detector.pointrend_cfg", {}) or {})
            pr_cfg.pop("name", None)
            install_mask_stack_on_cascade(
                self.roi_heads, mods, pointrend=pointrend,
                point_loss_weight=plw, target_resolution=res, pr_cfg=pr_cfg,
            )


__all__ = ["install_mask_stack_on_cascade", "_make_mask_loss_with_mods"]