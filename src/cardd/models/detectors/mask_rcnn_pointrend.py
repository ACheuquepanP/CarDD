"""CarDD · src/cardd/models/detectors/mask_rcnn_pointrend.py

Dueño: Equipo C · Fase: 5 (fila a3_pointrend, sub-paso 2) · Contrato: C3 (DETECTORS)

Companion de PointRend: aplica el refinamiento por puntos dentro de `RoIHeads`,
SIN editar `mask_rcnn.py` de M. Subclase de `CarddMaskRCNN` (igual que focal).

Qué hace, en dos mitades:
  · TRAIN     -> suma un point-loss al loss de máscara (puntos inciertos).
  · INFERENCE -> subdivisión coarse-to-fine (refina bordes en alta resolución).

Cómo, sin perturbar a M (mismo criterio que mask_rcnn_focal.py):
  1. Envuelve `mask_predictor.forward` para *stashear* las features finas (su
     input), que `maskrcnn_loss`/`maskrcnn_inference` no reciben pero PointRend
     necesita.
  2. Intercambia los free-functions `maskrcnn_loss` y `maskrcnn_inference` de
     torchvision SOLO durante el forward de esta instancia (reversible, finally).
  3. El point-loss se SUMA dentro del escalar `loss_mask` -> no se agregan claves
     al dict de pérdidas, el trainer de M queda intacto. El coarse loss se reusa
     verbatim del `maskrcnn_loss` original (riesgo cero en esa mitad). La inference
     devuelve el mismo formato que `maskrcnn_inference` (C2 sin cambios).

Suposición única sobre M (a confirmar en integración, mismo punto que focal):
  el detector expone `self.roi_heads` con `.mask_head` y `.mask_predictor`
  estándar de torchvision, y `mask_predictor` es el `PointRendMaskPredictor` de
  heads/pointrend.py (lo selecciona la config a3).

Confianza / validación:
  · `point_rendering_loss` y `subdivision_inference` son standalone y van con
    tests (tests/test_pointrend_companion.py).
  · El instalador y la mitad que usa `project_masks_on_boxes` se validan contra
    el `mask_rcnn.py` real de M en el venv (torch/torchvision). Es la pieza a
    integrar de verdad cuando M aterrice.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from cardd.utils.config import get
from cardd.utils.registry import DETECTORS
from cardd.models.heads.pointrend import (
    point_sample,
    calculate_uncertainty,
    get_uncertain_point_coords_with_randomness,
    get_uncertain_point_coords_on_grid,
)


# --------------------------------------------------------------------------- #
# Algoritmos core (standalone, testeables sin torchvision-internals)           #
# --------------------------------------------------------------------------- #
def point_rendering_loss(
    coarse_logits: torch.Tensor,
    fine_features: torch.Tensor,
    mask_targets_hr: torch.Tensor,
    labels: torch.Tensor,
    point_head,
    num_points: int,
    oversample_ratio: float,
    importance_sample_ratio: float,
) -> torch.Tensor:
    """Point-loss de PointRend (BCE en puntos inciertos).

    Args:
        coarse_logits:   [N, C, h, w] logits coarse del predictor.
        fine_features:   [N, Cf, h, w] features finas (input del predictor).
        mask_targets_hr: [N, R, R] GT binaria recortada a la caja, alta resolución.
        labels:          [N] clase GT por instancia (1..N).
        point_head:      el StandardPointHead del predictor.
    Returns:
        Escalar. Con N=0 -> 0 (no NaN).
    """
    if coarse_logits.numel() == 0:
        return coarse_logits.sum() * 0.0

    with torch.no_grad():
        point_coords = get_uncertain_point_coords_with_randomness(
            coarse_logits,
            lambda pl: calculate_uncertainty(pl, labels),
            num_points, oversample_ratio, importance_sample_ratio,
        )
    fine = point_sample(fine_features, point_coords)        # [N, Cf, P]
    coarse = point_sample(coarse_logits, point_coords)      # [N, C, P]
    point_logits = point_head(fine, coarse)                 # [N, C|1, P]

    with torch.no_grad():
        gt = point_sample(mask_targets_hr[:, None].float(), point_coords)  # [N,1,P]
        gt = (gt.squeeze(1) >= 0.5).float()                 # [N, P] binaria

    if point_logits.shape[1] == 1:                          # class-agnostic
        sel = point_logits[:, 0, :]
    else:
        idx = torch.arange(point_logits.shape[0], device=point_logits.device)
        sel = point_logits[idx, labels, :]                  # [N, P]
    return F.binary_cross_entropy_with_logits(sel, gt)


@torch.no_grad()
def subdivision_inference(
    coarse_logits: torch.Tensor,
    fine_features: torch.Tensor,
    point_head,
    classes: torch.Tensor,
    subdivision_steps: int,
    subdivision_num_points: int,
) -> torch.Tensor:
    """Subdivisión coarse-to-fine (INFERENCE). Devuelve logits refinados de mayor res.

    Empieza del coarse, y en cada paso: upsamplea x2, elige los puntos más
    inciertos de la grilla, los re-predice con el point-head y los inserta de
    vuelta (scatter). Solo usa salidas del modelo (sin GT).
    """
    mask_logits = coarse_logits
    for _ in range(subdivision_steps):
        mask_logits = F.interpolate(
            mask_logits, scale_factor=2, mode="bilinear", align_corners=False
        )
        uncertainty = calculate_uncertainty(mask_logits, classes)          # [N,1,H,W]
        point_indices, point_coords = get_uncertain_point_coords_on_grid(
            uncertainty, subdivision_num_points
        )
        fine = point_sample(fine_features, point_coords)                   # [N, Cf, P]
        coarse = point_sample(coarse_logits, point_coords)                 # [N, C, P]
        point_logits = point_head(fine, coarse)                            # [N, C|1, P]

        N, C, H, W = mask_logits.shape
        idx = point_indices.unsqueeze(1).expand(-1, C, -1)                 # [N, C, P]
        mask_logits = (
            mask_logits.reshape(N, C, H * W)
            .scatter_(2, idx, point_logits)
            .view(N, C, H, W)
        )
    return mask_logits


# --------------------------------------------------------------------------- #
# Drop-ins de torchvision (instance-scoped, reversibles)                       #
# --------------------------------------------------------------------------- #
def make_pointrend_maskrcnn_loss(orig_loss, predictor, weight: float, target_res: int):
    """`maskrcnn_loss` drop-in: coarse (verbatim) + weight * point_loss."""
    from torchvision.models.detection.roi_heads import project_masks_on_boxes

    def pointrend_maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
        coarse = orig_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs)
        fine = getattr(predictor, "_last_fine", None)
        if fine is None or mask_logits.numel() == 0:
            return coarse
        labels = torch.cat(
            [gl[idx] for gl, idx in zip(gt_labels, mask_matched_idxs)], dim=0
        )
        targets = torch.cat(
            [project_masks_on_boxes(m, p, i, target_res)
             for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)],
            dim=0,
        )  # [N, R, R]
        point = point_rendering_loss(
            mask_logits, fine, targets, labels, predictor.point_head,
            predictor.train_num_points, predictor.oversample_ratio,
            predictor.importance_sample_ratio,
        )
        return coarse + weight * point

    return pointrend_maskrcnn_loss


def make_pointrend_maskrcnn_inference(predictor):
    """`maskrcnn_inference` drop-in: subdivisión y MISMO formato de salida."""

    def pointrend_maskrcnn_inference(mask_logits, labels):
        boxes_per_image = [l.shape[0] for l in labels]
        cls = (
            torch.cat(labels) if len(labels)
            else torch.empty(0, dtype=torch.long, device=mask_logits.device)
        )
        fine = getattr(predictor, "_last_fine", None)
        if fine is None or mask_logits.numel() == 0:
            refined = mask_logits
        else:
            refined = subdivision_inference(
                mask_logits, fine, predictor.point_head, cls,
                predictor.subdivision_steps, predictor.subdivision_num_points,
            )
        prob = refined.sigmoid()
        index = torch.arange(refined.shape[0], device=cls.device)
        prob = prob[index, cls][:, None]                    # [num, 1, H, W]
        return prob.split(boxes_per_image, dim=0)

    return pointrend_maskrcnn_inference


def install_pointrend(roi_heads, point_loss_weight: float = 1.0, target_resolution: int = 112):
    """Activa PointRend en `roi_heads`. Idempotente, reversible por-forward."""
    import torchvision.models.detection.roi_heads as _rh

    if getattr(roi_heads, "_pointrend_installed", False):
        return roi_heads

    predictor = roi_heads.mask_predictor
    if not hasattr(predictor, "point_head"):
        raise AttributeError(
            "mask_predictor no es PointRendMaskPredictor (sin .point_head). "
            "Revisar que a3_pointrend.yaml seleccione 'pointrend_mask_predictor'."
        )

    # 1) stashear las features finas (input del predictor)
    orig_pred_forward = predictor.forward

    def _pred_forward(x):
        predictor._last_fine = x
        return orig_pred_forward(x)

    predictor.forward = _pred_forward

    # 2) drop-ins y wrap reversible del forward de roi_heads
    loss_fn = make_pointrend_maskrcnn_loss(_rh.maskrcnn_loss, predictor, point_loss_weight, target_resolution)
    infer_fn = make_pointrend_maskrcnn_inference(predictor)
    orig_forward = roi_heads.forward

    def _forward(*args, **kwargs):
        prev_loss, prev_infer = _rh.maskrcnn_loss, _rh.maskrcnn_inference
        _rh.maskrcnn_loss = loss_fn
        _rh.maskrcnn_inference = infer_fn
        try:
            return orig_forward(*args, **kwargs)
        finally:
            _rh.maskrcnn_loss, _rh.maskrcnn_inference = prev_loss, prev_infer

    roi_heads.forward = _forward
    roi_heads._pointrend_installed = True
    roi_heads._pointrend_params = {"point_loss_weight": point_loss_weight, "target_resolution": target_resolution}
    return roi_heads


def _resolve_roi_heads(detector):
    """Punto único de acoplamiento con M (mismo criterio que mask_rcnn_focal.py)."""
    if not hasattr(detector, "roi_heads"):
        raise AttributeError(
            "El detector base no expone `.roi_heads` (se asume torchvision "
            "GeneralizedRCNN). Confirmar con M, handoff §4.2."
        )
    return detector.roi_heads


# --------------------------------------------------------------------------- #
# Detector registrado (se define solo si la base de M existe)                  #
# --------------------------------------------------------------------------- #
try:
    from cardd.models.detectors.mask_rcnn import CarddMaskRCNN  # provisto por M
    _HAS_BASE = True
except Exception:
    _HAS_BASE = False


if _HAS_BASE:

    @DETECTORS.register("mask_rcnn_pointrend")
    class MaskRCNNPointRend(CarddMaskRCNN):
        """a0 + PointRend en la rama de máscara. Misma firma (cfg, **overrides)."""

        def __init__(self, cfg, **overrides):
            super().__init__(cfg, **overrides)
            weight = get(cfg, "model.detector.point_loss_weight", 1.0)
            res = get(cfg, "model.detector.point_target_resolution", 112)
            install_pointrend(_resolve_roi_heads(self), point_loss_weight=weight, target_resolution=res)


__all__ = [
    "point_rendering_loss",
    "subdivision_inference",
    "make_pointrend_maskrcnn_loss",
    "make_pointrend_maskrcnn_inference",
    "install_pointrend",
]