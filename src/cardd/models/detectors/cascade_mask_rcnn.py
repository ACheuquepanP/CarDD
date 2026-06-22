"""CarDD · src/cardd/models/detectors/cascade_mask_rcnn.py

Dueño: Equipo C · Fase: 5 (fila a4_cascade) · Contrato: C3 (registro DETECTORS)

Cascade Mask R-CNN (Cai & Vasconcelos, CVPR'18) sobre el Mask R-CNN de M.
Hipótesis de a4: mejor localización de cajas vía 3 stages de refinamiento con
umbral de IoU creciente (0.5 -> 0.6 -> 0.7).

A diferencia de focal (swap de loss) y pointrend (augment de la rama de máscara),
cascade NO tiene primitiva en torchvision: es un reemplazo de la rama de cajas de
`RoIHeads`. Para acotar riesgo, `CascadeRoIHeads` SUBCLASEA el `RoIHeads` de
torchvision y reusa sus helpers ya testeados (`assign_targets_to_proposals`,
`subsample`, `postprocess_detections`, NMS); encima solo va el loop de cascade.

Sin tocar a M (mismo criterio que focal/pointrend):
  · Subclase de `CarddMaskRCNN`, firma (cfg, **overrides).
  · En __init__, tras construir la base, REEMPLAZA `self.roi_heads` por
    `CascadeRoIHeads`, reusando del roi_heads original el box_roi_pool, el
    sampler, los umbrales de NMS y TODA la rama de máscara (la máscara corre una
    sola vez sobre las cajas del último stage, como en Cascade Mask R-CNN).
  · Los losses por-stage se PLIEGAN (suma ponderada por `stage_loss_weights`) en
    `loss_classifier` / `loss_box_reg` -> sin claves nuevas, trainer de M intacto.
  · C2 sin cambios: la salida sigue siendo {boxes,scores,labels,masks}.

Confianza / validación:
  · Decode de refinamiento, promediado de scores y plegado de loss: standalone +
    numpy (tests/test_cascade.py).
  · El forward integrado se valida en el venv (torch/torchvision) contra el
    detector real de M. Es la fila que MÁS necesita la corrida real.

Suposición sobre M (misma de siempre): `self.roi_heads` es el `RoIHeads`
estándar de torchvision con sus atributos de caja y máscara.
"""
from __future__ import annotations

from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cardd.utils.config import get
from cardd.utils.registry import DETECTORS


# Defaults de Cascade R-CNN (paper)
_DEFAULT_IOUS = [0.5, 0.6, 0.7]
_DEFAULT_LOSS_W = [1.0, 0.5, 0.25]
_DEFAULT_REG_W = [
    (10.0, 10.0, 5.0, 5.0),
    (20.0, 20.0, 10.0, 10.0),
    (30.0, 30.0, 15.0, 15.0),
]


def _build_cascade_roi_heads(base, num_stages, iou_thresholds, stage_loss_weights, bbox_reg_weights):
    """Construye CascadeRoIHeads reusando los componentes del roi_heads base de M."""
    from torchvision.models.detection.roi_heads import RoIHeads
    from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
    from torchvision.models.detection._utils import BoxCoder, Matcher

    # dims leídas del roi_heads base (no se hardcodean)
    resolution = base.box_roi_pool.output_size[0]
    representation_size = base.box_predictor.cls_score.in_features
    num_classes = base.box_predictor.cls_score.out_features
    out_channels = base.box_head.fc6.in_features // (resolution * resolution)

    class CascadeRoIHeads(RoIHeads):
        def __init__(self):
            nn.Module.__init__(self)
            # --- reuso del base (compartido entre stages) ---
            self.box_roi_pool = base.box_roi_pool
            self.fg_bg_sampler = base.fg_bg_sampler
            self.box_similarity = __import__(
                "torchvision.ops.boxes", fromlist=["box_iou"]
            ).box_iou
            self.score_thresh = base.score_thresh
            self.nms_thresh = base.nms_thresh
            self.detections_per_img = base.detections_per_img
            # rama de máscara: tal cual la base (corre 1 vez sobre el último stage)
            self.mask_roi_pool = base.mask_roi_pool
            self.mask_head = base.mask_head
            self.mask_predictor = base.mask_predictor
            self.keypoint_roi_pool = None
            self.keypoint_head = None
            self.keypoint_predictor = None

            # --- por stage ---
            self.num_stages = num_stages
            self.stage_loss_weights = list(stage_loss_weights)
            self.box_heads = nn.ModuleList([
                TwoMLPHead(out_channels * resolution * resolution, representation_size)
                for _ in range(num_stages)
            ])
            self.box_predictors = nn.ModuleList([
                FastRCNNPredictor(representation_size, num_classes)
                for _ in range(num_stages)
            ])
            self.stage_box_coders = [BoxCoder(weights=tuple(w)) for w in bbox_reg_weights]
            self.stage_matchers = [
                Matcher(t, t, allow_low_quality_matches=False) for t in iou_thresholds
            ]
            # placeholders que algunos helpers heredados consultan
            self.proposal_matcher = self.stage_matchers[0]
            self.box_coder = self.stage_box_coders[0]
            self.box_head = self.box_heads[0]
            self.box_predictor = self.box_predictors[0]

        # -- helper: refinamiento de propuestas entre stages (decode + clip) --
        def _refine(self, proposals, class_logits, box_regression, box_coder, image_shapes):
            boxes_per_image = [p.shape[0] for p in proposals]
            pred_boxes = box_coder.decode(box_regression, proposals)      # [sum, C, 4]
            pred_boxes = pred_boxes.split(boxes_per_image, 0)
            logits = class_logits.split(boxes_per_image, 0)
            from torchvision.ops import boxes as box_ops
            refined = []
            for boxes_i, logits_i, prop_i, shape in zip(pred_boxes, logits, proposals, image_shapes):
                labels_i = logits_i.argmax(dim=1)                          # clase predicha
                idx = torch.arange(boxes_i.shape[0], device=boxes_i.device)
                chosen = boxes_i[idx, labels_i]                            # [n,4] caja de su clase
                # fondo (label 0): conservar la propuesta original
                bg = labels_i == 0
                chosen[bg] = prop_i[bg]
                chosen = box_ops.clip_boxes_to_image(chosen, shape)
                refined.append(chosen.detach())
            return refined

        def forward(self, features, proposals, image_shapes, targets=None):
            from torchvision.models.detection.roi_heads import (
                fastrcnn_loss, maskrcnn_loss, maskrcnn_inference,
            )

            cls_per_stage: List[torch.Tensor] = []
            stage_losses: List[Tuple[torch.Tensor, torch.Tensor]] = []
            last_matched_idxs = last_labels = None

            for s in range(self.num_stages):
                self.proposal_matcher = self.stage_matchers[s]
                self.box_coder = self.stage_box_coders[s]

                if self.training:
                    # asignar+samplear con el matcher/coder de este stage
                    p, matched_idxs, labels, reg_targets = self.select_training_samples(
                        proposals, targets
                    )
                    proposals = p
                    last_matched_idxs, last_labels = matched_idxs, labels
                else:
                    labels = reg_targets = None

                box_features = self.box_roi_pool(features, proposals, image_shapes)
                box_features = self.box_heads[s](box_features)
                class_logits, box_regression = self.box_predictors[s](box_features)

                if self.training:
                    lc, lb = fastrcnn_loss(class_logits, box_regression, labels, reg_targets)
                    stage_losses.append((lc, lb))
                else:
                    cls_per_stage.append(F.softmax(class_logits, dim=-1))

                if s < self.num_stages - 1:
                    proposals = self._refine(
                        proposals, class_logits, box_regression,
                        self.stage_box_coders[s], image_shapes,
                    )

            result: List[Dict[str, torch.Tensor]] = []
            losses: Dict[str, torch.Tensor] = {}

            if self.training:
                # plegar losses por-stage en las claves estándar (suma ponderada)
                w = self.stage_loss_weights
                losses["loss_classifier"] = sum(w[i] * lc for i, (lc, _) in enumerate(stage_losses))
                losses["loss_box_reg"] = sum(w[i] * lb for i, (_, lb) in enumerate(stage_losses))
            else:
                # promediar scores de los 3 stages sobre las cajas del último stage
                avg_scores = torch.stack(cls_per_stage, dim=0).mean(dim=0)
                eps = torch.finfo(avg_scores.dtype).eps
                final_logits = torch.log(avg_scores.clamp(min=eps))        # vuelta a "logits" para postprocess
                zeros = torch.zeros(
                    final_logits.shape[0], (final_logits.shape[1]) * 4, device=final_logits.device
                )
                boxes, scores, labels_out = self.postprocess_detections(
                    final_logits, zeros, proposals, image_shapes      # box_reg=0 -> usa las cajas ya refinadas
                )
                for b, sc, la in zip(boxes, scores, labels_out):
                    result.append({"boxes": b, "labels": la, "scores": sc})

            # --- rama de máscara (compartida, 1 vez) ---
            if self.has_mask():
                if self.training:
                    num_images = len(proposals)
                    mask_proposals, pos_matched_idxs = [], []
                    for i in range(num_images):
                        pos = torch.where(last_labels[i] > 0)[0]
                        mask_proposals.append(proposals[i][pos])
                        pos_matched_idxs.append(last_matched_idxs[i][pos])
                else:
                    mask_proposals = [r["boxes"] for r in result]
                    pos_matched_idxs = None

                mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
                mask_features = self.mask_head(mask_features)
                mask_logits = self.mask_predictor(mask_features)

                if self.training:
                    gt_masks = [t["masks"] for t in targets]
                    gt_labels = [t["labels"] for t in targets]
                    losses["loss_mask"] = maskrcnn_loss(
                        mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs
                    )
                else:
                    labels_r = [r["labels"] for r in result]
                    for prob, r in zip(maskrcnn_inference(mask_logits, labels_r), result):
                        r["masks"] = prob

            return result, losses

    return CascadeRoIHeads()


def _resolve_roi_heads(detector):
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

    @DETECTORS.register("cascade_mask_rcnn")
    class CascadeMaskRCNN(CarddMaskRCNN):
        """a0 con la rama de cajas en cascada (3 stages). Firma (cfg, **overrides)."""

        def __init__(self, cfg, **overrides):
            super().__init__(cfg, **overrides)
            num_stages = get(cfg, "model.detector.num_stages", 3)
            ious = get(cfg, "model.detector.iou_thresholds", _DEFAULT_IOUS)
            loss_w = get(cfg, "model.detector.stage_loss_weights", _DEFAULT_LOSS_W)
            reg_w = get(cfg, "model.detector.bbox_reg_weights", _DEFAULT_REG_W)
            base = _resolve_roi_heads(self)
            self.roi_heads = _build_cascade_roi_heads(base, num_stages, ious, loss_w, reg_w)


__all__ = ["_build_cascade_roi_heads"]