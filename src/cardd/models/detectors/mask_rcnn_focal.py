"""CarDD · src/cardd/models/detectors/mask_rcnn_focal.py

Dueño: Equipo C · Fase: 5 (fila a1_focal) · Contrato: C3 (registro DETECTORS)

Fila a1: reemplaza el loss de clasificación de Mask R-CNN por focal, SIN editar
`mask_rcnn.py` de M. Es la vía recomendada por M (handoff Fase 3/4 §4.4): el loss
de clasificación se computa dentro de `RoIHeads`, así que la forma limpia de
cambiarlo es registrar un detector que herede de `CarddMaskRCNN` y, en su
construcción, intervenir su `roi_heads`.

UN CAMBIO POR FILA (higiene de la ablación): solo se sustituye el término de
clasificación. El término de regresión de caja (`box_loss`) se reproduce VERBATIM
del `fastrcnn_loss` de torchvision 0.26 → a1 difiere de a0 únicamente en el loss
de clasif., nada más.

Suposición única sobre M (a confirmar en integración): el detector expone
`self.roi_heads` (el atributo estándar de `GeneralizedRCNN`; `CarddMaskRCNN`
ensambla `MaskRCNN`, que lo tiene). Si M renombrara ese atributo, esto es un
ajuste de una línea en `_resolve_roi_heads`.

Diseño del parche: instance-scoped y reversible. No se monkeypatchea el global
de torchvision de forma permanente; se intercambia `fastrcnn_loss` SOLO durante
el `forward` de ESTA instancia y se restaura en un `finally`. Así conviven varios
detectores en el mismo proceso y no se contamina a0 ni otras filas.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from cardd.utils.config import get
from cardd.utils.registry import DETECTORS
from cardd.models.losses.focal import softmax_focal_loss


# --------------------------------------------------------------------------- #
# Reemplazo fiel de torchvision.models.detection.roi_heads.fastrcnn_loss       #
# Misma firma y mismo camino de box_loss; solo cambia la clasificación.        #
# --------------------------------------------------------------------------- #
def make_focal_fastrcnn_loss(alpha, gamma: float) -> Callable:
    """Devuelve un drop-in de `fastrcnn_loss` con focal en la clasificación.

    Firma idéntica a la de torchvision:
        (class_logits, box_regression, labels, regression_targets)
            -> (classification_loss, box_loss)
    """

    def focal_fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        # --- ÚNICO cambio respecto de a0: focal en vez de cross_entropy ---
        classification_loss = softmax_focal_loss(
            class_logits, labels, alpha=alpha, gamma=gamma, reduction="mean"
        )

        # --- box_loss: VERBATIM de torchvision 0.26 (no se toca) ---
        sampled_pos_inds_subset = torch.where(labels > 0)[0]
        labels_pos = labels[sampled_pos_inds_subset]
        N, num_classes = class_logits.shape
        box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

        box_loss = F.smooth_l1_loss(
            box_regression[sampled_pos_inds_subset, labels_pos],
            regression_targets[sampled_pos_inds_subset],
            beta=1 / 9,
            reduction="sum",
        )
        box_loss = box_loss / labels.numel()

        return classification_loss, box_loss

    return focal_fastrcnn_loss


def install_focal_classification_loss(roi_heads, alpha=0.5, gamma: float = 2.0):
    """Hace que `roi_heads` use focal en su loss de clasificación.

    Envuelve `roi_heads.forward`: durante la llamada intercambia el `fastrcnn_loss`
    del módulo de torchvision por el focal y lo restaura al salir (reversible y
    seguro ante excepciones). Idempotente. Testeable de forma aislada sobre un
    `MaskRCNN` de torchvision stock (ver tests/test_focal.py).
    """
    import torchvision.models.detection.roi_heads as _rh

    if getattr(roi_heads, "_focal_installed", False):
        return roi_heads

    orig_forward = roi_heads.forward
    focal_loss_fn = make_focal_fastrcnn_loss(alpha, gamma)

    def _forward(*args, **kwargs):
        prev = _rh.fastrcnn_loss
        _rh.fastrcnn_loss = focal_loss_fn
        try:
            return orig_forward(*args, **kwargs)
        finally:
            _rh.fastrcnn_loss = prev

    roi_heads.forward = _forward
    roi_heads._focal_installed = True
    roi_heads._focal_params = {"alpha": alpha, "gamma": gamma}
    return roi_heads


def _resolve_roi_heads(detector):
    """Punto único de acoplamiento con M. Si M renombra el atributo, se ajusta aquí."""
    if not hasattr(detector, "roi_heads"):
        raise AttributeError(
            "El detector base no expone `.roi_heads`. La fila a1 asume el atributo "
            "estándar de torchvision GeneralizedRCNN. Confirmar con M (handoff §4.2)."
        )
    return detector.roi_heads


# --------------------------------------------------------------------------- #
# Detector registrado. Se define solo si la base de M (CarddMaskRCNN) existe;  #
# así este módulo se importa y testea (la maquinaria de loss de arriba) aun    #
# antes de que M aterrice su Fase 3/4.                                         #
# --------------------------------------------------------------------------- #
try:
    from cardd.models.detectors.mask_rcnn import CarddMaskRCNN  # provisto por M
    _HAS_BASE = True
except Exception:  # stub de M todavía sin clase -> la maquinaria sigue importable
    _HAS_BASE = False


if _HAS_BASE:

    @DETECTORS.register("mask_rcnn_focal")
    class FocalMaskRCNN(CarddMaskRCNN):
        """a0 + focal en el clasificador. Misma firma (cfg, **overrides) que la base."""

        def __init__(self, cfg, **overrides):
            super().__init__(cfg, **overrides)
            alpha = get(cfg, "model.detector.focal.alpha", 0.5)
            gamma = get(cfg, "model.detector.focal.gamma", 2.0)
            install_focal_classification_loss(_resolve_roi_heads(self), alpha, gamma)


__all__ = [
    "make_focal_fastrcnn_loss",
    "install_focal_classification_loss",
]