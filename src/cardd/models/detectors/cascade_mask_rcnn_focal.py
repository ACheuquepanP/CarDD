"""CarDD · src/cardd/models/detectors/cascade_mask_rcnn_focal.py

Dueño: Equipo C · Fase: B (experimento) · Contrato: C3 (registro DETECTORS)

EXPERIMENTO: Cascade Mask R-CNN + focal loss en la clasificación de las etapas.

El composed excluye focal por diseño (hipótesis: redundante con el re-muestreo
por etapas de cascade). Este detector existe para PROBAR esa hipótesis
empíricamente, sin tocar el composed ni relajar su ValueError. Si el experimento
confirma la redundancia, este archivo se retira; si la refuta, se discute su
integración formal.

Por qué funciona sin tocar cascade_mask_rcnn.py:
  El forward de CascadeRoIHeads hace, EN CADA LLAMADA:
      from torchvision.models.detection.roi_heads import fastrcnn_loss, ...
  Ese import resuelve el atributo del módulo en el momento de la llamada, así
  que el swap instance-scoped de `_rh.fastrcnn_loss` (mismo patrón que
  mask_rcnn_focal.py) es levantado automáticamente por LAS 3 ETAPAS. Focal se
  aplica a cada stage y luego los losses se pliegan igual (stage_loss_weights).

Composición completa (hereda de CascadeMaskRCNNComposed):
  cascade (3 stages) + focal (stages) [+ loss_mods de máscara] [+ pointrend]
  Los dos wraps del forward son independientes (intercambian funciones
  distintas: fastrcnn_loss vs maskrcnn_loss/inference) y se encadenan sin
  pisarse. El de focal envuelve al de la pila de máscara.

Config:
    model:
      detector:
        name: cascade_mask_rcnn_focal
        focal: {alpha: 0.5, gamma: 2.0}     # opcional, defaults abajo
        loss_mods: [cldice]                  # opcional (solo sitio mask)
        pointrend: true                      # opcional

Nota de ablación: comparar contra cascade_mask_rcnn_composed CON LA MISMA
config de máscara, para aislar el efecto de focal.
"""
from __future__ import annotations

from cardd.utils.config import get
from cardd.utils.registry import DETECTORS

# Reuso verbatim del drop-in de focal (a1): misma firma, mismo box_loss.
from cardd.models.detectors.mask_rcnn_focal import make_focal_fastrcnn_loss


def install_focal_on_cascade(roi_heads, alpha=0.5, gamma: float = 2.0):
    """Focal en la clasificación de TODAS las etapas de un CascadeRoIHeads.

    Instance-scoped, reversible, idempotente. Envuelve el forward vigente
    (posiblemente ya envuelto por install_mask_stack_on_cascade): intercambia
    `_rh.fastrcnn_loss` durante la llamada y restaura en finally. Como el
    forward de cascade importa fastrcnn_loss por llamada, el swap alcanza a
    las 3 etapas sin tocar su código.
    """
    import torchvision.models.detection.roi_heads as _rh

    if getattr(roi_heads, "_cascade_focal_installed", False):
        return roi_heads

    orig_forward = roi_heads.forward
    focal_fn = make_focal_fastrcnn_loss(alpha, gamma)

    def _forward(*args, **kwargs):
        prev = _rh.fastrcnn_loss
        _rh.fastrcnn_loss = focal_fn
        try:
            return orig_forward(*args, **kwargs)
        finally:
            _rh.fastrcnn_loss = prev

    roi_heads.forward = _forward
    roi_heads._cascade_focal_installed = True
    roi_heads._cascade_focal_params = {"alpha": alpha, "gamma": gamma}
    return roi_heads


try:
    from cardd.models.detectors.cascade_mask_rcnn_composed import (
        CascadeMaskRCNNComposed,
    )
    _HAS_COMPOSED = True
except Exception:
    _HAS_COMPOSED = False


if _HAS_COMPOSED:

    @DETECTORS.register("cascade_mask_rcnn_focal")
    class CascadeMaskRCNNFocal(CascadeMaskRCNNComposed):
        """Cascade + focal en los stages (+ pila de máscara del composed).
        Firma (cfg, **overrides). Lee model.detector.focal.{alpha,gamma}."""

        def __init__(self, cfg, **overrides):
            # super() arma cascade + (opcionalmente) mods de máscara + pointrend.
            super().__init__(cfg, **overrides)
            alpha = get(cfg, "model.detector.focal.alpha", 0.5)
            gamma = get(cfg, "model.detector.focal.gamma", 2.0)
            # Se instala DESPUÉS de la pila de máscara: nuestro wrap envuelve al
            # del composed. Independientes (funciones distintas), se encadenan.
            install_focal_on_cascade(self.roi_heads, alpha=alpha, gamma=gamma)


__all__ = ["install_focal_on_cascade"]
