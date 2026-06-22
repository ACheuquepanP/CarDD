"""Mask R-CNN base (a0) sobre backbone ConvNeXt-V2 + FPN.

dueño: M · fase: 3 · contrato: C2 (produce) · C3 (expone hooks)

Ensambla el detector que:
  - **consume C1**: recibe ``list[Tensor[3,H,W] en [0,1]]`` + targets nativos de
    torchvision (boxes xyxy, labels 1..6, masks uint8). La imagen llega SIN
    normalizar (D, C1); la normalización la aplica el ``GeneralizedRCNNTransform``
    interno con ``data.image_mean/std``.
  - **produce C2**: en ``eval()``, ``model(images)`` devuelve la salida nativa de
    ``MaskRCNN`` — ``list[dict]`` por imagen con ``boxes[K,4] xyxy``,
    ``scores[K]``, ``labels[K] en 1..N`` (fondo=0 reservado) y
    ``masks[K,1,H,W] float en [0,1]``. Sin umbral de score: se entregan todas
    las detecciones (E arma la curva PR).
  - **expone C3**: puntos de inserción para que C enchufe módulos sin editar
    este archivo (ablación limpia, un cambio por fila).

## Puntos de inserción C3 (cómo C enchufa, desde config C4)

    model:
      detector: {name: mask_rcnn}        # swap de detector entero → DETECTORS
      backbone: {name: convnextv2_tiny, pretrained: true}   # → BACKBONES
      neck:        {name: ...}            # opcional → NECKS  (ema, dsconv)
      rpn_head / box_head / box_predictor / mask_head / mask_predictor:
                   {name: ...}            # opcional → HEADS  (p.ej. pointrend = mask_head)

- **backbone**: siempre vía ``BACKBONES`` (ya lo trae M).
- **neck**: si se nombra, se envuelve el backbone (``_BackboneWithNeck``). El
  módulo de C recibe el dict de features y devuelve un dict de features,
  preservando/declarando ``out_channels``.
- **head**: cualquier componente nombrado se construye desde ``HEADS`` y se pasa
  al constructor de ``MaskRCNN`` (a3_pointrend reemplaza ``mask_head``).
- **detector entero** (a4_cascade): C registra su clase en ``DETECTORS`` y
  ``build_detector`` la despacha por ``model.detector.name``.
- **loss** (a1_focal): la clasificación se computa dentro de ``RoIHeads``; el
  hook limpio es un ``box_predictor`` registrado que lleve el loss, o un detector
  propio en ``DETECTORS``. M deja el pase-through; el módulo lo pone C.

Los componentes C3 son autodescriptivos por config: leen sus dims (p.ej.
``in_channels = out_channels del FPN``) del YAML, no de M.
"""

from __future__ import annotations

from collections import OrderedDict

from torch import Tensor, nn
from torchvision.models.detection import MaskRCNN

from cardd.utils.config import get
from cardd.utils.registry import BACKBONES, DETECTORS, HEADS, NECKS

# Importar el módulo del backbone fuerza su registro en BACKBONES: los decoradores
# @register solo corren al importarse el módulo. Sin esto, BACKBONES.build("convnextv2_tiny")
# lanza KeyError. C debe hacer lo mismo con sus módulos (neck/head/loss/detector).
from cardd.models.backbones import convnextv2_fpn  # noqa: F401

# Componentes opcionales que MaskRCNN acepta como kwargs y que C puede sustituir.
# nombre_en_config -> kwarg del constructor de MaskRCNN
_INJECTABLE_HEADS = {
    "rpn_anchor_generator": "rpn_anchor_generator",
    "rpn_head": "rpn_head",
    "box_head": "box_head",
    "box_predictor": "box_predictor",
    "mask_head": "mask_head",
    "mask_predictor": "mask_predictor",
}


class _BackboneWithNeck(nn.Module):
    """Envuelve un backbone con un neck C3 (NECKS): features -> features.

    Contrato del neck: ``forward(OrderedDict[str,Tensor]) -> OrderedDict[str,Tensor]``
    y atributo ``out_channels``. Si el neck no lo declara, se asume que conserva
    el ancho del backbone.
    """

    def __init__(self, backbone: nn.Module, neck: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.out_channels = getattr(neck, "out_channels", backbone.out_channels)

    def forward(self, x: Tensor) -> "OrderedDict[str, Tensor]":
        return self.neck(self.backbone(x))


@DETECTORS.register("mask_rcnn")
class CarddMaskRCNN(MaskRCNN):
    """Mask R-CNN del proyecto: backbone C3 + transform desde ``data.*`` + hooks C3.

    Se construye con la config completa::

        model = CarddMaskRCNN(cfg)              # o DETECTORS.build({"name":"mask_rcnn"}, cfg=cfg)
    """

    def __init__(self, cfg, **overrides) -> None:
        # --- backbone (C3) ---
        backbone = BACKBONES.build(get(cfg, "model.backbone"))

        # --- neck opcional (C3): ema / dsconv ---
        neck_cfg = get(cfg, "model.neck")
        if neck_cfg:
            neck = NECKS.build(neck_cfg, in_channels=backbone.out_channels)
            backbone = _BackboneWithNeck(backbone, neck)

        # --- nº de clases: data.num_classes (sin fondo) + 1 ---
        num_classes = int(get(cfg, "data.num_classes")) + 1

        # --- transform: normaliza acá (D entrega [0,1]); todo desde data.* ---
        kwargs = dict(
            num_classes=num_classes,
            min_size=int(get(cfg, "data.image_min_size")),
            max_size=int(get(cfg, "data.image_max_size")),
            image_mean=list(get(cfg, "data.image_mean")),
            image_std=list(get(cfg, "data.image_std")),
        )

        # --- heads opcionales (C3): pointrend reemplaza mask_head, etc. ---
        for cfg_key, mrcnn_kwarg in _INJECTABLE_HEADS.items():
            comp_cfg = get(cfg, f"model.{cfg_key}")
            if comp_cfg:
                kwargs[mrcnn_kwarg] = HEADS.build(comp_cfg)

        # overrides en runtime ganan sobre lo derivado de config (test/inferencia).
        kwargs.update(overrides)

        super().__init__(backbone, **kwargs)


def build_detector(cfg) -> nn.Module:
    """Punto de entrada único: despacha por ``model.detector.name`` vía DETECTORS.

    Usa ``DETECTORS.get(name) -> clase`` y la instancia con la config completa.
    No usamos ``DETECTORS.build(spec, cfg=cfg)`` porque el primer parámetro de
    ``Registry.build`` ya se llama ``cfg`` y colisionaría con el kwarg. Un swap
    de detector (a4_cascade) es solo cambiar el nombre; C registra su clase en
    ``DETECTORS`` con la misma firma ``__init__(self, cfg, **overrides)``.
    """
    det_name = get(cfg, "model.detector.name", "mask_rcnn")
    det_cls = DETECTORS.get(det_name)
    return det_cls(cfg)


if __name__ == "__main__":
    # Hito de Fase 3 (paso 9): "forward sobre un batch sin error".
    # Sin dataset ni descarga de pesos: cfg como dict plano, pretrained=False.
    import torch

    cfg = {
        "data": {
            "num_classes": 6,
            "image_min_size": 640,
            "image_max_size": 1024,
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
        },
        "model": {
            "detector": {"name": "mask_rcnn"},
            "backbone": {"name": "convnextv2_tiny", "pretrained": False},
        },
    }

    model = build_detector(cfg)

    # un "batch" C1: lista de imágenes en [0,1], sin apilar (collate de D)
    images = [torch.rand(3, 480, 640), torch.rand(3, 512, 512)]

    # entrenamiento: model(images, targets) -> dict de losses
    targets = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 200.0, 200.0]]),
            "labels": torch.tensor([1]),
            "masks": torch.zeros(1, 480, 640, dtype=torch.uint8),
        },
        {
            "boxes": torch.tensor([[20.0, 20.0, 300.0, 300.0]]),
            "labels": torch.tensor([3]),
            "masks": torch.zeros(1, 512, 512, dtype=torch.uint8),
        },
    ]
    model.train()
    losses = model(images, targets)
    print("train -> losses:", {k: round(float(v), 4) for k, v in losses.items()})

    # inferencia: model(images) -> C2
    model.eval()
    with torch.no_grad():
        preds = model(images)
    p0 = preds[0]
    print("eval  -> C2 keys:", sorted(p0.keys()))
    print("        boxes:", tuple(p0["boxes"].shape), "| masks:", tuple(p0["masks"].shape))
    if p0["labels"].numel():
        print("        labels range:", int(p0["labels"].min()), "..", int(p0["labels"].max()))