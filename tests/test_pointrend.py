"""CarDD · tests/test_pointrend_companion.py

Dueño: Equipo C · Fase: 5 (fila a3_pointrend, sub-paso 2)

Dos niveles:
  1. Algoritmos core standalone (solo torch): point_rendering_loss + subdivision.
  2. Round-trip del instalador sobre un MaskRCNN stock con el predictor PointRend
     (necesita torchvision): confirma que loss/inference quedan parcheados y que
     los globals de torchvision se restauran tras el forward (sin fuga), y que el
     formato de salida en eval es el C2 esperado.

    PYTHONPATH=src pytest tests/test_pointrend_companion.py -q
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from cardd.models.heads.pointrend import StandardPointHead  # noqa: E402
from cardd.models.detectors.mask_rcnn_pointrend import (  # noqa: E402
    point_rendering_loss,
    subdivision_inference,
)


# --------------------------------------------------------------------------- #
# 1 · Algoritmos core                                                          #
# --------------------------------------------------------------------------- #
def _point_head(C=7):
    torch.manual_seed(0)
    return StandardPointHead(in_channels=16, num_classes=C)


def test_point_loss_es_escalar_finito_y_no_negativo():
    ph = _point_head()
    coarse = torch.randn(4, 7, 14, 14)
    fine = torch.randn(4, 16, 14, 14)
    targets = (torch.rand(4, 112, 112) > 0.5).float()
    labels = torch.tensor([1, 2, 3, 4])
    loss = point_rendering_loss(coarse, fine, targets, labels, ph,
                                num_points=196, oversample_ratio=3.0, importance_sample_ratio=0.75)
    assert loss.dim() == 0 and math.isfinite(loss.item()) and loss.item() >= 0.0


def test_point_loss_baja_si_la_prediccion_es_perfecta():
    """Con un point-head que predice +inf donde el GT es 1, el BCE -> ~0."""
    C = 1
    ph = StandardPointHead(in_channels=4, num_classes=C, cls_agnostic_mask=True)
    # forzamos que prediga muy positivo siempre
    with torch.no_grad():
        for p in ph.parameters():
            p.zero_()
        ph.predictor.bias.fill_(20.0)
    coarse = torch.zeros(2, 1, 14, 14)
    fine = torch.randn(2, 4, 14, 14)
    targets = torch.ones(2, 112, 112)        # GT todo 1 -> predecir +20 acierta
    labels = torch.tensor([0, 0])
    loss = point_rendering_loss(coarse, fine, targets, labels, ph,
                                num_points=100, oversample_ratio=3.0, importance_sample_ratio=0.75)
    assert loss.item() < 1e-3


def test_point_loss_entrada_vacia_no_es_nan():
    ph = _point_head()
    loss = point_rendering_loss(torch.empty(0, 7, 14, 14), torch.empty(0, 16, 14, 14),
                                torch.empty(0, 112, 112), torch.empty(0, dtype=torch.long), ph,
                                196, 3.0, 0.75)
    assert loss.item() == 0.0 and math.isfinite(loss.item())


def test_subdivision_sube_resolucion_y_conserva_canales():
    ph = _point_head(C=7)
    coarse = torch.randn(3, 7, 28, 28)
    fine = torch.randn(3, 16, 14, 14)
    classes = torch.tensor([1, 4, 6])
    out = subdivision_inference(coarse, fine, ph, classes,
                                subdivision_steps=2, subdivision_num_points=196)
    # 2 pasos x2 -> 28 -> 56 -> 112
    assert out.shape == (3, 7, 112, 112)


def test_subdivision_solo_modifica_puntos_seleccionados():
    """Tras un paso, el mapa upsampleado solo cambia en las celdas re-predichas."""
    ph = _point_head(C=1)
    with torch.no_grad():
        for p in ph.parameters():
            p.zero_()
        ph.predictor.bias.fill_(999.0)        # marca reconocible
    coarse = torch.zeros(1, 1, 4, 4)
    fine = torch.randn(1, 16, 4, 4)
    out = subdivision_inference(coarse, fine, ph, torch.tensor([0]),
                                subdivision_steps=1, subdivision_num_points=3)
    # exactamente 3 celdas marcadas con 999 (los puntos re-predichos)
    assert int((out == 999.0).sum()) == 3


# --------------------------------------------------------------------------- #
# 2 · Round-trip del instalador sobre MaskRCNN stock                           #
# --------------------------------------------------------------------------- #
def _maskrcnn_pointrend():
    pytest.importorskip("torchvision")
    from torchvision.models.detection.mask_rcnn import MaskRCNN
    from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
    from torchvision.models.resnet import resnet18
    from cardd.models.heads.pointrend import PointRendMaskHead, PointRendMaskPredictor
    backbone = _resnet_fpn_extractor(resnet18(weights=None), trainable_layers=5)
    model = MaskRCNN(backbone, num_classes=7)
    model.roi_heads.mask_head = PointRendMaskHead(in_channels=256)
    model.roi_heads.mask_predictor = PointRendMaskPredictor(in_channels=256, num_classes=7)
    return model


def _sample():
    img = torch.rand(3, 128, 128)
    t = {
        "boxes": torch.tensor([[10.0, 10.0, 70.0, 70.0]]),
        "labels": torch.tensor([1], dtype=torch.int64),
        "masks": torch.zeros(1, 128, 128, dtype=torch.uint8),
    }
    t["masks"][0, 10:70, 10:70] = 1
    return [img], [t]


def test_instalador_parchea_y_restaura_globals():
    import torchvision.models.detection.roi_heads as rh
    from cardd.models.detectors.mask_rcnn_pointrend import install_pointrend

    orig_loss, orig_infer = rh.maskrcnn_loss, rh.maskrcnn_inference
    model = _maskrcnn_pointrend()
    install_pointrend(model.roi_heads, point_loss_weight=1.0, target_resolution=112)

    images, targets = _sample()
    model.train()
    losses = model(images, targets)
    assert "loss_mask" in losses and math.isfinite(float(losses["loss_mask"]))
    # globals restaurados tras el forward (sin fuga a otras filas)
    assert rh.maskrcnn_loss is orig_loss and rh.maskrcnn_inference is orig_infer


def test_inference_devuelve_formato_C2():
    from cardd.models.detectors.mask_rcnn_pointrend import install_pointrend
    model = _maskrcnn_pointrend()
    install_pointrend(model.roi_heads)
    model.eval()
    with torch.no_grad():
        out = model([torch.rand(3, 128, 128)])
    assert isinstance(out, list) and "masks" in out[0]
    if out[0]["masks"].numel():
        m = out[0]["masks"]
        assert m.dim() == 4 and m.shape[1] == 1          # [K,1,H,W]
        assert float(m.min()) >= 0.0 and float(m.max()) <= 1.0


def test_instalador_idempotente():
    from cardd.models.detectors.mask_rcnn_pointrend import install_pointrend
    model = _maskrcnn_pointrend()
    install_pointrend(model.roi_heads)
    f = model.roi_heads.forward
    install_pointrend(model.roi_heads)
    assert model.roi_heads.forward is f