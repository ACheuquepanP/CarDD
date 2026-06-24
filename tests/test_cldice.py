"""CarDD · tests/test_cldice.py

Dueño: Equipo C · Fase: B (§5.2 custom, clDice)

Niveles:
  1. Matemática del loss (solo torch): identidad -> ~0, rotura -> >0, registro.
  2. Round-trip del companion sobre un MaskRCNN stock (importorskip torchvision):
     en train agrega la clave `loss_mask_cldice` y restaura el global; en eval
     devuelve C2 sin tocar.

    PYTHONPATH=src pytest tests/test_cldice.py -q
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from cardd.models.losses.cldice import soft_skel, soft_cldice, SoftClDiceLoss  # noqa: E402


# --------------------------------------------------------------------------- #
# 1 · Matemática                                                               #
# --------------------------------------------------------------------------- #
def _cross(n=9):
    y = torch.zeros(1, 1, n, n)
    y[0, 0, n // 2, :] = 1.0
    y[0, 0, :, n // 2] = 1.0
    return y


def test_cldice_identica_es_cero():
    y = _cross()
    assert soft_cldice(y, y).item() < 1e-4


def test_cldice_penaliza_rotura():
    y = _cross()
    roto = y.clone()
    roto[0, 0, 4, 3:6] = 0.0           # corta la línea horizontal
    loss = soft_cldice(roto, y)
    assert loss.item() > 0.0 and math.isfinite(loss.item())


def test_cldice_mascara_ajena_es_alta():
    y = _cross()
    ajena = torch.zeros_like(y)
    ajena[0, 0, 0, 0] = 1.0
    assert soft_cldice(ajena, y).item() > 0.5


def test_skel_es_subconjunto_de_la_mascara():
    y = _cross()
    s = soft_skel(y, iters=3)
    # el esqueleto no aparece donde la máscara es 0
    assert float((s * (1 - y)).abs().max()) < 1e-6


def test_registro_en_LOSSES():
    from cardd.utils.registry import LOSSES
    assert "cldice" in LOSSES
    loss = LOSSES.build({"name": "cldice", "iters": 3})
    assert isinstance(loss, SoftClDiceLoss)


# --------------------------------------------------------------------------- #
# 2 · Companion sobre MaskRCNN stock                                           #
# --------------------------------------------------------------------------- #
def _maskrcnn():
    pytest.importorskip("torchvision")
    from torchvision.models.detection.mask_rcnn import MaskRCNN
    from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
    from torchvision.models.resnet import resnet18
    backbone = _resnet_fpn_extractor(resnet18(weights=None), trainable_layers=5)
    return MaskRCNN(backbone, num_classes=7)


def _sample():
    img = torch.rand(3, 128, 128)
    t = {
        "boxes": torch.tensor([[10.0, 10.0, 70.0, 70.0]]),
        "labels": torch.tensor([1], dtype=torch.int64),
        "masks": torch.zeros(1, 128, 128, dtype=torch.uint8),
    }
    t["masks"][0, 10:70, 10:70] = 1
    return [img], [t]


def test_companion_agrega_clave_y_restaura_global():
    import torchvision.models.detection.roi_heads as rh
    from cardd.models.detectors.mask_rcnn_cldice import install_cldice

    orig = rh.maskrcnn_loss
    model = _maskrcnn()
    install_cldice(model.roi_heads, weight=1.0, iters=3)

    images, targets = _sample()
    model.train()
    losses = model(images, targets)
    assert "loss_mask_cldice" in losses                       # clave propia (M la prefiere)
    assert math.isfinite(float(losses["loss_mask_cldice"]))
    assert rh.maskrcnn_loss is orig                           # global restaurado (sin fuga)


def test_companion_eval_no_agrega_clave():
    from cardd.models.detectors.mask_rcnn_cldice import install_cldice
    model = _maskrcnn()
    install_cldice(model.roi_heads)
    model.eval()
    with torch.no_grad():
        out = model([torch.rand(3, 128, 128)])
    assert isinstance(out, list) and "masks" in out[0]        # C2 intacto


def test_companion_idempotente():
    from cardd.models.detectors.mask_rcnn_cldice import install_cldice
    model = _maskrcnn()
    install_cldice(model.roi_heads)
    f = model.roi_heads.forward
    install_cldice(model.roi_heads)
    assert model.roi_heads.forward is f