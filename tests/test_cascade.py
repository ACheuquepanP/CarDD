"""CarDD · tests/test_cascade.py

Dueño: Equipo C · Fase: 5 (fila a4_cascade)

Niveles:
  1. Lógica aislable (solo torch): promediado de scores entre stages, plegado
     ponderado de losses, y el decode de refinamiento (vía BoxCoder de torchvision).
  2. Construcción + round-trip de integración sobre un MaskRCNN stock
     (importorskip torchvision): el detector corre en train (devuelve las claves
     de loss estándar, SIN claves nuevas) y en eval (devuelve formato C2).

    PYTHONPATH=src pytest tests/test_cascade.py -q
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402


# --------------------------------------------------------------------------- #
# 1 · Lógica aislable                                                          #
# --------------------------------------------------------------------------- #
def test_promediado_de_scores_entre_stages():
    s0 = torch.tensor([[0.2, 0.8]])
    s1 = torch.tensor([[0.4, 0.6]])
    s2 = torch.tensor([[0.6, 0.4]])
    avg = torch.stack([s0, s1, s2], dim=0).mean(dim=0)
    assert torch.allclose(avg, torch.tensor([[0.4, 0.6]]), atol=1e-6)
    assert torch.allclose(avg.sum(dim=1), torch.tensor([1.0]), atol=1e-6)  # sigue siendo distribución


def test_plegado_ponderado_de_losses():
    w = [1.0, 0.5, 0.25]
    lc = [torch.tensor(2.0), torch.tensor(4.0), torch.tensor(8.0)]
    folded = sum(w[i] * lc[i] for i in range(3))
    assert math.isclose(folded.item(), 1.0 * 2 + 0.5 * 4 + 0.25 * 8)       # 6.0


def test_decode_refinamiento_identidad():
    """Con delta=0, el BoxCoder devuelve la caja original (decode es inverso de encode)."""
    from torchvision.models.detection._utils import BoxCoder
    coder = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))
    proposals = [torch.tensor([[10.0, 20.0, 50.0, 80.0]])]
    deltas = torch.zeros(1, 4)
    decoded = coder.decode(deltas, proposals).reshape(-1, 4)
    assert torch.allclose(decoded, proposals[0], atol=1e-4)


def test_decode_refinamiento_traslada_con_delta():
    from torchvision.models.detection._utils import BoxCoder
    wx = 10.0
    coder = BoxCoder(weights=(wx, wx, 5.0, 5.0))
    proposals = [torch.tensor([[0.0, 0.0, 10.0, 10.0]])]   # centro (5,5), w=h=10
    # dx normalizado por weight: un dx real de +0.1 -> delta = 0.1*wx = 1.0 mueve cx en 0.1*w=1
    deltas = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    decoded = coder.decode(deltas, proposals).reshape(-1, 4)
    cx = (decoded[0, 0] + decoded[0, 2]) / 2
    assert torch.allclose(cx, torch.tensor(6.0), atol=1e-3)               # 5 + 0.1*10


# --------------------------------------------------------------------------- #
# 2 · Construcción + integración                                              #
# --------------------------------------------------------------------------- #
def _base_maskrcnn():
    pytest.importorskip("torchvision")
    from torchvision.models.detection.mask_rcnn import MaskRCNN
    from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
    from torchvision.models.resnet import resnet18
    backbone = _resnet_fpn_extractor(resnet18(weights=None), trainable_layers=5)
    return MaskRCNN(backbone, num_classes=7)


def _cascade_roi_heads(model):
    from cardd.models.detectors.cascade_mask_rcnn import (
        _build_cascade_roi_heads, _DEFAULT_IOUS, _DEFAULT_LOSS_W, _DEFAULT_REG_W,
    )
    return _build_cascade_roi_heads(model.roi_heads, 3, _DEFAULT_IOUS, _DEFAULT_LOSS_W, _DEFAULT_REG_W)


def test_build_crea_3_stages():
    model = _base_maskrcnn()
    cascade = _cascade_roi_heads(model)
    assert len(cascade.box_heads) == 3 and len(cascade.box_predictors) == 3
    assert len(cascade.stage_box_coders) == 3 and len(cascade.stage_matchers) == 3


def _sample():
    img = torch.rand(3, 128, 128)
    t = {
        "boxes": torch.tensor([[10.0, 10.0, 70.0, 70.0]]),
        "labels": torch.tensor([1], dtype=torch.int64),
        "masks": torch.zeros(1, 128, 128, dtype=torch.uint8),
    }
    t["masks"][0, 10:70, 10:70] = 1
    return [img], [t]


def test_train_devuelve_claves_estandar_sin_nuevas():
    model = _base_maskrcnn()
    model.roi_heads = _cascade_roi_heads(model)
    images, targets = _sample()
    model.train()
    losses = model(images, targets)
    # exactamente las claves de a0 (sin loss_*_stageN): trainer de M intacto
    assert set(losses) == {
        "loss_classifier", "loss_box_reg", "loss_mask",
        "loss_objectness", "loss_rpn_box_reg",
    }
    assert all(math.isfinite(float(v)) for v in losses.values())


def test_eval_devuelve_formato_C2():
    model = _base_maskrcnn()
    model.roi_heads = _cascade_roi_heads(model)
    model.eval()
    with torch.no_grad():
        out = model([torch.rand(3, 128, 128)])
    assert isinstance(out, list)
    for k in ("boxes", "scores", "labels", "masks"):
        assert k in out[0]
    if out[0]["masks"].numel():
        assert out[0]["masks"].dim() == 4 and out[0]["masks"].shape[1] == 1