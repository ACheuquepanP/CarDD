"""CarDD · tests/test_focal.py

Dueño: Equipo C · Fase: 5 (fila a1_focal)

Dos niveles, ambos standalone (no necesitan el dataset real ni a0):

  1. Invariantes matemáticos del focal (solo torch).
  2. Round-trip del instalador sobre un MaskRCNN de torchvision STOCK: confirma
     que `install_focal_classification_loss` redirige el loss de clasificación
     y que el global de torchvision queda restaurado tras el forward (sin fuga).

Se ejecuta en el venv del proyecto (con torch/torchvision cu130):
    PYTHONPATH=src pytest tests/test_focal.py -q
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from cardd.models.losses.focal import softmax_focal_loss, FocalLoss  # noqa: E402


# --------------------------------------------------------------------------- #
# 1 · Invariantes matemáticos                                                  #
# --------------------------------------------------------------------------- #
def test_gamma0_alpha1_es_cross_entropy():
    """gamma=0, alpha=1  ->  focal == cross_entropy (definición)."""
    torch.manual_seed(0)
    logits = torch.randn(64, 7)
    targets = torch.randint(0, 7, (64,))
    fl = softmax_focal_loss(logits, targets, alpha=1.0, gamma=0.0, reduction="mean")
    ce = F.cross_entropy(logits, targets)
    assert torch.allclose(fl, ce, atol=1e-6)


def test_escalar_alpha_escala_linealmente():
    torch.manual_seed(1)
    logits = torch.randn(32, 7)
    targets = torch.randint(0, 7, (32,))
    base = softmax_focal_loss(logits, targets, alpha=1.0, gamma=2.0)
    half = softmax_focal_loss(logits, targets, alpha=0.5, gamma=2.0)
    assert torch.allclose(half, 0.5 * base, atol=1e-6)


def test_focal_downweightea_ejemplos_faciles():
    """Para un ejemplo bien clasificado (p_t alto), focal << CE; gamma mayor baja más."""
    logits = torch.tensor([[5.0, 0.0, 0.0]])   # muy confiado en clase 0
    targets = torch.tensor([0])
    ce = F.cross_entropy(logits, targets)
    fl2 = softmax_focal_loss(logits, targets, alpha=1.0, gamma=2.0)
    fl5 = softmax_focal_loss(logits, targets, alpha=1.0, gamma=5.0)
    assert fl2 < ce
    assert fl5 < fl2


def test_alpha_por_clase():
    logits = torch.randn(10, 4)
    targets = torch.randint(0, 4, (10,))
    per_class = [0.1, 0.2, 0.3, 0.4]
    out = softmax_focal_loss(logits, targets, alpha=per_class, gamma=2.0, reduction="none")
    ref = torch.stack([
        softmax_focal_loss(logits[i:i+1], targets[i:i+1], alpha=per_class[int(targets[i])],
                           gamma=2.0, reduction="mean")
        for i in range(10)
    ])
    assert torch.allclose(out, ref, atol=1e-6)


def test_entrada_vacia_no_es_nan():
    logits = torch.empty(0, 7)
    targets = torch.empty(0, dtype=torch.long)
    out = softmax_focal_loss(logits, targets, reduction="mean")
    assert out.item() == 0.0 and math.isfinite(out.item())


def test_modulo_registrado_y_construible():
    from cardd.utils.registry import LOSSES
    assert "focal" in LOSSES
    loss = LOSSES.build({"name": "focal", "alpha": 0.5, "gamma": 2.0})
    assert isinstance(loss, FocalLoss)


# --------------------------------------------------------------------------- #
# 2 · Round-trip del instalador sobre un MaskRCNN stock                        #
# --------------------------------------------------------------------------- #
def _tiny_maskrcnn():
    """MaskRCNN mínimo de torchvision (sin pesos, sin backbone pesado) para testear el parche."""
    tv = pytest.importorskip("torchvision")
    from torchvision.models.detection.mask_rcnn import MaskRCNN
    from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
    from torchvision.models.resnet import resnet18
    backbone = _resnet_fpn_extractor(resnet18(weights=None), trainable_layers=5)
    return MaskRCNN(backbone, num_classes=7)


def _one_sample():
    img = torch.rand(3, 128, 128)
    target = {
        "boxes": torch.tensor([[10.0, 10.0, 60.0, 60.0]]),
        "labels": torch.tensor([1], dtype=torch.int64),
        "masks": torch.zeros(1, 128, 128, dtype=torch.uint8),
    }
    target["masks"][0, 10:60, 10:60] = 1
    return [img], [target]


def test_instalador_cambia_loss_y_restaura_global():
    import torchvision.models.detection.roi_heads as rh
    from cardd.models.detectors.mask_rcnn_focal import install_focal_classification_loss

    original_global = rh.fastrcnn_loss

    model = _tiny_maskrcnn()
    images, targets = _one_sample()

    torch.manual_seed(0)
    model.train()
    losses_ce = model(images, targets)
    ce_cls = float(losses_ce["loss_classifier"])

    install_focal_classification_loss(model.roi_heads, alpha=0.5, gamma=2.0)

    torch.manual_seed(0)
    losses_focal = model(images, targets)
    focal_cls = float(losses_focal["loss_classifier"])

    # El loss de clasificación cambió (ya no es CE)...
    assert not math.isclose(focal_cls, ce_cls, rel_tol=1e-4)
    # ...y el global de torchvision quedó restaurado tras el forward (sin fuga).
    assert rh.fastrcnn_loss is original_global


def test_instalador_es_idempotente():
    from cardd.models.detectors.mask_rcnn_focal import install_focal_classification_loss
    model = _tiny_maskrcnn()
    install_focal_classification_loss(model.roi_heads)
    f1 = model.roi_heads.forward
    install_focal_classification_loss(model.roi_heads)   # segunda vez: no-op
    assert model.roi_heads.forward is f1