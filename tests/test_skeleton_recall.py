"""CarDD · tests/test_skeleton_recall.py

Dueño: Equipo C · Fase: B (§5.2 custom, Skeleton Recall)

    PYTHONPATH=src pytest tests/test_skeleton_recall.py -q
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from cardd.models.losses.skeleton_recall import skeleton_recall, SkeletonRecallLoss  # noqa: E402


def _cross(n=9):
    y = torch.zeros(1, 1, n, n)
    y[0, 0, n // 2, :] = 1.0
    y[0, 0, :, n // 2] = 1.0
    return y


def test_recall_perfecto_es_cero():
    y = _cross()
    assert skeleton_recall(y, y).item() < 1e-4


def test_recall_penaliza_rotura():
    y = _cross()
    roto = y.clone()
    roto[0, 0, 4, 3:6] = 0.0
    loss = skeleton_recall(roto, y)
    assert loss.item() > 0.0 and math.isfinite(loss.item())


def test_recall_vacio_es_alto():
    y = _cross()
    assert skeleton_recall(torch.zeros_like(y), y).item() > 0.5


def test_gradiente_solo_por_pred():
    """El esqueleto del GT es detached: target no recibe gradiente, pred sí."""
    y = _cross()
    pred = torch.rand(1, 1, 9, 9, requires_grad=True)
    tgt = y.clone().requires_grad_(True)
    loss = skeleton_recall(pred, tgt)
    loss.backward()
    assert pred.grad is not None
    assert tgt.grad is None or float(tgt.grad.abs().sum()) == 0.0


def test_registro_en_LOSSES():
    from cardd.utils.registry import LOSSES
    assert "skeleton_recall" in LOSSES
    assert isinstance(LOSSES.build({"name": "skeleton_recall", "iters": 3}), SkeletonRecallLoss)


# --- companion sobre MaskRCNN stock ---
def _maskrcnn():
    pytest.importorskip("torchvision")
    from torchvision.models.detection.mask_rcnn import MaskRCNN
    from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
    from torchvision.models.resnet import resnet18
    backbone = _resnet_fpn_extractor(resnet18(weights=None), trainable_layers=5)
    return MaskRCNN(backbone, num_classes=7)


def _sample():
    img = torch.rand(3, 128, 128)
    t = {"boxes": torch.tensor([[10.0, 10.0, 70.0, 70.0]]),
         "labels": torch.tensor([1], dtype=torch.int64),
         "masks": torch.zeros(1, 128, 128, dtype=torch.uint8)}
    t["masks"][0, 10:70, 10:70] = 1
    return [img], [t]


def test_companion_agrega_clave_y_restaura_global():
    import torchvision.models.detection.roi_heads as rh
    from cardd.models.detectors.mask_rcnn_skeleton import install_skeleton
    orig = rh.maskrcnn_loss
    model = _maskrcnn()
    install_skeleton(model.roi_heads, weight=1.0, iters=3)
    images, targets = _sample()
    model.train()
    losses = model(images, targets)
    assert "loss_mask_skeleton" in losses
    assert math.isfinite(float(losses["loss_mask_skeleton"]))
    assert rh.maskrcnn_loss is orig


def test_companion_idempotente():
    from cardd.models.detectors.mask_rcnn_skeleton import install_skeleton
    model = _maskrcnn()
    install_skeleton(model.roi_heads)
    f = model.roi_heads.forward
    install_skeleton(model.roi_heads)
    assert model.roi_heads.forward is f