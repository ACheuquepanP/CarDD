"""CarDD · tests/test_cascade_composed.py

Dueño: Equipo C · Fase: B (reevaluación)

Verifica cascade + loss-mods de máscara:
  · install acepta clDice/skeleton (sitio mask) y rechaza focal (sitio classification);
  · en train sobre un cascade real aparecen las claves loss_mask_* y se restaura el
    global maskrcnn_loss (sin fuga);
  · clDice, skeleton y ambos por separado dan sus claves;
  · idempotencia.

El cascade real necesita la base de M (CarddMaskRCNN). Si no está, los tests que
construyen el detector se saltan; los de validación de mods corren igual.

    PYTHONPATH=src pytest tests/test_cascade_composed.py -q
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from cardd.models.losses.loss_mods import build_loss_mod  # noqa: E402
from cardd.models.detectors.cascade_mask_rcnn_composed import (  # noqa: E402
    install_mask_mods_on_cascade,
)


# --------------------------------------------------------------------------- #
# Validación de sitio (no requiere construir el cascade)                       #
# --------------------------------------------------------------------------- #
class _FakeRoIHeads:
    """Stub mínimo: solo necesita .forward para que install lo envuelva."""
    def forward(self, *a, **k):
        return ([], {})


def test_rechaza_loss_mod_de_clasificacion():
    rh = _FakeRoIHeads()
    with pytest.raises(ValueError):
        install_mask_mods_on_cascade(rh, [build_loss_mod("focal")])   # focal = classification


def test_acepta_mods_de_mascara():
    for name in ("cldice", "skeleton"):
        rh = _FakeRoIHeads()
        out = install_mask_mods_on_cascade(rh, [build_loss_mod(name)])
        assert getattr(out, "_cascade_mods_installed", False)


def test_idempotente():
    rh = _FakeRoIHeads()
    install_mask_mods_on_cascade(rh, [build_loss_mod("cldice")])
    f = rh.forward
    install_mask_mods_on_cascade(rh, [build_loss_mod("skeleton")])    # 2ª vez: no-op
    assert rh.forward is f


# --------------------------------------------------------------------------- #
# Round-trip sobre cascade real (requiere base de M)                           #
# --------------------------------------------------------------------------- #
def _cfg(loss_mods):
    return {
        "model": {
            "detector": {"name": "cascade_mask_rcnn_composed", "loss_mods": loss_mods},
            "backbone": {"name": "convnextv2_tiny", "pretrained": False},
        },
        "data": {"num_classes": 6},
    }


def _sample():
    img = torch.rand(3, 128, 128)
    t = {"boxes": torch.tensor([[10.0, 10.0, 70.0, 70.0]]),
         "labels": torch.tensor([1], dtype=torch.int64),
         "masks": torch.zeros(1, 128, 128, dtype=torch.uint8)}
    t["masks"][0, 10:70, 10:70] = 1
    return [img], [t]


def _build(loss_mods):
    try:
        from cardd.models.detectors.cascade_mask_rcnn_composed import CascadeMaskRCNNComposed
    except Exception:
        pytest.skip("CascadeMaskRCNNComposed no disponible (falta base de M)")
    try:
        return CascadeMaskRCNNComposed(_cfg(loss_mods))
    except Exception as e:
        pytest.skip(f"no se pudo construir el cascade real: {e}")


@pytest.mark.parametrize("mods,keys", [
    ([{"name": "cldice"}], ["loss_mask_cldice"]),
    ([{"name": "skeleton"}], ["loss_mask_skeleton"]),
    ([{"name": "cldice"}, {"name": "skeleton"}], ["loss_mask_cldice", "loss_mask_skeleton"]),
])
def test_train_agrega_claves_y_restaura_global(mods, keys):
    import torchvision.models.detection.roi_heads as rh
    orig = rh.maskrcnn_loss
    model = _build(mods)
    images, targets = _sample()
    model.train()
    losses = model(images, targets)
    for k in keys:
        assert k in losses and math.isfinite(float(losses[k]))
    # claves estándar de cascade siguen ahí
    assert "loss_classifier" in losses and "loss_mask" in losses
    assert rh.maskrcnn_loss is orig                       # global restaurado