"""CarDD · tests/test_composed.py

Dueño: Equipo C · Fase: B (combinables)

Verifica el mecanismo componible:
  - los mods registran y se construyen (str o dict);
  - el composer aplica clasificación (focal) + máscara (cldice/skeleton) a la vez;
  - en train aparecen las claves esperadas y se restauran los globals (sin fuga);
  - en eval() C2 no cambia;
  - dos mods de sitio 'classification' -> error claro.

    PYTHONPATH=src pytest tests/test_composed.py -q
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from cardd.models.losses.loss_mods import build_loss_mod, FocalMod, ClDiceMod  # noqa: E402


# --------------------------------------------------------------------------- #
# mods                                                                         #
# --------------------------------------------------------------------------- #
def test_mods_registrados_y_construibles():
    from cardd.utils.registry import Registry  # noqa: F401
    from cardd.models.losses.loss_mods import LOSS_MODS
    assert {"focal", "cldice", "skeleton"} <= set(LOSS_MODS.keys())
    assert isinstance(build_loss_mod("focal"), FocalMod)                       # str
    assert isinstance(build_loss_mod({"name": "cldice", "weight": 0.5}), ClDiceMod)  # dict


def test_sitios_declarados():
    assert build_loss_mod("focal").site == "classification"
    assert build_loss_mod("cldice").site == "mask"
    assert build_loss_mod("skeleton").site == "mask"


# --------------------------------------------------------------------------- #
# composer sobre MaskRCNN stock                                                #
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
    t = {"boxes": torch.tensor([[10.0, 10.0, 70.0, 70.0]]),
         "labels": torch.tensor([1], dtype=torch.int64),
         "masks": torch.zeros(1, 128, 128, dtype=torch.uint8)}
    t["masks"][0, 10:70, 10:70] = 1
    return [img], [t]


def test_combinada_focal_cldice_train():
    import torchvision.models.detection.roi_heads as rh
    from cardd.models.detectors.mask_rcnn_composed import install_composed

    orig_f, orig_m = rh.fastrcnn_loss, rh.maskrcnn_loss
    model = _maskrcnn()
    mods = [build_loss_mod("focal"), build_loss_mod("cldice")]
    install_composed(model.roi_heads, mods)

    images, targets = _sample()
    model.train()
    losses = model(images, targets)
    # clasificación reemplazada (la clave sigue siendo loss_classifier) + clave de máscara sumada
    assert "loss_classifier" in losses
    assert "loss_mask_cldice" in losses
    assert all(math.isfinite(float(v)) for v in losses.values())
    # globals restaurados (sin fuga a otras filas)
    assert rh.fastrcnn_loss is orig_f and rh.maskrcnn_loss is orig_m


def test_combinada_dos_mask_mods_suman_dos_claves():
    from cardd.models.detectors.mask_rcnn_composed import install_composed
    model = _maskrcnn()
    install_composed(model.roi_heads, [build_loss_mod("cldice"), build_loss_mod("skeleton")])
    model.train()
    images, targets = _sample()
    losses = model(images, targets)
    assert "loss_mask_cldice" in losses and "loss_mask_skeleton" in losses


def test_eval_no_cambia_C2():
    from cardd.models.detectors.mask_rcnn_composed import install_composed
    model = _maskrcnn()
    install_composed(model.roi_heads, [build_loss_mod("focal"), build_loss_mod("cldice")])
    model.eval()
    with torch.no_grad():
        out = model([torch.rand(3, 128, 128)])
    assert isinstance(out, list)
    for k in ("boxes", "scores", "labels", "masks"):
        assert k in out[0]


def test_dos_classification_mods_es_error():
    from cardd.models.detectors.mask_rcnn_composed import install_composed
    model = _maskrcnn()
    with pytest.raises(ValueError):
        install_composed(model.roi_heads, [build_loss_mod("focal"), build_loss_mod("focal")])


def test_composer_idempotente():
    from cardd.models.detectors.mask_rcnn_composed import install_composed
    model = _maskrcnn()
    install_composed(model.roi_heads, [build_loss_mod("cldice")])
    f = model.roi_heads.forward
    install_composed(model.roi_heads, [build_loss_mod("skeleton")])  # segunda vez: no-op
    assert model.roi_heads.forward is f