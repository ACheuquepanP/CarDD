"""CarDD · tests/test_dcn.py

Dueño: Equipo C · Fase: B (§5.2 custom, DCN — contraparte de DSConv)

    PYTHONPATH=src pytest tests/test_dcn.py -q
"""
from __future__ import annotations

from collections import OrderedDict

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from cardd.models.modules.dcn import DeformConvBlock, DCNNeck  # noqa: E402


def _feats(c=8, levels=("0", "1", "2", "3", "pool")):
    return OrderedDict((lvl, torch.randn(2, c, 16, 16)) for lvl in levels)


def test_bloque_v1_y_v2_shapes():
    x = torch.randn(2, 8, 16, 16)
    for modulated in (False, True):
        blk = DeformConvBlock(8, 8, kernel_size=3, modulated=modulated)
        y = blk(x)
        assert y.shape == x.shape


def test_v2_tiene_mask_conv_y_v1_no():
    assert DeformConvBlock(8, modulated=True).mask_conv is not None     # DCN+
    assert DeformConvBlock(8, modulated=False).mask_conv is None        # DCNv1


def test_neck_residual_arranca_identidad():
    """offsets/máscara init 0 -> al inicio el bloque ~ conv normal; el residual
    domina y la salida no es basura (shapes intactos, finito)."""
    neck = DCNNeck(8, modulated=True)
    feats = _feats()
    out = neck(feats)
    for k, v in feats.items():
        assert out[k].shape == v.shape
        assert torch.isfinite(out[k]).all()


def test_neck_pasa_niveles_no_listados():
    neck = DCNNeck(8, levels=("0", "1"), modulated=True)
    feats = _feats(levels=("0", "1", "2", "3", "pool"))
    out = neck(feats)
    # niveles fuera de la lista pasan sin tocar
    for lvl in ("2", "3", "pool"):
        assert torch.equal(out[lvl], feats[lvl])


def test_registro_en_NECKS():
    from cardd.utils.registry import NECKS
    assert "dcn" in NECKS
    neck = NECKS.build({"name": "dcn", "in_channels": 8, "levels": ["0", "1"], "modulated": True})
    assert isinstance(neck, DCNNeck)


def test_gradiente_fluye():
    neck = DCNNeck(8, levels=("0",), modulated=True)
    feats = _feats(levels=("0",))
    feats["0"].requires_grad_(True)
    neck(feats)["0"].sum().backward()
    # parámetros de la conv deformable reciben gradiente
    assert neck.blocks["0"].weight.grad is not None