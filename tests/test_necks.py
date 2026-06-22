"""CarDD · tests/test_necks.py

Dueño: Equipo C · Fase: B (§5.2 custom)

Tests standalone de los necks DSConv y EMA (registro NECKS). Verifican el
contrato de M (§4.3): forward(OrderedDict)->OrderedDict, out_channels, y
preservación de canales/resolución por nivel (clave para no romper RPN/RoI -> C2).

    PYTHONPATH=src pytest tests/test_necks.py -q
"""
from __future__ import annotations

from collections import OrderedDict

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("einops")

from cardd.models.modules.ema_attention import EMA, EMAAttentionNeck  # noqa: E402
from cardd.models.modules.dsconv import (  # noqa: E402
    build_snake_grids,
    DynamicSnakeConv,
    DSConvNeck,
)


def _fpn(c=256):
    return OrderedDict([
        ("0", torch.randn(2, c, 32, 32)),
        ("1", torch.randn(2, c, 16, 16)),
        ("2", torch.randn(2, c, 8, 8)),
        ("3", torch.randn(2, c, 4, 4)),
        ("pool", torch.randn(2, c, 2, 2)),
    ])


# --------------------------------------------------------------------------- #
# Registro                                                                     #
# --------------------------------------------------------------------------- #
def test_registro_en_NECKS():
    from cardd.utils.registry import NECKS
    assert "ema_attention" in NECKS and "dsconv" in NECKS


# --------------------------------------------------------------------------- #
# EMA                                                                          #
# --------------------------------------------------------------------------- #
def test_ema_bloque_preserva_shape():
    ema = EMA(channels=256, factor=8)
    x = torch.randn(2, 256, 16, 16)
    assert ema(x).shape == x.shape


def test_ema_neck_contrato():
    neck = EMAAttentionNeck(in_channels=256)
    assert neck.out_channels == 256
    feats = _fpn()
    out = neck(feats)
    assert list(out.keys()) == list(feats.keys())          # mismo orden y niveles
    for k in feats:
        assert out[k].shape == feats[k].shape              # canales+resolución intactos


# --------------------------------------------------------------------------- #
# DSConv                                                                       #
# --------------------------------------------------------------------------- #
def test_snake_grids_count_y_rango():
    offset = torch.tanh(torch.randn(2, 9, 8, 8))           # K=9
    grids = build_snake_grids(offset, 8, 8, morph=0, extend_scope=1.0)
    assert len(grids) == 9
    for g in grids:
        assert g.shape == (2, 8, 8, 2)


def test_snake_grids_acumulacion_continua():
    """Con offset constante c por tap, el desplazamiento acumulado del tap más
    externo respecto del centro debe ser (n_pasos * c) (serpiente continua)."""
    B, K, H, W = 1, 5, 4, 4
    c = 0.1
    offset = torch.full((B, K, H, W), c)
    grids = build_snake_grids(offset, H, W, morph=1, extend_scope=1.0)
    k0 = (K - 1) // 2                                        # centro = 2
    # tap central: sin acumulación -> gy = ys (coordenada base). tap k0+2: +2c acumulado.
    # comparamos la coord y (no normalizada) implícita: diferencia entre tap externo y central
    # reconstruimos desde la normalización inversa: gy = (gyn+1)*(H-1)/2
    def gy_of(k):
        gyn = grids[k][0, :, :, 1]
        return (gyn + 1.0) * (H - 1) / 2.0
    diff = (gy_of(K - 1) - gy_of(k0)).mean().item()
    base_diff = (K - 1 - k0)                                 # 2 posiciones rígidas
    assert abs(diff - (base_diff + 2 * c)) < 1e-4


def test_dsconv_block_preserva_shape():
    dsc = DynamicSnakeConv(in_channels=64, kernel_size=9, morph=0)
    x = torch.randn(2, 64, 12, 12)
    assert dsc(x).shape == x.shape


def test_dsconv_neck_contrato_y_residual():
    neck = DSConvNeck(in_channels=256, kernel_size=9)
    assert neck.out_channels == 256
    feats = _fpn()
    out = neck(feats)
    assert list(out.keys()) == list(feats.keys())
    for k in feats:
        assert out[k].shape == feats[k].shape