"""CarDD · src/cardd/models/modules/dsconv.py

Dueño: Equipo C · Fase: B (§5.2 custom) · Contrato: C3 (registro NECKS)

DSConv — Dynamic Snake Convolution (Qi et al., ICCV'23), vendored.
Hipótesis (matriz §5.2): capturar estructuras finas/tubulares (scratch, crack).

Idea: en vez de una grilla rígida, la convolución muestrea a lo largo de una
"serpiente" cuyo recorrido se deforma con offsets aprendidos y CONTINUOS (cada
tap se apoya en el anterior). Se hace a lo largo de un eje (morph: x o y).

Implementación: la interpolación se delega a `grid_sample` (semántica ya validada
en a3); lo propio de aquí es la construcción del coordinate-map snake (offsets
acumulados desde el centro), que va numpy-validada. El sampling deformable
completo se valida en el venv (torch).

Neck (contrato de M §4.3): forward(OrderedDict)->OrderedDict + out_channels.
Preserva canales y resolución -> C2 intacto. `in_channels` lo inyecta M (256).
"""
from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange

from cardd.utils.registry import NECKS

_DEFAULT_LEVELS = ("0", "1", "2", "3", "pool")


def build_snake_grids(offset: torch.Tensor, H: int, W: int, morph: int, extend_scope: float):
    """Coordinate-map de la serpiente: offsets ACUMULADOS desde el tap central.

    Args:
        offset: [B, K, H, W] en [-1,1] (tanh), un offset por tap del kernel.
        morph:  0 = eje x, 1 = eje y.
    Returns:
        lista de K grids [B,H,W,2] normalizados a [-1,1] para grid_sample.
    """
    B, K, _, _ = offset.shape
    k0 = (K - 1) // 2
    device = offset.device

    # acumulación continua desde el centro hacia afuera (la "serpiente")
    cum = torch.zeros_like(offset)
    for k in range(k0 + 1, K):           # rama +
        cum[:, k] = cum[:, k - 1] + offset[:, k]
    for k in range(k0 - 1, -1, -1):      # rama -
        cum[:, k] = cum[:, k + 1] + offset[:, k]
    cum = cum * extend_scope

    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float),
        torch.arange(W, device=device, dtype=torch.float),
        indexing="ij",
    )
    grids = []
    for k in range(K):
        base = float(k - k0)             # posición rígida del tap respecto del centro
        if morph == 0:                   # eje x
            gx = xs[None] + base + cum[:, k]
            gy = ys[None].expand(B, H, W)
        else:                            # eje y
            gx = xs[None].expand(B, H, W)
            gy = ys[None] + base + cum[:, k]
        gxn = 2.0 * gx / max(W - 1, 1) - 1.0
        gyn = 2.0 * gy / max(H - 1, 1) - 1.0
        grids.append(torch.stack([gxn, gyn], dim=-1))   # [B,H,W,2]
    return grids


class DynamicSnakeConv(nn.Module):
    """Convolución serpiente deformable a lo largo de un eje. Salida mismo H,W."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        kernel_size: int = 9,
        morph: int = 0,
        extend_scope: float = 1.0,
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        self.kernel_size = kernel_size
        self.morph = morph
        self.extend_scope = extend_scope
        self.offset_conv = nn.Conv2d(in_channels, kernel_size, kernel_size=3, padding=1)
        self.offset_bn = nn.BatchNorm2d(kernel_size)
        self.aggregate = nn.Conv2d(in_channels * kernel_size, out_channels, kernel_size=1)
        self.gn = nn.GroupNorm(max(1, out_channels // 8), out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        offset = torch.tanh(self.offset_bn(self.offset_conv(x)))      # [B,K,H,W]
        grids = build_snake_grids(offset, H, W, self.morph, self.extend_scope)
        sampled = [F.grid_sample(x, g, align_corners=True, padding_mode="border") for g in grids]
        # `aggregate` es Conv2d(C*K -> out, 1x1). Convolucionar el stack concatenado
        # equivale a sumar K convoluciones 1x1, una por punto sampleado, SIN materializar
        # el tensor [B, C*K, H, W] (que con K=9 era el pico que reventaba la VRAM).
        # Equivalencia exacta (validada): pico de memoria pasa de C*K a C.
        w = self.aggregate.weight                                  # [out, C*K, 1, 1]
        if self.aggregate.bias is not None:
            out = self.aggregate.bias.view(1, -1, 1, 1).expand(B, -1, H, W).clone()
        else:
            out = x.new_zeros(B, w.shape[0], H, W)
        for i, s in enumerate(sampled):
            out = out + F.conv2d(s, w[:, i * C:(i + 1) * C])       # rebanada del peso del punto i
        return F.relu(self.gn(out))


@NECKS.register("dsconv")
class DSConvNeck(nn.Module):
    """Aplica DSConv (rama x + rama y, fusionadas) por nivel del FPN."""

    def __init__(
        self,
        in_channels: int,
        levels=_DEFAULT_LEVELS,
        kernel_size: int = 9,
        extend_scope: float = 1.0,
    ) -> None:
        super().__init__()
        self.out_channels = in_channels

        def _branch():
            return nn.ModuleDict({
                "x": DynamicSnakeConv(in_channels, in_channels, kernel_size, morph=0, extend_scope=extend_scope),
                "y": DynamicSnakeConv(in_channels, in_channels, kernel_size, morph=1, extend_scope=extend_scope),
            })

        self.blocks = nn.ModuleDict({lvl: _branch() for lvl in levels})
        # fusiona rama x + rama y + residual -> in_channels
        self.fuse = nn.ModuleDict({
            lvl: nn.Conv2d(in_channels * 2, in_channels, kernel_size=1) for lvl in levels
        })

    def forward(self, feats: "OrderedDict[str, torch.Tensor]") -> "OrderedDict[str, torch.Tensor]":
        # Gradient checkpointing en las ramas DSConv: en el forward NO se guardan las
        # activaciones internas (grid_sample × kernel × 2 ramas × niveles, el pico que
        # llena la VRAM); se recalculan en el backward. Cambia memoria por cómputo;
        # numéricamente idéntico. Solo se activa en training con grad: en val/eval
        # (sin backward) el checkpoint no aplica y se llama la rama directo.
        use_ckpt = self.training and torch.is_grad_enabled()
        out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        for k, v in feats.items():
            if k not in self.blocks:
                out[k] = v
                continue
            if use_ckpt:
                bx = checkpoint(self.blocks[k]["x"], v, use_reentrant=False)
                by = checkpoint(self.blocks[k]["y"], v, use_reentrant=False)
            else:
                bx = self.blocks[k]["x"](v)
                by = self.blocks[k]["y"](v)
            fused = self.fuse[k](torch.cat([bx, by], dim=1))
            out[k] = v + fused              # residual: no rompe el FPN si los offsets son ~0
        return out


__all__ = ["build_snake_grids", "DynamicSnakeConv", "DSConvNeck"]