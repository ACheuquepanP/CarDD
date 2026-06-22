"""CarDD · src/cardd/models/modules/ema_attention.py

Dueño: Equipo C · Fase: B (§5.2 custom) · Contrato: C3 (registro NECKS)

EMA — Efficient Multi-scale Attention (Ouyang et al., ICASSP'23), vendored.
Hipótesis (matriz §5.2): suprimir reflejos metálicos que confunden la máscara.

Se usa como NECK que envuelve el FPN de M (contrato §4.3):
  forward(OrderedDict[str,Tensor]) -> OrderedDict[str,Tensor]  +  out_channels.
Preserva canales y resolución por nivel -> RPN/RoI sin cambios -> C2 intacto.
`in_channels` lo inyecta M (= out_channels del backbone, 256). einops en el porte.
"""
from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
from einops import rearrange

from cardd.utils.registry import NECKS

_DEFAULT_LEVELS = ("0", "1", "2", "3", "pool")   # 5 niveles FPN de M (§5)


class EMA(nn.Module):
    """Bloque EMA sobre un solo mapa [B, C, H, W]. Salida mismo shape."""

    def __init__(self, channels: int, factor: int = 8) -> None:
        super().__init__()
        assert channels % factor == 0 and channels // factor > 0, (
            f"channels={channels} no divisible por factor={factor}"
        )
        self.groups = factor
        gc = channels // self.groups
        self.softmax = nn.Softmax(dim=-1)
        self.agp = nn.AdaptiveAvgPool2d(1)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(gc, gc)
        self.conv1x1 = nn.Conv2d(gc, gc, kernel_size=1)
        self.conv3x3 = nn.Conv2d(gc, gc, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        gx = rearrange(x, "b (g c) h w -> (b g) c h w", g=self.groups)

        x_h = self.pool_h(gx)                          # (bg, c, h, 1)
        x_w = self.pool_w(gx).permute(0, 1, 3, 2)      # (bg, c, w, 1)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(gx * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(gx)

        x11 = self.softmax(rearrange(self.agp(x1), "n c 1 1 -> n 1 c"))   # (bg,1,c)
        x12 = rearrange(x2, "n c h w -> n c (h w)")                       # (bg,c,hw)
        x21 = self.softmax(rearrange(self.agp(x2), "n c 1 1 -> n 1 c"))
        x22 = rearrange(x1, "n c h w -> n c (h w)")
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)

        out = gx * weights.sigmoid()
        return rearrange(out, "(b g) c h w -> b (g c) h w", g=self.groups)


@NECKS.register("ema_attention")
class EMAAttentionNeck(nn.Module):
    """Aplica EMA por nivel del FPN. Contrato de neck de M (§4.3)."""

    def __init__(self, in_channels: int, levels=_DEFAULT_LEVELS, factor: int = 8) -> None:
        super().__init__()
        self.out_channels = in_channels
        self.blocks = nn.ModuleDict({lvl: EMA(in_channels, factor) for lvl in levels})

    def forward(self, feats: "OrderedDict[str, torch.Tensor]") -> "OrderedDict[str, torch.Tensor]":
        out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        for k, v in feats.items():
            out[k] = self.blocks[k](v) if k in self.blocks else v
        return out


__all__ = ["EMA", "EMAAttentionNeck"]