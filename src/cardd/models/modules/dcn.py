"""CarDD · src/cardd/models/modules/dcn.py

Dueño: Equipo C · Fase: B (§5.2 custom, reevaluación) · Contrato: C3 (NECKS)

Neck de Convolución Deformable (DCN), como contraparte de DSConv para la ablación.
Mismo patrón que DSConvNeck: se aplica por nivel del FPN, con conexión residual
(si los offsets/pesos arrancan ~0, no degrada el FPN). Comparable 1:1 contra dsconv.

Soporta las dos variantes con un flag `modulated`:
  - modulated=False -> DCNv1 (offsets aprendidos, muestreo libre).
  - modulated=True  -> DCNv2 / "DCN+" (DCNv1 + máscara de modulación sigmoide por
    punto, que pondera cada muestra; `deform_conv2d` la acepta vía `mask`).

A diferencia de DSConv (offsets restringidos a una serpiente continua, sesgo
tubular), DCN deja los offsets libres -> mejor en formas no-tubulares (dent, área),
sin el prior lineal. La ablación mide cuál conviene en la mezcla de formas de CarDD.
"""
from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d

from cardd.utils.registry import NECKS

_DEFAULT_LEVELS = ("0", "1", "2", "3", "pool")


class DeformConvBlock(nn.Module):
    """Conv deformable 3x3 (v1 o v2) con offsets (y máscara si modulated) aprendidos."""

    def __init__(self, in_channels: int, out_channels: int | None = None,
                 kernel_size: int = 3, modulated: bool = True) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        self.kernel_size = kernel_size
        self.modulated = modulated
        self.padding = kernel_size // 2
        n = kernel_size * kernel_size

        # offsets: 2 por punto del kernel (dx, dy)
        self.offset_conv = nn.Conv2d(in_channels, 2 * n, kernel_size=kernel_size,
                                     padding=self.padding)
        # máscara de modulación (solo v2): 1 por punto, en [0,1] vía sigmoid
        self.mask_conv = nn.Conv2d(in_channels, n, kernel_size=kernel_size,
                                   padding=self.padding) if modulated else None
        # pesos de la conv deformable
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

        # offsets/máscara arrancan en 0 -> al inicio = conv normal centrada (residual seguro)
        nn.init.zeros_(self.offset_conv.weight); nn.init.zeros_(self.offset_conv.bias)
        if self.mask_conv is not None:
            nn.init.zeros_(self.mask_conv.weight); nn.init.zeros_(self.mask_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x)) if self.mask_conv is not None else None
        return deform_conv2d(x, offset, self.weight, self.bias,
                             padding=(self.padding, self.padding), mask=mask)


@NECKS.register("dcn")
class DCNNeck(nn.Module):
    """Aplica DCN por nivel del FPN, residual. Registrable: NECKS.build({'name':'dcn',...})."""

    def __init__(self, in_channels: int, levels=_DEFAULT_LEVELS,
                 kernel_size: int = 3, modulated: bool = True) -> None:
        super().__init__()
        self.out_channels = in_channels
        self.blocks = nn.ModuleDict({
            lvl: DeformConvBlock(in_channels, in_channels, kernel_size, modulated)
            for lvl in levels
        })
        self.gn = nn.ModuleDict({
            lvl: nn.GroupNorm(max(1, in_channels // 8), in_channels) for lvl in levels
        })

    def forward(self, feats: "OrderedDict[str, torch.Tensor]") -> "OrderedDict[str, torch.Tensor]":
        out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        for k, v in feats.items():
            if k not in self.blocks:
                out[k] = v
                continue
            out[k] = v + torch.relu(self.gn[k](self.blocks[k](v)))   # residual
        return out


__all__ = ["DeformConvBlock", "DCNNeck"]