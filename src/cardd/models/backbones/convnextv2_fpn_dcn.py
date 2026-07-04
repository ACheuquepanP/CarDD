"""ConvNeXt-V2 + FPN con deformable conv (DCNv2) en los stages tardíos.

dueño: Equipo C · fase: B (§5.2 custom) · contrato: C3 (registry BACKBONES)

Subclase de ``ConvNeXtV2FPN`` (de M, fase 3). NO modifica el backbone base:
hereda y, tras construir ``self.body``, reemplaza in-place las convs depthwise
de los stages indicados por una versión deformable v2 (modulada). Es el
componente de arquitectura de DCN+ (paper CarDD), adaptado a ConvNeXt-V2:
allí la 3x3 de ResNet en C3-C5; aquí la depthwise 7x7 de ConvNeXt.

Opción seleccionable desde config como un backbone más::

    backbone:
      name: convnextv2_tiny_dcn
      pretrained: true
      dcn_stages: [1, 2, 3]      # análogo a C3-C5 (deja stage 0 intacto)
      dcn_offset_groups: 1

Ortogonal a focal/cascade (esos viven en roi_heads/cabezas; esto en backbone).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d

from cardd.utils.registry import BACKBONES
from cardd.models.backbones.convnextv2_fpn import ConvNeXtV2FPN  # ajustar ruta real


class DeformDepthwise2d(nn.Module):
    """Reemplazo deformable (DCNv2) de una depthwise existente; reusa sus pesos."""

    def __init__(self, dw_conv: nn.Conv2d, offset_groups: int = 1):
        super().__init__()
        assert dw_conv.groups == dw_conv.in_channels == dw_conv.out_channels, \
            "Se esperaba conv depthwise (groups == in == out)."
        self.in_channels = dw_conv.in_channels
        self.kernel_size = dw_conv.kernel_size
        self.stride = dw_conv.stride
        self.padding = dw_conv.padding
        self.dilation = dw_conv.dilation
        self.groups = dw_conv.groups
        self.offset_groups = offset_groups

        self.weight = dw_conv.weight          # reusa parámetros aprendidos
        self.bias = dw_conv.bias

        kh, kw = self.kernel_size
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels, offset_groups * 3 * kh * kw,
            kernel_size=self.kernel_size, stride=self.stride,
            padding=self.padding, dilation=self.dilation, bias=True,
        )
        nn.init.zeros_(self.conv_offset_mask.weight)  # arranque neutro:
        nn.init.zeros_(self.conv_offset_mask.bias)    # offsets 0, máscara ~1

    def forward(self, x):
        out = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = 2.0 * torch.sigmoid(mask)
        return deform_conv2d(
            x, offset, self.weight, bias=self.bias,
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, mask=mask,
        )


def _iter_depthwise(module: nn.Module):
    for name, child in module.named_modules():
        if isinstance(child, nn.Conv2d) \
                and child.groups == child.in_channels == child.out_channels \
                and child.in_channels > 1:
            yield name, child


def _set_submodule(root: nn.Module, dotted: str, new: nn.Module):
    *parents, last = dotted.split(".")
    obj = root
    for p in parents:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    if last.isdigit():
        obj[int(last)] = new
    else:
        setattr(obj, last, new)


def _resolve_body_stages(body: nn.Module):
    """Devuelve {idx: módulo_stage} para un timm ConvNeXt-V2 con features_only.

    timm cambia la estructura entre versiones. Probamos los layouts conocidos;
    si ninguno encaja, lanzamos con un volcado de top-level para que el equipo
    apunte a mano (handoff §4.2). Ejecuta el __main__ de abajo para inspeccionar.
    """
    # Layout A: body.stages_0, body.stages_1, ... (features_only frecuente)
    seq = [getattr(body, f"stages_{i}", None) for i in range(4)]
    if all(s is not None for s in seq):
        return {i: s for i, s in enumerate(seq)}
    # Layout B: body.stages es Sequential/ModuleList
    if hasattr(body, "stages"):
        st = body.stages
        return {i: st[i] for i in range(len(st))}
    # Layout C: body.model.stages (wrapper interno)
    if hasattr(body, "model") and hasattr(body.model, "stages"):
        st = body.model.stages
        return {i: st[i] for i in range(len(st))}
    raise AttributeError(
        "No reconozco el layout de stages de timm. Top-level modules: "
        + ", ".join(n for n, _ in body.named_children())
        + " — adaptar `_resolve_body_stages`."
    )


def install_dcn_on_body(body: nn.Module, stages=(1, 2, 3), offset_groups: int = 1):
    stage_map = _resolve_body_stages(body)
    originals = {}
    for s in stages:
        if s not in stage_map:
            raise IndexError(f"stage {s} fuera de rango; disponibles: {sorted(stage_map)}")
        stage = stage_map[s]
        for name, dw in list(_iter_depthwise(stage)):
            originals[f"{s}.{name}"] = dw
            _set_submodule(stage, name, DeformDepthwise2d(dw, offset_groups=offset_groups))
    return originals


@BACKBONES.register("convnextv2_tiny_dcn")
class ConvNeXtV2TinyDCN(ConvNeXtV2FPN):
    """ConvNeXt-V2 Tiny + FPN con DCNv2 en stages tardíos. Hereda todo de M."""

    def __init__(self, dcn_stages=(1, 2, 3), dcn_offset_groups: int = 1, **kwargs):
        kwargs.setdefault("variant", "tiny")
        super().__init__(**kwargs)  # construye self.body + self.fpn igual que el base
        if getattr(self, "_dcn_installed", False):
            return
        self._dcn_originals = install_dcn_on_body(
            self.body, stages=tuple(dcn_stages), offset_groups=dcn_offset_groups
        )
        self._dcn_installed = True
        self._dcn_params = {"stages": tuple(dcn_stages), "offset_groups": dcn_offset_groups}


__all__ = ["ConvNeXtV2TinyDCN", "DeformDepthwise2d", "install_dcn_on_body"]


if __name__ == "__main__":
    # Inspector: imprime el layout real de stages y verifica forward + deformación.
    import timm
    body = timm.create_model(
        "convnextv2_tiny", pretrained=False, features_only=True,
        out_indices=(0, 1, 2, 3), in_chans=3,
    )
    print("top-level children:", [n for n, _ in body.named_children()])
    sm = _resolve_body_stages(body)
    print("stages resueltos:", sorted(sm))
    for s in (1, 2, 3):
        dws = list(_iter_depthwise(sm[s]))
        print(f"  stage {s}: {len(dws)} depthwise (ej: {dws[0][0] if dws else '—'})")

    bb = ConvNeXtV2TinyDCN(pretrained=False)
    x = torch.randn(2, 3, 640, 640)
    out = bb(x)
    print("out_channels:", bb.out_channels, "| niveles:", list(out))
    print("DCN instalada en:", len(bb._dcn_originals), "depthwise")
