"""ConvNeXt-V2 + FPN backbone para Mask R-CNN.

dueño: M · fase: 3 · contrato: feat→FPN

Produce el backbone de pirámide de features que consume el ``MaskRCNN`` de
torchvision. Los tensores de entrada llegan ya reescalados y **normalizados por
el ``GeneralizedRCNNTransform`` del detector** (mean/std leídos de
``data.image_mean/std``); este módulo solo convierte imágenes en un dict
multi-escala de features. La normalización NO ocurre acá ni en D: D entrega
``[0,1]`` sin normalizar (C1) y M la aplica en el transform del detector.

Registrado en el registry ``BACKBONES`` (C3) para que el detector lo construya
por nombre desde config (C4)::

    backbone = BACKBONES.build(
        {"name": "convnextv2_tiny", "pretrained": True},
        in_channels=3,
    )

El módulo expone ``.out_channels`` (int, ancho del FPN) y su ``forward``
devuelve un ``OrderedDict`` de features con claves ``"0".."3"`` + ``"pool"`` —
exactamente los 5 niveles que espera el RPN por defecto de torchvision, misma
convención que ``resnet_fpn_backbone``.
"""

from __future__ import annotations

from collections import OrderedDict

import timm
import torch
from torch import Tensor, nn
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from cardd.utils.registry import BACKBONES

# variante → nombre de modelo timm. Extender acá para Base/Large (fase 6).
_TIMM_NAMES = {
    "tiny":  "convnextv2_tiny.fcmae_ft_in22k_in1k_384",
    "base":  "convnextv2_base.fcmae_ft_in22k_in1k_384",
    "large": "convnextv2_large",
    # sub-tiny:
    "nano":  "convnextv2_nano.fcmae_ft_in22k_in1k_384",   # 384 disponible
    "pico":  "convnextv2_pico.fcmae_ft_in1k",             # solo 224
    "femto": "convnextv2_femto.fcmae_ft_in1k",            # solo 224
    "atto":  "convnextv2_atto.fcmae_ft_in1k",             # solo 224
}


class ConvNeXtV2FPN(nn.Module):
    """Cuerpo ConvNeXt-V2 (timm) + cuello FPN (torchvision).

    Args:
        variant: ``{"tiny","base","large"}`` (default ``"tiny"``; 8 GB → tiny).
        pretrained: carga pesos ImageNet de timm. Para pesos NO-ImageNet, pasar
            ``checkpoint_path`` y mantener ``data.image_mean/std`` alineados al
            preentrenamiento de *esos* pesos (no mezclar).
        in_channels: canales de entrada (3 = RGB).
        out_channels: ancho del FPN, compartido por todos los niveles.
        out_indices: qué stages del backbone alimentan el FPN (4 por defecto).
        drop_path_rate: stochastic depth (regularización ConvNeXt); 0.0 = off.
        checkpoint_path: pesos locales opcionales (tiene prioridad sobre ``pretrained``).
        freeze_backbone: si True, congela todos los parámetros del cuerpo.
    """

    def __init__(
        self,
        variant: str = "tiny",
        pretrained: bool = True,
        in_channels: int = 3,
        out_channels: int = 256,
        out_indices: tuple[int, ...] = (0, 1, 2, 3),
        drop_path_rate: float = 0.0,
        checkpoint_path: str | None = None,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        if variant not in _TIMM_NAMES:
            raise ValueError(
                f"variant {variant!r} no soportado; usa uno de {sorted(_TIMM_NAMES)}"
            )

        # features_only → el forward del cuerpo devuelve una lista de feature maps,
        # uno por stage en out_indices. in_chans deja entrar imágenes no-RGB si hiciera falta.
        # drop_path_rate = stochastic depth (regularización de ConvNeXt); solo actúa en
        # train, identidad en eval. 0.0 = sin regularizar (default, no cambia nada).
        self.body = timm.create_model(
            _TIMM_NAMES[variant],
            pretrained=pretrained and checkpoint_path is None,
            checkpoint_path=checkpoint_path or "",
            features_only=True,
            out_indices=out_indices,
            in_chans=in_channels,
            drop_path_rate=drop_path_rate,
        )

        # No hardcodeamos los channels: los leemos del propio modelo, así Tiny/Base/Large
        # funcionan sin tocar nada (tiny → [96,192,384,768]).
        in_channels_list = self.body.feature_info.channels()
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),  # nivel "pool" (P6) → 5 mapas para el RPN
        )

        # torchvision lee esto para dimensionar RPN y heads.
        self.out_channels = out_channels

        if freeze_backbone:
            for p in self.body.parameters():
                p.requires_grad_(False)

    def forward(self, x: Tensor) -> "OrderedDict[str, Tensor]":
        feats = self.body(x)  # list[Tensor], uno por out_index
        od: "OrderedDict[str, Tensor]" = OrderedDict(
            (str(i), f) for i, f in enumerate(feats)
        )
        return self.fpn(od)  # claves "0".."N-1" + "pool"


@BACKBONES.register("convnextv2_tiny")
class ConvNeXtV2Tiny(ConvNeXtV2FPN):
    """ConvNeXt-V2 Tiny + FPN — backbone por defecto bajo el presupuesto de 8 GB."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("variant", "tiny")
        super().__init__(**kwargs)


@BACKBONES.register("convnextv2_base")
class ConvNeXtV2Base(ConvNeXtV2FPN):
    """ConvNeXt-V2 Base + FPN — fase 6, solo si la VRAM lo permite."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("variant", "base")
        super().__init__(**kwargs)


if __name__ == "__main__":
    # Hito de Fase 3 (paso 8): "forward sin error", sin dataset ni descarga de pesos.
    bb = ConvNeXtV2Tiny(pretrained=False)
    x = torch.randn(2, 3, 640, 640)
    out = bb(x)
    print("out_channels:", bb.out_channels)
    for k, v in out.items():
        print(f"  {k}: {tuple(v.shape)}")
