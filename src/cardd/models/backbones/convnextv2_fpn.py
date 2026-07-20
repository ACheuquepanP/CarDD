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

Variantes sub-tiny (Equipo C, reevaluación): nano/pico/femto/atto para modelos
más livianos. Solo nano tiene pesos a 384; el resto solo a 224 (fcmae_ft_in1k).
"""

from __future__ import annotations

from collections import OrderedDict

import timm
import torch
from torch import Tensor, nn
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from cardd.utils.registry import BACKBONES

# variante → nombre de modelo timm.
# Resolución de pesos: tiny/base/nano tienen fine-tune a 384; pico/femto/atto
# solo tienen 224 (fcmae_ft_in1k) — verificado con timm.list_models.
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
        variant: clave de ``_TIMM_NAMES`` (default ``"tiny"``; 8 GB → tiny o menor).
        pretrained: carga pesos de timm. Para pesos NO-ImageNet, pasar
            ``checkpoint_path`` y mantener ``data.image_mean/std`` alineados al
            preentrenamiento de *esos* pesos (no mezclar).
        in_channels: canales de entrada (3 = RGB).
        out_channels: ancho del FPN, compartido por todos los niveles.
        out_indices: qué stages del backbone alimentan el FPN (4 por defecto).
        drop_path_rate: stochastic depth (regularización ConvNeXt); 0.0 = off.
        checkpoint_path: pesos locales opcionales (tiene prioridad sobre ``pretrained``).
        freeze_backbone: si True, congela todos los parámetros del cuerpo.
        freeze_stages: congela stem + stages_0..N-1 (granular).
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
        freeze_stages: int = 0,
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

        # No hardcodeamos los channels: los leemos del propio modelo, así todas las
        # variantes funcionan sin tocar nada (tiny → [96,192,384,768];
        # nano → [80,160,320,640]; atto → [40,80,160,320]; etc.).
        in_channels_list = self.body.feature_info.channels()
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),  # nivel "pool" (P6) → 5 mapas para el RPN
        )

        # torchvision lee esto para dimensionar RPN y heads.
        self.out_channels = out_channels

        # Congelamiento granular. freeze_backbone = atajo para "todo" (= 4 stages).
        # freeze_stages=N congela stem + stages_0..N-1 (de abajo hacia arriba: las
        # features genéricas del preentrenamiento). Ahorra VRAM (menos activaciones
        # con grad + menos estado de optimizer).
        n_freeze = 4 if freeze_backbone else int(freeze_stages)
        self._frozen_module_names: list[str] = []
        self._apply_freeze(n_freeze)

    def _apply_freeze(self, n_stages: int) -> None:
        if n_stages <= 0:
            return
        names = ["stem_0", "stem_1"] + [f"stages_{i}" for i in range(min(n_stages, 4))]
        for name in names:
            mod = getattr(self.body, name, None)
            if mod is None:
                continue
            for p in mod.parameters():
                p.requires_grad_(False)
            mod.eval()
            self._frozen_module_names.append(name)

    def train(self, mode: bool = True):
        # Mantener las stages congeladas en eval aunque el modelo pase a train
        # (drop_path determinista; en ConvNeXt las norms son LayerNorm, sin running stats).
        super().train(mode)
        if mode:
            for name in self._frozen_module_names:
                mod = getattr(self.body, name, None)
                if mod is not None:
                    mod.eval()
        return self

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


@BACKBONES.register("convnextv2_nano")
class ConvNeXtV2Nano(ConvNeXtV2FPN):
    """ConvNeXt-V2 Nano + FPN — sub-tiny con pesos a 384 (in22k_in1k_384)."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("variant", "nano")
        super().__init__(**kwargs)


@BACKBONES.register("convnextv2_pico")
class ConvNeXtV2Pico(ConvNeXtV2FPN):
    """ConvNeXt-V2 Pico + FPN — sub-tiny, pesos solo a 224 (fcmae_ft_in1k)."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("variant", "pico")
        super().__init__(**kwargs)


@BACKBONES.register("convnextv2_femto")
class ConvNeXtV2Femto(ConvNeXtV2FPN):
    """ConvNeXt-V2 Femto + FPN — sub-tiny, pesos solo a 224 (fcmae_ft_in1k)."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("variant", "femto")
        super().__init__(**kwargs)


@BACKBONES.register("convnextv2_atto")
class ConvNeXtV2Atto(ConvNeXtV2FPN):
    """ConvNeXt-V2 Atto + FPN — el más liviano (~3.7M params), solo 224."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("variant", "atto")
        super().__init__(**kwargs)


if __name__ == "__main__":
    # Hito de Fase 3 (paso 8): "forward sin error", sin dataset ni descarga de pesos.
    bb = ConvNeXtV2Tiny(pretrained=False)
    x = torch.randn(2, 3, 640, 640)
    out = bb(x)
    print("out_channels:", bb.out_channels)
    for k, v in out.items():
        print(f"  {k}: {tuple(v.shape)}")
