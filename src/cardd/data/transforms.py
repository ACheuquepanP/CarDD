"""src/cardd/data/transforms.py

Dueno: Equipo D - Fase: 1 (aug base) / 5 (multi-scale a2) - Contrato: C1

Augmentations sobre transforms v2 de torchvision: operan sobre tv_tensors, asi
que mueven imagen, caja (BoundingBoxes) y mascara (Mask) de forma consistente.
Las piezas intercambiables se registran por nombre en TRANSFORMS (C3) para que
build.py las componga desde la config, sin hardcodear.

Orden recomendado en la config:
    [geometricas..., fotometricas..., sanitize]
`sanitize` va SIEMPRE al final: elimina las cajas degeneradas que la augmentation
pueda producir y descarta EN SINCRONIA sus labels y mascaras. Es el cierre que
permite "confiar en la bbox del COCO" sin recomputar nada.

a0 (baseline) usa solo `hflip`. a2 usa `multiscale` (ver nota en MultiScale).
"""
from __future__ import annotations

from torchvision.transforms import v2

from cardd.utils.registry import TRANSFORMS

# --- piezas intercambiables (se construyen por nombre desde la config, C3) ---

# Geometrica: mueve caja y mascara junto a la imagen.
TRANSFORMS.register("hflip")(v2.RandomHorizontalFlip)

# Fotometrica: toca color/brillo/contraste; la geometria (caja/mascara) intacta.
TRANSFORMS.register("photometric")(v2.RandomPhotometricDistort)

# Cierre estandar del pipeline (va de ultimo): saca cajas degeneradas y, con
# ellas, sus labels y mascaras alineadas.
TRANSFORMS.register("sanitize")(v2.SanitizeBoundingBoxes)


@TRANSFORMS.register("multiscale")
class MultiScale(v2.RandomShortestSize):
    """Hook de a2 (Fase 5): reescala el lado corto a un valor elegido al azar de
    `scales` en cada iteracion -> multi-scale training.

    DECISION PENDIENTE con M: el GeneralizedRCNNTransform del modelo TAMBIEN
    reescala a su min_size. Si ambos actuan, el multi-scale se ANULA. Para usar
    este transform hay que dejar el min_size del modelo en passthrough, o mover el
    multi-scale al modelo (lista de min_size). No cablear hasta acordar a2.
    """

    def __init__(self, scales, max_size: int | None = None, **kwargs) -> None:
        super().__init__(min_size=list(scales), max_size=max_size, **kwargs)


# --- preset por defecto (referencia / smoke test; build.py prefiere la config) ---

def default_pipeline(train: bool) -> v2.Compose | None:
    """Preset minimo estilo torchvision-detection: flip + sanitize en train; en
    eval no se augmenta (el modelo se encarga del resize)."""
    if train:
        return v2.Compose([v2.RandomHorizontalFlip(p=0.5), v2.SanitizeBoundingBoxes()])
    return None