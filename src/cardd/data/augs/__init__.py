"""Augmentaciones de composición, una por archivo. El orquestador (composition.py)
las llama por nombre vía este registro. Cada módulo expone `compose(...)` y, las que
componen desde varias fuentes, `num_sources(**params)`.
"""
from cardd.data.augs import mosaic, mosaic_crop, radial, copy_paste, mixup

# tipos que compiten en el sorteo (normal se maneja aparte, sin módulo)
COMPETITORS = {
    "mosaic": mosaic,
    "mosaic_crop": mosaic_crop,
    "radial": radial,
    "copy_paste": copy_paste,
}

__all__ = ["mosaic", "mosaic_crop", "radial", "copy_paste", "mixup", "COMPETITORS"]
