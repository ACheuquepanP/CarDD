# Ratificación → Equipo D · contrato de geometría (tiling)

**Para:** D · **cc:** C · **Estado:** M ratifica el contrato de `tiling.py` (§1 de
`HANDOFF_coordinacion_para_M_tiling_wbf.md`). Sin correcciones — `scale=1.0` es correcto tal
cual está.

---

## Confirmación

El detector (`CarddMaskRCNN`, en `models/detectors/mask_rcnn.py`) hereda de
`torchvision.models.detection.MaskRCNN`, que usa internamente
`GeneralizedRCNNTransform`. Ese transform:

1. Reescala la imagen de entrada según `data.image_min_size` / `data.image_max_size`.
2. **Remapea la salida de vuelta al tamaño de entrada original** antes de devolver
   `boxes`/`masks` — es comportamiento estándar de torchvision, no algo que M tenga
   que codear aparte.

Como cada tile entra con su tamaño real (`tw, th` — los que ya define D en el `meta`) y sale
remapeado a ese mismo tamaño, el forward **no** reescala de forma que quede pendiente en la
salida. `scale=1.0` que ya declara D en el contrato es correcto sin ajuste.

## Lo que esto implica para la reproyección

`box_entera = box_tile/scale + (x0,y0)` con `scale=1.0` es efectivamente
`box_entera = box_tile + (x0,y0)` — solo suma de offset, como ya asumía D. Implementado en
`engine/inference.py::reproject_boxes`.

## Estado del resto de §8 (para C)

- Política de máscara elegida: **`union`** (configurable, ver `eval.wbf.mask_policy`).
- Knobs de WBF expuestos en `eval.wbf.*` (`base.yaml`), con defaults — retrocompatible.
- Harness de barrido: en curso (`scripts/wbf_sweep.py`).

Sin bloqueos de este lado hacia D. Cierre de este punto.
