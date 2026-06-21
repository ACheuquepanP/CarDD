"""src/cardd/evaluation/per_scale_metrics.py

Dueno: Equipo E - Fase: 2 - Contrato: C5

Extrae el AP por escala (APS / APM / APL) de un COCOeval ya
acumulado+resumido. Usa las bandas de area ESTANDAR de COCO
(S < 32^2, 32^2 <= M <= 96^2, L > 96^2). Se mantienen los defaults
(no bandas propias) para que la fila sea comparable al paper y para medir
limpio la hipotesis a2 (multi-scale 'mueve el small'). Si en el futuro se
quisieran bandas a medida, viven en base.yaml (P) y se inyectan aqui.

Nota: COCOeval reporta -1.0 en una banda sin GT; se propaga tal cual para
no inventar un 0 que confundiria la tabla.
"""
from __future__ import annotations

from pycocotools.cocoeval import COCOeval


def extract_per_scale(coco_eval: COCOeval) -> dict[str, float]:
    """Lee APS/APM/APL de coco_eval.stats (requiere summarize() previo).

    Orden estandar de los 12 stats de COCO:
        [0]=AP@[.50:.95] [1]=AP50 [2]=AP75 [3]=APS [4]=APM [5]=APL  [6..11]=AR
    """
    s = coco_eval.stats
    return {"APS": float(s[3]), "APM": float(s[4]), "APL": float(s[5])}


__all__ = ["extract_per_scale"]