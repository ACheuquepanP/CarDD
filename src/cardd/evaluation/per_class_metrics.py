"""src/cardd/evaluation/per_class_metrics.py

Dueno: Equipo E - Fase: 2 - Contrato: C5

AP por clase para mascara y caja. Se lee del array de precision de un unico
COCOeval ya acumulado (no se re-evalua por clase): es la misma cuenta que
hace summarize(), restringida a cada categoria.

Por que importa para la ablacion: la hipotesis a1 (focal) es 'balancear
clases'; sin el AP por clase no se puede ver si la clase rara mejora. Las
llaves del dict son los NOMBRES de clase (legibles para la tabla), no ids.
"""
from __future__ import annotations

import numpy as np
from pycocotools.cocoeval import COCOeval


def _ap_per_category(coco_eval: COCOeval) -> dict[int, float]:
    """category_id -> AP@[.50:.95]. Promedia sobre umbrales IoU y recall en
    el area 'all' (idx 0) y el maxDet mayor (idx -1)."""
    # precision: [T, R, K, A, M]  (T=IoU, R=recall, K=cat, A=areaRng, M=maxDet)
    precision = coco_eval.eval["precision"]
    out: dict[int, float] = {}
    for k, cat_id in enumerate(coco_eval.params.catIds):
        p = precision[:, :, k, 0, -1]
        p = p[p > -1]
        out[int(cat_id)] = float(p.mean()) if p.size else float("nan")
    return out


def extract_per_class(
    eval_segm: COCOeval,
    eval_bbox: COCOeval,
    cat_id_to_name: dict[int, str],
    eval_boundary: COCOeval | None = None,
) -> dict[str, dict[str, float]]:
    """Devuelve {nombre_clase: {'mask_AP', 'box_AP'[, 'boundary_AP']}}.

    cat_id_to_name viene de invertir/combinar los mapas que expone D:
        cat_id_to_name = {cid: ds.label_to_name[label]
                          for cid, label in ds.cat_id_to_label.items()}

    Si se pasa eval_boundary (COCOevalBoundary ya acumulado), agrega boundary_AP
    por clase, clave para juzgar a3_pointrend en scratch/crack.
    """
    mask_ap = _ap_per_category(eval_segm)
    box_ap = _ap_per_category(eval_bbox)
    bnd_ap = _ap_per_category(eval_boundary) if eval_boundary is not None else None
    out: dict[str, dict[str, float]] = {}
    for cid, name in cat_id_to_name.items():
        entry = {
            "mask_AP": mask_ap.get(cid, float("nan")),
            "box_AP": box_ap.get(cid, float("nan")),
        }
        if bnd_ap is not None:
            entry["boundary_AP"] = bnd_ap.get(cid, float("nan"))
        out[name] = entry
    return out


__all__ = ["extract_per_class"]