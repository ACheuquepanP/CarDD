"""src/cardd/evaluation/per_scale_metrics.py

Dueno: Equipo E - Fase: 2 - Contrato: C5

Extrae el AP por escala de COCOevals ya acumulados+resumidos, usando las
bandas de area del PROYECTO inyectadas en coco_eval._run_one (AREA_RNG:
small<128^2, 128^2<=medium<256^2, large>=256^2), no las estandar de COCO.

v2: ademas de APS/APM/APL de mascara @[.50:.95] (compatibilidad), emite el
desglose completo por escala x metrica x umbral:
    per_scale = {
        "APS": float, "APM": float, "APL": float,        # mask @[.50:.95] (legacy)
        "mask": {"small"|"medium"|"large": {"AP","AP50","AP75"}},
        "box":  {"small"|"medium"|"large": {"AP","AP50","AP75"}},
    }

Nota: COCOeval reporta -1.0 en una banda sin GT; se propaga tal cual (aqui
como nan al promediar rebanadas vacias) para no inventar un 0 que confundiria
la tabla.
"""
from __future__ import annotations

import numpy as np
from pycocotools.cocoeval import COCOeval

_SCALES = ("small", "medium", "large")


def _iou_thr_index(coco_eval: COCOeval, iou_thr: float) -> int:
    idx = np.where(np.isclose(coco_eval.params.iouThrs, iou_thr))[0]
    if idx.size == 0:
        raise ValueError(f"iou_thr={iou_thr} no esta en params.iouThrs")
    return int(idx[0])


def _area_index(coco_eval: COCOeval, label: str) -> int:
    """Indice de la banda en params.areaRngLbl ('small'/'medium'/'large')."""
    lbls = list(coco_eval.params.areaRngLbl)
    if label not in lbls:
        raise ValueError(f"banda '{label}' no esta en areaRngLbl={lbls}")
    return lbls.index(label)


def _ap_at(coco_eval: COCOeval, area_label: str, iou_thr: float | None) -> float:
    """AP promediado sobre recall y clases, en una banda de area, para un
    umbral IoU (o todos si iou_thr=None). Misma cuenta que summarize()."""
    # precision: [T, R, K, A, M]
    precision = coco_eval.eval["precision"]
    if iou_thr is not None:
        precision = precision[[_iou_thr_index(coco_eval, iou_thr)]]
    a = _area_index(coco_eval, area_label)
    p = precision[:, :, :, a, -1]
    p = p[p > -1]
    return float(p.mean()) if p.size else -1.0


def _scale_block(coco_eval: COCOeval) -> dict[str, dict[str, float]]:
    return {
        s: {
            "AP": _ap_at(coco_eval, s, None),
            "AP50": _ap_at(coco_eval, s, 0.5),
            "AP75": _ap_at(coco_eval, s, 0.75),
        }
        for s in _SCALES
    }


def extract_per_scale(eval_segm: COCOeval, eval_bbox: COCOeval | None = None) -> dict:
    """Desglose por escala. Requiere accumulate()+summarize() previos.

    Mantiene APS/APM/APL (mask @[.50:.95], leidos de stats[3:6]) por
    compatibilidad con consumidores existentes (run_ablation, JSONs viejos),
    y agrega los bloques 'mask' y 'box' con AP/AP50/AP75 por banda.
    """
    s = eval_segm.stats
    out: dict = {
        "APS": float(s[3]),
        "APM": float(s[4]),
        "APL": float(s[5]),
        "mask": _scale_block(eval_segm),
    }
    if eval_bbox is not None:
        out["box"] = _scale_block(eval_bbox)
    return out


__all__ = ["extract_per_scale"]
