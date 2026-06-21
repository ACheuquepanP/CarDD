"""src/cardd/evaluation/coco_eval.py

Dueno: Equipo E - Fase: 2 - Contrato: C2 (consume) -> C5 (produce)

Frontera de evaluacion del proyecto. Dos responsabilidades:

  1) Serializar la prediccion del modelo (C2) a resultados COCO
     (xyxy->xywh, label 1..N -> category_id, mascara float -> RLE binaria).
  2) Calcular el AP con pycocotools contra el GT COCO ORIGINAL (ds.coco),
     nunca contra el target aumentado de D, y emitir el dict C5.

C5 (una sola fuente de verdad para la tabla de ablacion):
    {
      "mask_AP": float,   # segm AP @[.50:.95]   <- metrica primaria
      "box_AP":  float,   # bbox AP @[.50:.95]
      "AP50":    float,   # segm AP @.50  (proyecto de segmentacion)
      "AP75":    float,   # segm AP @.75
      "per_class": {nombre: {"mask_AP": float, "box_AP": float}, ...},
      "per_scale": {"APS": float, "APM": float, "APL": float},  # segm
    }

Decision: la mascara es la tarea primaria (Mask R-CNN; las hipotesis a2/a3
son sobre mascara). Por eso AP50/AP75/per_scale salen de segm; box_AP queda
como referencia de localizacion. El AP por caja completo vive en per_class.

Sin dependencia de torch: acepta tensores via duck-typing (.detach) pero solo
importa numpy + pycocotools, para que la evaluacion sea liviana y testeable
sin GPU ni modelo.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pycocotools.mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from cardd.evaluation.per_class_metrics import extract_per_class
from cardd.evaluation.per_scale_metrics import extract_per_scale

# Llaves del contrato C5 (se valida que evaluate() las emita todas).
C5_KEYS = ("mask_AP", "box_AP", "AP50", "AP75", "per_class", "per_scale")


def _to_numpy(x: Any) -> np.ndarray:
    """Tensor de torch o array-like -> np.ndarray, sin importar torch."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def predictions_to_coco_results(
    predictions: Sequence[Mapping[str, Any]],
    image_ids: Sequence[int],
    label_to_cat_id: Mapping[int, int],
    mask_threshold: float = 0.5,
) -> list[dict]:
    """Serializa la salida del modelo (C2) a la lista de resultados COCO.

    predictions: un dict por imagen (mismo orden que image_ids) con
        boxes:  [K,4] xyxy
        scores: [K]
        labels: [K] en 1..N (mismo espacio que C1)
        masks:  [K,1,H,W] o [K,H,W] float en [0,1] (opcional; binariza a 0.5)

    label_to_cat_id: inverso de ds.cat_id_to_label
        {label: cid for cid, label in ds.cat_id_to_label.items()}

    No se aplica umbral de score aqui: COCOeval necesita todas las detecciones
    para construir la curva precision-recall. El filtrado por score, si se
    quiere para visualizacion, es responsabilidad de quien dibuja, no del AP.
    """
    results: list[dict] = []
    for pred, image_id in zip(predictions, image_ids):
        boxes = _to_numpy(pred["boxes"]).reshape(-1, 4)
        scores = _to_numpy(pred["scores"]).reshape(-1)
        labels = _to_numpy(pred["labels"]).reshape(-1)

        masks = pred.get("masks")
        if masks is not None:
            masks = _to_numpy(masks)
            if masks.ndim == 4:          # [K,1,H,W] -> [K,H,W]
                masks = masks[:, 0]

        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = (float(v) for v in boxes[i])
            res: dict[str, Any] = {
                "image_id": int(image_id),
                "category_id": int(label_to_cat_id[int(labels[i])]),
                "bbox": [x1, y1, x2 - x1, y2 - y1],   # xyxy -> xywh
                "score": float(scores[i]),
            }
            if masks is not None:
                binary = (masks[i] >= mask_threshold).astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(binary))
                rle["counts"] = rle["counts"].decode("ascii")  # JSON-serializable
                res["segmentation"] = rle
            results.append(res)
    return results


def _run_one(coco_gt: COCO, coco_results: list[dict], iou_type: str) -> COCOeval:
    """Una pasada de COCOeval (evaluate+accumulate+summarize) para un iouType."""
    coco_dt = coco_gt.loadRes(coco_results)
    ev = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return ev


def _empty_c5(cat_id_to_name: Mapping[int, str]) -> dict:
    """C5 con ceros: el modelo no predijo nada (tipico al inicio del train)."""
    return {
        "mask_AP": 0.0,
        "box_AP": 0.0,
        "AP50": 0.0,
        "AP75": 0.0,
        "per_class": {name: {"mask_AP": 0.0, "box_AP": 0.0} for name in cat_id_to_name.values()},
        "per_scale": {"APS": 0.0, "APM": 0.0, "APL": 0.0},
    }


def evaluate(
    coco_gt: COCO,
    coco_results: list[dict],
    cat_id_to_name: Mapping[int, str],
) -> dict:
    """GT (ds.coco) + resultados COCO -> dict C5.

    cat_id_to_name = {cid: ds.label_to_name[label]
                      for cid, label in ds.cat_id_to_label.items()}
    """
    if not coco_results:
        return _empty_c5(cat_id_to_name)

    ev_segm = _run_one(coco_gt, coco_results, "segm")
    ev_bbox = _run_one(coco_gt, coco_results, "bbox")

    out = {
        "mask_AP": float(ev_segm.stats[0]),
        "box_AP": float(ev_bbox.stats[0]),
        "AP50": float(ev_segm.stats[1]),   # segm
        "AP75": float(ev_segm.stats[2]),   # segm
        "per_class": extract_per_class(ev_segm, ev_bbox, cat_id_to_name),
        "per_scale": extract_per_scale(ev_segm),  # escalas sobre mascara
    }
    assert set(C5_KEYS) <= out.keys(), "C5 incompleto"
    return out


def evaluate_from_dataset(dataset: Any, predictions, image_ids) -> dict:
    """Conveniencia para E: saca GT y mapas de clase del dataset C1 de D.

    Espera un dataset que exponga la interfaz que D congelo para E:
        dataset.coco             -> objeto COCO (GT del JSON ORIGINAL)
        dataset.cat_id_to_label  -> {category_id: label(1..N)}
        dataset.label_to_name    -> {label: nombre_clase}

    Invierte cat_id_to_label para serializar (label->category_id), arma el
    cat_id_to_name legible y delega en evaluate(). Pensado para usarse con
    `loader.dataset` dentro de scripts/evaluate.py.
    """
    label_to_cat_id = {label: cid for cid, label in dataset.cat_id_to_label.items()}
    cat_id_to_name = {
        cid: dataset.label_to_name[label] for cid, label in dataset.cat_id_to_label.items()
    }
    results = predictions_to_coco_results(predictions, image_ids, label_to_cat_id)
    return evaluate(dataset.coco, results, cat_id_to_name)


__all__ = ["predictions_to_coco_results", "evaluate", "evaluate_from_dataset", "C5_KEYS"]