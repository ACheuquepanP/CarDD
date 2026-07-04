"""src/cardd/evaluation/boundary_eval.py

Dueno: Equipo E - Fase: 2 - Contrato: C5 (métrica aditiva)

Boundary AP (Cheng et al., 2021): AP COCO usando Boundary IoU como criterio de
match, en vez del IoU de máscara estándar. Mide calidad de BORDE, que el
mask_AP estándar sub-premia en daños de área grande (el interior domina el IoU).
Es la métrica que hace interpretable la fila a3_pointrend ('refinar bordes').

NO cubre la hipótesis de CONTINUIDAD/topología de los custom clDice/SkelRecall
(grieta conexa vs cortada): eso es conectividad, otra métrica (gate ②-bis).

Única dependencia extra: cv2 (ya en el proyecto vía opencv-headless), solo para
la erosión morfológica. El resto de evaluation/ sigue siendo numpy+pycocotools.
"""
from __future__ import annotations

import cv2
import numpy as np
import pycocotools.mask as mask_utils
from pycocotools.cocoeval import COCOeval

DILATION_RATIO = 0.02  # ancho de la banda de borde = ratio * diagonal de la imagen (paper)


def mask_to_boundary(mask: np.ndarray, dilation_ratio: float = DILATION_RATIO) -> np.ndarray:
    """Banda de borde de una máscara binaria: mask - erode(mask, d).

    d = round(dilation_ratio * diagonal). Se pad-ea 1px para que el borde de la
    imagen no se confunda con el contorno del objeto.
    """
    mask = np.ascontiguousarray(mask, dtype=np.uint8)
    h, w = mask.shape
    d = max(1, int(round(dilation_ratio * float(np.sqrt(h * h + w * w)))))
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    eroded = cv2.erode(padded, np.ones((3, 3), np.uint8), iterations=d)[1 : h + 1, 1 : w + 1]
    return mask - eroded  # eroded ⊆ mask -> sin underflow, valores en {0,1}


def boundary_iou(gt: np.ndarray, dt: np.ndarray, dilation_ratio: float = DILATION_RATIO) -> float:
    """IoU calculado solo sobre las bandas de borde de gt y dt."""
    gt_b = mask_to_boundary(gt, dilation_ratio)
    dt_b = mask_to_boundary(dt, dilation_ratio)
    union = int(np.count_nonzero(gt_b | dt_b))
    if union == 0:
        return 0.0
    return int(np.count_nonzero(gt_b & dt_b)) / union


class COCOevalBoundary(COCOeval):
    """COCOeval cuyo computeIoU usa Boundary IoU para las máscaras.

    Hereda toda la maquinaria de matching/accumulate/summarize; solo cambia la
    medida de solape. Para GT 'crowd' cae al IoU estándar (IoA), como el paper.
    """

    def __init__(self, coco_gt, coco_dt, dilation_ratio: float = DILATION_RATIO):
        super().__init__(coco_gt, coco_dt, iouType="segm")
        self.dilation_ratio = dilation_ratio

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [g for cId in p.catIds for g in self._gts[imgId, cId]]
            dt = [d for cId in p.catIds for d in self._dts[imgId, cId]]
        if len(gt) == 0 or len(dt) == 0:
            return []

        inds = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0 : p.maxDets[-1]]

        # tras _prepare(), las segmentaciones ya son RLE comprimido
        g_rle = [g["segmentation"] for g in gt]
        d_rle = [d["segmentation"] for d in dt]
        iscrowd = [int(o["iscrowd"]) for o in gt]
        g_mask = [mask_utils.decode(s) for s in g_rle]
        d_mask = [mask_utils.decode(s) for s in d_rle]

        ious = np.zeros((len(d_mask), len(g_mask)), dtype=np.float64)
        for di, dm in enumerate(d_mask):
            for gi, gm in enumerate(g_mask):
                if iscrowd[gi]:
                    ious[di, gi] = float(mask_utils.iou([d_rle[di]], [g_rle[gi]], [1])[0, 0])
                else:
                    ious[di, gi] = boundary_iou(gm, dm, self.dilation_ratio)
        return ious


def run_boundary_eval(coco_gt, coco_results: list[dict], img_ids: list[int]) -> COCOevalBoundary:
    """Una pasada de Boundary AP (evaluate+accumulate+summarize).

    img_ids: imágenes inferidas; acota igual que en coco_eval._run_one, para no
    promediar contra todo el GT (mismo bug AP≈0 en subconjuntos).
    """
    coco_dt = coco_gt.loadRes(coco_results)
    ev = COCOevalBoundary(coco_gt, coco_dt)
    ev.params.imgIds = img_ids
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return ev


__all__ = ["mask_to_boundary", "boundary_iou", "COCOevalBoundary", "run_boundary_eval"]