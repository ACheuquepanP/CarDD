"""CarDD · tests/test_metrics.py · Equipo E · Contrato C5.

Cubre las dos piezas de E que producen/consumen C5, sin torch ni GPU ni
dataset real:

  - Fase 2 · maquinaria de AP: GT sintetico, hito GT = pred => AP ≈ 1.0,
    serializacion C2->COCO, per_class, per_scale, forma del contrato.
  - Fase 6 · tabla de ablacion (run_ablation): deltas vs baseline, centinela
    -1.0, render markdown, lectura de resultados parciales.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_utils
import pytest
from pycocotools.coco import COCO

from cardd.evaluation.coco_eval import (
    C5_KEYS,
    evaluate,
    evaluate_from_dataset,
    predictions_to_coco_results,
)

# run_ablation vive en scripts/ (no es paquete): se importa por ruta.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_ablation as ra  # noqa: E402


# =====================================================================
# Fase 2 — maquinaria de AP (C5 producido)
# =====================================================================

# --- mapas de clase como los expone D (cat_id no contiguos a proposito) ---
CAT_ID_TO_LABEL = {3: 1, 7: 2}                         # ds.cat_id_to_label
LABEL_TO_NAME = {1: "dent", 2: "scratch"}             # ds.label_to_name
LABEL_TO_CAT_ID = {v: k for k, v in CAT_ID_TO_LABEL.items()}
CAT_ID_TO_NAME = {cid: LABEL_TO_NAME[lab] for cid, lab in CAT_ID_TO_LABEL.items()}

H = W = 64


def _rect(x: int, y: int, w: int, h: int) -> np.ndarray:
    m = np.zeros((H, W), dtype=np.uint8)
    m[y : y + h, x : x + w] = 1
    return m


# Cada objeto: (image_id, category_id, x, y, w, h). 10x10 -> area 100 < 32^2 = small.
_OBJECTS = [
    (1, 3, 5, 5, 10, 10),
    (1, 7, 30, 30, 10, 10),
    (2, 3, 12, 40, 10, 10),
]


@pytest.fixture
def coco_gt(tmp_path: Path) -> COCO:
    """Construye un GT COCO sintetico en disco y lo carga con pycocotools."""
    images = [
        {"id": 1, "width": W, "height": H, "file_name": "a.png"},
        {"id": 2, "width": W, "height": H, "file_name": "b.png"},
    ]
    categories = [{"id": 3, "name": "dent"}, {"id": 7, "name": "scratch"}]
    annotations = []
    for aid, (img_id, cat_id, x, y, w, h) in enumerate(_OBJECTS, start=1):
        rle = mask_utils.encode(np.asfortranarray(_rect(x, y, w, h)))
        annotations.append(
            {
                "id": aid,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [x, y, w, h],
                "area": float(w * h),       # 100 -> small
                "iscrowd": 0,
                "segmentation": {"counts": rle["counts"].decode("ascii"), "size": rle["size"]},
            }
        )
    gt = {"images": images, "categories": categories, "annotations": annotations}
    path = tmp_path / "gt.json"
    path.write_text(json.dumps(gt))
    return COCO(str(path))


def _perfect_predictions() -> tuple[list[dict], list[int]]:
    """Prediccion = GT exacto (score 1.0), por imagen, lista para serializar."""
    by_img: dict[int, dict] = {}
    for img_id, cat_id, x, y, w, h in _OBJECTS:
        d = by_img.setdefault(img_id, {"boxes": [], "scores": [], "labels": [], "masks": []})
        d["boxes"].append([x, y, x + w, y + h])           # xyxy
        d["scores"].append(1.0)
        d["labels"].append(CAT_ID_TO_LABEL[cat_id])       # 1..N
        d["masks"].append(_rect(x, y, w, h).astype(np.float32))
    image_ids = sorted(by_img)
    preds = [
        {
            "boxes": np.array(by_img[i]["boxes"], dtype=np.float32),
            "scores": np.array(by_img[i]["scores"], dtype=np.float32),
            "labels": np.array(by_img[i]["labels"], dtype=np.int64),
            "masks": np.stack(by_img[i]["masks"]),
        }
        for i in image_ids
    ]
    return preds, image_ids


def test_perfect_prediction_gives_ap_1(coco_gt):
    """El hito: GT = pred => mask_AP y box_AP ≈ 1.0."""
    preds, image_ids = _perfect_predictions()
    results = predictions_to_coco_results(preds, image_ids, LABEL_TO_CAT_ID)
    c5 = evaluate(coco_gt, results, CAT_ID_TO_NAME)

    assert c5["mask_AP"] > 0.99, f"mask_AP={c5['mask_AP']}"
    assert c5["box_AP"] > 0.99, f"box_AP={c5['box_AP']}"
    assert c5["AP50"] > 0.99 and c5["AP75"] > 0.99
    assert c5["mask_boundary_AP"] > 0.99, f"boundary={c5['mask_boundary_AP']}"
    # objetos pequenos -> la banda small debe estar perfecta
    assert c5["per_scale"]["APS"] > 0.99, f"APS={c5['per_scale']['APS']}"


def test_per_class_keys_and_values(coco_gt):
    """per_class indexado por nombre, con mask_AP y box_AP por clase."""
    preds, image_ids = _perfect_predictions()
    results = predictions_to_coco_results(preds, image_ids, LABEL_TO_CAT_ID)
    c5 = evaluate(coco_gt, results, CAT_ID_TO_NAME)

    assert set(c5["per_class"]) == {"dent", "scratch"}
    for name, ap in c5["per_class"].items():
        assert ap["mask_AP"] > 0.99, f"{name} mask_AP={ap['mask_AP']}"
        assert ap["box_AP"] > 0.99, f"{name} box_AP={ap['box_AP']}"
        assert ap["boundary_AP"] > 0.99, f"{name} boundary_AP={ap['boundary_AP']}"


def test_c5_contract_shape(coco_gt):
    """evaluate() emite exactamente las llaves del contrato C5."""
    preds, image_ids = _perfect_predictions()
    results = predictions_to_coco_results(preds, image_ids, LABEL_TO_CAT_ID)
    c5 = evaluate(coco_gt, results, CAT_ID_TO_NAME)
    assert set(C5_KEYS) <= set(c5)
    assert set(c5["per_scale"]) == {"APS", "APM", "APL"}


def test_shifted_prediction_drops_ap(coco_gt):
    """Cajas/mascaras corridas lejos => IoU≈0 => AP colapsa (sanity inverso)."""
    preds, image_ids = _perfect_predictions()
    for p in preds:
        p["boxes"] = p["boxes"] + 25.0                    # corre la caja
        p["masks"] = np.zeros_like(p["masks"])
        p["masks"][:, 50:60, 50:60] = 1.0                 # mascara en otra esquina
    results = predictions_to_coco_results(preds, image_ids, LABEL_TO_CAT_ID)
    c5 = evaluate(coco_gt, results, CAT_ID_TO_NAME)
    assert c5["mask_AP"] < 0.5, f"mask_AP={c5['mask_AP']} (deberia colapsar)"


def test_empty_predictions_are_safe(coco_gt):
    """Sin detecciones (modelo al inicio del train) => C5 de ceros, sin crashear."""
    c5 = evaluate(coco_gt, [], CAT_ID_TO_NAME)
    assert c5["mask_AP"] == 0.0 and c5["box_AP"] == 0.0
    assert set(c5["per_class"]) == {"dent", "scratch"}


class _FakeDataset:
    """Dataset minimo que imita la interfaz que D congelo para E."""

    def __init__(self, coco):
        self.coco = coco
        self.cat_id_to_label = CAT_ID_TO_LABEL
        self.label_to_name = LABEL_TO_NAME


def test_evaluate_from_dataset(coco_gt):
    """La variante que usa scripts/evaluate.py: GT y mapas salen del dataset."""
    preds, image_ids = _perfect_predictions()
    ds = _FakeDataset(coco_gt)
    c5 = evaluate_from_dataset(ds, preds, image_ids)

    assert c5["mask_AP"] > 0.99 and c5["box_AP"] > 0.99
    assert set(c5["per_class"]) == {"dent", "scratch"}


# =====================================================================
# Fase 6 — tabla de ablacion (C5 consumido por run_ablation)
# =====================================================================


def _c5(mask, box, ap50, ap75, aps, apm, apl, per_class, bnd=None):
    return {
        "mask_AP": mask, "box_AP": box, "AP50": ap50, "AP75": ap75,
        "mask_boundary_AP": mask if bnd is None else bnd,
        "per_class": per_class,
        "per_scale": {"APS": aps, "APM": apm, "APL": apl},
    }


def _pc(dent_m, scr_m):
    return {"dent": {"mask_AP": dent_m, "box_AP": dent_m, "boundary_AP": dent_m},
            "scratch": {"mask_AP": scr_m, "box_AP": scr_m, "boundary_AP": scr_m}}


@pytest.fixture
def ablation_results():
    return {
        "a0_baseline": _c5(0.40, 0.45, 0.60, 0.42, 0.20, -1.0, 0.55, _pc(0.50, 0.30)),
        "a2_focal_multiscale": _c5(0.43, 0.46, 0.62, 0.44, 0.28, -1.0, 0.55, _pc(0.51, 0.40)),
    }


def test_baseline_first_and_deltas(ablation_results):
    table = ra.build_table(ablation_results, "a0_baseline")
    assert table[0]["name"] == "a0_baseline" and table[0]["baseline"]
    a2 = table[1]
    assert a2["deltas"]["mask_AP"] == pytest.approx(0.03)
    assert a2["deltas"]["APS"] == pytest.approx(0.08)   # 'mueve el small'
    # APM = -1.0 en ambas (sin GT en la banda) -> delta no definido
    assert a2["deltas"]["APM"] is None


def test_markdown_has_base_marker_and_sentinel(ablation_results):
    md = ra.render_markdown(ra.build_table(ablation_results, "a0_baseline"))
    assert "a0_baseline *(base)*" in md
    assert "+0.0300" in md          # delta mask_AP de a2
    assert "—" in md                # centinela APM y fila baseline en deltas


def test_per_class_table(ablation_results):
    md = ra.render_per_class(ablation_results)
    assert "dent" in md and "scratch" in md
    assert "0.4000" in md           # scratch mask_AP de a2


def test_missing_baseline_no_crash(ablation_results):
    table = ra.build_table(ablation_results, "no_existe")
    assert all(all(d is None for d in r["deltas"].values()) for r in table)


def test_load_results_partial(tmp_path, ablation_results):
    (tmp_path / "a0_baseline.json").write_text(json.dumps(ablation_results["a0_baseline"]))
    loaded = ra.load_results(tmp_path, ["a0_baseline.yaml", "a2_focal_multiscale.yaml"])
    assert set(loaded) == {"a0_baseline"}


def test_marginal_vs_cumulative_deltas():
    """Acumulativa (gate ③): Δ vs baseline = ganancia acumulada;
    Δ vs fila anterior = contribución marginal del módulo que entra."""
    R = {
        "a0_baseline": _c5(0.40, 0.45, 0.60, 0.42, 0.20, -1.0, 0.55, _pc(0.50, 0.30)),
        "a1_focal": _c5(0.44, 0.45, 0.61, 0.42, 0.21, -1.0, 0.55, _pc(0.50, 0.42)),
        "a2_focal_multiscale": _c5(0.46, 0.46, 0.62, 0.44, 0.29, -1.0, 0.56, _pc(0.51, 0.42)),
    }
    table = ra.build_table(R, "a0_baseline")
    a2 = table[2]
    assert a2["deltas"]["mask_AP"] == pytest.approx(0.06)        # acumulada vs a0
    assert a2["deltas_prev"]["mask_AP"] == pytest.approx(0.02)   # marginal vs a1
    assert a2["deltas_prev"]["APS"] == pytest.approx(0.08)       # el small se mueve EN el paso multiscale
    # baseline no tiene fila anterior
    assert all(v is None for v in table[0]["deltas_prev"].values())


# =====================================================================
# Fase 2 — Boundary AP (gate ②): mide calidad de borde
# =====================================================================

from cardd.evaluation.boundary_eval import boundary_iou, mask_to_boundary  # noqa: E402


def _dense_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return inter / union if union else 0.0


def test_mask_to_boundary_is_a_ring():
    """La banda de borde es no vacía en el contorno y vacía en el interior profundo."""
    m = np.zeros((40, 40), np.uint8)
    m[8:32, 8:32] = 1
    b = mask_to_boundary(m)
    assert b[20, 20] == 0          # centro: interior, fuera de la banda
    assert b[8, 20] == 1           # borde superior: dentro de la banda
    assert np.count_nonzero(b) > 0


def test_boundary_iou_identical_and_stricter_than_mask():
    """Idénticas -> 1.0; ante un corrimiento, boundary IoU castiga más que el de máscara."""
    gt = np.zeros((60, 60), np.uint8)
    gt[15:45, 15:45] = 1           # objeto grande (30x30)
    assert boundary_iou(gt, gt) == pytest.approx(1.0)

    dt = np.zeros((60, 60), np.uint8)
    dt[17:47, 17:47] = 1           # mismo tamaño, corrido 2px
    b_iou = boundary_iou(gt, dt)
    m_iou = _dense_iou(gt, dt)
    assert 0.0 < b_iou < 1.0
    assert b_iou < m_iou, f"boundary={b_iou} debería ser más estricto que mask={m_iou}"


def _coco_from_masks(tmp_path, objects):
    """objects: lista de (image_id, cat_id, mask 2D). Devuelve COCO GT."""
    sizes = {}
    anns, used_imgs = [], {}
    for aid, (img_id, cat_id, m) in enumerate(objects, start=1):
        h, w = m.shape
        used_imgs[img_id] = (h, w)
        rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
        anns.append({
            "id": aid, "image_id": img_id, "category_id": cat_id,
            "bbox": list(map(float, mask_utils.toBbox(rle))),
            "area": float(mask_utils.area(rle)), "iscrowd": 0,
            "segmentation": {"counts": rle["counts"].decode("ascii"), "size": rle["size"]},
        })
    images = [{"id": i, "height": hw[0], "width": hw[1], "file_name": f"{i}.png"}
              for i, hw in used_imgs.items()]
    gt = {"images": images, "categories": [{"id": 3, "name": "dent"}], "annotations": anns}
    path = tmp_path / "gtb.json"
    path.write_text(json.dumps(gt))
    return COCO(str(path))


def test_boundary_ap_below_mask_ap_on_degraded_border(tmp_path):
    """El punto de la métrica: una predicción con buen solape de bulto pero
    borde degradado mantiene mask_AP alto y baja mask_boundary_AP."""
    gt_mask = np.zeros((80, 80), np.uint8)
    gt_mask[20:60, 20:60] = 1                      # objeto grande (40x40)
    coco_gt = _coco_from_masks(tmp_path, [(1, 3, gt_mask)])

    # predicción = GT erosionado 2px: solape de bulto alto, borde peor
    eroded = mask_to_boundary  # noqa: F841  (solo para claridad del intent)
    import cv2
    pred_mask = cv2.erode(gt_mask, np.ones((3, 3), np.uint8), iterations=2)

    preds = [{
        "boxes": np.array([[20, 20, 60, 60]], np.float32),
        "scores": np.array([1.0], np.float32),
        "labels": np.array([1], np.int64),          # label 1 -> category_id 3
        "masks": pred_mask[None].astype(np.float32),
    }]
    results = predictions_to_coco_results(preds, [1], {1: 3})
    c5 = evaluate(coco_gt, results, {3: "dent"})

    assert c5["mask_AP"] > 0.5, f"mask_AP={c5['mask_AP']}"
    assert c5["mask_boundary_AP"] < c5["mask_AP"], (
        f"boundary={c5['mask_boundary_AP']} debería caer bajo mask={c5['mask_AP']}"
    )


# =====================================================================
# Fase 2 — regresión del bug de imgIds (subconjunto / eval parcial)
# =====================================================================


def _gt_three_images(tmp_path):
    """GT con 3 imágenes, un objeto por imagen (cat 3 / 7 / 3)."""
    objs = [(1, 3, 5, 5, 10, 10), (2, 7, 30, 30, 10, 10), (3, 3, 12, 40, 10, 10)]
    images = [{"id": i, "width": W, "height": H, "file_name": f"{i}.png"} for i in (1, 2, 3)]
    categories = [{"id": 3, "name": "dent"}, {"id": 7, "name": "scratch"}]
    anns = []
    for aid, (img_id, cat_id, x, y, w, h) in enumerate(objs, start=1):
        rle = mask_utils.encode(np.asfortranarray(_rect(x, y, w, h)))
        anns.append({
            "id": aid, "image_id": img_id, "category_id": cat_id,
            "bbox": [x, y, w, h], "area": float(w * h), "iscrowd": 0,
            "segmentation": {"counts": rle["counts"].decode("ascii"), "size": rle["size"]},
        })
    path = tmp_path / "gt3.json"
    path.write_text(json.dumps({"images": images, "categories": categories, "annotations": anns}))
    return COCO(str(path)), objs


def _preds_for(objs, subset):
    """Predicción perfecta solo para las imágenes en `subset`."""
    by_img: dict[int, dict] = {}
    for img_id, cat_id, x, y, w, h in objs:
        if img_id not in subset:
            continue
        d = by_img.setdefault(img_id, {"boxes": [], "scores": [], "labels": [], "masks": []})
        d["boxes"].append([x, y, x + w, y + h])
        d["scores"].append(1.0)
        d["labels"].append(CAT_ID_TO_LABEL[cat_id])
        d["masks"].append(_rect(x, y, w, h).astype(np.float32))
    ids = sorted(by_img)
    preds = [
        {
            "boxes": np.array(by_img[i]["boxes"], np.float32),
            "scores": np.array(by_img[i]["scores"], np.float32),
            "labels": np.array(by_img[i]["labels"], np.int64),
            "masks": np.stack(by_img[i]["masks"]),
        }
        for i in ids
    ]
    return preds, ids


def test_subset_eval_restricts_to_inferred_images(tmp_path):
    """El bug de M: GT de 3 imgs, inferencia perfecta sobre 2. Acotando a las
    inferidas el AP da ~1; sin acotar, las 2808 (aquí 1) sin inferir lo hunden."""
    coco_gt, objs = _gt_three_images(tmp_path)
    preds, ids = _preds_for(objs, {1, 2})
    results = predictions_to_coco_results(preds, ids, LABEL_TO_CAT_ID)
    c5 = evaluate(coco_gt, results, CAT_ID_TO_NAME, image_ids=[1, 2])
    assert c5["mask_AP"] > 0.99, f"mask_AP={c5['mask_AP']}"
    assert c5["mask_boundary_AP"] > 0.99


def test_inferred_empty_image_counts_as_recall_miss(tmp_path):
    """Imagen INFERIDA pero vacía sigue contando: su GT no matcheado baja el AP.
    Por eso se hilvanan las image_ids inferidas y no se derivan de coco_results
    (que perdería esa imagen e inflaría el AP del split)."""
    coco_gt, objs = _gt_three_images(tmp_path)
    preds, ids = _preds_for(objs, {1, 2})              # imagen 3 inferida-y-vacía
    results = predictions_to_coco_results(preds, ids, LABEL_TO_CAT_ID)
    c5 = evaluate(coco_gt, results, CAT_ID_TO_NAME, image_ids=[1, 2, 3])
    assert c5["mask_AP"] < 1.0, f"mask_AP={c5['mask_AP']} (img 3 sin detección debe penalizar)"


def test_streaming_serialization_matches_batch():
    """Fix OOM: serializar por imagen dentro del loop (y soltar las máscaras
    densas) debe dar exactamente los mismos COCO-results que serializar el split
    entero de golpe. Si difieren, el fix cambiaría la métrica."""
    preds, ids = _perfect_predictions()
    batch = predictions_to_coco_results(preds, ids, LABEL_TO_CAT_ID)
    streamed: list[dict] = []
    for p, i in zip(preds, ids):
        streamed.extend(predictions_to_coco_results([p], [i], LABEL_TO_CAT_ID))
    assert streamed == batch