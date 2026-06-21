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


def _c5(mask, box, ap50, ap75, aps, apm, apl, per_class):
    return {
        "mask_AP": mask, "box_AP": box, "AP50": ap50, "AP75": ap75,
        "per_class": per_class,
        "per_scale": {"APS": aps, "APM": apm, "APL": apl},
    }


def _pc(dent_m, scr_m):
    return {"dent": {"mask_AP": dent_m, "box_AP": dent_m},
            "scratch": {"mask_AP": scr_m, "box_AP": scr_m}}


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