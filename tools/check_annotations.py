"""tools/check_annotations.py

Dueno: Equipo D - Fase: 1 - Contrato: C1

Guardia anti-rotura-silenciosa: valida las anotaciones COCO de CarDD ANTES de
que `coco_dataset.py` las convierta en C1. Reporta y cuenta lo que romperia (o
falsearia) el entrenamiento sin avisar.

Clasificacion:
  Errores duros  (exit 1): imagen ausente en disco, category_id fuera del
                           conjunto, caja/area degenerada, mascara vacia o que
                           no decodifica.
  Avisos         (exit 0): imagenes sin anotaciones (negativos -> decision D3),
                           iscrowd=1, caja levemente fuera de borde (recortable).

Higiene: `data.root` sale de la config (C4); las clases se leen de las
anotaciones, no se hardcodean.

Uso:
    python tools/check_annotations.py --config configs/base.yaml [--no-masks]
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from pycocotools.coco import COCO

from cardd.utils.config import get, load_config
from cardd.utils.logging import get_logger

log = get_logger("cardd.data.check")

DEFAULT_SPLITS = ("train", "val", "test")
_MISSING = object()
_EPS = 1.0  # tolerancia en px para cajas levemente fuera de borde


def _cfg_get(cfg, key: str, default=_MISSING):
    try:
        return get(cfg, key)
    except Exception:
        if default is _MISSING:
            raise
        return default


def resolve_split(root: Path, split: str) -> tuple[Path | None, Path | None]:
    ann_candidates = [
        root / "annotations" / f"instances_{split}2017.json",
        root / "annotations" / f"instances_{split}.json",
        root / "annotations" / f"{split}.json",
    ]
    img_candidates = [root / f"{split}2017", root / split, root / "images" / split]
    ann = next((p for p in ann_candidates if p.exists()), None)
    img = next((p for p in img_candidates if p.is_dir()), None)
    return ann, img


def check_split(root: Path, split: str, decode_masks: bool = True) -> tuple[int, int]:
    """Valida un split. Devuelve (n_errores_duros, n_avisos)."""
    ann_file, img_dir = resolve_split(root, split)
    if ann_file is None or img_dir is None:
        log.error("split '%s': estructura no encontrada (json o carpeta de imagenes)", split)
        return 1, 0

    coco = COCO(str(ann_file))
    cat_ids = set(coco.getCatIds())

    hard: Counter = Counter()
    warn: Counter = Counter()
    examples: dict[str, list[int]] = {}

    def _flag(bucket: Counter, key: str, ref: int) -> None:
        bucket[key] += 1
        examples.setdefault(key, [])
        if len(examples[key]) < 5:
            examples[key].append(ref)

    # --- imagenes: existencia en disco + imagenes sin anotaciones (negativos) ---
    anns_per_img = Counter(a["image_id"] for a in coco.anns.values())
    for img_id, info in coco.imgs.items():
        if not (img_dir / info["file_name"]).exists():
            _flag(hard, "imagen_ausente", img_id)
        if anns_per_img.get(img_id, 0) == 0:
            _flag(warn, "imagen_sin_anotaciones", img_id)

    # --- anotaciones ---
    for ann_id, ann in coco.anns.items():
        img = coco.imgs[ann["image_id"]]
        W, H = img["width"], img["height"]

        if ann["category_id"] not in cat_ids:
            _flag(hard, "category_id_invalido", ann_id)

        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            _flag(hard, "caja_degenerada", ann_id)
        if ann.get("area", 0) <= 0:
            _flag(hard, "area_no_positiva", ann_id)
        if x < -_EPS or y < -_EPS or x + w > W + _EPS or y + h > H + _EPS:
            _flag(warn, "caja_fuera_de_borde", ann_id)

        if ann.get("iscrowd", 0):
            _flag(warn, "iscrowd", ann_id)

        if decode_masks:
            try:
                if coco.annToMask(ann).sum() == 0:
                    _flag(hard, "mascara_vacia", ann_id)
            except Exception:
                _flag(hard, "mascara_no_decodifica", ann_id)

    n_hard, n_warn = sum(hard.values()), sum(warn.values())
    log.info("split '%s': %d imagenes, %d anotaciones -> %d errores, %d avisos",
             split, len(coco.imgs), len(coco.anns), n_hard, n_warn)
    for key, n in hard.items():
        log.error("  [error] %s: %d (ej. ids %s)", key, n, examples[key])
    for key, n in warn.items():
        log.warning("  [aviso] %s: %d (ej. ids %s)", key, n, examples[key])
    return n_hard, n_warn


def main() -> int:
    parser = argparse.ArgumentParser(description="Valida las anotaciones COCO de CarDD (C1).")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--no-masks", action="store_true", help="No decodifica mascaras (mas rapido).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = Path(_cfg_get(cfg, "data.root"))
    if not root.exists():
        log.error("data.root no existe: %s", root)
        return 1

    splits = args.splits or list(_cfg_get(cfg, "data.splits", DEFAULT_SPLITS))

    total_hard = 0
    for split in splits:
        hard, _ = check_split(root, split, decode_masks=not args.no_masks)
        total_hard += hard

    print("OK - anotaciones limpias" if total_hard == 0 else f"FALLO - {total_hard} errores duros")
    return 0 if total_hard == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())