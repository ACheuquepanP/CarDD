"""tools/prepare_cardd.py

Dueno: Equipo D - Fase: 1 - Contrato: C1 (datos -> M/E)

Prepara y verifica el dataset CarDD en formato COCO. CarDD ya se distribuye en
layout COCO 2017, asi que esta herramienta NO convierte nada: localiza los
ficheros por split, los abre con pycocotools y emite un resumen (conteos + mapa
de categorias) que confirma que el dataset esta listo para `coco_dataset.py`.

Higiene (heredada de P):
  - No hardcodea rutas: `data.root` sale de la config (C4).
  - Las 6 clases se leen de las anotaciones (fuente de verdad), nunca a mano.
  - El mapa `category_id -> label` (0 = fondo) debe ser UNICO entre splits; eso
    es lo unico realmente decisivo de C1 y aqui se verifica.

Uso:
    python tools/prepare_cardd.py --config configs/base.yaml [--out manifest.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pycocotools.coco import COCO

from cardd.utils.config import get, load_config
from cardd.utils.logging import get_logger

log = get_logger("cardd.data.prepare")

DEFAULT_SPLITS = ("train", "val", "test")
_MISSING = object()


def _cfg_get(cfg, key: str, default=_MISSING):
    """Accesor tolerante: usa get() de C4 y cae al default si la clave falta."""
    try:
        return get(cfg, key)
    except Exception:
        if default is _MISSING:
            raise
        return default


def resolve_split(root: Path, split: str) -> tuple[Path | None, Path | None]:
    """Localiza (json_anotaciones, carpeta_imagenes) para un split del layout CarDD-COCO."""
    ann_candidates = [
        root / "annotations" / f"instances_{split}2017.json",
        root / "annotations" / f"instances_{split}.json",
        root / "annotations" / f"{split}.json",
    ]
    img_candidates = [root / f"{split}2017", root / split, root / "images" / split]
    ann = next((p for p in ann_candidates if p.exists()), None)
    img = next((p for p in img_candidates if p.is_dir()), None)
    return ann, img


def build_label_map(coco: COCO) -> dict[int, int]:
    """category_id (COCO) -> label contiguo 1..N. El 0 queda reservado a fondo."""
    cat_ids = sorted(coco.getCatIds())
    return {cid: i + 1 for i, cid in enumerate(cat_ids)}


def inspect_split(root: Path, split: str) -> dict | None:
    ann_file, img_dir = resolve_split(root, split)
    if ann_file is None:
        log.error("split '%s': no encuentro el JSON en %s", split, root / "annotations")
        return None
    if img_dir is None:
        log.error("split '%s': no encuentro la carpeta de imagenes bajo %s", split, root)
        return None

    coco = COCO(str(ann_file))
    cats = coco.loadCats(coco.getCatIds())
    label_map = build_label_map(coco)
    id_to_name = {c["id"]: c["name"] for c in cats}

    summary = {
        "split": split,
        "ann_file": str(ann_file),
        "img_dir": str(img_dir),
        "num_images": len(coco.imgs),
        "num_annotations": len(coco.anns),
        "num_categories": len(cats),
        "label_map": {str(cid): label_map[cid] for cid in sorted(label_map)},
        "id_to_name": {str(cid): id_to_name[cid] for cid in sorted(id_to_name)},
    }
    log.info(
        "split '%s': %d imagenes, %d anotaciones, %d clases",
        split, summary["num_images"], summary["num_annotations"], summary["num_categories"],
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Verifica y resume el dataset CarDD-COCO (C1).")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--splits", nargs="*", default=None, help="Override de splits a inspeccionar.")
    parser.add_argument("--out", default=None, help="Ruta opcional para escribir el manifiesto JSON.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = Path(_cfg_get(cfg, "data.root"))
    if not root.exists():
        log.error("data.root no existe: %s (el dataset esta en .gitignore; descargalo ahi)", root)
        return 1

    splits = args.splits or list(_cfg_get(cfg, "data.splits", DEFAULT_SPLITS))
    expected = _cfg_get(cfg, "data.num_classes", None)

    ok = True
    summaries: list[dict] = []
    for split in splits:
        s = inspect_split(root, split)
        if s is None:
            ok = False
            continue
        if expected is not None and s["num_categories"] != expected:
            log.warning("split '%s': %d clases, esperaba %d (data.num_classes)",
                        split, s["num_categories"], expected)
        summaries.append(s)

    # C1: el mapa category_id -> label tiene que ser identico en todos los splits.
    distinct_maps = {json.dumps(s["label_map"], sort_keys=True) for s in summaries}
    if len(distinct_maps) > 1:
        log.error("el mapa category_id->label DIFIERE entre splits; C1 exige uno solo")
        ok = False
    elif summaries:
        log.info("mapa category_id->label (0=fondo): %s", summaries[0]["label_map"])
        log.info("clases: %s", summaries[0]["id_to_name"])

    if args.out and summaries:
        Path(args.out).write_text(
            json.dumps({"root": str(root), "splits": summaries}, indent=2, ensure_ascii=False)
        )
        log.info("manifiesto escrito en %s", args.out)

    print("OK - dataset preparado" if ok else "FALLO - revisa los errores arriba")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())