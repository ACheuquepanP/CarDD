#!/usr/bin/env python3
"""Materializa splits de debug tileados a partir de splits COCO existentes.

Cada imagen LxM se corta en 9 tiles de (L/2 x M/2) con solapa (grilla 3x3,
starts en [0, L/4, L/2] x [0, M/4, M/2]). Ciego a los objetos: TODOS los tiles
se emiten, tengan o no anotaciones. Las cajas y mascaras se RECALCULAN por tile
para los objetos que quedan cortados (clip a la ventana + reproyeccion a coords
del tile). Salida = un COCO nuevo por split, apto para check_annotations.py /
analyze_dataset.py igual que los splits originales.

Uso:
    python tools/make_tiled_split.py --config configs/base.yaml --splits val test
    python tools/make_tiled_split.py --root data/CarDD_COCO --splits val test
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

log = logging.getLogger("cardd.data.make_tiled")

TILE_FRAC = 0.5          # lado del tile = mitad del lado de la imagen
N = 3                    # grilla n x n  ->  n^2 = 9 tiles
MIN_POLY_AREA = 4.0      # descarta esquirlas de mascara < 4 px^2 tras el clip


def tile_windows(W: int, H: int) -> list[tuple[int, int, int, int]]:
    """Devuelve las 9 ventanas (x0, y0, tw, th). Determinista, cubre 0..W/0..H."""
    tw, th = round(W * TILE_FRAC), round(H * TILE_FRAC)
    xs = sorted({int(round(v)) for v in np.linspace(0, W - tw, N)})
    ys = sorted({int(round(v)) for v in np.linspace(0, H - th, N)})
    return [(x0, y0, tw, th) for y0 in ys for x0 in xs]


def ann_to_mask(ann: dict, W: int, H: int) -> np.ndarray:
    """Decodifica una anotacion COCO (polygon o RLE) a mascara binaria HxW."""
    seg = ann["segmentation"]
    if isinstance(seg, list):                       # polygon(s)
        rles = mask_utils.frPyObjects(seg, H, W)
        rle = mask_utils.merge(rles)
    elif isinstance(seg["counts"], list):           # uncompressed RLE
        rle = mask_utils.frPyObjects(seg, H, W)
    else:                                           # compressed RLE
        rle = seg
    return mask_utils.decode(rle).astype(np.uint8)


def mask_to_polygons(m: np.ndarray) -> list[list[float]]:
    """Reextrae poligonos COCO de una mascara binaria (mantiene el formato de origen)."""
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        c = c.reshape(-1, 2)
        if c.shape[0] < 3 or cv2.contourArea(c) < MIN_POLY_AREA:
            continue
        polys.append(c.astype(float).flatten().tolist())
    return polys


def clip_annotation(ann: dict, W: int, H: int, win: tuple[int, int, int, int],
                    rle: bool = False):
    """Clipea una anotacion a la ventana del tile. Devuelve dict COCO nuevo o None."""
    x0, y0, tw, th = win
    m = ann_to_mask(ann, W, H)[y0:y0 + th, x0:x0 + tw]   # crop a coords del tile
    if m.sum() == 0:
        return None                                       # objeto fuera del tile
    if rle:                                               # mascara bit-exacta
        enc = mask_utils.encode(np.asfortranarray(m))
        enc["counts"] = enc["counts"].decode("ascii")
        seg = enc
    else:                                                 # polygon (matchea la fuente)
        seg = mask_to_polygons(m)
        if not seg:
            return None
    ys, xs = np.where(m)
    bx, by = int(xs.min()), int(ys.min())
    bw, bh = int(xs.max() - bx + 1), int(ys.max() - by + 1)
    return {
        "segmentation": seg,
        "bbox": [bx, by, bw, bh],           # tight box recalculado, coords del tile
        "area": int(m.sum()),               # area = pixeles de la mascara clipeada
        "category_id": ann["category_id"],
        "iscrowd": ann.get("iscrowd", 0),
    }


def _find_ann(root: Path, split: str) -> Path:
    """Localiza annotations/instances_{split}*.json (tolera el sufijo '2017')."""
    cands = sorted((root / "annotations").glob(f"instances_{split}*.json"))
    cands = [c for c in cands if "_tiled" not in c.stem]     # ignora salidas previas
    if not cands:
        raise SystemExit(f"no encontre annotations/instances_{split}*.json en {root}")
    return cands[0]


def _find_img_dir(root: Path, coco: dict) -> Path:
    """Ubica la carpeta que contiene las imagenes, buscando el primer file_name."""
    base = Path(coco["images"][0]["file_name"]).name
    hit = next((p for p in root.rglob(base) if "_tiled" not in str(p)), None)
    if hit is None:
        raise SystemExit(f"no encontre la imagen '{base}' bajo {root}")
    return hit.parent


def process_split(root: Path, split: str, out_suffix: str = "_tiled", rle: bool = False) -> None:
    ann_path = _find_ann(root, split)
    coco = json.loads(ann_path.read_text())
    img_dir = _find_img_dir(root, coco)

    # nombres de salida espejando la fuente:  instances_val2017.json -> instances_val2017_tiled.json
    out_ann_path = ann_path.with_name(f"{ann_path.stem}{out_suffix}.json")
    out_img_dir = img_dir.with_name(f"{img_dir.name}{out_suffix}")
    out_img_dir.mkdir(parents=True, exist_ok=True)
    log.info("split '%s': ann=%s  imgs=%s/", split, ann_path.name, img_dir.name)
    anns_by_img: dict[int, list] = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    out_images, out_anns = [], []
    next_img_id, next_ann_id = 1, 1

    for img in coco["images"]:
        W, H = img["width"], img["height"]
        pil = Image.open(img_dir / img["file_name"]).convert("RGB")
        stem = Path(img["file_name"]).stem
        parent_anns = anns_by_img.get(img["id"], [])

        for k, win in enumerate(tile_windows(W, H)):
            x0, y0, tw, th = win
            r, c = divmod(k, N)
            fname = f"{stem}__t{r}{c}.jpg"
            pil.crop((x0, y0, x0 + tw, y0 + th)).save(out_img_dir / fname, quality=95)
            out_images.append({
                "id": next_img_id, "file_name": fname, "width": tw, "height": th,
                "parent_image_id": img["id"], "tile_index": k, "offset": [x0, y0],
            })
            for a in parent_anns:
                clipped = clip_annotation(a, W, H, win, rle=rle)
                if clipped is None:
                    continue
                clipped.update({"id": next_ann_id, "image_id": next_img_id})
                out_anns.append(clipped)
                next_ann_id += 1
            next_img_id += 1

    out = {
        "images": out_images,
        "annotations": out_anns,
        "categories": coco["categories"],
    }
    out_ann_path.write_text(json.dumps(out))
    log.info("-> %s + %s/: %d imagenes (tiles), %d anotaciones",
             out_ann_path.name, out_img_dir.name, len(out_images), len(out_anns))


def _root_from_config(cfg_path: Path) -> Path:
    import re
    for line in cfg_path.read_text().splitlines():
        m = re.search(r"root:\s*(\S+)", line)
        if m:
            return Path(m.group(1))
    raise SystemExit(f"no encontre 'root:' en {cfg_path}")


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    p.add_argument("--root")
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    p.add_argument("--rle", action="store_true", help="mascaras RLE bit-exactas (default: polygon)")
    args = p.parse_args()

    if args.root:
        root = Path(args.root)
    elif args.config:
        root = _root_from_config(Path(args.config))
    else:
        raise SystemExit("pasa --root o --config")

    for split in args.splits:
        process_split(root, split, rle=args.rle)
    log.info("OK - splits tileados generados")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())