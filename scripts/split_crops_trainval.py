#!/usr/bin/env python3
"""CarDD · scripts/split_crops_trainval.py

Prueba de hipótesis (Equipo C, coordinado con D): parte el único
instances_train2017.json de CarDD_COCO_crop en train/val internos, para poder
validar el checkpoint rápido sin tocar el val/test reales.

- Split por IMAGEN (no por anotación), 80/20, semilla fija -> reproducible.
- Respeta el layout que espera resolve_split(): crea instances_val2017.json +
  instances_train2017.json (el train reducido) y carpetas val2017/ + una carpeta
  train nueva, con SYMLINKS a las imágenes originales (no duplica en disco).
- NO modifica el JSON original: escribe a un root de salida aparte.

Uso:
    python scripts/split_crops_trainval.py \
        --src data/CarDD_COCO_crop \
        --dst data/CarDD_COCO_crop_split \
        --val-frac 0.2 --seed 42
Después, en la config: data.root: data/CarDD_COCO_crop_split
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--src-split", default="train2017",
                    help="nombre del split fuente (carpeta img + instances_<>.json)")
    args = ap.parse_args()

    src_json = args.src / "annotations" / f"instances_{args.src_split}.json"
    src_imgs = args.src / args.src_split
    if not src_json.exists():
        raise SystemExit(f"no existe {src_json}")
    if not src_imgs.exists():
        raise SystemExit(f"no existe la carpeta de imágenes {src_imgs}")

    coco = json.load(open(src_json))
    images = coco["images"]
    anns = coco["annotations"]

    # --- split por imagen, semilla fija ---
    rng = random.Random(args.seed)
    ids = [im["id"] for im in images]
    rng.shuffle(ids)
    n_val = int(len(ids) * args.val_frac)
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])
    print(f"total imgs: {len(ids)} -> train {len(train_ids)} / val {len(val_ids)}")

    def subset(split_ids):
        imgs = [im for im in images if im["id"] in split_ids]
        ans = [a for a in anns if a["image_id"] in split_ids]
        return {**{k: v for k, v in coco.items() if k not in ("images", "annotations")},
                "images": imgs, "annotations": ans}

    (args.dst / "annotations").mkdir(parents=True, exist_ok=True)

    for split_name, split_ids in (("train2017", train_ids), ("val2017", val_ids)):
        # JSON
        out_json = args.dst / "annotations" / f"instances_{split_name}.json"
        sub = subset(split_ids)
        json.dump(sub, open(out_json, "w"))
        # carpeta de imágenes con symlinks (no duplica)
        out_dir = args.dst / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for im in sub["images"]:
            fn = im["file_name"]
            src_f = (src_imgs / fn).resolve()
            dst_f = out_dir / fn
            if dst_f.exists() or dst_f.is_symlink():
                dst_f.unlink()
            os.symlink(src_f, dst_f)
            n += 1
        print(f"  {split_name}: {len(sub['images'])} imgs, {len(sub['annotations'])} anns, {n} symlinks -> {out_dir}")

    print(f"\nlisto. usar en la config:  data.root: {args.dst}")


if __name__ == "__main__":
    main()