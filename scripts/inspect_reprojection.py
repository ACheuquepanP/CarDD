"""scripts/inspect_reprojection.py

Dueño: M · diagnóstico (no es parte del pipeline)

Toma UNA imagen del cache de tiles (generado por wbf_sweep.py cache) y dibuja,
sobre la imagen entera real, las cajas de cada tile reproyectadas a coords de
imagen entera — usando exactamente la misma reproyección que el pipeline
(cardd.engine.inference.reproject_boxes). Sirve para ver a ojo si las cajas
caen donde deben o están corridas/mal escaladas (el sospechoso del colapso de
AP en el régimen multi-tile).

Colorea las cajas por tile (para ver de qué tile vino cada una) y dibuja, en
línea punteada, el borde de cada tile (su rectángulo offset..offset+size en la
imagen entera). Si las cajas de un tile caen FUERA de su propio rectángulo de
tile, la reproyección está mal.

Además imprime, por tile: el tamaño que dice el meta (tw,th) y el offset, para
cotejar contra el tamaño real del recorte.

Uso:
    PYTHONPATH=src python scripts/inspect_reprojection.py \\
        --config configs/ablation/custom/base_dcnplus_cascade_ms.yaml \\
        --cache-dir cache/val_tiles_full \\
        --image-id 13 \\
        --split val \\
        --out results/inspect_13.png \\
        --score-thr 0.3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cardd.utils.config import get, load_config
from cardd.engine.inference import reproject_boxes


# paleta simple para distinguir tiles (RGB 0..255)
_COLORS = [
    (255, 0, 0), (0, 200, 0), (0, 0, 255), (255, 165, 0), (200, 0, 200),
    (0, 200, 200), (150, 75, 0), (255, 0, 128), (128, 128, 0),
]


def _load_pil_image(cfg, split: str, image_id: int):
    from PIL import Image
    from cardd.data.coco_dataset import CarDDCocoDataset

    root = Path(get(cfg, "data.root"))
    ds = CarDDCocoDataset(root, split)
    info = ds.coco.imgs[image_id]
    path = ds.img_dir / info["file_name"]
    return Image.open(path).convert("RGB"), info


def main() -> None:
    ap = argparse.ArgumentParser(description="Dibuja cajas reproyectadas de una imagen cacheada.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--image-id", type=int, required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default=None)
    ap.add_argument("--score-thr", type=float, default=0.3, help="solo dibuja cajas con score >= esto")
    args = ap.parse_args()

    from PIL import Image, ImageDraw

    cfg = load_config(args.config)

    cache_file = Path(args.cache_dir) / f"{args.image_id}.pt"
    if not cache_file.exists():
        raise SystemExit(f"no existe {cache_file}; revisá --image-id / --cache-dir")

    cached = torch.load(cache_file, weights_only=False)
    outputs, metas = cached["outputs"], cached["metas"]

    pil, info = _load_pil_image(cfg, args.split, args.image_id)
    W, H = pil.size
    print(f"imagen {args.image_id}: {W}x{H} (segun COCO: {info['width']}x{info['height']})")
    print(f"tiles en cache: {len(outputs)}")

    draw = ImageDraw.Draw(pil)

    for t, (out, meta) in enumerate(zip(outputs, metas)):
        color = _COLORS[t % len(_COLORS)]
        x0, y0 = meta["offset"]
        tw, th = meta["tile_size"]
        ow, oh = meta["orig_size"]
        n_det = out["boxes"].shape[0]
        print(
            f"  tile {t}: offset=({x0},{y0}) tile_size=({tw},{th}) orig_size=({ow},{oh}) "
            f"scale={meta.get('scale', 1.0)} n_det={n_det}"
        )

        # borde del tile (punteado manual: rectángulo simple)
        draw.rectangle([x0, y0, x0 + tw, y0 + th], outline=color, width=1)

        if n_det == 0:
            continue
        boxes = reproject_boxes(out["boxes"], meta["offset"], meta.get("scale", 1.0))
        scores = out["scores"]
        for i in range(n_det):
            if float(scores[i]) < args.score_thr:
                continue
            x1, y1, x2, y2 = (float(v) for v in boxes[i])
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

    out_path = Path(args.out) if args.out else Path(f"results/inspect_{args.image_id}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_path)
    print(f"\nguardado en {out_path}")
    print("Leé la imagen: si las cajas gruesas caen dentro/cerca de su rectángulo de tile")
    print("del mismo color y sobre objetos reales, la reproyección está bien. Si están")
    print("corridas, escaladas o amontonadas en una esquina, ahí está el bug.")


if __name__ == "__main__":
    main()