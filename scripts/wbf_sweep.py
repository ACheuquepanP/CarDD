"""scripts/wbf_sweep.py

Dueño: M · Fase: val con tiling · harness de barrido de WBF (HANDOFF §4)

Dos subcomandos, para no re-correr el forward del modelo por cada combinación
de parámetros de fusión (caro en GPU):

  cache : corre el modelo UNA vez sobre todos los tiles de un split y cachea a
          disco (un archivo por imagen) las predicciones crudas por tile + su
          meta. Memory-safe: nunca tiene el split completo en RAM a la vez.
  grid  : barre una grilla de eval.wbf.* sobre ese cache, sin volver a tocar la
          GPU (puro post-proceso: reproyección + WBF + serialización + C5), y
          escribe una tabla param -> métrica en CSV.

Uso:
    python scripts/wbf_sweep.py cache --config CFG --weights best.pth --split val --cache-dir cache/val_tiles
    python scripts/wbf_sweep.py grid  --config CFG --split val --cache-dir cache/val_tiles \\
        --out results/wbf_grid.csv \\
        --iou-thr 0.45,0.55,0.65 --skip-box-thr 0.0,0.1 \\
        --conf-type avg,max --mask-policy union,top1,weighted_avg

La tabla resultante es lo que M entrega a C — C elige la fila óptima, no M
(HANDOFF §4: "Tú entregás la tabla param→métrica; C fija el valor").
"""
from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import torch

from cardd.utils.config import get, load_config
from cardd.utils.logging import get_logger


def _pick_device(cfg) -> str:
    dev = get(cfg, "eval.device")
    if dev:
        return dev
    return "cuda" if torch.cuda.is_available() else "cpu"


def _parse_list(s: str, cast):
    return [cast(x) for x in s.split(",")]


def cmd_cache(args) -> None:
    """Un forward pass por tile, UNA vez. Un archivo por imagen (nunca todo el
    split en RAM simultáneamente — la lección de OOM del handoff aplica igual
    acá: el pico es una imagen, no el split).

    El forward por imagen se hace en sub-batches de ``eval.tile_batch_size``
    (por default hereda ``data.tiling.n`` — ver
    ``cardd.engine.inference.resolve_tile_batch_size``): mismo resultado que
    meter todos los tiles juntos, menos pico de VRAM a costa de algo de
    velocidad.
    """
    from cardd.data.tiling import build_tile_dataloader
    from cardd.engine.inference import build_eval_model, resolve_tile_batch_size, run_tiles_chunked

    cfg = load_config(args.config)
    # override opcional desde CLI (no toca el yaml) — util para el test passthrough
    # (n=1, tile_frac=1.0 -> un solo "tile" = imagen entera, reproyeccion identidad).
    if args.n is not None:
        cfg.setdefault("data", {}).setdefault("tiling", {})["n"] = int(args.n)
    if args.tile_frac is not None:
        cfg.setdefault("data", {}).setdefault("tiling", {})["tile_frac"] = float(args.tile_frac)
    logger = get_logger("cardd.wbf_sweep.cache")
    device = _pick_device(cfg)

    loader = build_tile_dataloader(cfg, args.split)
    model = build_eval_model(cfg, args.weights, device)
    model.eval()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    device_t = torch.device(device)
    use_amp = args.amp and device_t.type == "cuda"
    chunk_size = resolve_tile_batch_size(cfg)
    logger.info("tile_batch_size efectivo: %s", chunk_size if chunk_size else "todos los tiles juntos")
    is_cuda = device_t.type == "cuda"
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device_t)

    n_images = 0
    with torch.no_grad():
        for tiles, metas in loader:  # un batch = todos los tiles de UNA imagen (collate de D)
            outputs = run_tiles_chunked(model, list(tiles), chunk_size, device_t, use_amp)
            outputs_cpu = [{k: v.detach().cpu() for k, v in o.items()} for o in outputs]
            img_id = int(metas[0]["image_id"])
            torch.save({"outputs": outputs_cpu, "metas": list(metas)}, cache_dir / f"{img_id}.pt")
            n_images += 1
            if is_cuda:
                peak_now = torch.cuda.max_memory_allocated(device_t) / 1e9
                logger.info(
                    "imagen %d cacheada (image_id=%d, %d tiles) | pico VRAM hasta ahora: %.2f GB",
                    n_images, img_id, len(tiles), peak_now,
                )
            else:
                logger.info("imagen %d cacheada (image_id=%d, %d tiles)", n_images, img_id, len(tiles))
            del outputs, outputs_cpu  # se libera antes del próximo batch
            if args.max_images is not None and n_images >= args.max_images:
                logger.info("corte por --max-images=%d (smoke)", args.max_images)
                break

    logger.info("cache lista: %d imagenes en %s", n_images, cache_dir)
    if is_cuda:
        peak_alloc = torch.cuda.max_memory_allocated(device_t) / 1e9
        peak_reserved = torch.cuda.max_memory_reserved(device_t) / 1e9
        logger.info(
            "VRAM pico -> allocated: %.2f GB | reserved (con fragmentacion): %.2f GB",
            peak_alloc, peak_reserved,
        )


def cmd_grid(args) -> None:
    """Barre la grilla de eval.wbf.* sobre el cache de `cache`. Sin GPU: solo
    reproyección + WBF + serialización + C5, por cada combinación.

    IMPORTANTE — orden del loop invertido a propósito: se recorre el cache
    en disco UNA sola vez (imagen por imagen); para cada imagen ya cargada
    en memoria se aplican TODAS las combinaciones de la grilla antes de
    pasar a la próxima imagen. Evita releer 42GB de disco una vez por
    combinación (lo que sí pasaría si el loop fuera "por combinación, leer
    todo el cache") — con 36 combinaciones eso sería 36x la lectura de disco,
    el cuello de botella real. Así, el disco se lee 1x; lo que se repite 36
    veces es solo la fusión en memoria (barata, CPU).

    El mapa de clases (`cat_id_to_label` / `label_to_name`) vive en
    `CarDDCocoDataset` (C1), NO en `CarDDTileDataset` (que solo expone
    `.coco`/`.ann_file`/`.img_dir`/`.index` — geometría de tiles, no clases).
    Se instancia aparte, es liviano: solo lee el JSON, no carga imágenes.
    """
    from cardd.data.coco_dataset import CarDDCocoDataset
    from cardd.engine.inference import fuse_tile_predictions
    from cardd.evaluation.coco_eval import evaluate, predictions_to_coco_results

    cfg = load_config(args.config)
    logger = get_logger("cardd.wbf_sweep.grid")

    root = Path(get(cfg, "data.root"))
    dataset = CarDDCocoDataset(root, args.split)
    label_to_cat_id = {label: cid for cid, label in dataset.cat_id_to_label.items()}
    cat_id_to_name = {
        cid: dataset.label_to_name[label] for cid, label in dataset.cat_id_to_label.items()
    }

    cache_dir = Path(args.cache_dir)
    cache_files = sorted(cache_dir.glob("*.pt"))
    if not cache_files:
        raise SystemExit(f"cache vacio en {cache_dir}; correr 'cache' primero.")

    grid = list(
        itertools.product(
            _parse_list(args.iou_thr, float),
            _parse_list(args.skip_box_thr, float),
            _parse_list(args.conf_type, str),
            _parse_list(args.mask_policy, str),
        )
    )
    logger.info(
        "barrido: %d combinaciones sobre %d imagenes cacheadas (disco leido 1 vez)",
        len(grid), len(cache_files),
    )

    # un acumulador de resultados COCO por combinación (livianos: RLE, no denso)
    coco_results_per_combo: list[list[dict]] = [[] for _ in grid]
    image_ids: list[int] = []

    for i, f in enumerate(cache_files):
        cached = torch.load(f, weights_only=False)  # única lectura de esta imagen
        outputs, metas = cached["outputs"], cached["metas"]
        img_id = int(metas[0]["image_id"])
        image_ids.append(img_id)

        for combo_idx, (iou_thr, skip_box_thr, conf_type, mask_policy) in enumerate(grid):
            wbf_cfg = {
                "iou_thr": iou_thr,
                "skip_box_thr": skip_box_thr,
                "weights": None,
                "conf_type": conf_type,
                "mask_policy": mask_policy,
            }
            fused = fuse_tile_predictions(outputs, metas, wbf_cfg)
            coco_results_per_combo[combo_idx].extend(
                predictions_to_coco_results([fused], [img_id], label_to_cat_id)
            )
            del fused

        del cached, outputs, metas  # esta imagen se libera antes de pasar a la próxima
        if (i + 1) % 50 == 0 or (i + 1) == len(cache_files):
            logger.info("procesadas %d/%d imagenes (todas las combinaciones)", i + 1, len(cache_files))

    rows = []
    for (iou_thr, skip_box_thr, conf_type, mask_policy), coco_results in zip(grid, coco_results_per_combo):
        c5 = evaluate(dataset.coco, coco_results, cat_id_to_name, image_ids=image_ids, with_boundary=False)
        row = {
            "iou_thr": iou_thr,
            "skip_box_thr": skip_box_thr,
            "conf_type": conf_type,
            "mask_policy": mask_policy,
            "mask_AP": c5["mask_AP"],
            "box_AP": c5["box_AP"],
            "AP50": c5["AP50"],
            "AP75": c5["AP75"],
        }
        rows.append(row)
        logger.info(
            "iou_thr=%s skip_box_thr=%s conf_type=%s mask_policy=%s -> mask_AP=%.4f",
            iou_thr, skip_box_thr, conf_type, mask_policy, row["mask_AP"],
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("tabla param->metrica guardada en %s (%d filas)", out, len(rows))


def cmd_ablation(args) -> None:
    """Ablation check: sobre el MISMO cache, evalúa dos caminos y los compara —
    (A) sin-fusión (detecciones crudas reproyectadas, con duplicados de costura)
    (B) WBF (fusión, con los parámetros dados).

    Aísla si el AP bajo viene de WBF o de algo previo (reproyección/geometría o
    las detecciones por tile). No toca GPU: solo lee el cache. Lee el disco 1 vez.
    """
    from cardd.data.coco_dataset import CarDDCocoDataset
    from cardd.engine.inference import (
        concat_tile_predictions_no_fusion,
        fuse_tile_predictions,
    )
    from cardd.evaluation.coco_eval import evaluate, predictions_to_coco_results

    cfg = load_config(args.config)
    logger = get_logger("cardd.wbf_sweep.ablation")

    root = Path(get(cfg, "data.root"))
    dataset = CarDDCocoDataset(root, args.split)
    label_to_cat_id = {label: cid for cid, label in dataset.cat_id_to_label.items()}
    cat_id_to_name = {
        cid: dataset.label_to_name[label] for cid, label in dataset.cat_id_to_label.items()
    }

    cache_dir = Path(args.cache_dir)
    cache_files = sorted(cache_dir.glob("*.pt"))
    if not cache_files:
        raise SystemExit(f"cache vacio en {cache_dir}; correr 'cache' primero.")

    wbf_cfg = {
        "iou_thr": float(args.iou_thr),
        "skip_box_thr": float(args.skip_box_thr),
        "weights": None,
        "conf_type": args.conf_type,
        "mask_policy": args.mask_policy,
    }
    logger.info("ablation sobre %d imagenes: (A) sin-fusion vs (B) WBF %s", len(cache_files), wbf_cfg)

    results_nofuse: list[dict] = []
    results_wbf: list[dict] = []
    image_ids: list[int] = []

    for i, f in enumerate(cache_files):
        cached = torch.load(f, weights_only=False)
        outputs, metas = cached["outputs"], cached["metas"]
        img_id = int(metas[0]["image_id"])
        image_ids.append(img_id)

        nofuse = concat_tile_predictions_no_fusion(outputs, metas)
        results_nofuse.extend(predictions_to_coco_results([nofuse], [img_id], label_to_cat_id))

        fused = fuse_tile_predictions(outputs, metas, wbf_cfg)
        results_wbf.extend(predictions_to_coco_results([fused], [img_id], label_to_cat_id))

        del cached, outputs, metas, nofuse, fused
        if (i + 1) % 50 == 0 or (i + 1) == len(cache_files):
            logger.info("procesadas %d/%d imagenes", i + 1, len(cache_files))

    c5_nofuse = evaluate(dataset.coco, results_nofuse, cat_id_to_name, image_ids=image_ids, with_boundary=False)
    c5_wbf = evaluate(dataset.coco, results_wbf, cat_id_to_name, image_ids=image_ids, with_boundary=False)

    logger.info("=========== ABLATION ===========")
    logger.info(
        "(A) SIN FUSION -> mask_AP=%.4f  box_AP=%.4f  AP50=%.4f  AP75=%.4f",
        c5_nofuse["mask_AP"], c5_nofuse["box_AP"], c5_nofuse["AP50"], c5_nofuse["AP75"],
    )
    logger.info(
        "(B) CON WBF    -> mask_AP=%.4f  box_AP=%.4f  AP50=%.4f  AP75=%.4f",
        c5_wbf["mask_AP"], c5_wbf["box_AP"], c5_wbf["AP50"], c5_wbf["AP75"],
    )
    logger.info("================================")


def main() -> None:
    ap = argparse.ArgumentParser(description="Harness de barrido de WBF (M construye, C elige el valor).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_cache = sub.add_parser("cache", help="corre el modelo una vez y cachea predicciones por tile")
    p_cache.add_argument("--config", required=True)
    p_cache.add_argument("--weights", required=True)
    p_cache.add_argument("--split", default="val")
    p_cache.add_argument("--cache-dir", required=True)
    p_cache.add_argument(
        "--no-amp", dest="amp", action="store_false", default=True,
        help="desactiva autocast (float32 puro) — útil para descartar jitter numérico del chunking",
    )
    p_cache.add_argument("--max-images", type=int, default=None, help="corta tras N imagenes (smoke)")
    p_cache.add_argument("--n", type=int, default=None, help="override de data.tiling.n (sin tocar yaml)")
    p_cache.add_argument("--tile-frac", type=float, default=None, help="override de data.tiling.tile_frac")
    p_cache.set_defaults(func=cmd_cache)

    p_grid = sub.add_parser("grid", help="barre parametros de fusion sobre el cache (sin GPU)")
    p_grid.add_argument("--config", required=True)
    p_grid.add_argument("--split", default="val")
    p_grid.add_argument("--cache-dir", required=True)
    p_grid.add_argument("--out", default="results/wbf_grid.csv")
    p_grid.add_argument("--iou-thr", default="0.45,0.55,0.65")
    p_grid.add_argument("--skip-box-thr", default="0.0,0.1")
    p_grid.add_argument("--conf-type", default="avg,max")
    p_grid.add_argument("--mask-policy", default="union,top1,weighted_avg")
    p_grid.set_defaults(func=cmd_grid)

    p_abl = sub.add_parser("ablation", help="compara sin-fusion vs WBF sobre el mismo cache")
    p_abl.add_argument("--config", required=True)
    p_abl.add_argument("--split", default="val")
    p_abl.add_argument("--cache-dir", required=True)
    p_abl.add_argument("--iou-thr", default="0.55")
    p_abl.add_argument("--skip-box-thr", default="0.0")
    p_abl.add_argument("--conf-type", default="avg")
    p_abl.add_argument("--mask-policy", default="union")
    p_abl.set_defaults(func=cmd_ablation)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()