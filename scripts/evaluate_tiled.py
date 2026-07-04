"""CarDD · scripts/evaluate_tiled.py

Dueño: Equipo M - Fase: val/test con tiling - Contrato: C2 (produce, vía WBF) -> C5 (consume evaluación de E)

Analogo de scripts/evaluate.py pero para el camino de val/test con TILING:
usa el feed de D (cardd.data.tiling) en vez del loader C1 normal, infiere por
tile, reproyecta y fusiona con WBF (cardd.engine.inference), y evalúa igual que
siempre contra la GT ENTERA (ds.coco) — la frontera con E no cambia, solo cómo
se llega a la predicción de imagen completa.

Es un archivo NUEVO: scripts/evaluate.py no se modifica, así que el camino sin
tiling sigue funcionando exactamente igual (retrocompatible).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cardd.evaluation.coco_eval import evaluate
from cardd.utils.config import get, load_config
from cardd.utils.logging import get_logger
from cardd.utils.seed import set_seed


def _pick_device(cfg) -> str:
    dev = get(cfg, "eval.device")
    if dev:
        return dev
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def collect_coco_results_tiled(cfg, split: str, weights: str, logger):
    """Igual que collect_coco_results de evaluate.py pero con tiling+WBF.

    El dataset de tiles (D) expone cat_id_to_label igual que CarDDCocoDataset,
    así que run_tiled_inference_coco_results (M) lo consume sin adaptador.
    """
    from cardd.data.tiling import build_tile_dataloader, CarDDTileDataset
    from cardd.engine.inference import build_eval_model, run_tiled_inference_coco_results
    from cardd.utils.config import get as _get

    device = _pick_device(cfg)
    loader = build_tile_dataloader(cfg, split)
    dataset: CarDDTileDataset = loader.dataset  # expone .coco / .cat_id_to_label / .label_to_name
    model = build_eval_model(cfg, weights, device)
    model.eval()

    coco_results, image_ids = run_tiled_inference_coco_results(
        model, loader, dataset, cfg, device=device, amp=bool(_get(cfg, "eval.amp", True))
    )
    logger.info(
        "inferencia tileada '%s': %d imagenes, %d detecciones (post-WBF)",
        split, len(image_ids), len(coco_results),
    )
    return dataset, coco_results, image_ids


def _format_c5(c5: dict) -> str:
    head = (
        f"mask_AP={c5['mask_AP']:.4f}  box_AP={c5['box_AP']:.4f}  "
        f"AP50={c5['AP50']:.4f}  AP75={c5['AP75']:.4f}"
    )
    sc = c5["per_scale"]
    scale = f"APS={sc['APS']:.4f}  APM={sc['APM']:.4f}  APL={sc['APL']:.4f}"
    per_cls = "\n".join(
        f"    {name:<16} mask_AP={v['mask_AP']:.4f}  box_AP={v['box_AP']:.4f}"
        for name, v in c5["per_class"].items()
    )
    return f"{head}\n  {scale}\n  per_class:\n{per_cls}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Evalua un checkpoint con tiling+WBF y emite C5.")
    ap.add_argument("--config", required=True, help="ruta a la config YAML")
    ap.add_argument("--weights", required=True, help="checkpoint a evaluar")
    ap.add_argument("--split", default="test", help="split a evaluar (default: test)")
    ap.add_argument("--out", default=None, help="ruta del JSON C5 (default: output.dir)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(get(cfg, "seed", 0))
    logger = get_logger("cardd.evaluate_tiled")

    dataset, coco_results, image_ids = collect_coco_results_tiled(cfg, args.split, args.weights, logger)
    cat_id_to_name = {
        cid: dataset.label_to_name[label] for cid, label in dataset.cat_id_to_label.items()
    }
    c5 = evaluate(dataset.coco, coco_results, cat_id_to_name, image_ids=image_ids)

    print(_format_c5(c5))

    out = Path(args.out) if args.out else Path(get(cfg, "output.dir", "outputs")) / f"eval_tiled_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(c5, indent=2, ensure_ascii=False))
    logger.info("C5 (tiled) guardado en %s", out)


if __name__ == "__main__":
    main()