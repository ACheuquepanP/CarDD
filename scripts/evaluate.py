"""scripts/evaluate.py — evalúa un checkpoint sobre un split → C5.

dueño: M·E · fase: 4 · contrato: C5

Punta a punta, conecta las dos mitades:
    build_eval_model (M) → run_inference (M) → evaluate_from_dataset (E)

NOTA de integración: E ya tiene su mitad de este script (formato C5 a stdout +
JSON, modo árbitro). Este archivo es el **punto de encuentro M·E**: M aporta la
costura de inferencia; reconciliar el reporte/flags con la versión de E para no
duplicar. La frontera estable es C2→C5: M entrega C2, E calcula C5.

Uso:
    PYTHONPATH=src python scripts/evaluate.py \
        --config configs/ablation/a0_baseline.yaml \
        --weights outputs/a0_baseline/last.pth --split val

    # SMOKE end-to-end (tras `train.py --overfit 8`): AP≈1 sobre esas 8 imágenes
    PYTHONPATH=src python scripts/evaluate.py \
        --config configs/ablation/a0_baseline.yaml \
        --weights outputs/a0_baseline/last.pth --split train --overfit 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cardd.utils.config import load_config, get
from cardd.utils.seed import set_seed
from cardd.utils.logging import get_logger
from cardd.data.build import build_dataloader
from cardd.engine.inference import build_eval_model, run_inference
from cardd.evaluation.coco_eval import evaluate_from_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description="Evalúa un checkpoint sobre un split → C5.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--overfit", type=int, default=0,
                    help="smoke end-to-end: evalúa solo las primeras N imágenes (AP≈1 esperado)")
    ap.add_argument("--out", default=None, help="ruta JSON de salida para el dict C5")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(get(cfg, "seed", 0)))  # mismo seed que train → mismas N imágenes en --overfit
    log = get_logger("cardd.evaluate")
    device = get(cfg, "train.device", "cuda")

    loader = build_dataloader(cfg, args.split)
    model = build_eval_model(cfg, weights=args.weights, device=device)
    preds, image_ids = run_inference(model, loader, device=device,
                                     max_images=args.overfit or None)

    # E: C2 → C5. El GT sale del JSON COCO del dataset (ds.coco); image_ids
    # restringe la evaluación a las imágenes realmente inferidas.
    c5 = evaluate_from_dataset(loader.dataset, preds, image_ids)

    print(json.dumps(c5, indent=2, default=float))
    out = args.out or str(
        Path(get(cfg, "train.output_dir", "outputs/run")) / f"metrics_{args.split}.json"
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(c5, indent=2, default=float))
    log.info("C5 → %s · mask_AP=%s", out, c5.get("mask_AP"))


if __name__ == "__main__":
    main()