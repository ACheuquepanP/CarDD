"""scripts/train.py — entrena un detector CarDD desde un config.

dueño: M · fase: 4

Uso:
    # entrenamiento normal de la línea base
    PYTHONPATH=src python scripts/train.py --config configs/ablation/a0_baseline.yaml

    # SMOKE TEST (hito paso 13): loss→0 sobreajustando N imágenes
    PYTHONPATH=src python scripts/train.py --config configs/ablation/a0_baseline.yaml --overfit 8
"""

from __future__ import annotations

import argparse

import torch

from cardd.utils.config import load_config, get
from cardd.utils.seed import set_seed
from cardd.utils.logging import get_logger
from cardd.data.build import build_dataloader
from cardd.models.detectors.mask_rcnn import build_detector
from cardd.engine.trainer import Trainer
from cardd.engine.inference import _load_weights


def _materialize_overfit(loader, n: int, log):
    """Junta los primeros >= n imágenes en una lista de batches en memoria.

    Devolver una lista (no un DataLoader) permite repetir el mismo subconjunto
    muchas epochs sin volver a leer disco. El Trainer solo necesita un iterable
    con ``len()``, así que una lista de ``(tuple(images), tuple(targets))`` sirve.
    """
    batches, count = [], 0
    for images, targets in loader:
        batches.append((images, targets))
        count += len(images)
        if count >= n:
            break
    log.info("overfit: %d imágenes en %d batches (se repiten cada epoch)", count, len(batches))
    return batches


def _tune_cfg_for_overfit(cfg, n_batches: int, target_steps: int, log) -> None:
    """Sube las epochs para alcanzar ~target_steps y fija lr constante."""
    accum = max(1, int(get(cfg, "train.grad_accum_steps", 1)))
    steps_per_epoch = max(1, -(-n_batches // accum))  # ceil
    epochs = max(int(get(cfg, "train.max_epochs", 1)), -(-target_steps // steps_per_epoch))
    cfg.setdefault("train", {})["max_epochs"] = epochs
    cfg.setdefault("optim", {})["scheduler"] = "none"  # sin coseno: el loss baja a 0 más rápido
    log.info("overfit: %d epochs (~%d opt-steps) · scheduler=none", epochs, epochs * steps_per_epoch)


def main() -> None:
    ap = argparse.ArgumentParser(description="Entrena un detector CarDD desde un config.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--overfit", type=int, default=0,
                    help="smoke loss→0: entrena sobre N imágenes (hito paso 13)")
    ap.add_argument("--resume", default=None, help="checkpoint del que cargar pesos antes de entrenar")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(get(cfg, "seed", 0)))
    log = get_logger("cardd.train")

    train_loader = build_dataloader(cfg, "train")
    if args.overfit:
        train_loader = _materialize_overfit(train_loader, args.overfit, log)
        _tune_cfg_for_overfit(cfg, len(train_loader), target_steps=300, log=log)

    model = build_detector(cfg)
    if args.resume:
        device = torch.device(get(cfg, "train.device", "cuda"))
        _load_weights(model, args.resume, device)
        log.info("pesos cargados desde %s", args.resume)

    ckpt = Trainer(cfg, model, train_loader).train()
    log.info("listo · checkpoint final: %s", ckpt)


if __name__ == "__main__":
    main()