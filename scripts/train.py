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
from torch.utils.data import DataLoader, Subset

from cardd.utils.config import load_config, get
from cardd.utils.seed import set_seed
from cardd.utils.logging import get_logger
from cardd.data.build import build_dataloader
from cardd.models.detectors.mask_rcnn import build_detector
from cardd.engine.trainer import Trainer
from cardd.engine.inference import _load_weights


def _force_eval_transforms(cfg, log) -> None:
    """Overfit: usar los transforms de eval (sin augmentation) para que el modelo
    pueda MEMORIZAR las imágenes y para que train/eval vean el mismo preprocesamiento.
    Guardado contra estructuras de config distintas (si no encuentra el bloque, no toca)."""
    t = cfg.get("data", {}).get("transforms")
    if isinstance(t, dict) and "eval" in t:
        t["train"] = t["eval"]
        log.info("overfit: train usa transforms de eval (sin augmentation)")


def _overfit_loader(base_loader, n: int, log):
    """Loader DETERMINISTA de las primeras n imágenes (índices 0..n-1, sin shuffle).

    Idéntico al subconjunto que arma evaluate.py → el smoke evalúa EXACTAMENTE lo
    que entrenó, sin depender del RNG ni del shuffle. Se materializa en memoria
    (frozen) para repetir el mismo subconjunto muchas epochs sin releer disco.
    """
    ds = base_loader.dataset
    n = min(n, len(ds))
    loader = DataLoader(
        Subset(ds, list(range(n))),
        batch_size=base_loader.batch_size or 1,
        shuffle=False,
        num_workers=0,
        collate_fn=base_loader.collate_fn,
    )
    log.info("overfit: %d imágenes deterministas (índices 0..%d)", n, n - 1)
    return list(loader)


def _tune_cfg_for_overfit(cfg, n_batches: int, target_steps: int, log) -> None:
    """Para sobreajustar rápido: paso por batch (accum=1), lr constante, y epochs
    suficientes para ~target_steps."""
    cfg.setdefault("train", {})["grad_accum_steps"] = 1
    cfg["optim"] = {**cfg.get("optim", {}), "scheduler": "none"}
    steps_per_epoch = max(1, n_batches)  # accum=1 → un step por batch
    epochs = max(1, -(-target_steps // steps_per_epoch))  # ceil
    cfg["train"]["epochs"] = epochs
    log.info("overfit: accum=1 · %d epochs (~%d opt-steps) · scheduler=none",
             epochs, epochs * steps_per_epoch)


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

    if args.overfit:
        _force_eval_transforms(cfg, log)          # sin aug, antes de construir el loader
    base = build_dataloader(cfg, "train")
    if args.overfit:
        train_loader = _overfit_loader(base, args.overfit, log)
        _tune_cfg_for_overfit(cfg, len(train_loader), target_steps=300, log=log)
    else:
        train_loader = base

    model = build_detector(cfg)
    if args.resume:
        device = torch.device(get(cfg, "train.device", "cuda"))
        _load_weights(model, args.resume, device)
        log.info("pesos cargados desde %s", args.resume)

    ckpt = Trainer(cfg, model, train_loader).train()
    log.info("listo · checkpoint final: %s", ckpt)


if __name__ == "__main__":
    main()