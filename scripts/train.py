"""scripts/train.py — entrena un detector CarDD desde un config.

dueño: M · fase: 4

Uso:
    # entrenamiento normal de la línea base (1 GPU/CPU, igual que siempre)
    PYTHONPATH=src python scripts/train.py --config configs/ablation/a0_baseline.yaml

    # MULTI-GPU (Kaggle T4 x2): lanzar con torchrun → un proceso por GPU (DDP)
    PYTHONPATH=src torchrun --standalone --nproc_per_node=2 scripts/train.py \
        --config configs/ablation/a0_baseline.yaml

    # SMOKE TEST (hito paso 13): loss→0 sobreajustando N imágenes (correr SIN torchrun)
    PYTHONPATH=src python scripts/train.py --config configs/ablation/a0_baseline.yaml --overfit 8
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from cardd.utils.config import load_config, get
from cardd.utils.seed import set_seed
from cardd.utils.logging import get_logger
from cardd.data.build import build_dataloader
from cardd.models.detectors.mask_rcnn import build_detector
from cardd.engine.trainer import Trainer


# --------------------------- setup distribuido (DDP) ---------------------------

def _init_distributed():
    """Inicializa DDP si el script fue lanzado con torchrun.

    torchrun setea RANK/LOCAL_RANK/WORLD_SIZE en el entorno; si no están,
    corremos en modo normal (1 proceso, 1 GPU/CPU) sin tocar nada.
    Devuelve (distributed, rank, local_rank)."""
    if "RANK" not in os.environ:
        return False, 0, 0
    from datetime import timedelta
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    # timeout generoso: rank 0 valida solo y los demás esperan en un barrier;
    # con val lenta, el timeout por defecto de NCCL podría cortar la corrida.
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    return True, rank, local_rank


def _distributed_train_loader(base_loader, log):
    """Reconstruye el train loader con DistributedSampler.

    Reusa dataset/batch_size/collate/num_workers del loader que armó
    data.build (D) — solo cambia el sampler para que cada rank vea una
    porción DISTINTA del dataset por época (sin duplicar imágenes)."""
    ds = base_loader.dataset
    sampler = DistributedSampler(ds, shuffle=True)
    loader = DataLoader(
        ds,
        batch_size=base_loader.batch_size or 1,
        sampler=sampler,          # reemplaza el shuffle: el sampler reparte y baraja
        num_workers=base_loader.num_workers,
        collate_fn=base_loader.collate_fn,
        pin_memory=getattr(base_loader, "pin_memory", False),
    )
    log.info("DDP: train loader con DistributedSampler · %d imágenes repartidas entre %d ranks",
             len(ds), dist.get_world_size())
    return loader


# -------------------------------------------------------------------------------

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
    cfg["train"]["scheduler"] = {"name": "none"}  # lr constante en overfit
    steps_per_epoch = max(1, n_batches)  # accum=1 → un step por batch
    epochs = max(1, -(-target_steps // steps_per_epoch))  # ceil
    cfg["train"]["epochs"] = epochs
    log.info("overfit: accum=1 · %d epochs (~%d opt-steps) · scheduler=none",
             epochs, epochs * steps_per_epoch)


def main() -> None:
    ap = argparse.ArgumentParser(description="Entrena un detector CarDD desde un config.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--overfit", type=int, default=0,
                    help="smoke loss→0: entrena sobre N imágenes (hito paso 13). "
                         "Correr SIN torchrun (single-GPU).")
    ap.add_argument("--resume", default=None,
                    help="checkpoint para RESUME estricto (modelo+optimizer+scheduler+época+best). "
                         "Sin --resume, el backbone parte de los pesos de timm.")
    ap.add_argument("--warm-start", default=None,
                    help="checkpoint para WARM-START (solo pesos; schedule nuevo desde epoch 0). "
                         "Para extender un modelo ya entrenado con un schedule corto.")
    args = ap.parse_args()

    # DDP: si torchrun lanzó este script, inicializar el process group ANTES de todo
    distributed, rank, local_rank = _init_distributed()

    cfg = load_config(args.config)
    set_seed(int(get(cfg, "seed", 0)))
    log = get_logger("cardd.train")

    if distributed:
        # cada proceso trabaja en SU GPU (rank 0 → cuda:0, rank 1 → cuda:1)
        cfg.setdefault("train", {})["device"] = f"cuda:{local_rank}"
        if args.overfit:
            raise SystemExit("--overfit es single-GPU: correr sin torchrun")

    if args.overfit:
        _force_eval_transforms(cfg, log)          # sin aug, antes de construir el loader
    base = build_dataloader(cfg, "train")
    if args.overfit:
        train_loader = _overfit_loader(base, args.overfit, log)
        _tune_cfg_for_overfit(cfg, len(train_loader), target_steps=300, log=log)
        val_loader = None  # overfit: la selección best.pth por val no aplica
    else:
        if distributed:
            train_loader = _distributed_train_loader(base, log)
            # DDP: SOLO rank 0 valida (evita evaluar val duplicado en cada GPU);
            # los demás ranks esperan en el barrier del Trainer.
            val_loader = build_dataloader(cfg, "val") if rank == 0 else None
        else:
            train_loader = base
            val_loader = build_dataloader(cfg, "val")

    model = build_detector(cfg)
    trainer = Trainer(cfg, model, train_loader, val_loader)
    if args.warm_start:
        trainer.warm_start(args.warm_start)   # warm-start: solo pesos, schedule nuevo
    if args.resume:
        trainer.resume(args.resume)   # resume ESTRICTO: modelo+optimizer+scheduler+época+best

    ckpt = trainer.train()
    log.info("listo · checkpoint final: %s", ckpt)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
