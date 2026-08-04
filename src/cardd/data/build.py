"""src/cardd/data/build.py

Dueno: Equipo D - Fase: 1 - Contrato: C1

Arma Dataset y DataLoader desde la config (C4). El DataLoader respeta C1:
  - collate_fn que NO apila imagenes (cada una tiene su H,W): viajan como listas
    (images, targets), que es lo que espera MaskRCNN.forward.
  - reproducibilidad de la augmentation: generator + worker_init_fn sembrados,
    para que comparar filas de ablacion mida senal y no ruido de aug (higiene P).
  - transforms: se construyen por nombre desde el registry (C3), nunca a mano.

Nada se hardcodea: root, batch, workers y semilla salen de la config. Claves que
lee: data.root, data.batch_size, data.num_workers, seed,
data.transforms.{train,eval} (lista de specs {name, ...} para el registry).
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cardd.data.coco_dataset import CarDDCocoDataset
from cardd.utils.config import get
from cardd.utils.logging import get_logger
from cardd.utils.registry import TRANSFORMS

log = get_logger("cardd.data.build")
_MISSING = object()


def _cfg_get(cfg, key: str, default=_MISSING):
    try:
        return get(cfg, key)
    except Exception:
        if default is _MISSING:
            raise
        return default


def collate_fn(batch):
    """(image, target)* -> (tuple(images), tuple(targets)). MaskRCNN espera listas."""
    return tuple(zip(*batch))


def _worker_init_fn(worker_id: int) -> None:
    """Siembra numpy/random por worker desde la semilla de torch (aug determinista)."""
    seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(seed)
    random.seed(seed)


def build_transforms(cfg, split: str):
    """Compone el pipeline de transforms del split desde el registry (C3)."""
    key = "data.transforms.train" if split == "train" else "data.transforms.eval"
    specs = _cfg_get(cfg, key, [])
    if not specs:
        return None
    # import perezoso: registra las clases en TRANSFORMS (transforms.py, paso 3b)
    from torchvision.transforms import v2

    from cardd.data import transforms as _transforms  # noqa: F401
    from cardd.data import transforms_paperms as _transforms_paperms  # noqa: F401

    return v2.Compose([TRANSFORMS.build(spec) for spec in specs])


def _split_train_transform_specs(cfg):
    """Separa las specs de train en (scale, per_sample) para el wrapper de composición.

    La composición multi-muestra (mosaic/mixup/copy-paste) necesita aplicar el resize
    del multiscale ELLA MISMA, con la misma altura para las muestras que compone. Por
    eso `paper_multiscale` se extrae como `scale_transform` y el resto (flip, etc.)
    queda como `per_sample_transforms`. Devuelve (scale_obj_or_None, [per_sample_objs]).
    """
    from cardd.data import transforms as _t  # noqa: F401  (registra en TRANSFORMS)
    from cardd.data import transforms_paperms as _tp  # noqa: F401

    specs = _cfg_get(cfg, "data.transforms.train", [])
    scale_obj, per_sample = None, []
    for spec in specs:
        name = spec.get("name") if isinstance(spec, dict) else None
        obj = TRANSFORMS.build(spec)
        if name == "paper_multiscale":
            scale_obj = obj
        else:
            per_sample.append(obj)
    return scale_obj, per_sample


def build_dataset(cfg, split: str):
    """Construye el dataset del split.

    Para train con `data.composition_aug.enabled=true`, el `CarDDCocoDataset` se crea
    SIN transforms y se envuelve en `CompositionAugDataset` (que aplica scale+flip y la
    composición con schedule por época). En cualquier otro caso, comportamiento previo:
    transforms aplicadas dentro del dataset.

    Cuando envuelve, expone `cfg["_epoch_state"]` (un EpochState compartido) para que el
    trainer publique la época actual a los workers. Es el ÚNICO canal del schedule.
    """
    root = Path(_cfg_get(cfg, "data.root"))
    ca_enabled = bool(_cfg_get(cfg, "data.composition_aug.enabled", False))

    # formato NUEVO (programable): si el bloque composition_aug trae una lista `augs`,
    # se usa el orquestador modular CompositionAug (augs/*.py). Si no, el legacy
    # CompositionAugDataset (retrocompatible con configs viejas de early_mode/p_*).
    ca_spec = _cfg_get(cfg, "data.composition_aug", {}) or {}
    ca_new_format = isinstance(ca_spec.get("augs"), list)

    if split == "train" and ca_enabled and ca_new_format:
        from cardd.data.composition import CompositionAug, EpochState

        base = CarDDCocoDataset(root=root, split=split, transforms=None)
        scale_obj, per_sample = _split_train_transform_specs(cfg)

        epoch_state = EpochState(0)
        if isinstance(cfg, dict):
            cfg.setdefault("_runtime", {})["epoch_state"] = epoch_state

        ds = CompositionAug(
            base, scale_transform=scale_obj, per_sample_transforms=per_sample,
            spec=ca_spec, total_epochs=int(_cfg_get(cfg, "train.epochs", 1)),
            epoch_state=epoch_state,
        )
        log.info(
            "dataset 'train' (composition_aug PROGRAMABLE): %d imagenes, %d clases · "
            "%d augs, mixup=%s, class_weights=%s",
            len(ds), ds.num_classes, len(ca_spec.get("augs", [])),
            bool((ca_spec.get("mixup", {}) or {}).get("enabled", False)),
            bool((ca_spec.get("class_weights", {}) or {}).get("enabled", False)),
        )
        return ds

    if split == "train" and ca_enabled:
        from cardd.data.composition_aug import CompositionAugDataset, EpochState

        base = CarDDCocoDataset(root=root, split=split, transforms=None)
        scale_obj, per_sample = _split_train_transform_specs(cfg)

        epoch_state = EpochState(0)
        # canal trainer -> workers (se escribe en trainer.train(), se lee en __getitem__)
        if isinstance(cfg, dict):
            cfg.setdefault("_runtime", {})["epoch_state"] = epoch_state

        def _cg(key, default):
            return _cfg_get(cfg, f"data.composition_aug.{key}", default)

        ds = CompositionAugDataset(
            base,
            scale_transform=scale_obj,
            per_sample_transforms=per_sample,
            total_epochs=int(_cfg_get(cfg, "train.epochs", 1)),
            epoch_state=epoch_state,
            mosaic_end_frac=float(_cg("mosaic_end_frac", 0.8)),
            early_mode=str(_cg("early_mode", "mosaic")),
            p_radial=float(_cg("p_radial", 0.5)),
            p_mosaic=float(_cg("p_mosaic", 1.0)),
            mosaic_center_jitter=float(_cg("mosaic_center_jitter", 0.25)),
            n_sources=int(_cg("n_sources", 4)),
            m_wedges=int(_cg("m_wedges", 4)),
            radial_center_jitter=float(_cg("radial_center_jitter", 0.15)),
            crop_frac_min=float(_cg("crop_frac_min", 0.5)),
            crop_frac_max=float(_cg("crop_frac_max", 1.0)),
            p_mixup=float(_cg("p_mixup", 0.5)),
            mixup_alpha_beta=float(_cg("mixup_alpha_beta", 1.5)),
            cp_p_early=float(_cg("cp_p_early", 0.15)),
            cp_p_late=float(_cg("cp_p_late", 0.5)),
            cp_max_paste_instances=_cg("cp_max_paste_instances", None),
            cp_min_paste_instances=int(_cg("cp_min_paste_instances", 1)),
            cp_min_visible_ratio=float(_cg("cp_min_visible_ratio", 0.10)),
            cp_blend=str(_cg("cp_blend", "gaussian")),
            cp_blur_sigma=float(_cg("cp_blur_sigma", 1.5)),
            cp_blur_ksize=int(_cg("cp_blur_ksize", 5)),
            cp_random_translate=bool(_cg("cp_random_translate", False)),
            min_area=int(_cg("min_area", 16)),
            avoid_self_pair=bool(_cg("avoid_self_pair", True)),
            source_sampling=bool(_cg("source_sampling.enabled", False)),
            source_weights=_cg("source_sampling.weights", None),
            source_mode=str(_cg("source_sampling.mode", "max")),
            debug_sample=float(_cg("debug_sample", 0.005)),
        )
        log.info(
            "dataset 'train' (composition_aug ON): %d imagenes, %d clases · "
            "mosaic_end_frac=%.2f p_mosaic=%.2f p_mixup=%.2f cp_early=%.2f cp_late=%.2f",
            len(ds), ds.num_classes, float(_cg("mosaic_end_frac", 0.8)),
            float(_cg("p_mosaic", 1.0)), float(_cg("p_mixup", 0.5)),
            float(_cg("cp_p_early", 0.15)), float(_cg("cp_p_late", 0.5)),
        )
        return ds

    transforms = build_transforms(cfg, split)
    ds = CarDDCocoDataset(root=root, split=split, transforms=transforms)
    log.info("dataset '%s': %d imagenes, %d clases", split, len(ds), ds.num_classes)
    return ds


def build_dataloader(cfg, split: str) -> DataLoader:
    ds = build_dataset(cfg, split)
    is_train = split == "train"

    batch_size = int(_cfg_get(cfg, "data.batch_size", 2))
    num_workers = int(_cfg_get(cfg, "data.num_workers", 4))
    seed = int(_cfg_get(cfg, "seed", 0))

    generator = torch.Generator()
    generator.manual_seed(seed)

    # persistent_workers: con augmentación de composición pesada, re-forkear los
    # workers cada época es caro. Persistirlos lo evita; el schedule sigue llegando
    # porque la época viaja por el EpochState (mp.Value compartido), no por re-fork.
    # Requiere num_workers>0. Default False (comportamiento previo).
    persistent = bool(_cfg_get(cfg, "data.persistent_workers", False)) and num_workers > 0

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        collate_fn=collate_fn,
        worker_init_fn=_worker_init_fn,
        generator=generator,
        pin_memory=True,
        drop_last=is_train,
        persistent_workers=persistent,
    )