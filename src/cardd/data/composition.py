"""CarDD · src/cardd/data/composition.py

Orquestador de la augmentación de composición programable.

MODELO (ver README_composition.md):
  - Progreso global prog = epoch / total_epochs (vía EpochState compartido, a prueba de
    workers).
  - Cada imagen del training pasa por un SORTEO ponderado entre {normal + augs cuya
    ventana está activa en prog}. 'normal' siempre compite con peso normal_weight.
  - copy_paste, mosaic, mosaic_crop, radial compiten como opciones excluyentes por
    imagen. La ÚNICA forma de combinar dos es mixup.
  - Cada aug tiene: window, weight (con decay temporal), enhance (mueve un param de
    intensidad en el tiempo), params. Puede haber varias instancias del mismo type.
  - mixup: operador aparte con su ventana y probabilidad; si dispara, hace 2 sorteos
    (sin mixup) y los mezcla.
  - class_weights: sesga selección de fuente y posición de parche (mosaic_crop), con
    decay temporal propio.
  - Constantes (flip, multiscale) se aplican SIEMPRE por muestra, fuera del sorteo.

Arquitectura B: dataset base SIN transforms; este wrapper aplica scale+flip y compone.
"""
from __future__ import annotations

import multiprocessing as mp
import random
from typing import Any, Callable, Sequence

import torch

from cardd.data.augs import COMPETITORS, mixup as mixup_mod
from cardd.data.augs._core import as_t, resize_image, resize_masks
from cardd.data.augs import mosaic_crop as crop_mod
from cardd.data.augs._schedule import (eval_schedule, in_window, resolve_schedule)
from torchvision import tv_tensors


# ── estado de época compartido (a prueba de workers / DDP por rank) ──
class EpochState:
    def __init__(self, initial: int = 0):
        self._v = mp.Value("i", int(initial))

    def set(self, epoch: int):
        with self._v.get_lock():
            self._v.value = int(epoch)

    def get(self) -> int:
        return int(self._v.value)


class CompositionAug(torch.utils.data.Dataset):
    """Wrapper programable. Lee la spec (dict del YAML composition_aug) y orquesta."""

    def __init__(self, base_dataset, scale_transform=None, per_sample_transforms=None, *,
                 spec: dict, total_epochs: int, epoch_state: EpochState | None = None):
        self.base = base_dataset
        self.scale = scale_transform
        if per_sample_transforms is None:
            self.per_sample = []
        elif callable(per_sample_transforms) and not isinstance(per_sample_transforms, (list, tuple)):
            self.per_sample = [per_sample_transforms]
        else:
            self.per_sample = list(per_sample_transforms)

        self.total_epochs = max(1, int(total_epochs))
        self.epoch_state = epoch_state if epoch_state is not None else EpochState(0)
        self.spec = spec or {}

        self.normal_weight = float(self.spec.get("normal_weight", 1.0))
        self.min_area = int(self.spec.get("min_area", 16))
        self.avoid_self_pair = bool(self.spec.get("avoid_self_pair", True))
        self.debug_sample = float(self.spec.get("debug_sample", 0.005))

        # augs: lista de instancias; precomputa sus schedules
        self.augs = []
        for a in self.spec.get("augs", []):
            self.augs.append(self._prepare_aug(a))

        # mixup
        mx = self.spec.get("mixup", {}) or {}
        self.mixup_enabled = bool(mx.get("enabled", False))
        self.mixup_window = mx.get("window", [0.0, 1.0])
        self.mixup_p_base = float(mx.get("p", 0.5))
        self.mixup_p_sched = resolve_schedule(mx.get("decay"), self.mixup_p_base, self.mixup_window)
        self.mixup_params = dict(mx.get("params", {}))

        # class weights
        cw = self.spec.get("class_weights", {}) or {}
        self.cw_enabled = bool(cw.get("enabled", False))
        self.cw_mode = str(cw.get("mode", "max"))
        self.cw_weights = {str(k): float(v) for k, v in (cw.get("weights", {}) or {}).items()}
        self.cw_window = (cw.get("decay", {}) or {}).get("window", [0.0, 1.0])
        self.cw_curve = str((cw.get("decay", {}) or {}).get("curve", "none")).lower()
        self.cw_end_factor = float((cw.get("decay", {}) or {}).get("end_factor", 1.0))
        self._cw_per_image = self._precompute_class_weights() if self.cw_enabled else None

        for attr in ("coco", "ann_file", "img_dir", "ids", "num_classes",
                     "label_to_name", "cat_id_to_label", "root", "split"):
            if hasattr(base_dataset, attr):
                setattr(self, attr, getattr(base_dataset, attr))

    # ── preparación ──
    def _prepare_aug(self, a: dict) -> dict:
        typ = a["type"]
        window = a.get("window", [0.0, 1.0])
        weight = float(a.get("weight", 1.0))
        params = dict(a.get("params", {}))
        weight_sched = resolve_schedule(a.get("decay"), weight, window)
        enh = a.get("enhance")
        enh_param = None
        enh_sched = None
        if enh and str(enh.get("curve", "none")).lower() not in ("none", ""):
            enh_param = enh.get("param")
            base = float(params.get(enh_param, 0)) if enh_param in params else float(enh.get("start", 0))
            enh_sched = resolve_schedule(enh, base, window)
        return {"type": typ, "window": window, "weight_base": weight,
                "weight_sched": weight_sched, "params": params,
                "enh_param": enh_param, "enh_sched": enh_sched}

    def _precompute_class_weights(self):
        n = len(self.base)
        weights = [1.0] * n
        coco = getattr(self, "coco", None)
        c2l = getattr(self, "cat_id_to_label", None)
        l2n = getattr(self, "label_to_name", None)
        ids = getattr(self, "ids", None)
        if not (coco and c2l and l2n and ids):
            return weights

        def combine(vals):
            if not vals: return 1.0
            if self.cw_mode == "mean": return sum(vals) / len(vals)
            if self.cw_mode == "sum": return sum(vals)
            return max(vals)

        for i, img_id in enumerate(ids):
            anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
            names = set()
            for an in anns:
                lb = c2l.get(an["category_id"])
                if lb is not None:
                    names.add(l2n.get(lb))
            vals = [self.cw_weights.get(nm, 1.0) for nm in names if nm is not None]
            weights[i] = combine(vals)
        return weights

    def __len__(self):
        return len(self.base)

    # ── transforms (constantes) ──
    def _apply(self, image, target, force_h):
        if self.scale is not None:
            image, target = (self.scale(image, target, force_h=force_h)
                             if force_h is not None else self.scale(image, target))
        for t in self.per_sample:
            image, target = t(image, target)
        return image, target

    def _shared_h(self):
        st = self.scale
        if st is None: return None
        if hasattr(st, "sample_h"): return int(st.sample_h())
        if hasattr(st, "h_min"): return random.randint(int(st.h_min), int(st.h_max))
        return None

    def _out_size(self, h):
        st = self.scale
        if h is not None and st is not None and hasattr(st, "_round"):
            return st._round(h), st._round(st.width)
        return None

    # ── muestreo de índices (con sesgo de clase si aplica) ──
    def _cw_factor(self, prog):
        """Factor [end_factor..1] que escala el sesgo de clase con el tiempo."""
        if self.cw_curve in ("none", ""):
            return 1.0
        # 1 -> end_factor a lo largo de cw_window
        return eval_schedule(
            resolve_schedule({"curve": self.cw_curve, "start": 1.0, "end": self.cw_end_factor,
                              "window": self.cw_window}, 1.0, self.cw_window), prog, 1.0)

    def _weighted_source_index(self, prog):
        if not self.cw_enabled or self._cw_per_image is None:
            return random.randrange(len(self.base))
        f = self._cw_factor(prog)
        # peso efectivo = 1 + (peso-1)*f  (decae el exceso sobre 1)
        w = [1.0 + (pw - 1.0) * f for pw in self._cw_per_image]
        return random.choices(range(len(self.base)), weights=w, k=1)[0]

    def _guide_fn(self, prog):
        """Devuelve una guide_fn para mosaic_crop que posiciona la ventana para
        CONTENER (no centrar) una instancia de clase pesada, elegida ponderada por su
        peso de clase escalado por el factor temporal."""
        if not self.cw_enabled:
            return None
        f = self._cw_factor(prog)
        cw_weights = self.cw_weights
        l2n = getattr(self, "label_to_name", {})

        def guide(masks, labels, H, W, cw, ch):
            cand, cw_ = [], []
            for k in range(masks.shape[0]):
                nm = l2n.get(int(labels[k])) if l2n else None
                w = 1.0 + (cw_weights.get(nm, 1.0) - 1.0) * f
                if w > 1.0:
                    cand.append(k); cw_.append(w)
            if not cand:
                return None
            k = random.choices(cand, weights=cw_, k=1)[0]
            m = as_t(masks)[k].bool()
            ys, xs = torch.where(m.any(1))[0], torch.where(m.any(0))[0]
            if not (ys.numel() and xs.numel()):
                return None
            iy0, iy1, ix0, ix1 = int(ys[0]), int(ys[-1]), int(xs[0]), int(xs[-1])
            lo_x, hi_x = max(0, ix1 - cw + 1), min(W - cw, ix0)
            lo_y, hi_y = max(0, iy1 - ch + 1), min(H - ch, iy0)
            if lo_x <= hi_x and lo_y <= hi_y:
                return random.randint(lo_x, hi_x), random.randint(lo_y, hi_y)
            return None
        return guide

    # ── enhance: params efectivos de una aug en prog ──
    def _eff_params(self, aug, prog):
        params = dict(aug["params"])
        if aug["enh_sched"] is not None and aug["enh_param"] is not None:
            base = float(params.get(aug["enh_param"], aug["enh_sched"]["start"]))
            val = eval_schedule(aug["enh_sched"], prog, base)
            # params enteros conocidos -> redondear
            if aug["enh_param"] in ("grid", "m_wedges", "n_sources", "max_paste_instances"):
                val = int(round(val))
            params[aug["enh_param"]] = val
        return params

    # ── sorteo de una opción (sin mixup) -> (img, tgt, label) ──
    def _active_augs(self, prog):
        return [a for a in self.augs if in_window(prog, a["window"])]

    def _draw_one(self, prog, h, out_hw, exclude_dst=None):
        """Elige una opción del sorteo {normal + augs activas} y la produce.
        Devuelve (img, tgt, mode_label)."""
        active = self._active_augs(prog)
        weights = [self.normal_weight]
        options = ["normal"]
        for a in active:
            w = eval_schedule(a["weight_sched"], prog, a["weight_base"])
            if w > 0:
                weights.append(w); options.append(a)
        choice = random.choices(options, weights=weights, k=1)[0]

        if choice == "normal":
            idx = self._weighted_source_index(prog)
            img, tgt = self._apply(*self.base[idx], h)
            return img, tgt, "normal"

        aug = choice
        typ = aug["type"]
        params = self._eff_params(aug, prog)

        if typ == "copy_paste":
            di = exclude_dst if exclude_dst is not None else random.randrange(len(self.base))
            img_dst, tgt_dst = self._apply(*self.base[di], h)
            sj = self._weighted_source_index(prog)
            if self.avoid_self_pair and len(self.base) > 1:
                while sj == di:
                    sj = self._weighted_source_index(prog)
            img_src, tgt_src = self._apply(*self.base[sj], h)
            img, tgt = COMPETITORS["copy_paste"].compose(
                (img_dst, tgt_dst), (img_src, tgt_src), min_paste_area=self.min_area, **params)
            return img, tgt, "copy_paste"

        # mosaic / mosaic_crop / radial: componen desde varias fuentes
        mod = COMPETITORS[typ]
        if out_hw is not None:
            out_h, out_w = out_hw
        else:
            probe, _ = self._apply(*self.base[random.randrange(len(self.base))], h)
            out_h, out_w = int(probe.shape[-2]), int(probe.shape[-1])

        nsrc = mod.num_sources(**params)
        if typ == "mosaic_crop":
            guide = self._guide_fn(prog)
            samples = []
            for _ in range(nsrc):
                si = self._weighted_source_index(prog)
                im, tg = self._apply(*self.base[si], h)
                samples.append(crop_mod.crop_sample(
                    im, tg, crop_frac_min=params.get("crop_frac_min", 0.5),
                    crop_frac_max=params.get("crop_frac_max", 1.0),
                    min_area=self.min_area, guide_fn=guide))
            cparams = {k: v for k, v in params.items()
                       if k not in ("crop_frac_min", "crop_frac_max")}
            img, tgt = mod.compose(samples, out_h=out_h, out_w=out_w,
                                   min_area=self.min_area, **cparams)
        else:
            samples = [self._apply(*self.base[self._weighted_source_index(prog)], h)
                       for _ in range(nsrc)]
            # 'n_sources' solo lo usa num_sources(), no compose() -> quitarlo
            cparams = {k: v for k, v in params.items() if k != "n_sources"}
            img, tgt = mod.compose(samples, out_h=out_h, out_w=out_w,
                                   min_area=self.min_area, **cparams)
        return img, tgt, typ

    def __getitem__(self, index):
        prog = self.epoch_state.get() / self.total_epochs
        h = self._shared_h()
        out_hw = self._out_size(h)
        used = []

        # ¿mixup dispara?
        do_mixup = (self.mixup_enabled and in_window(prog, self.mixup_window)
                    and random.random() < eval_schedule(self.mixup_p_sched, prog, self.mixup_p_base))

        if do_mixup:
            n_img = int(self.mixup_params.get("n_images", 2))
            items = []
            for _ in range(max(2, n_img)):
                im, tg, lab = self._draw_one(prog, h, out_hw, exclude_dst=index)
                items.append((im, tg)); used.append(lab)
            # asegurar mismo shape (out_hw fija esto para composiciones; para 'normal'
            # el shape ya es (h,width) del multiscale -> coincide)
            H, W = items[0][0].shape[-2:]
            fixed = []
            for im, tg in items:
                if im.shape[-2:] != (H, W):
                    im2 = resize_image(im, H, W)
                    m = as_t(tg["masks"])
                    if m.numel():
                        tg = dict(tg); tg["masks"] = tv_tensors.Mask(
                            resize_masks(m, H, W).to(as_t(tg["masks"]).dtype))
                    fixed.append((im2, tg))
                else:
                    fixed.append((im, tg))
            img, tgt = mixup_mod.compose(
                fixed, alpha_beta=float(self.mixup_params.get("alpha_beta", 1.5)),
                n_images=len(fixed), min_area=self.min_area)
            used.append("mixup")
        else:
            img, tgt, lab = self._draw_one(prog, h, out_hw, exclude_dst=index)
            used.append(lab)

        if self.debug_sample > 0 and random.random() < self.debug_sample:
            print(f"[comp] prog={prog:.2f} used={used} shape={tuple(img.shape[-2:])} "
                  f"inst={int(as_t(tgt['masks']).shape[0])}", flush=True)
        return img, tgt


__all__ = ["CompositionAug", "EpochState"]
