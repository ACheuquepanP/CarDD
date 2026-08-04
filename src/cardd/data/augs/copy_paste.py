"""CarDD · src/cardd/data/augs/copy_paste.py

COPY-PASTE: pega instancias de una fuente sobre un destino (imagen normal). Compite en
el sorteo como una opción más (una imagen 'copy_paste' es imagen normal + instancias
pegadas). No se apila sobre otras composiciones (solo se combina vía mixup).

Parámetro de intensidad (enhance): `max_paste_instances`.
"""
from __future__ import annotations

import random

import torch
from torchvision import tv_tensors

from cardd.data.augs._core import (as_t, empty_target_like, gaussian_blur,
                                   masks_to_boxes)


def compose(dst: tuple[torch.Tensor, dict], src: tuple[torch.Tensor, dict], *,
            max_paste_instances=None, min_paste_instances: int = 1,
            min_visible_ratio: float = 0.10, min_paste_area: int = 16,
            blend: str = "gaussian", blur_sigma: float = 1.5, blur_ksize: int = 5,
            random_translate: bool = False, max_translate_frac: float = 0.25
            ) -> tuple[torch.Tensor, dict]:
    """Pega instancias de src sobre dst. Ambas del MISMO shape espacial."""
    image_dst, target_dst = dst
    image_src, target_src = src
    assert image_dst.shape[-2:] == image_src.shape[-2:], "copy_paste: shapes distintos"
    H, W = int(image_dst.shape[-2]), int(image_dst.shape[-1])

    masks_src = as_t(target_src["masks"])
    n_src = int(masks_src.shape[0])
    if n_src == 0:
        return image_dst, target_dst

    hi = n_src if max_paste_instances is None else min(int(max_paste_instances), n_src)
    lo = max(1, min(int(min_paste_instances), hi))
    k = random.randint(lo, hi)
    idxs = random.sample(range(n_src), k)

    labels_src = as_t(target_src["labels"])
    iscrowd_src = as_t(target_src.get("iscrowd", torch.zeros(n_src, dtype=torch.uint8)))
    mask_dtype = as_t(target_dst["masks"]).dtype
    img = image_dst.clone()

    kept_masks, kept_labels, kept_iscrowd = [], [], []
    union = torch.zeros((H, W), dtype=torch.bool)

    for i in idxs:
        m = masks_src[i].bool()
        if not bool(m.any()):
            continue
        if random_translate:
            dy = random.randint(-int(H * max_translate_frac), int(H * max_translate_frac))
            dx = random.randint(-int(W * max_translate_frac), int(W * max_translate_frac))
            m_sh = torch.roll(m, shifts=(dy, dx), dims=(0, 1))
            if dy > 0: m_sh[:dy, :] = False
            elif dy < 0: m_sh[dy:, :] = False
            if dx > 0: m_sh[:, :dx] = False
            elif dx < 0: m_sh[:, dx:] = False
            src_px = torch.roll(image_src, shifts=(dy, dx), dims=(1, 2))
            m = m_sh
        else:
            src_px = image_src
        if int(m.sum()) < min_paste_area:
            continue
        if blend == "gaussian":
            alpha = gaussian_blur(m.float().view(1, 1, H, W), blur_sigma, blur_ksize
                                  ).view(H, W).clamp_(0.0, 1.0)
        elif blend == "none":
            alpha = m.float()
        else:
            raise ValueError(f"blend desconocido: {blend!r}")
        img = img * (1.0 - alpha) + src_px * alpha
        union |= m
        kept_masks.append(m)
        kept_labels.append(int(labels_src[i]))
        kept_iscrowd.append(int(iscrowd_src[i]) if iscrowd_src.numel() else 0)

    if not kept_masks:
        return image_dst, target_dst
    img = img.clamp_(0.0, 1.0)

    # oclusión de instancias del destino
    masks_dst = as_t(target_dst["masks"]).bool()
    labels_dst = as_t(target_dst["labels"])
    iscrowd_dst = as_t(target_dst.get("iscrowd", torch.zeros(masks_dst.shape[0], dtype=torch.uint8)))
    surv_masks, surv_labels, surv_iscrowd = [], [], []
    for j in range(masks_dst.shape[0]):
        m0 = masks_dst[j]; a0 = int(m0.sum())
        if a0 == 0:
            continue
        m1 = m0 & ~union; a1 = int(m1.sum())
        if a1 == 0 or (a1 / a0) < min_visible_ratio or a1 < min_paste_area:
            continue
        surv_masks.append(m1)
        surv_labels.append(int(labels_dst[j]))
        surv_iscrowd.append(int(iscrowd_dst[j]) if iscrowd_dst.numel() else 0)

    all_masks = surv_masks + kept_masks
    all_labels = surv_labels + kept_labels
    all_iscrowd = surv_iscrowd + kept_iscrowd
    if not all_masks:
        return img, empty_target_like(target_dst, H, W)

    masks_t = torch.stack(all_masks).to(mask_dtype)
    boxes_t = masks_to_boxes(masks_t)
    wh = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
    valid = wh > 0
    if not bool(valid.all()):
        masks_t = masks_t[valid]; boxes_t = boxes_t[valid]
        all_labels = [l for l, v in zip(all_labels, valid.tolist()) if v]
        all_iscrowd = [c for c, v in zip(all_iscrowd, valid.tolist()) if v]
    if masks_t.shape[0] == 0:
        return img, empty_target_like(target_dst, H, W)

    out = dict(target_dst)
    out["boxes"] = tv_tensors.BoundingBoxes(boxes_t, format=tv_tensors.BoundingBoxFormat.XYXY,
                                            canvas_size=(H, W))
    out["labels"] = torch.tensor(all_labels, dtype=torch.int64)
    out["masks"] = tv_tensors.Mask(masks_t)
    out["area"] = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
    out["iscrowd"] = torch.tensor(all_iscrowd, dtype=torch.uint8)
    return img, out
