"""CarDD · src/cardd/data/copy_paste.py

Dueño: Equipo C · Contrato: C1 (I/O por muestra)

Copy-Paste augmentation para segmentación de instancias, según
"Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation"
(Ghiasi et al., 2020).

Idea: se toman dos muestras del dataset, se recorta un subconjunto de instancias de
la FUENTE usando su máscara y se pegan sobre la imagen DESTINO. Las instancias del
destino que quedan tapadas se recortan (se les resta el área ocluida) y se descartan
si su visibilidad cae bajo un umbral. El paper demuestra que el pegado NO necesita
modelar el contexto de la escena: la colocación aleatoria funciona igual o mejor que
métodos de composición contextual.

ARQUITECTURA (decidida para este pipeline, variante "B"):
  El dataset base se construye SIN transforms. `CopyPasteDataset` las aplica él mismo
  para poder imponer la MISMA altura a las dos muestras (`force_h`), de modo que
  fuente y destino salen del multiscale ya alineadas y la composición es una
  operación elemento a elemento sin un segundo resize.

  Orden de operaciones por muestra compuesta:
      1. sortea new_h UNA vez (compartido)
      2. escala destino y fuente a (new_h, width)      <- PaperMultiScale(force_h)
      3. flip independiente a cada una                 <- PaperHFlip
      4. compone                                       <- copy_paste_compose()

  Esto se aparta del paper en un punto: allí cada imagen sortea su propio jitter de
  escala. Aquí comparten altura. A cambio se evita la interpolación extra que exigiría
  reescalar la fuente al destino después de que cada una escaló por su cuenta.

DESVIACIONES DEL PAPER (conscientes, todas parametrizables):
  - `blend="gaussian"` por defecto. El paper evaluó el suavizado gaussiano del borde
    y lo encontró marginal en COCO, donde el borde de la máscara coincide con un borde
    real de objeto. En CarDD las clases son daños (dent, scratch, crack...) que a
    menudo NO tienen borde neto: se desvanecen en la chapa sana. Un pegado binario
    introduce una discontinuidad de alta frecuencia justo en el perímetro de la
    máscara, y la red puede aprender "discontinuidad brusca => frontera de instancia",
    atajo que funciona en train y no existe en test. Poner `blend="none"` para
    reproducir el paper al pie de la letra.
  - `min_visible_ratio`: el paper elimina las instancias COMPLETAMENTE ocluidas pero no
    fija umbral para las casi-ocluidas. Una instancia con el 2% de área visible produce
    una caja diminuta y una máscara casi vacía: es ruido de supervisión. `0.0` reproduce
    el paper.

NOTA sobre `area`: se recalcula desde la caja nueva, igual que hace PaperMultiScale.
Es un campo inerte en train (MaskRCNN de torchvision lee boxes/labels/masks; `area` la
usa CocoEvaluator sobre val/test, que no pasa por aquí). Se mantiene coherente por
higiene, no porque afecte a ninguna métrica reportada.

I/O según C1 — entrada y salida idénticas al dataset base:
    image:  Tensor[3,H,W] float32 en [0,1]
    target: {boxes: BoundingBoxes[N,4] xyxy, labels: int64[N], masks: Mask[N,H,W] uint8,
             image_id: int, area: float32[N], iscrowd: uint8[N]}
"""
from __future__ import annotations

import random
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import tv_tensors

# ---------------------------------------------------------------------------
# utilidades
# ---------------------------------------------------------------------------


def _masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """[N,H,W] bool/uint8 -> [N,4] xyxy float32. Filas vacías -> ceros.

    No se usa torchvision.ops.masks_to_boxes porque esa función asume que toda máscara
    tiene al menos un píxel; aquí pueden llegar máscaras vacías tras la oclusión y hay
    que devolver una caja nula sin reventar (el filtrado por área ocurre después).
    """
    n = masks.shape[0]
    if n == 0:
        return torch.zeros((0, 4), dtype=torch.float32)

    m = masks.bool()
    boxes = torch.zeros((n, 4), dtype=torch.float32)
    rows = m.any(dim=2)  # [N,H]
    cols = m.any(dim=1)  # [N,W]
    for i in range(n):
        r = torch.where(rows[i])[0]
        c = torch.where(cols[i])[0]
        if r.numel() == 0 or c.numel() == 0:
            continue
        # +1 en el extremo: xyxy en píxeles con el borde derecho/inferior exclusivo,
        # igual convención que masks_to_boxes de torchvision.
        boxes[i] = torch.tensor(
            [c[0].item(), r[0].item(), c[-1].item() + 1.0, r[-1].item() + 1.0],
            dtype=torch.float32,
        )
    return boxes


def _gaussian_kernel1d(sigma: float, ksize: int) -> torch.Tensor:
    x = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2.0
    k = torch.exp(-(x**2) / (2.0 * sigma**2))
    return k / k.sum()


def _gaussian_blur(x: torch.Tensor, sigma: float, ksize: int) -> torch.Tensor:
    """Blur gaussiano separable sobre [1,1,H,W]. Sin dependencia de cv2."""
    if ksize % 2 == 0:
        ksize += 1
    k = _gaussian_kernel1d(sigma, ksize).to(x.device, x.dtype)
    pad = ksize // 2
    x = F.pad(x, (pad, pad, 0, 0), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1))
    x = F.pad(x, (0, 0, pad, pad), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, -1, 1))
    return x


def _empty_target_like(target: dict, h: int, w: int) -> dict:
    out = dict(target)
    out["boxes"] = tv_tensors.BoundingBoxes(
        torch.zeros((0, 4), dtype=torch.float32),
        format=tv_tensors.BoundingBoxFormat.XYXY,
        canvas_size=(h, w),
    )
    out["labels"] = torch.zeros((0,), dtype=torch.int64)
    out["masks"] = tv_tensors.Mask(torch.zeros((0, h, w), dtype=torch.uint8))
    out["area"] = torch.zeros((0,), dtype=torch.float32)
    out["iscrowd"] = torch.zeros((0,), dtype=torch.uint8)
    return out


# ---------------------------------------------------------------------------
# composición pura
# ---------------------------------------------------------------------------


def copy_paste_compose(
    image_dst: torch.Tensor,
    target_dst: dict,
    image_src: torch.Tensor,
    target_src: dict,
    *,
    max_paste_instances: int | None = None,
    min_paste_instances: int = 1,
    min_visible_ratio: float = 0.10,
    min_paste_area: int = 16,
    blend: str = "gaussian",
    blur_sigma: float = 1.5,
    blur_ksize: int = 5,
    random_translate: bool = False,
    max_translate_frac: float = 0.25,
) -> tuple[torch.Tensor, dict]:
    """Pega instancias de (image_src, target_src) sobre (image_dst, target_dst).

    Requiere que ambas imágenes tengan el MISMO shape espacial. `CopyPasteDataset` lo
    garantiza escalando las dos con la misma altura; si se llama a mano, es
    responsabilidad del llamante.

    Devuelve la muestra compuesta respetando C1 (tv_tensors reconstruidos).
    """
    assert image_dst.shape[-2:] == image_src.shape[-2:], (
        f"copy_paste_compose exige mismo shape: dst={tuple(image_dst.shape[-2:])} "
        f"src={tuple(image_src.shape[-2:])}"
    )
    H, W = int(image_dst.shape[-2]), int(image_dst.shape[-1])

    masks_src = torch.as_tensor(target_src["masks"])
    n_src = int(masks_src.shape[0])
    if n_src == 0:
        return image_dst, target_dst

    # --- selección aleatoria del subconjunto a pegar -----------------------
    hi = n_src if max_paste_instances is None else min(int(max_paste_instances), n_src)
    lo = max(1, min(int(min_paste_instances), hi))
    k = random.randint(lo, hi)
    idxs = random.sample(range(n_src), k)

    labels_src = torch.as_tensor(target_src["labels"])
    iscrowd_src = torch.as_tensor(
        target_src.get("iscrowd", torch.zeros(n_src, dtype=torch.uint8))
    )

    mask_dtype = torch.as_tensor(target_dst["masks"]).dtype
    img = image_dst.clone()

    kept_masks: list[torch.Tensor] = []
    kept_labels: list[int] = []
    kept_iscrowd: list[int] = []

    # unión de lo pegado: define la oclusión sobre las instancias del destino
    union = torch.zeros((H, W), dtype=torch.bool)

    for i in idxs:
        m = masks_src[i].bool()
        if not bool(m.any()):
            continue

        # --- traslación opcional (el paper NO traslada: el jitter de posición
        # viene de que las dos imágenes se aumentan de forma independiente) ---
        if random_translate:
            dy = random.randint(
                -int(H * max_translate_frac), int(H * max_translate_frac)
            )
            dx = random.randint(
                -int(W * max_translate_frac), int(W * max_translate_frac)
            )
            m_shift = torch.roll(m, shifts=(dy, dx), dims=(0, 1))
            # roll es circular: hay que borrar lo que dio la vuelta
            if dy > 0:
                m_shift[:dy, :] = False
            elif dy < 0:
                m_shift[dy:, :] = False
            if dx > 0:
                m_shift[:, :dx] = False
            elif dx < 0:
                m_shift[:, dx:] = False
            src_pixels = torch.roll(image_src, shifts=(dy, dx), dims=(1, 2))
            m = m_shift
        else:
            src_pixels = image_src

        if int(m.sum()) < min_paste_area:
            continue

        # --- alpha para el compositing RGB ---------------------------------
        # El alpha puede ser continuo (blend suavizado), pero la MÁSCARA DE
        # ANOTACIÓN y la que alimenta la oclusión son siempre la binaria. Mezclarlas
        # produce máscaras no binarias en el loss y una resta de oclusión que no
        # cuadra con lo que se ve en el RGB.
        if blend == "gaussian":
            alpha = _gaussian_blur(
                m.float().view(1, 1, H, W), blur_sigma, blur_ksize
            ).view(H, W).clamp_(0.0, 1.0)
        elif blend == "none":
            alpha = m.float()
        else:
            raise ValueError(f"blend desconocido: {blend!r} (usa 'gaussian' o 'none')")

        img = img * (1.0 - alpha) + src_pixels * alpha

        union |= m
        kept_masks.append(m)
        kept_labels.append(int(labels_src[i]))
        kept_iscrowd.append(int(iscrowd_src[i]) if iscrowd_src.numel() else 0)

    if not kept_masks:
        return image_dst, target_dst

    img = img.clamp_(0.0, 1.0)

    # --- oclusión de las instancias del destino ----------------------------
    masks_dst = torch.as_tensor(target_dst["masks"]).bool()
    labels_dst = torch.as_tensor(target_dst["labels"])
    iscrowd_dst = torch.as_tensor(
        target_dst.get("iscrowd", torch.zeros(masks_dst.shape[0], dtype=torch.uint8))
    )

    surv_masks: list[torch.Tensor] = []
    surv_labels: list[int] = []
    surv_iscrowd: list[int] = []
    n_dropped = 0

    for j in range(masks_dst.shape[0]):
        m0 = masks_dst[j]
        a0 = int(m0.sum())
        if a0 == 0:
            n_dropped += 1
            continue
        m1 = m0 & ~union
        a1 = int(m1.sum())
        if a1 == 0 or (a1 / a0) < min_visible_ratio or a1 < min_paste_area:
            n_dropped += 1
            continue
        surv_masks.append(m1)
        surv_labels.append(int(labels_dst[j]))
        surv_iscrowd.append(int(iscrowd_dst[j]) if iscrowd_dst.numel() else 0)

    # --- ensamblado del target ---------------------------------------------
    all_masks = surv_masks + kept_masks
    all_labels = surv_labels + kept_labels
    all_iscrowd = surv_iscrowd + kept_iscrowd

    if not all_masks:
        return img, _empty_target_like(target_dst, H, W)

    masks_t = torch.stack(all_masks).to(mask_dtype)
    boxes_t = _masks_to_boxes(masks_t)

    # guarda final: ninguna caja degenerada puede llegar a RoIHeads
    wh = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
    valid = wh > 0
    if not bool(valid.all()):
        n_dropped += int((~valid).sum())
        masks_t = masks_t[valid]
        boxes_t = boxes_t[valid]
        all_labels = [l for l, v in zip(all_labels, valid.tolist()) if v]
        all_iscrowd = [c for c, v in zip(all_iscrowd, valid.tolist()) if v]

    if masks_t.shape[0] == 0:
        return img, _empty_target_like(target_dst, H, W)

    out = dict(target_dst)
    out["boxes"] = tv_tensors.BoundingBoxes(
        boxes_t, format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=(H, W)
    )
    out["labels"] = torch.tensor(all_labels, dtype=torch.int64)
    out["masks"] = tv_tensors.Mask(masks_t)
    out["area"] = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
    out["iscrowd"] = torch.tensor(all_iscrowd, dtype=torch.uint8)
    out["_cp_stats"] = {
        "pasted": len(kept_masks),
        "survived": len(surv_masks),
        "dropped": n_dropped,
    }
    return img, out


# ---------------------------------------------------------------------------
# wrapper de dataset
# ---------------------------------------------------------------------------


class CopyPasteDataset(Dataset):
    """Envuelve un dataset C1 SIN transforms y aplica multiscale+flip+copy-paste.

    El dataset base debe construirse con `transforms=None`: este wrapper es quien
    aplica `scale_transform` (con altura compartida entre las dos muestras) y
    `per_sample_transforms` (independientes por muestra), en ese orden.

    Cuando copy-paste no dispara (prob. 1-p) la muestra recorre EXACTAMENTE el mismo
    camino de escala y flip, para no sesgar la distribución de tamaños del batch.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        scale_transform: Any | None = None,
        per_sample_transforms: Sequence[Callable] | Callable | None = None,
        *,
        p: float = 0.5,
        max_paste_instances: int | None = None,
        min_paste_instances: int = 1,
        min_visible_ratio: float = 0.10,
        min_paste_area: int = 16,
        blend: str = "gaussian",
        blur_sigma: float = 1.5,
        blur_ksize: int = 5,
        random_translate: bool = False,
        max_translate_frac: float = 0.25,
        avoid_self_pair: bool = True,
        debug_sample: float = 0.005,
    ) -> None:
        self.base = base_dataset
        self.scale_transform = scale_transform
        if per_sample_transforms is None:
            self.per_sample = []
        elif callable(per_sample_transforms) and not isinstance(
            per_sample_transforms, (list, tuple)
        ):
            self.per_sample = [per_sample_transforms]
        else:
            self.per_sample = list(per_sample_transforms)

        self.p = float(p)
        self.max_paste_instances = max_paste_instances
        self.min_paste_instances = int(min_paste_instances)
        self.min_visible_ratio = float(min_visible_ratio)
        self.min_paste_area = int(min_paste_area)
        self.blend = str(blend)
        self.blur_sigma = float(blur_sigma)
        self.blur_ksize = int(blur_ksize)
        self.random_translate = bool(random_translate)
        self.max_translate_frac = float(max_translate_frac)
        self.avoid_self_pair = bool(avoid_self_pair)
        self.debug_sample = float(debug_sample)

        # se propagan los atributos que M/E consumen del dataset base
        for attr in ("coco", "ann_file", "img_dir", "ids", "num_classes",
                     "label_to_name", "cat_id_to_label", "root", "split"):
            if hasattr(base_dataset, attr):
                setattr(self, attr, getattr(base_dataset, attr))

    def __len__(self) -> int:
        return len(self.base)

    # -- aplicación de transforms ------------------------------------------
    def _apply(self, image, target, force_h: int | None):
        if self.scale_transform is not None:
            if force_h is not None:
                image, target = self.scale_transform(image, target, force_h=force_h)
            else:
                image, target = self.scale_transform(image, target)
        for t in self.per_sample:
            image, target = t(image, target)
        return image, target

    def _shared_h(self) -> int | None:
        """Altura compartida por las dos muestras de una composición."""
        st = self.scale_transform
        if st is None:
            return None
        if hasattr(st, "sample_h"):
            return int(st.sample_h())
        if hasattr(st, "h_min") and hasattr(st, "h_max"):
            return random.randint(int(st.h_min), int(st.h_max))
        return None

    def __getitem__(self, index: int):
        # --- sin copy-paste: mismo camino de transforms ---------------------
        if random.random() >= self.p:
            image, target = self.base[index]
            return self._apply(image, target, force_h=None)

        # --- con copy-paste: altura compartida ------------------------------
        h = self._shared_h()

        img_dst, tgt_dst = self.base[index]
        img_dst, tgt_dst = self._apply(img_dst, tgt_dst, force_h=h)

        j = index
        if len(self.base) > 1 and self.avoid_self_pair:
            while j == index:
                j = random.randrange(len(self.base))
        else:
            j = random.randrange(len(self.base))

        img_src, tgt_src = self.base[j]
        img_src, tgt_src = self._apply(img_src, tgt_src, force_h=h)

        img, tgt = copy_paste_compose(
            img_dst,
            tgt_dst,
            img_src,
            tgt_src,
            max_paste_instances=self.max_paste_instances,
            min_paste_instances=self.min_paste_instances,
            min_visible_ratio=self.min_visible_ratio,
            min_paste_area=self.min_paste_area,
            blend=self.blend,
            blur_sigma=self.blur_sigma,
            blur_ksize=self.blur_ksize,
            random_translate=self.random_translate,
            max_translate_frac=self.max_translate_frac,
        )

        stats = tgt.pop("_cp_stats", None)
        if self.debug_sample > 0 and random.random() < self.debug_sample:
            n_in = int(torch.as_tensor(tgt_dst["masks"]).shape[0])
            n_out = int(torch.as_tensor(tgt["masks"]).shape[0])
            print(
                f"[copy_paste] dst={index} src={j} "
                f"shape={tuple(img.shape[-2:])} inst {n_in}->{n_out} stats={stats}",
                flush=True,
            )
        return img, tgt


__all__ = ["copy_paste_compose", "CopyPasteDataset"]
