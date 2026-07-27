"""CarDD · src/cardd/data/composition_aug.py

Dueño: Equipo C · Contrato: C1 (I/O por muestra)

Augmentación de composición multi-muestra con SCHEDULE por época, unificada en un
solo wrapper de Dataset. Reúne tres técnicas que comparten el mismo problema —
necesitan más de una muestra y hay que reescribir boxes/masks de forma consistente —
y las gobierna con una única tabla de fases dependiente de la época:

  - MOSAIC  : 4 imágenes en una grilla 2x2 recortada -> densidad y variación de escala.
  - MIXUP   : mezcla alpha de dos muestras con UNIÓN de anotaciones.
  - COPY-PASTE: pega instancias de una fuente sobre un destino (copy_paste.py).

Anidamiento (estilo YOLO agresivo, elegido para este proyecto): mixup se aplica SOBRE
el resultado de mosaic. Es decir, una muestra puede ser el mixup de dos mosaicos.
Copy-paste se aplica como capa final e independiente sobre la muestra ya compuesta.

SCHEDULE (por qué existe): la augmentación agresiva regulariza pronto pero desajusta la
distribución respecto a test. La receta —agresivo temprano, limpio tarde— es la de YOLO
("close mosaic"). Aquí se expresa como fases: en la fracción inicial mosaic+mixup están
activos; pasado el umbral se apagan y copy-paste (moderado) toma el relevo para las
épocas finales, que es cuando el modelo afina la precisión de máscara sobre datos reales.

ESTADO DE ÉPOCA A PRUEBA DE WORKERS: el schedule depende de la época, pero con
`num_workers>0` cada worker recibe una COPIA del dataset al arrancar la época y no vería
un atributo mutado en el proceso principal. Por eso la época vive en un
`multiprocessing.Value` COMPARTIDO que el trainer escribe y todos los workers leen en
cada __getitem__. No se guarda como estado mutable del objeto.

ARQUITECTURA (variante "B", consistente con copy_paste.py): el dataset base se construye
SIN transforms. Este wrapper aplica scale (PaperMultiScale) y per-sample (flip, etc.) él
mismo, para poder imponer la MISMA altura a las muestras que se van a componer y para
controlar el orden respecto a la composición.

I/O según C1 — idéntico al dataset base:
    image:  Tensor[3,H,W] float32 en [0,1]
    target: {boxes: BoundingBoxes[N,4] xyxy, labels: int64[N], masks: Mask[N,H,W] uint8,
             image_id: int, area: float32[N], iscrowd: uint8[N]}
"""
from __future__ import annotations

import multiprocessing as mp
import random
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import tv_tensors

from cardd.data.copy_paste import copy_paste_compose, _masks_to_boxes, _empty_target_like


# ---------------------------------------------------------------------------
# estado de época compartido entre procesos
# ---------------------------------------------------------------------------


class EpochState:
    """Entero compartido (multiprocessing) con la época actual.

    El trainer llama `set(epoch)` en el proceso principal; los workers leen `get()`
    dentro de __getitem__. `mp.Value` es lock-libre para lectura de un int y su valor
    se hereda por fork, así que las copias del dataset en cada worker comparten el mismo
    contador subyacente. Con `num_workers=0` funciona igual (mismo proceso).
    """

    def __init__(self, initial: int = 0) -> None:
        self._v = mp.Value("i", int(initial))

    def set(self, epoch: int) -> None:
        with self._v.get_lock():
            self._v.value = int(epoch)

    def get(self) -> int:
        return int(self._v.value)


# ---------------------------------------------------------------------------
# helpers de anotación (comparten convención con copy_paste.py)
# ---------------------------------------------------------------------------


def _as_t(x) -> torch.Tensor:
    return torch.as_tensor(x)


def _clip_boxes_to_canvas(boxes: torch.Tensor, h: int, w: int) -> torch.Tensor:
    boxes = boxes.clone()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)
    return boxes


def _rebuild_target(
    masks: torch.Tensor,
    labels: Sequence[int],
    iscrowd: Sequence[int],
    h: int,
    w: int,
    mask_dtype: torch.dtype,
    template: dict,
    *,
    min_area: int = 16,
) -> dict:
    """Ensambla un target C1 desde máscaras + labels, recalculando cajas desde la
    máscara y descartando instancias degeneradas (área nula o < min_area)."""
    if masks.shape[0] == 0:
        return _empty_target_like(template, h, w)

    masks = masks.to(mask_dtype)
    boxes = _masks_to_boxes(masks)
    wh = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    areas = masks.bool().flatten(1).sum(1)
    keep = (wh > 0) & (areas >= min_area)
    if not bool(keep.all()):
        idx = torch.where(keep)[0].tolist()
        masks = masks[keep]
        boxes = boxes[keep]
        labels = [labels[i] for i in idx]
        iscrowd = [iscrowd[i] for i in idx]

    if masks.shape[0] == 0:
        return _empty_target_like(template, h, w)

    out = dict(template)
    out["boxes"] = tv_tensors.BoundingBoxes(
        boxes, format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=(h, w)
    )
    out["labels"] = torch.tensor(list(labels), dtype=torch.int64)
    out["masks"] = tv_tensors.Mask(masks)
    out["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    out["iscrowd"] = torch.tensor(list(iscrowd), dtype=torch.uint8)
    return out


# ---------------------------------------------------------------------------
# MOSAIC
# ---------------------------------------------------------------------------


def mosaic_compose(
    samples: Sequence[tuple[torch.Tensor, dict]],
    *,
    out_h: int,
    out_w: int,
    center_jitter: float = 0.25,
    min_area: int = 16,
) -> tuple[torch.Tensor, dict]:
    """Compone 4 muestras en una grilla 2x2 recortada a (out_h, out_w).

    Cada cuadrante recibe una muestra reescalada para llenar su celda; el punto de
    corte (cx, cy) se jitterea alrededor del centro para variar cuánto se ve de cada
    imagen. Las instancias se trasladan al cuadrante y se recortan al marco final.

    Las 4 imágenes deben poder reescalarse a cualquier tamaño (no exige mismo shape de
    entrada). El resultado tiene EXACTAMENTE (out_h, out_w).
    """
    assert len(samples) == 4, "mosaic requiere 4 muestras"

    # centro de corte con jitter, dentro de una banda central [±center_jitter]
    cx = int(out_w * (0.5 + random.uniform(-center_jitter, center_jitter)))
    cy = int(out_h * (0.5 + random.uniform(-center_jitter, center_jitter)))
    cx = max(1, min(out_w - 1, cx))
    cy = max(1, min(out_h - 1, cy))

    # celdas: (x0,y0,x1,y1) en el lienzo de salida, orden TL, TR, BL, BR
    cells = [
        (0, 0, cx, cy),
        (cx, 0, out_w, cy),
        (0, cy, cx, out_h),
        (cx, cy, out_w, out_h),
    ]

    canvas = torch.zeros(3, out_h, out_w, dtype=samples[0][0].dtype)
    mask_dtype = _as_t(samples[0][1]["masks"]).dtype

    all_masks: list[torch.Tensor] = []
    all_labels: list[int] = []
    all_iscrowd: list[int] = []

    for (img, tgt), (x0, y0, x1, y1) in zip(samples, cells):
        cw, ch = x1 - x0, y1 - y0
        if cw <= 0 or ch <= 0:
            continue

        # reescala la imagen de la fuente para llenar la celda (deforma; coherente con
        # el multiscale deformante del proyecto)
        img_r = F.interpolate(
            img.unsqueeze(0), size=(ch, cw), mode="bilinear", align_corners=False
        ).squeeze(0).clamp_(0.0, 1.0)
        canvas[:, y0:y1, x0:x1] = img_r

        m = _as_t(tgt["masks"])
        if m.numel() == 0 or m.shape[0] == 0:
            continue
        Hs, Ws = m.shape[-2], m.shape[-1]
        # reescala máscaras a la celda (nearest) y colócalas en un lienzo full-size
        m_r = F.interpolate(
            m.unsqueeze(1).float(), size=(ch, cw), mode="nearest"
        ).squeeze(1) > 0.5
        labels = _as_t(tgt["labels"]).tolist()
        iscrowd = _as_t(tgt.get("iscrowd", torch.zeros(m.shape[0]))).tolist()
        for k in range(m_r.shape[0]):
            full = torch.zeros(out_h, out_w, dtype=torch.bool)
            full[y0:y1, x0:x1] = m_r[k]
            if not bool(full.any()):
                continue
            all_masks.append(full)
            all_labels.append(int(labels[k]))
            all_iscrowd.append(int(iscrowd[k]) if k < len(iscrowd) else 0)

    masks_t = (
        torch.stack(all_masks) if all_masks
        else torch.zeros((0, out_h, out_w), dtype=torch.bool)
    )
    tgt = _rebuild_target(
        masks_t, all_labels, all_iscrowd, out_h, out_w, mask_dtype,
        samples[0][1], min_area=min_area,
    )
    return canvas, tgt


# ---------------------------------------------------------------------------
# RADIAL (prisma de cuñas triangulares) — variante experimental de mosaic
# ---------------------------------------------------------------------------


def radial_compose(
    samples: Sequence[tuple[torch.Tensor, dict]],
    *,
    out_h: int,
    out_w: int,
    m_wedges: int,
    center_jitter: float = 0.0,
    min_area: int = 16,
) -> tuple[torch.Tensor, dict]:
    """Compone una imagen partiendo el lienzo en `m_wedges` cuñas angulares que
    convergen en un punto (centro ± jitter) y se extienden radialmente HASTA EL BORDE
    del cuadrado — sin zonas vacías, incluidas las esquinas ("warp total", opción A).

    Cada cuña se rellena con una muestra tomada al azar (con reemplazo) de `samples`;
    el número de FUENTES (n = len(samples)) es independiente de `m_wedges`. Con n=1 y
    m=4 sale la imagen repetida en aspa. Con m=4 la partición es un aspa (X), NO la
    rejilla de mosaic.

    Geometría: cada píxel del lienzo pertenece a la cuña dada por su ángulo respecto al
    punto de convergencia (atan2). Esto rasteriza exactamente m sectores que cubren todo
    el cuadrado; la "deformación tipo pentágono/hexágono estirado al cuadrado" es
    justamente que los sectores llegan al borde en vez de formar un m-gono inscrito.

    Las instancias de la fuente de cada cuña se conservan SOLO donde caen dentro del
    sector (intersección máscara ∩ cuña), se recalcula la caja desde la máscara
    recortada y se descartan las que quedan bajo `min_area`.
    """
    assert m_wedges >= 3, "radial requiere al menos 3 cuñas (m>=3); usa 4 para el aspa"
    n = len(samples)
    assert n >= 1

    dtype = samples[0][0].dtype
    mask_dtype = _as_t(samples[0][1]["masks"]).dtype

    # punto de convergencia con jitter
    cx = out_w * (0.5 + random.uniform(-center_jitter, center_jitter))
    cy = out_h * (0.5 + random.uniform(-center_jitter, center_jitter))

    # ángulo por píxel -> índice de cuña [0, m)
    ys, xs = torch.meshgrid(
        torch.arange(out_h, dtype=torch.float32),
        torch.arange(out_w, dtype=torch.float32),
        indexing="ij",
    )
    ang = torch.atan2(ys - cy, xs - cx)
    ang = (ang + 2 * torch.pi) % (2 * torch.pi)
    # offset de rotación aleatorio: evita que las costuras caigan siempre igual
    rot = random.uniform(0.0, 2 * torch.pi / m_wedges)
    wedge_of_pixel = (((ang + rot) % (2 * torch.pi)) / (2 * torch.pi) * m_wedges).floor().long()
    wedge_of_pixel = wedge_of_pixel.clamp_(0, m_wedges - 1)

    canvas = torch.zeros(3, out_h, out_w, dtype=dtype)
    all_masks: list[torch.Tensor] = []
    all_labels: list[int] = []
    all_iscrowd: list[int] = []

    for w in range(m_wedges):
        img, tgt = samples[random.randrange(n)]  # muestreo con reemplazo
        # cada fuente se escala al lienzo completo; luego se recorta a su cuña
        img_r = F.interpolate(
            img.unsqueeze(0), size=(out_h, out_w), mode="bilinear", align_corners=False
        ).squeeze(0).clamp_(0.0, 1.0)

        sector = wedge_of_pixel == w  # [H,W] bool
        canvas[:, sector] = img_r[:, sector]

        m = _as_t(tgt["masks"])
        if m.numel() == 0 or m.shape[0] == 0:
            continue
        m_r = F.interpolate(
            m.unsqueeze(1).float(), size=(out_h, out_w), mode="nearest"
        ).squeeze(1) > 0.5
        labels = _as_t(tgt["labels"]).tolist()
        iscrowd = _as_t(tgt.get("iscrowd", torch.zeros(m.shape[0]))).tolist()
        for k in range(m_r.shape[0]):
            inst = m_r[k] & sector  # intersección con la cuña
            if not bool(inst.any()):
                continue
            all_masks.append(inst)
            all_labels.append(int(labels[k]))
            all_iscrowd.append(int(iscrowd[k]) if k < len(iscrowd) else 0)

    masks_t = (
        torch.stack(all_masks) if all_masks
        else torch.zeros((0, out_h, out_w), dtype=torch.bool)
    )
    tgt = _rebuild_target(
        masks_t, all_labels, all_iscrowd, out_h, out_w, mask_dtype,
        samples[0][1], min_area=min_area,
    )
    return canvas, tgt


# ---------------------------------------------------------------------------
# MIXUP
# ---------------------------------------------------------------------------


def mixup_compose(
    a: tuple[torch.Tensor, dict],
    b: tuple[torch.Tensor, dict],
    *,
    alpha_beta: float = 1.5,
    min_area: int = 16,
) -> tuple[torch.Tensor, dict]:
    """Mezcla alpha de dos muestras del MISMO shape espacial. Une las anotaciones.

    lambda ~ Beta(alpha_beta, alpha_beta). La imagen resultante es
    lam*A + (1-lam)*B; las instancias de ambas se conservan tal cual (unión), que es
    la convención estándar de mixup en detección: el modelo debe detectar los objetos
    de las dos imágenes en la superposición.
    """
    img_a, tgt_a = a
    img_b, tgt_b = b
    assert img_a.shape == img_b.shape, (
        f"mixup exige mismo shape: {tuple(img_a.shape)} vs {tuple(img_b.shape)}"
    )
    H, W = img_a.shape[-2], img_a.shape[-1]

    lam = float(torch.distributions.Beta(alpha_beta, alpha_beta).sample())
    img = (lam * img_a + (1.0 - lam) * img_b).clamp_(0.0, 1.0)

    mask_dtype = _as_t(tgt_a["masks"]).dtype
    ma = _as_t(tgt_a["masks"]).bool()
    mb = _as_t(tgt_b["masks"]).bool()
    masks = torch.cat([ma, mb], dim=0) if (ma.shape[0] or mb.shape[0]) else \
        torch.zeros((0, H, W), dtype=torch.bool)

    labels = _as_t(tgt_a["labels"]).tolist() + _as_t(tgt_b["labels"]).tolist()
    ica = _as_t(tgt_a.get("iscrowd", torch.zeros(ma.shape[0]))).tolist()
    icb = _as_t(tgt_b.get("iscrowd", torch.zeros(mb.shape[0]))).tolist()
    iscrowd = ica + icb

    tgt = _rebuild_target(
        masks, labels, iscrowd, H, W, mask_dtype, tgt_a, min_area=min_area
    )
    return img, tgt


# ---------------------------------------------------------------------------
# wrapper unificado
# ---------------------------------------------------------------------------


class CompositionAugDataset(Dataset):
    """Envuelve un dataset C1 SIN transforms y aplica scale+flip+composición con
    schedule por época.

    Fases (leídas de la config, ver README):
      - mosaic/mixup activos mientras epoch < mosaic_end_frac * total_epochs.
      - copy-paste activo con probabilidad `cp_p_early` en la fase temprana y
        `cp_p_late` en la fase tardía (típicamente cp_p_late > cp_p_early).

    Orden por muestra:
        1. resuelve fase según EpochState
        2. (fase temprana) con prob p_mosaic: compón mosaic de 4
           y con prob p_mixup: mézclalo (mixup) con OTRO mosaic/muestra  -> anidado
        3. con prob cp_p(fase): copy-paste sobre el resultado
        4. si nada disparó: muestra simple (scale+flip)
    """

    def __init__(
        self,
        base_dataset: Dataset,
        scale_transform: Any | None = None,
        per_sample_transforms: Sequence[Callable] | Callable | None = None,
        *,
        total_epochs: int,
        epoch_state: EpochState | None = None,
        # --- schedule ---
        mosaic_end_frac: float = 0.8,
        # --- modo de composición temprana: 'mosaic' (rejilla 2x2) | 'radial' (cuñas) ---
        early_mode: str = "mosaic",
        # --- mosaic ---
        p_mosaic: float = 1.0,
        mosaic_center_jitter: float = 0.25,
        # --- radial (prisma de cuñas; solo si early_mode='radial') ---
        n_sources: int = 4,          # nº de imágenes fuente (independiente de m)
        m_wedges: int = 4,           # nº de cuñas triangulares (m=4 -> aspa)
        radial_center_jitter: float = 0.15,
        # --- mixup (anidado sobre la composición temprana) ---
        p_mixup: float = 0.5,
        mixup_alpha_beta: float = 1.5,
        # --- copy-paste (rama de copy_paste.py) ---
        cp_p_early: float = 0.15,
        cp_p_late: float = 0.5,
        cp_max_paste_instances: int | None = None,
        cp_min_paste_instances: int = 1,
        cp_min_visible_ratio: float = 0.10,
        cp_blend: str = "gaussian",
        cp_blur_sigma: float = 1.5,
        cp_blur_ksize: int = 5,
        cp_random_translate: bool = False,
        # --- común ---
        min_area: int = 16,
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

        self.total_epochs = max(1, int(total_epochs))
        self.epoch_state = epoch_state if epoch_state is not None else EpochState(0)

        self.mosaic_end_frac = float(mosaic_end_frac)
        self.early_mode = str(early_mode).lower()
        assert self.early_mode in ("mosaic", "radial"), self.early_mode
        self.p_mosaic = float(p_mosaic)
        self.mosaic_center_jitter = float(mosaic_center_jitter)
        self.n_sources = max(1, int(n_sources))
        self.m_wedges = max(3, int(m_wedges))
        self.radial_center_jitter = float(radial_center_jitter)
        self.p_mixup = float(p_mixup)
        self.mixup_alpha_beta = float(mixup_alpha_beta)

        self.cp_p_early = float(cp_p_early)
        self.cp_p_late = float(cp_p_late)
        self.cp_kwargs = dict(
            max_paste_instances=cp_max_paste_instances,
            min_paste_instances=int(cp_min_paste_instances),
            min_visible_ratio=float(cp_min_visible_ratio),
            min_paste_area=int(min_area),
            blend=str(cp_blend),
            blur_sigma=float(cp_blur_sigma),
            blur_ksize=int(cp_blur_ksize),
            random_translate=bool(cp_random_translate),
        )

        self.min_area = int(min_area)
        self.avoid_self_pair = bool(avoid_self_pair)
        self.debug_sample = float(debug_sample)

        for attr in ("coco", "ann_file", "img_dir", "ids", "num_classes",
                     "label_to_name", "cat_id_to_label", "root", "split"):
            if hasattr(base_dataset, attr):
                setattr(self, attr, getattr(base_dataset, attr))

    def __len__(self) -> int:
        return len(self.base)

    # -- transforms --------------------------------------------------------
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
        st = self.scale_transform
        if st is None:
            return None
        if hasattr(st, "sample_h"):
            return int(st.sample_h())
        if hasattr(st, "h_min") and hasattr(st, "h_max"):
            return random.randint(int(st.h_min), int(st.h_max))
        return None

    def _rand_index(self, exclude: int | None = None) -> int:
        n = len(self.base)
        if n <= 1 or exclude is None or not self.avoid_self_pair:
            return random.randrange(n)
        j = exclude
        while j == exclude:
            j = random.randrange(n)
        return j

    def _sample(self, index: int, force_h: int | None):
        img, tgt = self.base[index]
        return self._apply(img, tgt, force_h)

    def _mosaic_at(self, force_h: int | None, out_h: int, out_w: int):
        """Un mosaico de 4 muestras aleatorias, todas escaladas con force_h."""
        idxs = [random.randrange(len(self.base)) for _ in range(4)]
        samples = [self._sample(i, force_h) for i in idxs]
        return mosaic_compose(
            samples, out_h=out_h, out_w=out_w,
            center_jitter=self.mosaic_center_jitter, min_area=self.min_area,
        )

    def _radial_at(self, force_h: int | None, out_h: int, out_w: int):
        """Un prisma radial de `n_sources` fuentes repartidas en `m_wedges` cuñas."""
        idxs = [random.randrange(len(self.base)) for _ in range(self.n_sources)]
        samples = [self._sample(i, force_h) for i in idxs]
        return radial_compose(
            samples, out_h=out_h, out_w=out_w, m_wedges=self.m_wedges,
            center_jitter=self.radial_center_jitter, min_area=self.min_area,
        )

    def _compose_early(self, force_h, out_h, out_w):
        """Composición temprana según early_mode: mosaic (rejilla) o radial (cuñas)."""
        if self.early_mode == "radial":
            return self._radial_at(force_h, out_h, out_w)
        return self._mosaic_at(force_h, out_h, out_w)

    # -- fase --------------------------------------------------------------
    def _is_early(self) -> bool:
        epoch = self.epoch_state.get()
        return epoch < self.mosaic_end_frac * self.total_epochs

    def __getitem__(self, index: int):
        early = self._is_early()
        h = self._shared_h()
        used = []  # para debug

        if early and random.random() < self.p_mosaic:
            # tamaño de salida del mosaico = (h, width) del multiscale, para encajar
            # con el resto del batch. Si no hay scale_transform, usa el de la muestra.
            if h is not None and self.scale_transform is not None:
                out_h = self.scale_transform._round(h) if hasattr(
                    self.scale_transform, "_round") else h
                out_w = self.scale_transform._round(self.scale_transform.width) \
                    if hasattr(self.scale_transform, "_round") \
                    else self.scale_transform.width
            else:
                probe_img, _ = self._sample(index, h)
                out_h, out_w = int(probe_img.shape[-2]), int(probe_img.shape[-1])

            img, tgt = self._compose_early(h, out_h, out_w)
            used.append(self.early_mode)

            # MIXUP anidado: mezcla esta composición con OTRA del mismo modo y tamaño
            if random.random() < self.p_mixup:
                img2, tgt2 = self._compose_early(h, out_h, out_w)
                img, tgt = mixup_compose(
                    (img, tgt), (img2, tgt2),
                    alpha_beta=self.mixup_alpha_beta, min_area=self.min_area,
                )
                used.append("mixup")
        else:
            # muestra simple (destino de un posible copy-paste)
            img, tgt = self._sample(index, h)

        # COPY-PASTE (capa final e independiente)
        cp_p = self.cp_p_early if early else self.cp_p_late
        if random.random() < cp_p:
            j = self._rand_index(exclude=index)
            img_src, tgt_src = self._sample(j, h if h is not None else int(img.shape[-2]))
            # la fuente debe igualar el shape del destino (mosaic ya lo fijó a out_h/out_w)
            if img_src.shape[-2:] != img.shape[-2:]:
                img_src = F.interpolate(
                    img_src.unsqueeze(0), size=img.shape[-2:],
                    mode="bilinear", align_corners=False,
                ).squeeze(0).clamp_(0.0, 1.0)
                m = _as_t(tgt_src["masks"])
                if m.numel():
                    m = F.interpolate(
                        m.unsqueeze(1).float(), size=img.shape[-2:], mode="nearest"
                    ).squeeze(1)
                    tgt_src = dict(tgt_src)
                    tgt_src["masks"] = tv_tensors.Mask((m > 0.5).to(_as_t(tgt["masks"]).dtype))
            img, tgt = copy_paste_compose(img, tgt, img_src, tgt_src, **self.cp_kwargs)
            tgt.pop("_cp_stats", None)
            used.append("copy_paste")

        if self.debug_sample > 0 and random.random() < self.debug_sample:
            n_out = int(_as_t(tgt["masks"]).shape[0])
            print(
                f"[comp_aug] epoch={self.epoch_state.get()} "
                f"{'early' if early else 'late'} idx={index} "
                f"used={used or ['simple']} shape={tuple(img.shape[-2:])} inst={n_out}",
                flush=True,
            )
        return img, tgt


__all__ = [
    "EpochState",
    "CompositionAugDataset",
    "mosaic_compose",
    "radial_compose",
    "mixup_compose",
]
