"""src/cardd/data/coco_dataset.py

Dueno: Equipo D - Fase: 1 - Contrato: C1 (datos -> M/E)

Dataset que lee CarDD en formato COCO y produce el sample C1 en el formato
`target` nativo de torchvision MaskRCNN. Decisiones congeladas con M y E:
  - caja: se toma de la bbox del COCO (xywh -> xyxy); no se recomputa.
  - clases: 0 = fondo; clases reales 1..N (contiguas, leidas de las anotaciones).
  - imagen: float32 RGB en [0,1] SIN normalizar (la normaliza el modelo).
  - tv_tensors: caja y mascara viajan como BoundingBoxes/Mask, de modo que los
    transforms v2 las mueven juntas y de forma consistente.

C1 (por muestra):
    image:  Tensor[3,H,W] float32 en [0,1]
    target: {
        boxes:    BoundingBoxes[N,4] xyxy float32
        labels:   Tensor[N] int64 en 1..N
        masks:    Mask[N,H,W] uint8 {0,1}
        image_id: int
        area:     Tensor[N] float32
        iscrowd:  Tensor[N] uint8
    }
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


def _resolve_split(root: Path, split: str) -> tuple[Path, Path]:
    """Localiza (json_anotaciones, carpeta_imagenes) del layout CarDD-COCO."""
    ann_candidates = [
        root / "annotations" / f"instances_{split}2017.json",
        root / "annotations" / f"instances_{split}.json",
        root / "annotations" / f"{split}.json",
    ]
    img_candidates = [root / f"{split}2017", root / split, root / "images" / split]
    ann = next((p for p in ann_candidates if p.exists()), None)
    img = next((p for p in img_candidates if p.is_dir()), None)
    if ann is None or img is None:
        raise FileNotFoundError(
            f"No encuentro el split '{split}' bajo {root} "
            f"(esperaba annotations/instances_{split}2017.json + {split}2017/)."
        )
    return ann, img


class CarDDCocoDataset(Dataset):
    """CarDD-COCO -> C1. Las clases se leen de las anotaciones (fuente de verdad)."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        transforms: Callable | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transforms = transforms

        ann_file, self.img_dir = _resolve_split(self.root, split)
        self.coco = COCO(str(ann_file))
        self.ids = sorted(self.coco.imgs.keys())

        cat_ids = sorted(self.coco.getCatIds())
        # 0 reservado a fondo -> clases reales en 1..N
        self.cat_id_to_label: dict[int, int] = {cid: i + 1 for i, cid in enumerate(cat_ids)}
        self.label_to_name: dict[int, str] = {
            self.cat_id_to_label[c["id"]]: c["name"] for c in self.coco.loadCats(cat_ids)
        }
        self.num_classes = len(cat_ids)  # sin contar el fondo (M usa num_classes + 1)

    def __len__(self) -> int:
        return len(self.ids)

    def _load_image(self, img_id: int) -> Image.Image:
        info = self.coco.imgs[img_id]
        return Image.open(self.img_dir / info["file_name"]).convert("RGB")

    def __getitem__(self, index: int):
        img_id = self.ids[index]
        pil = self._load_image(img_id)
        W, H = pil.size

        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))

        boxes: list[list[float]] = []
        labels: list[int] = []
        masks: list[np.ndarray] = []
        areas: list[float] = []
        iscrowd: list[int] = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:  # guarda: nunca emitimos una caja degenerada
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_label[ann["category_id"]])
            masks.append(self.coco.annToMask(ann))
            areas.append(ann.get("area", float(w * h)))
            iscrowd.append(ann.get("iscrowd", 0))

        image = F.to_dtype(F.to_image(pil), torch.float32, scale=True)  # [3,H,W] en [0,1]

        boxes_t = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        masks_np = np.stack(masks) if masks else np.zeros((0, H, W), dtype=np.uint8)
        target = {
            "boxes": tv_tensors.BoundingBoxes(
                boxes_t, format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=(H, W)
            ),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": tv_tensors.Mask(torch.as_tensor(masks_np, dtype=torch.uint8)),
            "image_id": int(img_id),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.uint8),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target