"""src/cardd/data/tiling.py

Dueno: Equipo D - Fase: val/test (feed de inferencia con tiling) - Contrato: geometria de tile -> M

Tilea cada imagen ENTERA de val/test de forma determinista para inferencia por
tile + WBF (lo hace M). NO toca train (crops en ruta fija) ni la GT (las metricas
se calculan sobre la imagen entera contra el COCO original, ds.coco).

Geometria (impuesta por C, acordada con M):
  - tile rectangular = (W/2, H/2): la mitad de cada eje, con el aspect ratio de
    la imagen (tile_frac = 0.5).
  - matriz n x n de posiciones: starts = linspace(0, W/2, n) x linspace(0, H/2, n),
    redondeados a entero; el ultimo start de cada eje se clava al borde para que
    el barrido cubra de 0 a W (y 0 a H) sin huecos.
  - n impar -> existe el tile central (linspace incluye el punto medio).
  - determinista: sin azar; dos corridas -> exactamente los mismos tiles.

Contrato de geometria (lo que M consume para reproyectar tile -> imagen entera):
    meta = {
        image_id:  int            # imagen padre (COCO original)
        offset:    (x0, y0) int   # esquina sup-izq del tile en coords de la imagen entera
        tile_size: (tw, th) int   # tamano real del tile en px
        orig_size: (W, H)   int   # tamano de la imagen padre (canvas de pegado)
        scale:     float = 1.0    # reservado; 1.0 si no se reescala el tile
    }
Reproyeccion de M:  box_entera = box_tile / scale + (x0, y0)
                    mask: pegar el tile (th, tw) en el canvas (H, W) en (x0, y0)
"""
from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import functional as F

from cardd.data.coco_dataset import resolve_split
from cardd.utils.config import get

_MISSING = object()


def _cfg_get(cfg, key: str, default=_MISSING):
    try:
        return get(cfg, key)
    except Exception:
        if default is _MISSING:
            raise
        return default


class TileMeta(TypedDict):
    image_id: int
    offset: tuple[int, int]
    tile_size: tuple[int, int]
    orig_size: tuple[int, int]
    scale: float


def _axis_starts(length: int, tile: int, n: int) -> list[int]:
    """n posiciones de inicio en un eje: linspace(0, length-tile, n) redondeado,
    con el ultimo clavado al borde. length-tile == length//2 cuando tile=length/2,
    asi que el barrido cubre 0..length sin huecos."""
    if n <= 1:
        return [0]
    last = length - tile  # = floor(length/2) con tile = round(length/2)
    starts = [round(i * last / (n - 1)) for i in range(n)]
    starts[-1] = last  # garantiza cierre exacto al borde
    return starts


def tile_geometry(W: int, H: int, n: int, tile_frac: float = 0.5) -> list[TileMeta]:
    """Genera la matriz n x n de tiles (solo geometria, sin pixeles). Determinista."""
    tw = max(1, round(W * tile_frac))
    th = max(1, round(H * tile_frac))
    xs = _axis_starts(W, tw, n)
    ys = _axis_starts(H, th, n)
    tiles: list[TileMeta] = []
    for y0 in ys:
        for x0 in xs:
            tiles.append(
                {
                    "image_id": -1,  # lo fija el dataset al conocer la imagen
                    "offset": (x0, y0),
                    "tile_size": (tw, th),
                    "orig_size": (W, H),
                    "scale": 1.0,
                }
            )
    return tiles


def tile_image(image: torch.Tensor, n: int, tile_frac: float = 0.5):
    """Corta `image` [C,H,W] en n x n tiles segun la geometria. Devuelve
    [(tile_tensor, meta), ...]. La imagen de entrada se asume ya en [0,1] RGB."""
    _, H, W = image.shape
    out = []
    for meta in tile_geometry(W, H, n, tile_frac):
        x0, y0 = meta["offset"]
        tw, th = meta["tile_size"]
        tile = image[:, y0 : y0 + th, x0 : x0 + tw]
        out.append((tile, meta))
    return out


class CarDDTileDataset(Dataset):
    """Feed de inferencia: una entrada por TILE. El agrupamiento por imagen padre
    lo hace el collate (ver build_tile_dataloader). GT NO se tilea."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        n: int = 3,
        tile_frac: float = 0.5,
    ) -> None:
        if split == "train":
            raise ValueError("El tiling es solo para val/test; train usa crops (C1).")
        self.root = Path(root)
        self.split = split
        self.n = n
        self.tile_frac = tile_frac

        from pycocotools.coco import COCO

        self.ann_file, self.img_dir = resolve_split(self.root, split)
        self.coco = COCO(str(self.ann_file))

        # mapa category_id -> label contiguo (1..C; 0 = fondo), igual convencion
        # que el dataset C1. El driver de inferencia lo consume para invertirlo
        # a label -> category_id al serializar a COCO-results.
        cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_label: dict[int, int] = {cid: i + 1 for i, cid in enumerate(cat_ids)}
        _id_to_name = {c["id"]: c["name"] for c in self.coco.loadCats(cat_ids)}
        self.label_to_name: dict[int, str] = {
            label: _id_to_name[cid] for cid, label in self.cat_id_to_label.items()
        }

        # indice plano: (image_id, tile_meta) por cada tile de cada imagen
        self.index: list[tuple[int, TileMeta]] = []
        for img_id in sorted(self.coco.imgs.keys()):
            info = self.coco.imgs[img_id]
            W, H = info["width"], info["height"]
            for meta in tile_geometry(W, H, n, tile_frac):
                meta["image_id"] = img_id
                self.index.append((img_id, meta))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        img_id, meta = self.index[idx]
        info = self.coco.imgs[img_id]
        pil = Image.open(self.img_dir / info["file_name"]).convert("RGB")
        image = F.to_dtype(F.to_image(pil), torch.float32, scale=True)  # [3,H,W] en [0,1]
        x0, y0 = meta["offset"]
        tw, th = meta["tile_size"]
        tile = image[:, y0 : y0 + th, x0 : x0 + tw]
        return tile, meta


def tile_collate(batch):
    """Todos los tiles de UNA imagen padre -> (tuple(tiles), tuple(metas)).
    M infiere por tile y fusiona por imagen (WBF)."""
    tiles, metas = zip(*batch)
    return tuple(tiles), tuple(metas)


def build_tile_dataloader(cfg, split: str) -> DataLoader:
    """DataLoader de inferencia con tiling. Un batch = todos los tiles de una
    imagen padre (collate agrupa por image_id). Determinista (sin shuffle).
    Lee data.tiling.n (default 3) y data.tiling.tile_frac (default 0.5)."""
    root = Path(_cfg_get(cfg, "data.root"))
    n = int(_cfg_get(cfg, "data.tiling.n", 3))
    tile_frac = float(_cfg_get(cfg, "data.tiling.tile_frac", 0.5))
    num_workers = int(_cfg_get(cfg, "data.num_workers", 4))

    ds = CarDDTileDataset(root, split, n=n, tile_frac=tile_frac)

    # agrupa indices planos por imagen padre -> un batch por imagen
    groups: dict[int, list[int]] = {}
    for i, (img_id, _meta) in enumerate(ds.index):
        groups.setdefault(img_id, []).append(i)
    batch_sampler = [groups[k] for k in sorted(groups)]

    return DataLoader(
        ds,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=tile_collate,
        pin_memory=True,
    )