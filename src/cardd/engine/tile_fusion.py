"""src/cardd/engine/tile_fusion.py

Dueno: M - Fase: val/test (fusion por mascara, alternativa a WBF)

Estrategia de fusion de detecciones por tile ALTERNATIVA a WBF. Mismo hueco que
`fuse_tile_predictions` en inference.py: misma firma y misma salida C2, asi que
el empalme en `run_tiled_inference_coco_results` es elegir 'wbf' o 'graph' por
config. No toca WBF ni ninguna firma existente.

Diferencia con WBF: WBF empareja y fusiona CAJAS; aca el emparejamiento y la
fusion son a nivel de MASCARA. La caja de salida se DERIVA de la mascara
fusionada (en segmentacion de instancias la caja es un subproducto de la
mascara, no al reves). Por eso el `boxes` de entrada por tile no se usa.

--- Mini-contrato (entrada / salida) ---

Entrada (identica a fuse_tile_predictions):
  tile_outputs: list[dict]  # un dict por tile, alineado con tile_metas
      out["scores"]: Tensor[N]              # confianza por deteccion
      out["labels"]: Tensor[N] int64        # clase 1..C
      out["masks"]:  Tensor[N,1,th,tw] f32  # mascara blanda [0,1], coords del tile
      out["boxes"]:  Tensor[N,4]            # presente por compatibilidad; NO se usa
  tile_metas: list[dict]    # TileMeta de tiling.py: offset(x0,y0), tile_size(tw,th),
                            # orig_size(W,H), scale. (scale no se usa: mascaras ya en px de tile)
  fusion_cfg: dict          # ver _fusion_cfg / claves eval.fusion.*

Salida (C2 de imagen entera, identica a fuse_tile_predictions):
  { "boxes":  Tensor[K,4]  xyxy f32   # bbox ajustado de cada mascara fusionada
    "scores": Tensor[K]    f32
    "labels": Tensor[K]    int64
    "masks":  Tensor[K,1,H,W] f32 }   # mascara blanda [0,1] (se umbraliza aguas abajo, igual que WBF)

--- Algoritmo ---
1. Aplanar detecciones de todos los tiles; binarizar cada mascara a `mask_thr`.
2. Grafo: nodo = deteccion. Arista i-j si  misma clase  Y  tiles distintos  Y
   solape de mascara (en coords enteras) >= `match_thr`. Componentes conexas
   (union-find) = instancias.
3. Por grupo: promedio ponderado de las mascaras blandas (peso uniforme o
   feathered) -> mascara fusionada; caja = bbox de esa mascara; score = maximo
   del grupo (o ponderado por area).

Metrica de solape (`match_metric`):
  - 'containment' (default):  inter / min(area_i, area_j)  -- discrimina mejor
    las costuras (una viruta chica queda contenida en la mascara grande del
    tile vecino -> solape alto aunque el IoU seria bajo).
  - 'iou':  inter / (area_i + area_j - inter)  -- clasico.
"""
from __future__ import annotations

import numpy as np
import torch

_MATCH_METRICS = ("containment", "iou")
_MASK_WEIGHTS = ("uniform", "feathered")
_SCORE_REDUCE = ("max", "wavg_area")


def _fusion_cfg(cfg) -> dict:
    """Knobs de fusion por grafo desde config. Todas con default -> un config sin
    `eval.fusion.*` corre igual. Mismo patron mecanismo/valor que _wbf_cfg."""
    from cardd.utils.config import get

    return {
        "match_metric": get(cfg, "eval.fusion.match_metric", "containment"),
        "match_thr": float(get(cfg, "eval.fusion.match_thr", 0.5)),
        "mask_weight": get(cfg, "eval.fusion.mask_weight", "uniform"),
        "mask_thr": float(get(cfg, "eval.fusion.mask_thr", 0.5)),
        "score_reduce": get(cfg, "eval.fusion.score_reduce", "max"),
    }


class _DSU:
    """Union-find para componentes conexas (sin dependencias externas)."""

    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def _feather_window(th: int, tw: int, floor: float = 0.1) -> np.ndarray:
    """Ventana separable tipo Hann (sin^2) con piso: ~`floor` en los bordes del
    tile, 1.0 en el centro. Hace que en el solape cada tile aporte mas donde es
    confiable (centro) y casi nada en el borde -> la costura se difumina sola."""

    def w1(n: int) -> np.ndarray:
        if n <= 1:
            return np.ones(n, dtype=np.float32)
        x = np.linspace(0.0, np.pi, n, dtype=np.float32)
        return (floor + (1.0 - floor) * np.sin(x) ** 2).astype(np.float32)

    return np.outer(w1(th), w1(tw)).astype(np.float32)


def _pair_intersection(di: dict, dj: dict) -> int:
    """Pixeles en comun (mascaras binarias) sobre el rectangulo donde los dos
    tiles se solapan, en coords de imagen entera. 0 si los tiles no se tocan."""
    ax0, ay0, atw, ath = di["x0"], di["y0"], di["tw"], di["th"]
    bx0, by0, btw, bth = dj["x0"], dj["y0"], dj["tw"], dj["th"]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax0 + atw, bx0 + btw), min(ay0 + ath, by0 + bth)
    if ix0 >= ix1 or iy0 >= iy1:
        return 0
    si = di["bin"][iy0 - ay0 : iy1 - ay0, ix0 - ax0 : ix1 - ax0]
    sj = dj["bin"][iy0 - by0 : iy1 - by0, ix0 - bx0 : ix1 - bx0]
    return int(np.logical_and(si, sj).sum())


def _fuse_group_mask(members: list[int], dets: list[dict], H: int, W: int, mask_weight: str) -> np.ndarray:
    """Promedio ponderado, pixel a pixel, de las mascaras blandas del grupo,
    contado solo donde cada tile cubre. Peso uniforme (1) o feathered (ventana)."""
    acc = np.zeros((H, W), dtype=np.float32)
    wsum = np.zeros((H, W), dtype=np.float32)
    for m in members:
        d = dets[m]
        x0, y0, tw, th = d["x0"], d["y0"], d["tw"], d["th"]
        w = _feather_window(th, tw) if mask_weight == "feathered" else np.ones((th, tw), dtype=np.float32)
        acc[y0 : y0 + th, x0 : x0 + tw] += w * d["soft"]
        wsum[y0 : y0 + th, x0 : x0 + tw] += w
    return np.divide(acc, wsum, out=np.zeros_like(acc), where=wsum > 0)


def _empty(H: int, W: int) -> dict:
    return {
        "boxes": torch.zeros(0, 4, dtype=torch.float32),
        "scores": torch.zeros(0, dtype=torch.float32),
        "labels": torch.zeros(0, dtype=torch.int64),
        "masks": torch.zeros(0, 1, H, W, dtype=torch.float32),
    }


@torch.no_grad()
def fuse_tile_predictions_graph(tile_outputs: list[dict], tile_metas: list[dict], fusion_cfg: dict) -> dict:
    """Fusion por grafo de mascaras. Drop-in de fuse_tile_predictions (misma
    firma y misma salida C2). Ver docstring del modulo para el algoritmo."""
    W, H = tile_metas[0]["orig_size"]
    metric = fusion_cfg.get("match_metric", "containment")
    match_thr = float(fusion_cfg.get("match_thr", 0.5))
    mask_weight = fusion_cfg.get("mask_weight", "uniform")
    mask_thr = float(fusion_cfg.get("mask_thr", 0.5))
    score_reduce = fusion_cfg.get("score_reduce", "max")
    if metric not in _MATCH_METRICS:
        raise ValueError(f"match_metric desconocida: {metric!r} (usar {_MATCH_METRICS})")
    if mask_weight not in _MASK_WEIGHTS:
        raise ValueError(f"mask_weight desconocida: {mask_weight!r} (usar {_MASK_WEIGHTS})")
    if score_reduce not in _SCORE_REDUCE:
        raise ValueError(f"score_reduce desconocida: {score_reduce!r} (usar {_SCORE_REDUCE})")

    # 1 - aplanar detecciones (una por mascara), binarizar para el matching
    dets: list[dict] = []
    for ti, (out, meta) in enumerate(zip(tile_outputs, tile_metas)):
        masks = out["masks"]
        n = int(masks.shape[0])
        if n == 0:
            continue
        x0, y0 = meta["offset"]
        tw, th = meta["tile_size"]
        labels = out["labels"].detach().cpu().numpy()
        scores = out["scores"].detach().cpu().numpy().astype(np.float32)
        masks_np = masks.detach().cpu().numpy()
        for k in range(n):
            soft = masks_np[k, 0].astype(np.float32)
            binm = soft >= mask_thr
            area = int(binm.sum())
            if area == 0:
                continue  # mascara vacia tras umbral -> no aporta
            dets.append(
                {
                    "label": int(labels[k]),
                    "score": float(scores[k]),
                    "tile": ti,
                    "x0": x0, "y0": y0, "tw": tw, "th": th,
                    "soft": soft, "bin": binm, "area": area,
                }
            )

    if not dets:
        return _empty(H, W)

    # 2 - grafo: misma clase + tiles distintos + solape >= umbral
    dsu = _DSU(len(dets))
    for i in range(len(dets)):
        for j in range(i + 1, len(dets)):
            di, dj = dets[i], dets[j]
            if di["label"] != dj["label"]:
                continue
            if di["tile"] == dj["tile"]:
                continue  # dentro de un tile ya corrio NMS: objetos genuinamente distintos
            inter = _pair_intersection(di, dj)
            if inter == 0:
                continue
            if metric == "iou":
                overlap = inter / (di["area"] + dj["area"] - inter)
            else:  # containment
                overlap = inter / min(di["area"], dj["area"])
            if overlap >= match_thr:
                dsu.union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(len(dets)):
        groups.setdefault(dsu.find(i), []).append(i)

    # 3 - fusion por grupo -> mascara blanda, caja desde mascara, score=max
    boxes, out_scores, out_labels, mask_tensors = [], [], [], []
    for members in groups.values():
        soft = _fuse_group_mask(members, dets, H, W, mask_weight)
        ys, xs = np.where(soft >= mask_thr)
        if xs.size == 0:
            continue
        boxes.append([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)])
        member_scores = [dets[m]["score"] for m in members]
        if score_reduce == "wavg_area":
            aw = np.array([dets[m]["area"] for m in members], dtype=np.float32)
            sc = float((np.array(member_scores, dtype=np.float32) * aw).sum() / aw.sum())
        else:  # max
            sc = float(max(member_scores))
        out_scores.append(sc)
        out_labels.append(dets[members[0]]["label"])
        mask_tensors.append(torch.from_numpy(soft))

    if not boxes:
        return _empty(H, W)

    masks_out = torch.zeros(len(mask_tensors), 1, H, W, dtype=torch.float32)
    for i, m in enumerate(mask_tensors):
        masks_out[i, 0] = m
    return {
        "boxes": torch.as_tensor(boxes, dtype=torch.float32),
        "scores": torch.as_tensor(out_scores, dtype=torch.float32),
        "labels": torch.as_tensor(out_labels, dtype=torch.int64),
        "masks": masks_out,
    }