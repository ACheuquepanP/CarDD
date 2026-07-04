"""Inferencia y construcción del modelo de evaluación.

dueño: M · fase: 4 · contrato: C2 (produce)

Aquí vive la **costura** que pidió E: ``build_eval_model(cfg, weights, device)``
devuelve el detector en ``eval()``, en ``device``, con pesos cargados, tal que
``model(images)`` entregue C2 nativo. ``run_inference`` recorre un loader (C1) y
junta las predicciones C2 + ``image_ids`` para pasárselas a E
(``evaluation.coco_eval.evaluate_from_dataset``).

La serialización a COCO la hace E; M solo entrega los tensores nativos del
detector (boxes xyxy, scores, labels 1..N, masks [K,1,H,W] en [0,1]).

--- Fase val/test con tiling ---

El camino tileado (``run_tiled_inference_coco_results``) fusiona las detecciones
por tile con la estrategia de GRAFO DE MASCARAS (``cardd.engine.tile_fusion``):
empareja por solape de mascara (misma clase + tiles distintos), promedia las
mascaras blandas del grupo y deriva la caja de la mascara fusionada. Reemplaza a
WBF (fusion de cajas), que no daba el resultado esperado en segmentacion.

Estas funciones son ADITIVAS: no tocan ninguna firma del motor base ni el camino
sin tiling (``build_eval_model`` y ``run_inference*`` siguen igual), asi que
volver al modelo plano no requiere nada. El camino tileado se consume desde
``scripts/evaluate_tiled.py``.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from cardd.engine.tile_fusion import fuse_tile_predictions_graph, _fusion_cfg
from cardd.utils.config import get
from cardd.utils.logging import get_logger
from cardd.models.detectors.mask_rcnn import build_detector

log = get_logger("cardd.engine.inference")


def _load_weights(model: nn.Module, weights: str | Path, device: torch.device) -> nn.Module:
    ckpt = torch.load(weights, map_location=device)
    # acepta tanto un checkpoint del trainer ({"model": state_dict, ...}) como un state_dict crudo
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        log.warning("state_dict: %d claves faltantes (p.ej. %s)", len(missing), missing[:3])
    if unexpected:
        log.warning("state_dict: %d claves inesperadas (p.ej. %s)", len(unexpected), unexpected[:3])
    return model


def build_eval_model(cfg, weights: str | Path | None = None, device: str | torch.device = "cuda") -> nn.Module:
    """Detector en eval(), en device, con pesos cargados (la costura de E).

    Args:
        cfg: config del proyecto (C4).
        weights: ruta a un checkpoint del trainer o a un state_dict. Si es None,
            el modelo queda sin entrenar (la salida C2 es válida en forma pero el
            AP será ≈0 — útil para validar el *cableado* de C2, no la calidad).
        device: 'cuda' / 'cpu' / torch.device.
    """
    device = torch.device(device)
    model = build_detector(cfg)
    if weights is not None:
        _load_weights(model, weights, device)
        log.info("modelo de evaluación listo con pesos de %s", weights)
    else:
        log.warning("build_eval_model SIN pesos: C2 de un modelo no entrenado (AP≈0).")
    return model.to(device).eval()


@torch.no_grad()
def run_inference(
    model: nn.Module,
    dataloader,
    device: str | torch.device = "cuda",
    amp: bool = True,
    max_images: int | None = None,
):
    """Recorre un loader C1 y devuelve ``(predictions, image_ids)`` para E.

    - ``predictions``: ``list[dict]`` C2 por imagen, en CPU (libera VRAM).
    - ``image_ids``: ``list[int]`` alineado, tomado de ``target['image_id']`` (C1).
    - ``max_images``: si se da, corta tras N imágenes. Sirve para el smoke
      end-to-end (AP≈1) sobre el mismo subconjunto sobreajustado: a E se le pasa
      igual el dataset completo, y ``image_ids`` restringe la evaluación a esas N.

    El orden se preserva 1:1, que es lo que ``evaluate_from_dataset`` espera.
    """
    device = torch.device(device)
    model.eval()
    use_amp = amp and device.type == "cuda"

    predictions: list[dict] = []
    image_ids: list[int] = []
    for images, targets in dataloader:  # collate de D: (tuple(images), tuple(targets))
        images = [img.to(device, non_blocking=True) for img in images]
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
        for out, tgt in zip(outputs, targets):
            predictions.append({k: v.detach().cpu() for k, v in out.items()})
            image_ids.append(int(tgt["image_id"]))
        if max_images is not None and len(image_ids) >= max_images:
            return predictions[:max_images], image_ids[:max_images]
    return predictions, image_ids


@torch.no_grad()
def run_inference_coco_results(
    model: nn.Module,
    dataloader,
    dataset,
    device: str | torch.device = "cuda",
    amp: bool = True,
):
    """Inferencia **memory-safe**: serializa cada imagen a COCO-results (RLE) en el
    momento y suelta la máscara densa por iteración. Devuelve ``(coco_results, image_ids)``.

    A diferencia de ``run_inference`` (que acumula las predicciones C2 DENSAS de
    todo el split en RAM y puede dar OOM en val/test completos), aquí solo se
    acumulan los dicts compactos. ``dataset`` debe exponer ``cat_id_to_label`` (C1)
    para el mapa ``label→category_id``. La serialización es la de E
    (``predictions_to_coco_results``); este es el único loop de inferencia que
    debería usarse para evaluar splits grandes (val en loop, evaluate.py).
    """
    from cardd.evaluation.coco_eval import predictions_to_coco_results

    device = torch.device(device)
    model.eval()
    use_amp = amp and device.type == "cuda"
    label_to_cat_id = {label: cid for cid, label in dataset.cat_id_to_label.items()}

    coco_results: list[dict] = []
    image_ids: list[int] = []
    for images, targets in dataloader:
        images = [img.to(device, non_blocking=True) for img in images]
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
        for out, tgt in zip(outputs, targets):
            iid = int(tgt["image_id"])
            pred = {k: v.detach().cpu() for k, v in out.items()}   # densa, de UNA imagen
            coco_results.extend(predictions_to_coco_results([pred], [iid], label_to_cat_id))
            image_ids.append(iid)
            # `pred` (incluida la máscara densa) se libera al pasar de iteración
    return coco_results, image_ids


# --------------------------------------------------------------------------- #
# Tiling (val/test) — fusión por grafo de máscaras; no afecta el motor base    #
# --------------------------------------------------------------------------- #

def resolve_tile_batch_size(cfg) -> int | None:
    """Tamaño del sub-batch para el forward por tile (trade-off VRAM/velocidad).

    - ``eval.tile_batch_size`` explícito en config → se usa tal cual (``0`` o
      valores ``<=0`` significan "todos los tiles juntos", ver abajo).
    - Sin esa clave (default) → **hereda ``data.tiling.n``**: procesa de a
      ``n`` tiles por forward (una "fila" de la grilla n×n) en vez de los
      ``n²`` juntos.
    - Si tampoco hay ``data.tiling.n`` → ``None`` = todos los tiles de la
      imagen en un solo forward (comportamiento previo a este knob).

    Nunca cambia el resultado final (mismas detecciones, misma fusión) — solo
    el pico de VRAM y la velocidad del forward.
    """
    explicit = get(cfg, "eval.tile_batch_size", None)
    if explicit is not None:
        explicit = int(explicit)
        return explicit if explicit > 0 else None
    n = get(cfg, "data.tiling.n", None)
    return int(n) if n else None


def run_tiles_chunked(
    model: nn.Module,
    tiles: list[torch.Tensor],
    chunk_size: int | None,
    device: torch.device,
    use_amp: bool,
) -> list[dict]:
    """Corre ``model`` sobre ``tiles`` en sub-batches de ``chunk_size`` (o todos
    juntos si es ``None``/``>= len(tiles)``). Mismo resultado que un solo
    forward con todos los tiles — solo cambia cuánta VRAM pico se usa."""
    if not chunk_size or chunk_size >= len(tiles):
        with torch.autocast(device_type=device.type, enabled=use_amp):
            return model([t.to(device, non_blocking=True) for t in tiles])
    outputs: list[dict] = []
    for i in range(0, len(tiles), chunk_size):
        chunk = [t.to(device, non_blocking=True) for t in tiles[i : i + chunk_size]]
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs.extend(model(chunk))
    return outputs


def reproject_boxes(boxes: torch.Tensor, offset: tuple[int, int], scale: float = 1.0) -> torch.Tensor:
    """box_tile -> box_entera, según el contrato de D: ``box/scale + (x0,y0)``."""
    x0, y0 = offset
    out = boxes.detach().to(torch.float32).clone()
    if out.numel():
        out[:, 0::2] = out[:, 0::2] / scale + x0
        out[:, 1::2] = out[:, 1::2] / scale + y0
    return out


@torch.no_grad()
def concat_tile_predictions_no_fusion(tile_outputs: list[dict], tile_metas: list[dict]) -> dict:
    """Baseline de ablación (fusión vs. sin-fusión). Reproyecta TODAS las
    detecciones de todos los tiles a coords de imagen entera y las concatena
    **sin fusionar** — los duplicados de costura quedan tal cual (doble conteo).

    Sirve para aislar el efecto de la fusión: si esta salida y la de
    ``fuse_tile_predictions_graph`` dan un AP igual de bajo, el problema NO es la
    fusión (es reproyección/geometría o las detecciones por tile). Si esta da
    razonable y la fusión lo derrumba, el problema está en la fusión.

    Mismo formato de salida C2 (boxes/scores/labels/masks[K,1,H,W] en coords
    enteras). Cada detección conserva su propia máscara, pegada en el offset de
    su tile.
    """
    W, H = tile_metas[0]["orig_size"]

    boxes_parts, scores_parts, labels_parts = [], [], []
    mask_lookup: list[tuple[dict, torch.Tensor]] = []

    for out, meta in zip(tile_outputs, tile_metas):
        n = out["boxes"].shape[0]
        if n == 0:
            continue
        boxes_parts.append(reproject_boxes(out["boxes"], meta["offset"], meta.get("scale", 1.0)))
        scores_parts.append(out["scores"].detach().to(torch.float32))
        labels_parts.append(out["labels"].detach().to(torch.int64))
        for k in range(n):
            mask_lookup.append((meta, out["masks"][k]))

    if not boxes_parts:
        return {
            "boxes": torch.zeros(0, 4),
            "scores": torch.zeros(0),
            "labels": torch.zeros(0, dtype=torch.int64),
            "masks": torch.zeros(0, 1, H, W),
        }

    boxes = torch.cat(boxes_parts)
    scores = torch.cat(scores_parts)
    labels = torch.cat(labels_parts)

    masks_out = torch.zeros(len(mask_lookup), 1, H, W, dtype=torch.float32)
    for i, (meta, tile_mask) in enumerate(mask_lookup):
        x0, y0 = meta["offset"]
        tw, th = meta["tile_size"]
        masks_out[i, 0, y0 : y0 + th, x0 : x0 + tw] = tile_mask[0]

    return {
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
        "masks": masks_out,
    }


@torch.no_grad()
def run_tiled_inference_coco_results(
    model: nn.Module,
    tile_dataloader,
    dataset,
    cfg,
    device: str | torch.device = "cuda",
    amp: bool = True,
):
    """Análogo tileado de ``run_inference_coco_results``: memory-safe, RLE por
    imagen. Cada batch de ``tile_dataloader`` = TODOS los tiles de UNA imagen
    padre (collate de D, ``tile_collate``). Por batch: inferencia por tile →
    fusión por grafo de máscaras → C2 de imagen entera → serializa a
    COCO-results → suelta la máscara densa antes del próximo batch.

    ``dataset`` es el dataset de tiles de D (``CarDDTileDataset``); solo se usa
    para ``cat_id_to_label`` (mismo contrato que ``run_inference_coco_results``).
    ``cfg`` es la config del proyecto, de donde se leen los knobs de fusión
    (``eval.fusion.*`` — ver ``_fusion_cfg``; todos tienen default) y el tamaño
    de sub-batch por tile (``eval.tile_batch_size`` — ver
    ``resolve_tile_batch_size``; por default hereda ``data.tiling.n``).
    """
    from cardd.evaluation.coco_eval import predictions_to_coco_results

    device = torch.device(device)
    model.eval()
    use_amp = amp and device.type == "cuda"
    fusion_cfg = _fusion_cfg(cfg)
    chunk_size = resolve_tile_batch_size(cfg)
    label_to_cat_id = {label: cid for cid, label in dataset.cat_id_to_label.items()}

    coco_results: list[dict] = []
    image_ids: list[int] = []
    for tiles, metas in tile_dataloader:  # collate de D: todos los tiles de UNA imagen
        outputs = run_tiles_chunked(model, list(tiles), chunk_size, device, use_amp)
        outputs_cpu = [{k: v.detach().cpu() for k, v in o.items()} for o in outputs]
        fused = fuse_tile_predictions_graph(outputs_cpu, list(metas), fusion_cfg)
        img_id = int(metas[0]["image_id"])
        coco_results.extend(predictions_to_coco_results([fused], [img_id], label_to_cat_id))
        image_ids.append(img_id)
        del outputs, outputs_cpu, fused  # el canvas (H,W) de esta imagen se libera ya
    return coco_results, image_ids


if __name__ == "__main__":
    # Smoke sin dataset ni pesos: confirma que la costura arma un modelo eval que produce C2.
    cfg = {
        "data": {
            "num_classes": 6,
            "image_min_size": 640,
            "image_max_size": 1024,
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
        },
        "model": {
            "detector": {"name": "mask_rcnn"},
            "backbone": {"name": "convnextv2_tiny", "pretrained": False},
        },
    }
    model = build_eval_model(cfg, weights=None, device="cpu")
    imgs = [torch.rand(3, 480, 640)]
    with torch.no_grad():
        out = model(imgs)[0]
    print("C2 keys:", sorted(out.keys()))
    print("modelo en eval:", not model.training)