"""Inferencia y construcción del modelo de evaluación.

dueño: M · fase: 4 · contrato: C2 (produce)

Aquí vive la **costura** que pidió E: ``build_eval_model(cfg, weights, device)``
devuelve el detector en ``eval()``, en ``device``, con pesos cargados, tal que
``model(images)`` entregue C2 nativo. ``run_inference`` recorre un loader (C1) y
junta las predicciones C2 + ``image_ids`` para pasárselas a E
(``evaluation.coco_eval.evaluate_from_dataset``).

La serialización a COCO la hace E; M solo entrega los tensores nativos del
detector (boxes xyxy, scores, labels 1..N, masks [K,1,H,W] en [0,1]).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

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