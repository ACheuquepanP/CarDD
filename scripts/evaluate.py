"""CarDD · scripts/evaluate.py

Dueno: Equipo M · E - Fase: 4 - Contrato: C2 (consume) -> C5 (produce)

Evalua un checkpoint sobre un split y emite el dict C5 (a stdout + JSON).
Reparto del trabajo en este archivo compartido:

  Mitad de E (lista, abajo):
    - construye el loader de eval (D) y saca el GT de loader.dataset.coco
    - corre el loop de inferencia generico de torchvision (model(images) -> C2)
    - serializa C2 a COCO y calcula C5 con cardd.evaluation
    - imprime y guarda el resultado

  Costura de M (Fase 3, un solo punto):
    build_eval_model(cfg, weights) -> nn.Module en eval(), en device, tal que
    model(list[Tensor[3,H,W]]) devuelva la salida nativa de MaskRCNN:
    list[dict] por imagen con boxes(xyxy), scores, labels(1..N), masks[K,1,H,W].
    Mientras M no la publique en cardd.engine.inference, el script avisa claro
    y sale; la parte de E ya esta testeada con un dataset sintetico.

Nada se hardcodea: split, salida y device salen de la config / CLI.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cardd.evaluation.coco_eval import evaluate
from cardd.utils.config import get, load_config
from cardd.utils.logging import get_logger
from cardd.utils.seed import set_seed


def _pick_device(cfg) -> str:
    """Device de la config si esta; si no, cuda cuando exista, si no cpu."""
    dev = get(cfg, "eval.device")
    if dev:
        return dev
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_eval_model(cfg, weights: str, device: str):
    """Costura de M: trae el modelo listo para inferencia.

    Espera cardd.engine.inference.build_eval_model(cfg, weights, device).
    Si M aun no la implemento, corta con un mensaje accionable en vez de un
    traceback opaco.
    """
    try:
        from cardd.engine.inference import build_eval_model
    except ImportError as exc:  # el modulo es un stub todavia
        raise SystemExit(
            "scripts/evaluate.py: falta la costura de M. Se necesita "
            "cardd.engine.inference.build_eval_model(cfg, weights, device) -> "
            "nn.Module en eval() que produzca C2 (salida nativa de MaskRCNN). "
            f"Detalle: {exc}"
        ) from exc
    return build_eval_model(cfg, weights, device)


def collect_coco_results(cfg, split: str, weights: str, logger):
    """Corre la inferencia sobre el split y devuelve (dataset, coco_results, image_ids).

    STREAMING: serializa la predicción de cada imagen a COCO-results (RLE) DENTRO
    del loop y acumula solo eso; las máscaras densas se sueltan en cada iteración.
    Acumular C2 crudo (máscaras del tamaño de la imagen) de todas las imágenes del
    split desborda la RAM en filas con muchas detecciones (p. ej. a1_focal); en RLE
    es órdenes de magnitud menos. torchvision ya capa en box_detections_per_img=100,
    así que el peso real era el formato denso, no la cantidad.

    El loop es genérico de torchvision; lo único de M es el modelo. El GT para
    evaluar sale del dataset (loader.dataset.coco), no del target augmentado.
    """
    import torch

    from cardd.data.build import build_dataloader
    from cardd.evaluation.coco_eval import predictions_to_coco_results

    device = _pick_device(cfg)
    loader = build_dataloader(cfg, split)
    dataset = loader.dataset  # CarDDCocoDataset: expone .coco / .cat_id_to_label / .label_to_name
    label_to_cat_id = {label: cid for cid, label in dataset.cat_id_to_label.items()}
    model = _build_eval_model(cfg, weights, device)
    model.eval()

    coco_results: list[dict] = []
    image_ids: list[int] = []
    with torch.inference_mode():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            outputs = model(images)  # C2: list[dict] por imagen
            for out, tgt in zip(outputs, targets):
                img_id = int(tgt["image_id"])
                # serializa YA a RLE; la máscara densa de esta imagen se libera al
                # salir del scope, no se acumula
                coco_results.extend(predictions_to_coco_results([out], [img_id], label_to_cat_id))
                image_ids.append(img_id)
    logger.info("inferencia '%s': %d imagenes, %d detecciones",
                split, len(image_ids), len(coco_results))
    return dataset, coco_results, image_ids


def _format_c5(c5: dict) -> str:
    """Resumen legible para stdout: agregados, por escala y por clase,
    en mask y box, a AP@[.50:.95] / AP50 / AP75."""
    f = lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else "  --  "

    lines: list[str] = []
    lines.append(
        f"mask_AP={f(c5['mask_AP'])}  box_AP={f(c5['box_AP'])}  "
        f"AP50={f(c5['AP50'])}  AP75={f(c5['AP75'])}"
    )
    lines.append(
        "  agregado:"
        f"  mask 50:95={f(c5['mask_AP'])}  50={f(c5['AP50'])}  75={f(c5['AP75'])}"
        f"  |  box 50:95={f(c5['box_AP'])}"
        f"  50={f(c5.get('box_AP50'))}  75={f(c5.get('box_AP75'))}"
    )

    sc = c5["per_scale"]
    lines.append(f"  APS={f(sc['APS'])}  APM={f(sc['APM'])}  APL={f(sc['APL'])}")
    lines.append("  per_scale (50:95 / 50 / 75):")
    for kind in ("mask", "box"):
        block = sc.get(kind)
        if not block:
            continue
        for band in ("small", "medium", "large"):
            b = block[band]
            lines.append(
                f"    {kind:<5} {band:<7} "
                f"AP={f(b['AP'])}  AP50={f(b['AP50'])}  AP75={f(b['AP75'])}"
            )

    lines.append("  per_class (50:95 / 50 / 75):")
    for name, v in c5["per_class"].items():
        lines.append(
            f"    {name:<16} "
            f"mask={f(v['mask_AP'])}/{f(v.get('mask_AP50'))}/{f(v.get('mask_AP75'))}  "
            f"box={f(v['box_AP'])}/{f(v.get('box_AP50'))}/{f(v.get('box_AP75'))}"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evalua un checkpoint y emite C5.")
    ap.add_argument("--config", required=True, help="ruta a la config YAML")
    ap.add_argument("--weights", required=True, help="checkpoint a evaluar")
    ap.add_argument("--split", default="test", help="split a evaluar (default: test)")
    ap.add_argument("--out", default=None, help="ruta del JSON C5 (default: output.dir)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(get(cfg, "seed", 0))
    logger = get_logger("cardd.evaluate")

    dataset, coco_results, image_ids = collect_coco_results(cfg, args.split, args.weights, logger)
    cat_id_to_name = {
        cid: dataset.label_to_name[label] for cid, label in dataset.cat_id_to_label.items()
    }
    c5 = evaluate(dataset.coco, coco_results, cat_id_to_name, image_ids=image_ids)

    print(_format_c5(c5))

    out = Path(args.out) if args.out else Path(get(cfg, "output.dir", "outputs")) / f"eval_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(c5, indent=2, ensure_ascii=False))
    logger.info("C5 guardado en %s", out)


if __name__ == "__main__":
    main()
