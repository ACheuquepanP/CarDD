"""tools/analyze_dataset.py

Dueno: Equipo D - Fase: 1 - Contrato: - (EDA, no produce C1)

Analisis exploratorio de CarDD-COCO. Hito (orden de creacion, paso 4):
CONFIRMAR que el dataset esta dominado por objetos pequenos -> justifica la fila
de ablacion a2 (multi-scale). De paso reporta el desbalance de clases (motiva a1
focal) y la densidad de instancias por imagen.

Buckets de escala = umbrales COCO sobre el area de segmentacion:
    small  : area <  32^2   (1024 px^2)
    medium : 32^2 <= area < 96^2  (9216 px^2)
    large  : area >= 96^2
Son los mismos cortes que usara per_scale_metrics de E (APS/APM/APL).

Higiene: data.root sale de config; las clases se leen de las anotaciones.

Uso:
    python tools/analyze_dataset.py --config configs/base.yaml [--plot out_dir]
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO

from cardd.utils.config import get, load_config
from cardd.utils.logging import get_logger

log = get_logger("cardd.data.analyze")

DEFAULT_SPLITS = ("train", "val", "test")
SMALL = 32 ** 2   # 1024
LARGE = 96 ** 2   # 9216
_MISSING = object()


def _cfg_get(cfg, key: str, default=_MISSING):
    try:
        return get(cfg, key)
    except Exception:
        if default is _MISSING:
            raise
        return default


def resolve_split(root: Path, split: str) -> Path | None:
    for p in (
        root / "annotations" / f"instances_{split}2017.json",
        root / "annotations" / f"instances_{split}.json",
        root / "annotations" / f"{split}.json",
    ):
        if p.exists():
            return p
    return None


def _bucket(area: float) -> str:
    if area < SMALL:
        return "small"
    if area < LARGE:
        return "medium"
    return "large"


def analyze_split(ann_file: Path, split: str) -> dict:
    coco = COCO(str(ann_file))
    id_to_name = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    scale: Counter = Counter()
    per_class: Counter = Counter()
    per_class_scale: dict[str, Counter] = defaultdict(Counter)
    areas: list[float] = []
    anns_per_img: Counter = Counter()

    for ann in coco.anns.values():
        name = id_to_name[ann["category_id"]]
        area = float(ann.get("area", 0.0))
        b = _bucket(area)
        scale[b] += 1
        per_class[name] += 1
        per_class_scale[name][b] += 1
        areas.append(area)
        anns_per_img[ann["image_id"]] += 1

    inst_per_img = np.asarray([anns_per_img.get(i, 0) for i in coco.imgs], dtype=float)
    return {
        "split": split,
        "num_images": len(coco.imgs),
        "num_annotations": len(coco.anns),
        "scale": scale,
        "per_class": per_class,
        "per_class_scale": per_class_scale,
        "areas": np.asarray(areas, dtype=float),
        "inst_per_img": inst_per_img,
    }


def report(s: dict) -> None:
    n = max(s["num_annotations"], 1)
    log.info("== split '%s' ==", s["split"])
    log.info("imagenes: %d | anotaciones: %d", s["num_images"], s["num_annotations"])

    for b in ("small", "medium", "large"):
        c = s["scale"][b]
        log.info("  escala %-6s: %6d  (%4.1f%%)", b, c, 100 * c / n)

    ipi = s["inst_per_img"]
    if len(ipi):
        log.info("instancias/imagen: media %.2f, max %d", ipi.mean(), int(ipi.max()))

    a = s["areas"]
    if len(a):
        p10, p50, p90 = np.percentile(a, [10, 50, 90])
        log.info("area px^2: p10 %.0f | mediana %.0f | p90 %.0f", p10, p50, p90)

    log.info("por clase (n | %% small):")
    for name, c in s["per_class"].most_common():
        sm = s["per_class_scale"][name]["small"]
        log.info("  %-14s %6d  | %4.0f%% small", name, c, 100 * sm / max(c, 1))


def plot(all_stats: list[dict], out_dir: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    areas = np.concatenate([s["areas"] for s in all_stats if len(s["areas"])])
    per_class: Counter = Counter()
    for s in all_stats:
        per_class.update(s["per_class"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    sqrt_area = np.sqrt(areas[areas > 0])
    ax1.hist(sqrt_area, bins=60)
    ax1.axvline(np.sqrt(SMALL), color="k", ls="--", lw=1, label="small|medium")
    ax1.axvline(np.sqrt(LARGE), color="k", ls=":", lw=1, label="medium|large")
    ax1.set_xlabel("sqrt(area) [px]")
    ax1.set_ylabel("# instancias")
    ax1.set_title("Distribucion de escala")
    ax1.legend()

    names = [n for n, _ in per_class.most_common()]
    counts = [per_class[n] for n in names]
    ax2.barh(names[::-1], counts[::-1])
    ax2.set_xlabel("# instancias")
    ax2.set_title("Instancias por clase")

    fig.tight_layout()
    path = out / "analyze_dataset.png"
    fig.savefig(path, dpi=120)
    log.info("figura escrita en %s", path)


def main() -> int:
    parser = argparse.ArgumentParser(description="EDA de CarDD-COCO (confirma objetos pequenos).")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--plot", default=None, help="Carpeta donde guardar las figuras (opcional).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = Path(_cfg_get(cfg, "data.root"))
    if not root.exists():
        log.error("data.root no existe: %s", root)
        return 1

    splits = args.splits or list(_cfg_get(cfg, "data.splits", DEFAULT_SPLITS))

    all_stats: list[dict] = []
    total_scale: Counter = Counter()
    total_n = 0
    for split in splits:
        ann_file = resolve_split(root, split)
        if ann_file is None:
            log.warning("split '%s': sin JSON de anotaciones, lo salto", split)
            continue
        s = analyze_split(ann_file, split)
        report(s)
        all_stats.append(s)
        total_scale += s["scale"]
        total_n += s["num_annotations"]

    if total_n:
        small_pct = 100 * total_scale["small"] / total_n
        med_pct = 100 * total_scale["medium"] / total_n
        large_pct = 100 * total_scale["large"] / total_n
        log.info("== agregado (%d anotaciones) ==", total_n)
        log.info("small %.1f%% | medium %.1f%% | large %.1f%%", small_pct, med_pct, large_pct)
        if small_pct >= 25:
            log.info("objetos pequenos confirmados -> a2 (multi-scale) justificado")
        else:
            log.info("small por debajo del 25%%; revisar si a2 (multi-scale) aporta")

    if args.plot and all_stats:
        plot(all_stats, args.plot)

    print("OK - analisis completo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())