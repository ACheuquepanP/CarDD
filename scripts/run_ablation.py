"""CarDD · scripts/run_ablation.py

Dueno: Equipo E - Fase: 6 - Contrato: C5 (consume) -> tabla comparativa

Orquesta el estudio de ablacion (una fila = un cambio) y produce la tabla
comparativa, el entregable de E como arbitro. Dos responsabilidades separadas:

  Motor de tabla (E, puro, testeable sin torch):
    lee los C5 (eval_*.json) de cada fila, calcula deltas vs baseline y
    renderiza markdown (absolutos + deltas). Cada hipotesis se juzga contra su
    metrica objetivo: a1_focal -> per_class; a2 -> APS ('mueve el small');
    a4 -> box_AP / AP75; etc.

  Orquestacion (costura, no importa torch):
    por cada config, lanza por subprocess  scripts/train.py (M)  y luego
    scripts/evaluate.py (E), que deja el C5 de la fila en results/<fila>.json.
    Con --from-results se SALTA correr y solo tabula lo ya producido (modo
    util hoy, antes de que exista el train de M).

No hardcodea la ciencia: no asume que tal fila 'gana'; solo muestra numeros y
deltas. La lectura de si la hipotesis se cumplio la hace la persona sobre la
tabla.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

# Columnas de la tabla = contrato C5 aplanado (mascara primaria + escalas).
_HEAD = ("mask_AP", "box_AP", "AP50", "AP75")
_SCALE = ("APS", "APM", "APL")
_COLS = _HEAD + _SCALE


def row_name(config: str | Path) -> str:
    """Nombre de fila = stem del YAML (a0_baseline.yaml -> a0_baseline)."""
    return Path(config).stem


def _valid(v: float) -> bool:
    """-1.0 es el centinela COCO de 'sin GT en la banda'; NaN es clase ausente.
    Ninguno cuenta para un delta."""
    return v is not None and not math.isnan(v) and v != -1.0


def _row_metrics(c5: dict) -> dict[str, float]:
    vals = {m: float(c5[m]) for m in _HEAD}
    vals.update({s: float(c5["per_scale"][s]) for s in _SCALE})
    return vals


def build_table(results: dict[str, dict], baseline: str) -> list[dict]:
    """results: {fila: C5}. Devuelve filas ordenadas (baseline primero) con
    valores y deltas vs baseline por columna."""
    base = _row_metrics(results[baseline]) if baseline in results else None
    table: list[dict] = []
    for name, c5 in results.items():
        vals = _row_metrics(c5)
        deltas = {
            c: (vals[c] - base[c] if base and _valid(vals[c]) and _valid(base[c]) else None)
            for c in _COLS
        }
        table.append({"name": name, "values": vals, "deltas": deltas, "baseline": name == baseline})
    table.sort(key=lambda r: (not r["baseline"]))  # estable: baseline primero
    return table


def _fmt(v: float) -> str:
    return "—" if not _valid(v) else f"{v:.4f}"


def _fmt_delta(v) -> str:
    return "—" if v is None else f"{v:+.4f}"


def _md_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def render_markdown(table: list[dict]) -> str:
    """Dos tablas: valores absolutos y deltas vs baseline."""
    header = _md_row(["config", *_COLS])
    sep = _md_row(["---"] * (len(_COLS) + 1))

    abs_rows = [
        _md_row([r["name"] + (" *(base)*" if r["baseline"] else ""),
                 *[_fmt(r["values"][c]) for c in _COLS]])
        for r in table
    ]
    delta_rows = [
        _md_row([r["name"], *["—" if r["baseline"] else _fmt_delta(r["deltas"][c]) for c in _COLS]])
        for r in table
    ]

    return (
        "### Ablación — valores absolutos\n\n"
        + "\n".join([header, sep, *abs_rows])
        + "\n\n### Ablación — Δ vs baseline\n\n"
        + "\n".join([header, sep, *delta_rows])
        + "\n"
    )


def render_per_class(results: dict[str, dict]) -> str:
    """Tabla aparte de mask_AP por clase (para juzgar a1_focal y los custom)."""
    classes = sorted({c for c5 in results.values() for c in c5["per_class"]})
    header = _md_row(["config", *classes])
    sep = _md_row(["---"] * (len(classes) + 1))
    rows = [
        _md_row([name, *[_fmt(c5["per_class"].get(cl, {}).get("mask_AP", float("nan"))) for cl in classes]])
        for name, c5 in results.items()
    ]
    return "### Ablación — mask_AP por clase\n\n" + "\n".join([header, sep, *rows]) + "\n"


def load_results(results_dir: Path, configs: list[str]) -> dict[str, dict]:
    """Lee results_dir/<fila>.json por cada config. Omite (con aviso) las que
    falten, para poder tabular un estudio parcial."""
    out: dict[str, dict] = {}
    for cfg in configs:
        name = row_name(cfg)
        path = results_dir / f"{name}.json"
        if path.exists():
            out[name] = json.loads(path.read_text())
        else:
            print(f"[run_ablation] aviso: falta {path}, fila '{name}' omitida", file=sys.stderr)
    return out


def run_row(config: str, weights_dir: Path, results_dir: Path, split: str) -> None:
    """Costura de orquestacion: train (M) + evaluate (E) por subprocess."""
    name = row_name(config)
    ckpt = weights_dir / f"{name}.pth"
    here = Path(__file__).parent
    # 1) entrenar la fila — entrega de M (scripts/train.py). Si es stub, falla aqui.
    subprocess.run(
        [sys.executable, str(here / "train.py"), "--config", config, "--out", str(ckpt)],
        check=True,
    )
    # 2) evaluar la fila — E (ya implementado) -> results_dir/<fila>.json
    subprocess.run(
        [sys.executable, str(here / "evaluate.py"), "--config", config,
         "--weights", str(ckpt), "--split", split, "--out", str(results_dir / f"{name}.json")],
        check=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Corre el estudio de ablación y arma la tabla.")
    ap.add_argument("--configs", nargs="+", default=None,
                    help="YAMLs de ablación (default: configs/ablation/*.yaml)")
    ap.add_argument("--results-dir", default="outputs/ablation", help="donde viven/van los <fila>.json")
    ap.add_argument("--weights-dir", default="outputs/weights", help="checkpoints por fila")
    ap.add_argument("--baseline", default="a0_baseline", help="fila de referencia para los deltas")
    ap.add_argument("--split", default="test", help="split de evaluación")
    ap.add_argument("--from-results", action="store_true",
                    help="no corre nada: solo tabula los <fila>.json ya existentes")
    ap.add_argument("--out", default=None, help="markdown de salida (default: results-dir/ablation_table.md)")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    configs = args.configs or sorted(str(p) for p in (repo / "configs/ablation").glob("*.yaml"))
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not args.from_results:
        weights_dir = Path(args.weights_dir)
        weights_dir.mkdir(parents=True, exist_ok=True)
        for cfg in configs:
            run_row(cfg, weights_dir, results_dir, args.split)

    results = load_results(results_dir, configs)
    if not results:
        raise SystemExit("[run_ablation] no hay resultados que tabular.")

    table = build_table(results, args.baseline)
    md = render_markdown(table) + "\n" + render_per_class(results)
    print(md)

    out = Path(args.out) if args.out else results_dir / "ablation_table.md"
    out.write_text(md, encoding="utf-8")
    print(f"[run_ablation] tabla guardada en {out}", file=sys.stderr)


if __name__ == "__main__":
    main()