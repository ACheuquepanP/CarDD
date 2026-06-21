"""CarDD · utils/config.py — carga YAML + merge base←override (Fase 0, P).

Contrato C4: el código no cambia entre experimentos, solo el YAML. Cada config
de ablación hereda de base.yaml vía la clave especial '_base_' y solo declara
sus deltas. base.yaml es inmutable entre filas.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

_BASE_KEY = "_base_"


def _deep_merge(base: dict, override: dict) -> dict:
    """Mezcla recursiva: override gana. Los dicts anidados se fusionan; listas
    y escalares se reemplazan. Devuelve un dict nuevo (no muta los argumentos)."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(
            f"El YAML '{path}' debe ser un mapa en la raíz, no {type(data).__name__}."
        )
    return data


def load_config(path: str | Path, _seen: set[Path] | None = None) -> dict:
    """Carga un YAML resolviendo la herencia '_base_'.

    Si el YAML declara '_base_: ruta' (relativa al propio archivo, o una lista
    de rutas), carga esa(s) base(s) primero y mezcla el archivo actual encima
    (override gana). Detecta herencia circular.
    """
    path = Path(path).resolve()
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"Herencia circular en _base_ detectada en {path}.")
    _seen = _seen | {path}

    raw = _load_yaml(path)
    bases = raw.pop(_BASE_KEY, None)
    if bases is None:
        return raw

    if isinstance(bases, str):
        bases = [bases]

    merged: dict = {}
    for b in bases:
        merged = _deep_merge(merged, load_config(path.parent / b, _seen))
    return _deep_merge(merged, raw)


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    """Acceso por clave punteada: get(cfg, 'train.batch_size')."""
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


__all__ = ["load_config", "get"]
