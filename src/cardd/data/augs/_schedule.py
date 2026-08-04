"""CarDD · src/cardd/data/augs/_schedule.py

Sistema de programación temporal (schedule) de la augmentación.

Modelo:
  - El training tiene un progreso global `prog ∈ [0,1]` (epoch/total_epochs).
  - Cada aug tiene una VENTANA [inicio, fin] en fracción de épocas: solo compite en
    el sorteo mientras prog ∈ ventana.
  - Un valor programado (peso, o un parámetro de intensidad) va de `start` a `end` a
    lo largo de la ventana, con la FORMA dada por una curva. La DIRECCIÓN la da el
    signo de (end - start): sube si end>start, baja si end<start, mantiene si iguales.

Curvas (forma de la interpolación, todas van de 0 a 1 conforme el progreso local
va de 0 a 1):
  - linear : c(t) = t                          (transición uniforme)
  - cosine : c(t) = (1 - cos(pi*t)) / 2         (S suave, simétrica; sirve subiendo o
             bajando sin necesidad de seno/espejo)
  - log    : c(t) = log(1 + t*(e-1)) ...        (rápido al principio, lento al final)
  - none   : c(t) = 0  -> el valor se queda en `start` (sin cambio)

La curva NO sabe si el valor sube o baja: solo dicta el ritmo. La dirección emerge de
start vs end en `interp`. Por eso no hacen falta seno ni exponencial: cosine es
simétrica y log fija su asimetría, la dirección la pone el signo de (end-start).
"""
from __future__ import annotations

import math


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def curve_value(curve: str, t: float) -> float:
    """Forma de la curva evaluada en t∈[0,1] -> [0,1]."""
    t = _clamp01(t)
    if curve == "linear":
        return t
    if curve == "cosine":
        return (1.0 - math.cos(math.pi * t)) / 2.0
    if curve == "log":
        # normalizado: log(1 + t*(e-1)) va de log(1)=0 a log(e)=1
        return math.log(1.0 + t * (math.e - 1.0))
    if curve in ("none", None, ""):
        return 0.0
    raise ValueError(f"curva desconocida: {curve!r} (linear|cosine|log|none)")


def local_progress(prog: float, window) -> float:
    """Progreso local dentro de la ventana [w0,w1] -> [0,1]. Fuera de la ventana:
    0 si antes, 1 si después (para que valores 'end' persistan tras cerrarse)."""
    w0, w1 = float(window[0]), float(window[1])
    if prog <= w0:
        return 0.0
    if prog >= w1:
        return 1.0
    if w1 <= w0:
        return 1.0
    return (prog - w0) / (w1 - w0)


def in_window(prog: float, window) -> bool:
    """True si prog está dentro de [w0, w1] (inclusive)."""
    return float(window[0]) <= prog <= float(window[1])


def interp(start: float, end: float, curve: str, prog: float, window) -> float:
    """Valor programado en el instante `prog`:
        v = start + (end - start) * curve(local_progress)
    La dirección la da (end - start). Fuera de ventana devuelve start (antes) o end
    (después) por cómo local_progress satura."""
    t = local_progress(prog, window)
    return start + (end - start) * curve_value(curve, t)


def resolve_schedule(spec: dict, base_value: float, window):
    """Normaliza un bloque decay/enhance a (start, end, curve, param).

    - `start` implícito = base_value (el peso de la aug, o el valor del param en
      params). Se puede sobrescribir con `start` explícito.
    - `end` explícito; si falta, = start (sin cambio).
    - `curve` default 'none' (sin cambio).
    - `param` (solo enhance): qué parámetro mueve. None para decay de peso.
    - `window` propia opcional; si falta, usa la de la aug.
    Devuelve dict {start, end, curve, param, window} o None si el spec es vacío/none.
    """
    if not spec:
        return None
    curve = str(spec.get("curve", "none")).lower()
    if curve in ("none", ""):
        return None
    start = float(spec.get("start", base_value))
    end = float(spec.get("end", start))
    param = spec.get("param", None)
    win = spec.get("window", window)
    return {"start": start, "end": end, "curve": curve, "param": param, "window": win}


def eval_schedule(sched, prog: float, base_value: float) -> float:
    """Evalúa un schedule resuelto (de resolve_schedule) en `prog`. Si sched es None,
    devuelve base_value (sin cambio)."""
    if sched is None:
        return base_value
    return interp(sched["start"], sched["end"], sched["curve"], prog, sched["window"])
