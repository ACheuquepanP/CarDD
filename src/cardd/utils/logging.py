"""CarDD · utils/logging.py — logging de consola/archivo (Fase 0, Equipo P).

Logs de texto reproducibles. El SummaryWriter de TensorBoard NO va aquí: lo
crea el trainer (Equipo M, Fase 4); esto es solo el log textual transversal.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def get_logger(
    name: str = "cardd",
    level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Devuelve un logger con salida a consola y, opcionalmente, a archivo.

    Idempotente: si el logger ya tiene handlers, los reutiliza (no duplica
    líneas si se llama varias veces).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger

    fmt = logging.Formatter(_FMT, datefmt=_DATEFMT)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger
