"""CarDD · utils/registry.py — registro nombre→clase (Fase 0, Equipo P).

Contrato C3: cada módulo enchufable se registra por nombre y se construye por
nombre desde el YAML (C4). Así la ablación queda limpia —un módulo por fila,
sin tocar el código del Equipo M—.

Puntos de inserción: backbone, neck, head, loss (los 4 del contrato), más
detector (para Cascade, a4) y transform (para multi-scale, a2, del Equipo D).
"""
from __future__ import annotations

from typing import Any, Callable, Iterator, KeysView


class Registry:
    """Mapa nombre -> clase/función, con construcción desde config.

    Ejemplo:
        BACKBONES = Registry("backbone")

        @BACKBONES.register()                  # usa el __name__ de la clase
        class ConvNeXtV2Tiny: ...

        @BACKBONES.register("convnextv2_t")    # nombre explícito
        class ConvNeXtV2TinyAlias: ...

        obj = BACKBONES.build({"name": "convnextv2_t", "depth": 9})
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._registry: dict[str, Callable[..., Any]] = {}

    @property
    def name(self) -> str:
        return self._name

    def register(self, name: str | None = None) -> Callable[[Callable], Callable]:
        """Decorador: registra la clase/función bajo `name` (o su __name__)."""

        def _decorator(obj: Callable[..., Any]) -> Callable[..., Any]:
            key = name or obj.__name__
            if key in self._registry:
                raise KeyError(
                    f"'{key}' ya está registrado en el registry '{self._name}'."
                )
            self._registry[key] = obj
            return obj

        return _decorator

    def get(self, key: str) -> Callable[..., Any]:
        """Devuelve la clase/función registrada bajo `key`."""
        if key not in self._registry:
            disponibles = ", ".join(sorted(self._registry)) or "(vacío)"
            raise KeyError(
                f"'{key}' no está en el registry '{self._name}'. "
                f"Disponibles: {disponibles}."
            )
        return self._registry[key]

    def build(self, cfg: dict[str, Any], **kwargs: Any) -> Any:
        """Construye un objeto desde un dict de config.

        `cfg` debe traer la clave 'name' (el nombre registrado); el resto de
        claves se pasan al constructor. `kwargs` inyecta dependencias en
        runtime (p. ej. in_channels que solo M conoce al construir el modelo).
        """
        if "name" not in cfg:
            raise KeyError(
                f"La config para el registry '{self._name}' necesita la clave "
                f"'name'. Recibido: {cfg}."
            )
        cfg = dict(cfg)  # copia defensiva: no mutamos el dict del usuario
        key = cfg.pop("name")
        return self.get(key)(**cfg, **kwargs)

    def __contains__(self, key: str) -> bool:
        return key in self._registry

    def __iter__(self) -> Iterator[str]:
        return iter(self._registry)

    def __len__(self) -> int:
        return len(self._registry)

    def keys(self) -> KeysView[str]:
        return self._registry.keys()

    def __repr__(self) -> str:
        return f"Registry(name='{self._name}', items={sorted(self._registry)})"


# Puntos de inserción del contrato C3. M expone estos hooks; C/D registran piezas.
BACKBONES = Registry("backbone")
NECKS = Registry("neck")
HEADS = Registry("head")
LOSSES = Registry("loss")
DETECTORS = Registry("detector")
TRANSFORMS = Registry("transform")

__all__ = [
    "Registry",
    "BACKBONES",
    "NECKS",
    "HEADS",
    "LOSSES",
    "DETECTORS",
    "TRANSFORMS",
]