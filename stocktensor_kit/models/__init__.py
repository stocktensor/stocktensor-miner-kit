"""Baseline model zoo and model loading.

Every model is a plain function with the signature in
:class:`stocktensor_kit.models.base.Model`. Load one by zoo name
(``vol_bands``) or by import path (``my_pkg.my_module:predict``).
"""

from __future__ import annotations

import importlib
from types import ModuleType

from . import gbm, momentum, session_aware, vol_bands
from .base import Model

ZOO: dict[str, ModuleType] = {m.NAME: m for m in (vol_bands, momentum, session_aware, gbm)}


def available_zoo() -> dict[str, Model]:
    """Zoo models that can run here (``gbm`` needs the ``[ml]`` extra)."""
    return {name: module.predict for name, module in ZOO.items() if name != "gbm" or gbm.available()}


def load_model(spec: str) -> tuple[str, Model]:
    """Resolve ``spec`` to ``(name, predict)``."""
    if spec in ZOO:
        return spec, ZOO[spec].predict
    if ":" not in spec:
        raise ValueError(f"unknown model {spec!r}: use one of {sorted(ZOO)} or module:callable")
    module_name, attr = spec.split(":", 1)
    module = importlib.import_module(module_name)
    model = getattr(module, attr)
    if not callable(model):
        raise TypeError(f"{spec} is not callable")
    return spec, model


__all__ = ["ZOO", "Model", "available_zoo", "load_model"]
