"""Adapters that plug kit models into the subnet miner (``neurons.miner``).

The miner loads ``--model package.module:callable`` and calls it as
``model(request: ForecastRequest, history)``, where ``history`` is
``[(updated_at, price), ...]`` for the last 7 days. Kit models use the
signature ``predict(asset, horizon_seconds, history, reference_price, as_of)``,
so this module exposes one adapter per zoo model as a module attribute:

    stx-miner --model stocktensor_kit.serve:momentum ...

For your own kit-style model, set ``STX_KIT_MODEL=my_pkg.my_module:predict``
and pass ``--model stocktensor_kit.serve:from_env`` (``stx-serve`` does this).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence

from stocktensor.protocol import HORIZONS, Forecast, ForecastRequest

from .models import ZOO, load_model
from .models.base import Model

MinerModel = Callable[[ForecastRequest, Sequence[tuple[float, float]]], Forecast]
ENV_MODEL = "STX_KIT_MODEL"


def adapt(model: Model) -> MinerModel:
    """Wrap a kit model so the subnet miner can call it."""

    def miner_model(request: ForecastRequest, history: Sequence[tuple[float, float]]) -> Forecast:
        points = [(int(t), float(p)) for t, p in history if t <= request.as_of]
        return model(
            request.asset,
            HORIZONS[request.horizon],
            points,
            float(request.reference_price),
            request.as_of,
        )

    return miner_model


def from_env(request: ForecastRequest, history: Sequence[tuple[float, float]]) -> Forecast:
    spec = os.environ.get(ENV_MODEL)
    if not spec:
        raise RuntimeError(f"set {ENV_MODEL}=module:callable")
    _, model = load_model(spec)
    return adapt(model)(request, history)


def __getattr__(name: str) -> MinerModel:
    if name in ZOO:
        return adapt(ZOO[name].predict)
    raise AttributeError(name)
