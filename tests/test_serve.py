from __future__ import annotations

import pytest
from stocktensor.protocol import ForecastRequest

from stocktensor_kit import serve
from stocktensor_kit.models import ZOO

from .conftest import synthetic_points

REQUEST = ForecastRequest(
    task_id="t",
    asset="NVDA",
    horizon="1d",
    as_of=1_790_000_000,
    reference_price="236.22",
    deadline=1_790_000_012,
)


def history():
    return [
        (float(t), p) for t, p in synthetic_points(start=1_789_000_000, days=12) if t <= REQUEST.as_of + 999
    ]


def test_zoo_adapters_match_direct_calls() -> None:
    points = [(int(t), p) for t, p in history() if t <= REQUEST.as_of]
    for name in ("vol_bands", "momentum", "session_aware"):
        adapted = getattr(serve, name)(REQUEST, history())
        direct = ZOO[name].predict("NVDA", 86_400, points, 236.22, REQUEST.as_of)
        assert adapted == direct


def test_unknown_attribute() -> None:
    with pytest.raises(AttributeError):
        serve.nope  # noqa: B018


def test_from_env(monkeypatch) -> None:
    monkeypatch.delenv(serve.ENV_MODEL, raising=False)
    with pytest.raises(RuntimeError):
        serve.from_env(REQUEST, history())
    monkeypatch.setenv(serve.ENV_MODEL, "stocktensor_kit.models.vol_bands:predict")
    assert serve.from_env(REQUEST, history()) == serve.vol_bands(REQUEST, history())


def test_subnet_miner_can_load_adapter() -> None:
    miner = pytest.importorskip("neurons.miner")
    model = miner.load_model("stocktensor_kit.serve:momentum")
    forecast = model(REQUEST, history())
    forecast.validate(236.22)
