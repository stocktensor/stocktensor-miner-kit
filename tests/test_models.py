from __future__ import annotations

import pytest
from stocktensor.protocol import HORIZONS, MAX_PRICE_RATIO, Forecast

from stocktensor_kit.models import ZOO, available_zoo, gbm, load_model
from stocktensor_kit.models.base import make_forecast

from .conftest import synthetic_points

MODELS = sorted(ZOO)


def check(forecast: Forecast, reference: float) -> None:
    low, point, high, p_up = forecast.validate(reference)  # raises if invalid
    assert 0 < low <= point <= high
    assert 0.0 <= p_up <= 1.0
    assert reference / MAX_PRICE_RATIO <= low and high <= reference * MAX_PRICE_RATIO


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("horizon", sorted(HORIZONS))
def test_valid_on_synthetic(name: str, horizon: str) -> None:
    if name == "gbm" and not gbm.available():
        pytest.skip("scikit-learn not installed ([ml] extra)")
    points = synthetic_points()
    predict = ZOO[name].predict
    for as_of in (points[-1][0], points[len(points) // 2][0] + 1_234):
        history = [p for p in points if p[0] <= as_of]
        reference = history[-1][1]
        check(predict("SYN", HORIZONS[horizon], history, reference, as_of), reference)


@pytest.mark.parametrize("name", MODELS)
def test_valid_on_recorded(name: str, fixture_histories) -> None:
    if name == "gbm" and not gbm.available():
        pytest.skip("scikit-learn not installed ([ml] extra)")
    for symbol, history in fixture_histories.items():
        for as_of in history.times[50::40]:
            reference = history.price_at(as_of)
            for seconds in HORIZONS.values():
                check(
                    ZOO[name].predict(symbol, seconds, history.points_until(as_of), reference, as_of),
                    reference,
                )


@pytest.mark.parametrize("name", MODELS)
def test_valid_without_history(name: str) -> None:
    check(ZOO[name].predict("NEW", 3_600, [], 42.0, 1_790_000_000), 42.0)


def test_make_forecast_keeps_order_after_rounding() -> None:
    forecast = make_forecast(0.0000004, 0.0, 1e-9, 0.99)
    check(forecast, 0.0000004)
    assert forecast.p_up == "0.9800"


def test_load_model_by_name_and_path() -> None:
    assert load_model("vol_bands")[1] is ZOO["vol_bands"].predict
    name, model = load_model("stocktensor_kit.models.momentum:predict")
    assert model is ZOO["momentum"].predict and name.endswith(":predict")
    with pytest.raises(ValueError):
        load_model("nope")


def test_available_zoo_matches_ml_extra() -> None:
    assert ("gbm" in available_zoo()) == gbm.available()
