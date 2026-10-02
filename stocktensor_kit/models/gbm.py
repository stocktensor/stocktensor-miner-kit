"""Gradient-boosting quantiles (optional, ``pip install stocktensor-miner-kit[ml]``).

Three scikit-learn ``GradientBoostingRegressor(loss="quantile")`` models learn
the 10th, 50th and 90th percentile of the log return over the horizon from
simple features (recent returns, recent volatility, session, hour). They are
trained only on outcomes that were known at ``as_of`` and retrained at most
once a day. Falls back to ``vol_bands`` when scikit-learn is missing or there
is too little history. An example of the plumbing, not a tuned model.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any

from stocktensor.protocol import Forecast
from stocktensor.sessions import session_at

from . import vol_bands
from .base import STEP, Points, grid_returns, make_forecast, normal_cdf, rms

NAME = "gbm"
DESCRIPTION = "scikit-learn quantile gradient boosting on simple features (optional [ml] extra)."
TRAIN_LOOKBACK = 60 * 86_400
MIN_SAMPLES = 200
RETRAIN_EVERY = 86_400
SESSIONS = ("regular", "pre", "post", "overnight")


def _sklearn() -> Any:
    try:
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError:
        return None
    return GradientBoostingRegressor


def available() -> bool:
    return _sklearn() is not None


@dataclass
class _Fit:
    trained_at: int
    models: tuple[Any, Any, Any]


_cache: dict[tuple[str, int], _Fit] = {}


def _features(times: list[int], prices: list[float], t: int) -> list[float] | None:
    def at(u: int) -> float | None:
        i = bisect_right(times, u) - 1
        return prices[i] if i >= 0 else None

    hourly = [at(t - k * 3_600) for k in range(25)]
    if any(p is None or p <= 0 for p in hourly):
        return None
    series = [float(p) for p in hourly if p is not None]
    now = series[0]
    vol = rms([math.log(series[k] / series[k + 1]) for k in range(24)])
    session = session_at(t)
    return [
        math.log(now / series[1]),
        math.log(now / series[6]),
        math.log(now / series[24]),
        vol,
        *[1.0 if session == s else 0.0 for s in SESSIONS],
        float((t // 3_600) % 24),
    ]


def _train(times: list[int], prices: list[float], horizon: int, as_of: int) -> _Fit | None:
    regressor = _sklearn()
    if regressor is None:
        return None
    rows, targets = [], []
    for t, _, _ in grid_returns(list(zip(times, prices, strict=True)), as_of - horizon, TRAIN_LOOKBACK):
        if t + horizon > as_of:
            break
        features = _features(times, prices, t)
        later = prices[bisect_right(times, t + horizon) - 1]
        now = prices[bisect_right(times, t) - 1]
        if features is None or now <= 0 or later <= 0:
            continue
        rows.append(features)
        targets.append(math.log(later / now))
    if len(rows) < MIN_SAMPLES:
        return None
    models = tuple(
        regressor(
            loss="quantile", alpha=a, n_estimators=80, max_depth=3, learning_rate=0.05, random_state=0
        ).fit(rows, targets)
        for a in (0.1, 0.5, 0.9)
    )
    return _Fit(as_of, models)  # type: ignore[arg-type]


def predict(
    asset: str, horizon_seconds: int, history: Points, reference_price: float, as_of: int
) -> Forecast:
    fallback = vol_bands.predict(asset, horizon_seconds, history, reference_price, as_of)
    if not available() or len(history) < 2:
        return fallback
    times = [t for t, _ in history]
    prices = [p for _, p in history]
    key = (asset, horizon_seconds)
    fit = _cache.get(key)
    if fit is None or as_of < fit.trained_at or as_of - fit.trained_at >= RETRAIN_EVERY:
        fit = _train(times, prices, horizon_seconds, as_of - (as_of % STEP))
        if fit is None:
            return fallback
        _cache[key] = fit
    features = _features(times, prices, as_of)
    if features is None:
        return fallback
    q10, q50, q90 = sorted(float(m.predict([features])[0]) for m in fit.models)
    sigma = max((q90 - q10) / 2.0 / 1.2815515655446004, 1e-5)
    return make_forecast(reference_price, q50, sigma, normal_cdf(q50 / sigma))


def reset() -> None:
    """Forget trained models (tests, or switching datasets)."""
    _cache.clear()
