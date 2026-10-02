"""Momentum: extrapolate a shrunk recent drift, with volatility bands around it.

Drift = mean hourly log return over the last 3 days, multiplied by 0.5
(shrinkage: short-term drift is mostly noise). The interval is centred on the
drifted price, and p_up = Φ(drift_h / σ_h).
"""

from __future__ import annotations

import math

from stocktensor.protocol import Forecast

from .base import Points, grid_returns, make_forecast, normal_cdf, open_steps, step_sigma

NAME = "momentum"
DESCRIPTION = "Shrunk 3-day drift plus 14-day volatility bands."
VOL_LOOKBACK = 14 * 86_400
DRIFT_LOOKBACK = 3 * 86_400
SHRINK = 0.5


def predict(
    asset: str, horizon_seconds: int, history: Points, reference_price: float, as_of: int
) -> Forecast:
    returns = [r for _, r, _ in grid_returns(history, as_of, VOL_LOOKBACK)]
    recent = [r for _, r, _ in grid_returns(history, as_of, DRIFT_LOOKBACK)]
    sigma = step_sigma(returns)
    drift = SHRINK * (sum(recent) / len(recent)) if recent else 0.0
    steps = max(1, sum(n for s, n in open_steps(as_of, horizon_seconds).items() if s != "closed"))
    mu = drift * steps
    sigma_h = sigma * math.sqrt(steps)
    p_up = normal_cdf(mu / sigma_h) if sigma_h > 0 else 0.5
    return make_forecast(reference_price, mu, sigma_h, p_up)
