"""Volatility bands: no view on direction, interval from recent realised volatility.

Point = reference price, p_up = 0.5, interval = reference × exp(±z·σ) where σ
is the hourly realised volatility of the last 14 days scaled by the number of
open-market hours inside the horizon (the feed does not move on weekends).
This is the honest floor: a model that cannot beat it is not adding anything.
"""

from __future__ import annotations

import math

from stocktensor.protocol import Forecast

from .base import Points, grid_returns, make_forecast, open_steps, step_sigma

NAME = "vol_bands"
DESCRIPTION = "No direction; 80% band from 14-day hourly realised volatility."
LOOKBACK = 14 * 86_400


def predict(
    asset: str, horizon_seconds: int, history: Points, reference_price: float, as_of: int
) -> Forecast:
    returns = [r for _, r, _ in grid_returns(history, as_of, LOOKBACK)]
    sigma = step_sigma(returns)
    steps = sum(n for session, n in open_steps(as_of, horizon_seconds).items() if session != "closed")
    return make_forecast(reference_price, 0.0, sigma * math.sqrt(max(steps, 1)), 0.5)
