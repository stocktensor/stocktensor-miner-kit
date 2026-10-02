"""Session-aware volatility: separate volatility for each US session.

Overnight and pre-market hours move far less than the regular session, and the
weekend not at all. This model estimates hourly volatility per session over
the last 21 days and sums the variance of the sessions the horizon actually
covers. No direction view (p_up = 0.5).
"""

from __future__ import annotations

import math

from stocktensor.protocol import Forecast

from .base import MIN_RETURNS, Points, grid_returns, make_forecast, open_steps, rms, step_sigma

NAME = "session_aware"
DESCRIPTION = "No direction; variance summed per US session the horizon covers."
LOOKBACK = 21 * 86_400


def predict(
    asset: str, horizon_seconds: int, history: Points, reference_price: float, as_of: int
) -> Forecast:
    grid = grid_returns(history, as_of, LOOKBACK)
    overall = step_sigma([r for _, r, _ in grid])
    by_session: dict[str, list[float]] = {}
    for _, r, session in grid:
        by_session.setdefault(session, []).append(r)
    variance = 0.0
    for session, steps in open_steps(as_of, horizon_seconds).items():
        if session == "closed":
            continue
        returns = by_session.get(session, [])
        sigma = rms(returns) if len(returns) >= MIN_RETURNS // 2 else overall
        variance += steps * sigma * sigma
    return make_forecast(reference_price, 0.0, math.sqrt(variance) if variance > 0 else overall, 0.5)
