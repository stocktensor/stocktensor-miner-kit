"""Shared helpers for the baseline models.

A model is any callable::

    predict(asset, horizon_seconds, history, reference_price, as_of) -> Forecast

``history`` is ``[(updated_at, price), ...]`` sorted by time, containing only
rounds published at or before ``as_of`` (no look-ahead). Prices are the
Chainlink token price in USD.

Chainlink stock feeds update on deviation (0.5%) or heartbeat, and not at all
over the weekend, so the helpers resample onto an hourly grid and only count
steps where the market is open.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Sequence
from typing import Protocol

from stocktensor.protocol import Forecast, format_price
from stocktensor.sessions import session_at

Points = Sequence[tuple[int, float]]

STEP = 3_600
Z80 = 1.2815515655446004  # standard normal quantile at 0.9 -> central 80% interval
DEFAULT_STEP_SIGMA = 0.02 / math.sqrt(24)  # 2% a day, used only without enough history
MIN_RETURNS = 24


class Model(Protocol):
    def __call__(
        self, asset: str, horizon_seconds: int, history: Points, reference_price: float, as_of: int
    ) -> Forecast: ...


def price_at(points: Points, unix: int) -> float | None:
    times = [t for t, _ in points]
    i = bisect_right(times, unix) - 1
    return points[i][1] if i >= 0 else None


def grid_returns(points: Points, as_of: int, lookback: int, step: int = STEP) -> list[tuple[int, float, str]]:
    """Hourly log returns ``(end_time, return, session)`` over ``lookback``, open-market steps only."""
    if not points:
        return []
    times = [t for t, _ in points]
    start = max(as_of - lookback, times[0])
    end = as_of - (as_of - start) % step
    out: list[tuple[int, float, str]] = []
    t = start - (start % step) + step
    previous: float | None = None
    previous_t = t - step
    while t <= end:
        i = bisect_right(times, t) - 1
        price = points[i][1] if i >= 0 else None
        session = session_at(t)
        if (
            price is not None
            and previous is not None
            and session != "closed"
            and session_at(previous_t) != "closed"
            and price > 0
            and previous > 0
        ):
            out.append((t, math.log(price / previous), session))
        previous, previous_t = price, t
        t += step
    return out


def open_steps(as_of: int, horizon_seconds: int, step: int = STEP) -> dict[str, int]:
    """How many grid steps of each session fall inside ``(as_of, as_of + horizon]``."""
    counts: dict[str, int] = {}
    steps = max(1, horizon_seconds // step)
    for k in range(1, steps + 1):
        session = session_at(as_of + k * step)
        counts[session] = counts.get(session, 0) + 1
    return counts


def rms(values: Sequence[float]) -> float:
    return math.sqrt(sum(v * v for v in values) / len(values)) if values else 0.0


def step_sigma(returns: Sequence[float]) -> float:
    """Per-step volatility (zero-mean RMS), falling back to a default without data."""
    if len(returns) < MIN_RETURNS:
        return DEFAULT_STEP_SIGMA
    return max(rms(returns), 1e-5)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _places(value: float) -> int:
    """Decimal places that keep ~8 significant digits (at least 6, at most 18)."""
    if value <= 0:
        return 6
    return min(18, max(6, 8 - math.floor(math.log10(value))))


def make_forecast(reference: float, mu: float, sigma: float, p_up: float) -> Forecast:
    """Lognormal 80% interval around ``reference * exp(mu)`` with log-volatility ``sigma``."""
    sigma = max(sigma, 0.0)
    point = reference * math.exp(mu)
    low = reference * math.exp(mu - Z80 * sigma)
    high = reference * math.exp(mu + Z80 * sigma)
    p_up = min(max(p_up, 0.02), 0.98)
    low_s, point_s, high_s = (format_price(v, _places(v)) for v in (low, point, high))
    # Rounding must not break low <= point <= high.
    if float(low_s) > float(point_s):
        low_s = point_s
    if float(high_s) < float(point_s):
        high_s = point_s
    return Forecast(low=low_s, point=point_s, high=high_s, p_up=f"{p_up:.4f}")
