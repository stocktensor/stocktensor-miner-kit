from __future__ import annotations

import math
from pathlib import Path

import pytest
from stocktensor.assets import load_assets

from stocktensor_kit.data import History, load_history

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixture_histories() -> dict[str, History]:
    assets = load_assets()
    return {s: load_history(FIXTURES, assets[s]) for s in ("NVDA", "AAPL")}


def synthetic_points(start: int = 1_788_000_000, days: int = 30, seed: int = 7) -> list[tuple[int, float]]:
    """Deterministic random-walk prices, one round every 47 minutes, weekends included."""
    import random

    rng = random.Random(seed)
    price, out = 150.0, []
    for k in range(days * 24 * 60 // 47):
        price *= math.exp(rng.gauss(0.00002, 0.003))
        out.append((start + k * 47 * 60, round(price, 6)))
    return out
