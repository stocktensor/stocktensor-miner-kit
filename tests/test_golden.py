"""The installed `stocktensor` scoring must reproduce the subnet's golden vectors exactly."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from stocktensor.protocol import Forecast
from stocktensor.scoring import (
    SCORING_VERSION,
    TaskResult,
    direction_loss,
    interval_loss,
    point_loss,
    rank_scores,
    rolling_scores,
    score_task,
    weights,
)

GOLDEN = Path(__file__).parent / "golden"
SIBLING = Path(__file__).resolve().parents[2] / "stocktensor-subnet" / "tests" / "golden"
VECTORS = json.loads((GOLDEN / "scoring.json").read_text())


def test_scoring_version() -> None:
    assert VECTORS["scoring_version"] == SCORING_VERSION


def test_losses() -> None:
    for case in VECTORS["losses"]:
        r, y = case["reference"], case["realised"]
        assert interval_loss(case["low"], case["high"], y, r) == case["interval"]
        assert direction_loss(case["p_up"], y, r) == case["direction"]
        assert point_loss(case["point"], y, r) == case["point_loss"]


def test_ranks() -> None:
    for case in VECTORS["ranks"]:
        assert rank_scores(case["losses"]) == case["expected"]


def test_tasks_rolling_and_weights() -> None:
    results = []
    for task in VECTORS["tasks"]:
        responses = {m: (Forecast.from_json(f) if f else None) for m, f in task["responses"].items()}
        scores = score_task(task["reference"], task["realised"], responses)
        assert scores == task["expected"]
        results.append(TaskResult(task["task_id"], task["as_of"], scores))
    rolling = rolling_scores(results, VECTORS["rolling"]["now"])
    assert rolling == pytest.approx(VECTORS["rolling"]["expected"], rel=1e-12)
    assert weights(rolling) == pytest.approx(VECTORS["weights"], rel=1e-12)


@pytest.mark.skipif(not SIBLING.is_dir(), reason="no stocktensor-subnet checkout next to this repo")
def test_vendored_copy_is_current() -> None:
    for path in SIBLING.glob("*.json"):
        assert (GOLDEN / path.name).read_text() == path.read_text(), (
            f"run scripts/sync_golden.py ({path.name})"
        )
