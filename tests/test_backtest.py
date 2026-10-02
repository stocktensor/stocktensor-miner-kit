from __future__ import annotations

import json

import pytest
from stocktensor.sessions import session_at

from stocktensor_kit.backtest import generate_tasks, run_backtest
from stocktensor_kit.cli import backtest_main
from stocktensor_kit.models import ZOO

from .conftest import FIXTURES

FIELD = {name: ZOO[name].predict for name in ("vol_bands", "session_aware")}


def test_task_rules(fixture_histories) -> None:
    history = fixture_histories["NVDA"]
    taskset = generate_tasks("NVDA", history, ["1h", "1d"], 3_600)
    assert taskset.tasks and taskset.void > 0
    last = history.times[-1]
    for task in taskset.tasks:
        assert session_at(task.as_of) != "closed"
        assert task.as_of + {"1h": 3_600, "1d": 86_400}[task.horizon] <= last
        ref = history.round_at(task.as_of)
        end = history.round_at(task.as_of + {"1h": 3_600, "1d": 86_400}[task.horizon])
        assert ref is not None and end is not None and ref.round_id != end.round_id
        assert task.reference_price == history.price_at(task.as_of)


def run(fixture_histories):
    return run_backtest(
        ("momentum", ZOO["momentum"].predict), FIELD, fixture_histories, ["1h", "1d"], 3 * 3_600
    )


def test_backtest_is_deterministic(fixture_histories) -> None:
    first, second = run(fixture_histories), run(fixture_histories)
    assert first.to_json() == second.to_json()
    assert first.tasks > 100
    assert set(first.mean) == {"momentum", "vol_bands", "session_aware"}
    # Ranked scores: the three models' mean task scores average to 0.5 when every forecast is valid.
    assert sum(first.mean.values()) / 3 == pytest.approx(0.5, abs=1e-9)
    assert all(v == 0 for v in first.failures.values())


def test_backtest_snapshot(fixture_histories) -> None:
    report = run(fixture_histories)
    assert (report.tasks, report.void) == SNAPSHOT["counts"]
    assert report.mean == pytest.approx(SNAPSHOT["mean"], rel=1e-12)


def test_crashing_model_scores_zero(fixture_histories) -> None:
    def broken(*_args):
        raise RuntimeError("boom")

    report = run_backtest(("broken", broken), FIELD, fixture_histories, ["1d"], 6 * 3_600)
    assert report.mean["broken"] == 0.0
    assert report.failures["broken"] == report.tasks


def test_cli_json(capsys) -> None:
    backtest_main(
        [
            "vol_bands",
            "--data",
            str(FIXTURES),
            "--horizons",
            "1d",
            "--every",
            "6h",
            "--field",
            "momentum",
            "--json",
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert out["target"] == "vol_bands" and out["field"] == ["momentum", "vol_bands"]
    assert out["tasks"] > 0 and out["breakdown"]


# Recorded from the fixture CSVs (real NVDA/AAPL Chainlink rounds). Changes here mean the
# models, the task rules or the subnet scoring changed: check which before updating.
SNAPSHOT: dict = {
    "counts": (344, 259),
    "mean": {
        "momentum": 0.4670058139534883,
        "session_aware": 0.4754360465116277,
        "vol_bands": 0.5575581395348835,
    },
}
