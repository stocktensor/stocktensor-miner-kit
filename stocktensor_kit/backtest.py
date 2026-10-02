"""Replay forecast tasks from recorded Chainlink history and score them like a validator.

Tasks follow the validator rules (``docs/PROTOCOL.md`` in stocktensor-subnet):

- no tasks while the session is ``closed`` (weekend);
- reference = last round at ``as_of``, realised = last round at ``as_of + horizon``;
- void (not scored) when no new round arrived during the horizon;
- tasks whose horizon ends after the last recorded round are skipped.

Scores come from ``stocktensor.scoring`` itself. Because a task score is a rank
among the forecasts for that task, your model is scored against a field: by
default every zoo baseline that can run here.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from stocktensor.protocol import HORIZONS, Forecast
from stocktensor.scoring import TaskResult, rolling_scores, score_task
from stocktensor.sessions import session_at

from .data import History
from .models.base import Model

WARMUP = 7 * 86_400


@dataclass(frozen=True)
class Task:
    task_id: str
    asset: str
    horizon: str
    as_of: int
    session: str
    reference_price: float
    realised_price: float


@dataclass
class TaskSet:
    tasks: list[Task] = field(default_factory=list)
    void: int = 0


def generate_tasks(
    asset: str,
    history: History,
    horizons: Sequence[str],
    every: int,
    *,
    warmup: int = WARMUP,
) -> TaskSet:
    out = TaskSet()
    if len(history) < 2:
        return out
    first, last = history.times[0] + warmup, history.times[-1]
    as_of = math.ceil(first / every) * every
    while as_of <= last:
        session = session_at(as_of)
        reference = history.round_at(as_of)
        if session != "closed" and reference is not None:
            for name in horizons:
                end = as_of + HORIZONS[name]
                if end > last:
                    continue
                realised = history.round_at(end)
                if realised is None or realised.round_id == reference.round_id:
                    out.void += 1
                    continue
                out.tasks.append(
                    Task(
                        task_id=f"{asset}-{name}-{as_of}",
                        asset=asset,
                        horizon=name,
                        as_of=as_of,
                        session=session,
                        reference_price=history.price_at(as_of),  # type: ignore[arg-type]
                        realised_price=history.price_at(end),  # type: ignore[arg-type]
                    )
                )
        as_of += every
    return out


def _valid(forecast: Forecast | None, reference: float) -> bool:
    if forecast is None:
        return False
    try:
        forecast.validate(reference)
    except ValueError:
        return False
    return True


@dataclass
class Report:
    target: str
    field: list[str]
    tasks: int
    void: int
    mean: dict[str, float]  # model -> mean task score
    rolling: dict[str, float]  # model -> validator-style rolling score at the last task
    breakdown: list[dict[str, object]]  # target only: per asset/horizon/session
    failures: dict[str, int]  # model -> forecasts that raised or were invalid (scored 0)

    def to_json(self) -> dict[str, object]:
        return {
            "target": self.target,
            "field": self.field,
            "tasks": self.tasks,
            "void": self.void,
            "mean": self.mean,
            "rolling": self.rolling,
            "breakdown": self.breakdown,
            "failures": self.failures,
        }


def run_backtest(
    target: tuple[str, Model],
    field_models: Mapping[str, Model],
    histories: Mapping[str, History],
    horizons: Sequence[str],
    every: int,
) -> Report:
    models: dict[str, Model] = dict(field_models)
    models[target[0]] = target[1]
    names = sorted(models)

    results: list[TaskResult] = []
    totals: dict[str, float] = defaultdict(float)
    failures: dict[str, int] = defaultdict(int)
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    count = void = 0
    last_as_of = 0

    for asset in sorted(histories):
        history = histories[asset]
        taskset = generate_tasks(asset, history, horizons, every)
        void += taskset.void
        for task in taskset.tasks:
            points = history.points_until(task.as_of)
            responses: dict[str, Forecast | None] = {}
            for name in names:
                try:
                    responses[name] = models[name](
                        asset, HORIZONS[task.horizon], points, task.reference_price, task.as_of
                    )
                except Exception:  # a crashing model just scores 0 for the task
                    responses[name] = None
            scores = score_task(task.reference_price, task.realised_price, responses)
            for name in names:
                totals[name] += scores[name]
                if not _valid(responses[name], task.reference_price):
                    failures[name] += 1
            groups[(task.asset, task.horizon, task.session)].append(scores[target[0]])
            results.append(TaskResult(task.task_id, task.as_of, scores))
            count += 1
            last_as_of = max(last_as_of, task.as_of)

    mean = {name: (totals[name] / count if count else 0.0) for name in names}
    rolling = rolling_scores(results, last_as_of) if results else {}
    breakdown = [
        {"asset": a, "horizon": h, "session": s, "tasks": len(v), "mean": sum(v) / len(v)}
        for (a, h, s), v in sorted(groups.items())
    ]
    return Report(
        target=target[0],
        field=names,
        tasks=count,
        void=void,
        mean=mean,
        rolling=rolling,
        breakdown=breakdown,
        failures={name: failures.get(name, 0) for name in names},
    )


def format_report(report: Report) -> str:
    lines = [
        f"target: {report.target}   tasks: {report.tasks}   void: {report.void}",
        "",
        f"{'model':<28}{'mean':>8}{'rolling':>10}{'failed':>8}",
    ]
    for name in sorted(report.field, key=lambda n: -report.mean[n]):
        marker = " *" if name == report.target else ""
        lines.append(
            f"{name + marker:<28}{report.mean[name]:>8.4f}"
            f"{report.rolling.get(name, 0.0):>10.4f}{report.failures[name]:>8}"
        )
    lines += [
        "",
        f"{report.target} by asset / horizon / session",
        f"{'asset':<8}{'hz':<5}{'session':<11}{'n':>6}{'mean':>8}",
    ]
    for row in report.breakdown:
        lines.append(
            f"{row['asset']!s:<8}{row['horizon']!s:<5}{row['session']!s:<11}{row['tasks']:>6}{row['mean']:>8.4f}"
        )
    lines += ["", "Scores are ranks within the field (1 = best on every component), not returns."]
    return "\n".join(lines)
