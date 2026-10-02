"""CSV storage for round history: one file per asset, ``<dir>/<SYMBOL>.csv``.

Columns: ``round_id,answer,updated_at`` (integers, Chainlink units). Pulls are
incremental: a re-run only fetches rounds newer than the newest one on disk.
"""

from __future__ import annotations

import csv
import time
from bisect import bisect_right
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from stocktensor.assets import Asset

from .chainlink import FeedReader, JsonRpc, Round, clean_rounds, to_price

FIELDS = ("round_id", "answer", "updated_at")


def csv_path(directory: Path, symbol: str) -> Path:
    return directory / f"{symbol}.csv"


def read_rounds(path: Path) -> list[Round]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return [
            Round(int(row["round_id"]), int(row["answer"]), int(row["updated_at"]))
            for row in csv.DictReader(handle)
        ]


def write_rounds(path: Path, rounds: Iterable[Round]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        for r in sorted(rounds, key=lambda r: (r.updated_at, r.round_id)):
            writer.writerow((r.round_id, r.answer, r.updated_at))
    tmp.replace(path)


@dataclass(frozen=True)
class PullResult:
    symbol: str
    new_rounds: int
    total_rounds: int


def pull_asset(
    rpc: JsonRpc,
    asset: Asset,
    directory: Path,
    *,
    since: int,
    log: Callable[[str], None] = lambda _: None,
) -> PullResult:
    """Fetch rounds for one asset newer than what is on disk (and not older than ``since``)."""
    path = csv_path(directory, asset.symbol)
    existing = read_rounds(path)
    newest = max((r.round_id for r in existing), default=0)
    reader = FeedReader(rpc, asset.feed)
    fresh: list[Round] = []
    for round_ in reader.walk_back(stop_before=since, stop_round=newest):
        fresh.append(round_)
        if len(fresh) % 500 == 0:
            log(f"{asset.symbol}: {len(fresh)} rounds…")
    merged = {r.round_id: r for r in existing}
    merged.update({r.round_id: r for r in fresh})
    write_rounds(path, merged.values())
    return PullResult(asset.symbol, len(fresh), len(merged))


class History:
    """Cleaned price history of one asset, queryable by time."""

    def __init__(self, rounds: Iterable[Round], decimals: int) -> None:
        cleaned = clean_rounds(list(rounds))
        self.rounds = cleaned
        self.times = [r.updated_at for r in cleaned]
        self.prices = [to_price(r.answer, decimals) for r in cleaned]
        self.decimals = decimals

    def __len__(self) -> int:
        return len(self.rounds)

    def index_at(self, unix: int) -> int:
        """Index of the last round with ``updated_at <= unix`` (-1 if none)."""
        return bisect_right(self.times, unix) - 1

    def round_at(self, unix: int) -> Round | None:
        i = self.index_at(unix)
        return self.rounds[i] if i >= 0 else None

    def price_at(self, unix: int) -> float | None:
        i = self.index_at(unix)
        return self.prices[i] if i >= 0 else None

    def points_until(self, unix: int) -> list[tuple[int, float]]:
        """``(updated_at, price)`` for every round up to ``unix``: what a model may see."""
        i = self.index_at(unix)
        return list(zip(self.times[: i + 1], self.prices[: i + 1], strict=True))


def load_history(directory: Path, asset: Asset) -> History:
    return History(read_rounds(csv_path(directory, asset.symbol)), asset.decimals)


def parse_since(text: str, now: int | None = None) -> int:
    """``30d`` / ``12h`` / ``2w`` relative to now, or a unix timestamp."""
    now = int(time.time()) if now is None else now
    units = {"h": 3_600, "d": 86_400, "w": 604_800}
    if text and text[-1] in units:
        return now - int(float(text[:-1]) * units[text[-1]])
    return int(text)
