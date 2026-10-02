"""Read Chainlink round history from Robinhood Chain over plain JSON-RPC.

Only ``eth_call`` is used, so any RPC (including pruned ones) works: Chainlink
aggregators keep every round in contract storage. Calls are batched and the
client backs off on 429/5xx so a public endpoint is not hammered.

Round ids on a proxy are ``(phase << 64) | aggregator_round``. Walking back
past round 1 of a phase continues in the previous phase.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

DEFAULT_RPC = "https://robinhood-rpc.publicnode.com"

LATEST_ROUND_DATA = "0xfeaf968c"
GET_ROUND_DATA = "0x9a6fc8f5"
PHASE_ID = "0x58303b10"
PHASE_AGGREGATORS = "0xc1597304"
LATEST_ROUND = "0x668a0f02"

PHASE_SHIFT = 64


@dataclass(frozen=True)
class Round:
    round_id: int
    answer: int
    updated_at: int


class RpcError(RuntimeError):
    pass


class JsonRpc:
    """Minimal batched ``eth_call`` client with retry and pacing (stdlib only)."""

    def __init__(
        self,
        url: str = DEFAULT_RPC,
        *,
        min_interval: float = 0.15,
        retries: int = 5,
        timeout: float = 20.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.url = url
        self.min_interval = min_interval
        self.retries = retries
        self.timeout = timeout
        self._sleep = sleep
        self._last = 0.0

    def _post(self, payload: object) -> object:
        body = json.dumps(payload).encode()
        delay = 1.0
        for attempt in range(self.retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                self._sleep(wait)
            self._last = time.monotonic()
            request = urllib.request.Request(
                self.url,
                data=body,
                headers={"content-type": "application/json", "user-agent": "stocktensor-miner-kit"},
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504) or attempt == self.retries:
                    raise RpcError(f"HTTP {exc.code} from RPC") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt == self.retries:
                    raise RpcError(f"RPC unreachable: {exc}") from exc
            self._sleep(delay)
            delay = min(delay * 2, 30.0)
        raise RpcError("unreachable")

    def eth_call_batch(self, calls: Sequence[tuple[str, str]]) -> list[str | None]:
        """Run ``eth_call``s in one batch; a reverted call yields ``None``."""
        if not calls:
            return []
        payload = [
            {"jsonrpc": "2.0", "id": i, "method": "eth_call", "params": [{"to": to, "data": data}, "latest"]}
            for i, (to, data) in enumerate(calls)
        ]
        reply = self._post(payload)
        if isinstance(reply, dict):  # some nodes answer a batch error with one object
            raise RpcError(str(reply.get("error", reply)))
        by_id = {item.get("id"): item for item in reply}  # type: ignore[union-attr]
        out: list[str | None] = []
        for i in range(len(calls)):
            item = by_id.get(i, {})
            out.append(item.get("result") if "error" not in item else None)
        return out


def _word(value: int) -> str:
    return f"{value:064x}"


def _words(hex_data: str) -> list[int]:
    raw = hex_data[2:] if hex_data.startswith("0x") else hex_data
    return [int(raw[i : i + 64], 16) for i in range(0, len(raw), 64)]


def _signed(value: int) -> int:
    return value - (1 << 256) if value >= 1 << 255 else value


def _decode_round(hex_data: str | None) -> Round | None:
    if not hex_data or len(hex_data) < 2 + 64 * 5:
        return None
    words = _words(hex_data)
    round_id, answer, _started, updated_at = words[0], _signed(words[1]), words[2], words[3]
    if updated_at == 0:
        return None
    return Round(round_id, answer, updated_at)


class FeedReader:
    """Round history for one Chainlink proxy."""

    def __init__(self, rpc: JsonRpc, proxy: str, *, batch_size: int = 50) -> None:
        self.rpc = rpc
        self.proxy = proxy
        self.batch_size = batch_size

    def latest(self) -> Round:
        (data,) = self.rpc.eth_call_batch([(self.proxy, LATEST_ROUND_DATA)])
        round_ = _decode_round(data)
        if round_ is None:
            raise RpcError(f"no latestRoundData for {self.proxy}")
        return round_

    def _phase_last_round(self, phase: int) -> int:
        """Last aggregator round id of an older phase (0 if the phase is unknown)."""
        (agg,) = self.rpc.eth_call_batch([(self.proxy, PHASE_AGGREGATORS + _word(phase))])
        if not agg or int(agg, 16) == 0:
            return 0
        address = "0x" + agg[-40:]
        (last,) = self.rpc.eth_call_batch([(address, LATEST_ROUND)])
        return int(last, 16) if last else 0

    def walk_back(self, *, stop_before: int = 0, stop_round: int = 0) -> Iterator[Round]:
        """Yield rounds newest-first until ``updated_at < stop_before`` or ``round_id <= stop_round``."""
        latest = self.latest()
        phase, agg = latest.round_id >> PHASE_SHIFT, latest.round_id & ((1 << PHASE_SHIFT) - 1)
        while phase > 0:
            while agg > 0:
                ids = [(phase << PHASE_SHIFT) | a for a in range(agg, max(agg - self.batch_size, 0), -1)]
                calls = [(self.proxy, GET_ROUND_DATA + _word(i)) for i in ids]
                for round_id, data in zip(ids, self.rpc.eth_call_batch(calls), strict=True):
                    if round_id <= stop_round:
                        return
                    round_ = _decode_round(data)
                    if round_ is None:
                        continue
                    if round_.updated_at < stop_before:
                        return
                    yield round_
                agg -= len(ids)
            phase -= 1
            if phase > 0:
                agg = self._phase_last_round(phase)


def to_price(answer: int, decimals: int) -> float:
    return answer / 10**decimals


def clean_rounds(rounds: Sequence[Round], *, window: int = 10, max_ratio: float = 2.0) -> list[Round]:
    """Drop rounds whose answer is far from its neighbours' median.

    Some feeds carry placeholder answers from before launch (orders of
    magnitude off). Validators only ever read current rounds, so for history
    these are noise and are removed. Input and output are sorted by time.
    """
    ordered = sorted(rounds, key=lambda r: (r.updated_at, r.round_id))
    kept: list[Round] = []
    for i, round_ in enumerate(ordered):
        lo, hi = max(0, i - window), min(len(ordered), i + window + 1)
        neighbours = sorted(r.answer for j, r in enumerate(ordered[lo:hi], start=lo) if j != i)
        if not neighbours or round_.answer <= 0:
            if round_.answer > 0:
                kept.append(round_)
            continue
        median = neighbours[len(neighbours) // 2]
        if median > 0 and 1 / max_ratio <= round_.answer / median <= max_ratio:
            kept.append(round_)
    return kept
