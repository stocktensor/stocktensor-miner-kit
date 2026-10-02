from __future__ import annotations

from stocktensor.assets import Asset

from stocktensor_kit.chainlink import (
    GET_ROUND_DATA,
    LATEST_ROUND,
    LATEST_ROUND_DATA,
    PHASE_AGGREGATORS,
    FeedReader,
    JsonRpc,
    Round,
    clean_rounds,
)
from stocktensor_kit.data import History, pull_asset, read_rounds

PROXY = "0x" + "11" * 20
AGG1 = "0x" + "a1" * 20
AGG2 = "0x" + "a2" * 20


def word(v: int) -> str:
    return f"{v % (1 << 256):064x}"


def encode(r: Round) -> str:
    return (
        "0x" + word(r.round_id) + word(r.answer) + word(r.updated_at) + word(r.updated_at) + word(r.round_id)
    )


class FakeRpc(JsonRpc):
    """Proxy with phase 1 (rounds 1..30) and phase 2 (rounds 1..20), one round per hour."""

    def __init__(self) -> None:
        super().__init__("http://fake", min_interval=0)
        self.rounds: dict[int, Round] = {}
        t = 1_789_000_000
        for phase, count in ((1, 30), (2, 20)):
            for agg in range(1, count + 1):
                rid = (phase << 64) | agg
                self.rounds[rid] = Round(rid, 100_00000000 + agg * 1_000_000, t)
                t += 3_600
        self.calls = 0
        self.batches = 0

    def latest_id(self) -> int:
        return max(self.rounds)

    def eth_call_batch(self, calls):
        self.batches += 1
        out = []
        for to, data in calls:
            self.calls += 1
            if to == PROXY and data == LATEST_ROUND_DATA:
                out.append(encode(self.rounds[self.latest_id()]))
            elif to == PROXY and data.startswith(GET_ROUND_DATA):
                rid = int(data[len(GET_ROUND_DATA) :], 16)
                r = self.rounds.get(rid)
                out.append(encode(r) if r else "0x" + "0" * 320)
            elif to == PROXY and data.startswith(PHASE_AGGREGATORS):
                phase = int(data[len(PHASE_AGGREGATORS) :], 16)
                out.append("0x" + "0" * 24 + {1: AGG1, 2: AGG2}.get(phase, "0x" + "00" * 20)[2:])
            elif to == AGG1 and data == LATEST_ROUND:
                out.append("0x" + word(30))
            else:
                out.append(None)
        return out


def test_walk_back_crosses_phase_boundary() -> None:
    rpc = FakeRpc()
    rounds = list(FeedReader(rpc, PROXY, batch_size=7).walk_back())
    assert len(rounds) == 50
    assert [r.round_id >> 64 for r in rounds[:20]] == [2] * 20 and rounds[20].round_id == (1 << 64) | 30
    assert [r.updated_at for r in rounds] == sorted((r.updated_at for r in rounds), reverse=True)


def test_walk_back_stops_at_time_and_round() -> None:
    rpc = FakeRpc()
    newest = rpc.rounds[rpc.latest_id()].updated_at
    by_time = list(FeedReader(rpc, PROXY).walk_back(stop_before=newest - 5 * 3_600))
    assert len(by_time) == 6
    by_round = list(FeedReader(rpc, PROXY).walk_back(stop_round=(2 << 64) | 17))
    assert [r.round_id & 0xFFFF for r in by_round] == [20, 19, 18]


def test_pull_is_incremental(tmp_path) -> None:
    rpc = FakeRpc()
    asset = Asset("FAKE", PROXY, 8, "Fake / USD")
    first = pull_asset(rpc, asset, tmp_path, since=0)
    assert (first.new_rounds, first.total_rounds) == (50, 50)
    for agg in (21, 22):
        rid = (2 << 64) | agg
        rpc.rounds[rid] = Round(rid, 101_00000000, 1_790_000_000 + agg)
    calls_before = rpc.calls
    second = pull_asset(rpc, asset, tmp_path, since=0)
    assert (second.new_rounds, second.total_rounds) == (2, 52)
    assert rpc.calls - calls_before < 60  # stopped at the newest stored round
    stored = read_rounds(tmp_path / "FAKE.csv")
    assert [r.updated_at for r in stored] == sorted(r.updated_at for r in stored)


def test_clean_rounds_drops_placeholders() -> None:
    good = [Round(i, 236_00000000 + i, 1_000 + i) for i in range(1, 30)]
    bogus = [Round(100, 2_082_200_000_000_000_000, 999), Round(101, 2_092_699_999_900_000_000, 1_015)]
    cleaned = clean_rounds(good + bogus)
    assert cleaned == good
    history = History(good + bogus, 8)
    assert history.price_at(1_015) == good[14].answer / 1e8


def test_negative_answer_decodes_and_is_dropped() -> None:
    from stocktensor_kit.chainlink import _decode_round

    r = _decode_round(encode(Round(5, -7, 123)))
    assert r == Round(5, -7, 123)
    assert clean_rounds([r]) == []


def test_batch_reply_parsing(monkeypatch) -> None:
    rpc = JsonRpc("http://fake", min_interval=0)
    monkeypatch.setattr(
        rpc, "_post", lambda payload: [{"id": 1, "error": {"message": "revert"}}, {"id": 0, "result": "0x01"}]
    )
    assert rpc.eth_call_batch([(PROXY, "0x"), (PROXY, "0x")]) == ["0x01", None]
