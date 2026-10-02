"""Command line entry points: ``stx-data``, ``stx-backtest``, ``stx-serve``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from stocktensor.assets import load_assets
from stocktensor.protocol import HORIZONS

from .backtest import format_report, run_backtest
from .chainlink import DEFAULT_RPC, JsonRpc, RpcError
from .data import load_history, parse_since, pull_asset
from .models import ZOO, available_zoo, load_model
from .serve import ENV_MODEL


def _assets(text: str | None) -> list[str]:
    known = load_assets()
    if not text:
        return sorted(known)
    wanted = [s.strip().upper() for s in text.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in known]
    if unknown:
        raise SystemExit(f"unknown assets: {', '.join(unknown)} (known: {', '.join(sorted(known))})")
    return wanted


def _duration(text: str) -> int:
    units = {"m": 60, "h": 3_600, "d": 86_400}
    if text[-1] in units:
        return int(float(text[:-1]) * units[text[-1]])
    return int(text)


def data_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="stx-data", description="Download Chainlink round history")
    sub = parser.add_subparsers(dest="command", required=True)
    pull = sub.add_parser("pull", help="fetch rounds into one CSV per asset (incremental)")
    pull.add_argument("--assets", help="comma separated, default: all scored assets")
    pull.add_argument("--since", default="30d", help="how far back on first pull: 30d, 12h, 2w or unix time")
    pull.add_argument("--out", default="data", type=Path)
    pull.add_argument("--rpc", default=os.environ.get("STX_RPC", DEFAULT_RPC))
    pull.add_argument("--interval", type=float, default=0.15, help="seconds between RPC requests")
    args = parser.parse_args(argv)

    assets = load_assets()
    rpc = JsonRpc(args.rpc, min_interval=args.interval)
    since = parse_since(args.since)
    failed = 0
    for symbol in _assets(args.assets):
        try:
            result = pull_asset(rpc, assets[symbol], args.out, since=since, log=print)
        except RpcError as exc:
            failed += 1
            print(f"{symbol}: failed: {exc}", file=sys.stderr)
            continue
        print(f"{symbol}: +{result.new_rounds} rounds ({result.total_rounds} on disk)")
    if failed:
        raise SystemExit(1)


def backtest_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="stx-backtest", description="Score a model on recorded history")
    parser.add_argument("model", help=f"zoo name ({', '.join(ZOO)}) or module:callable")
    parser.add_argument("--data", default="data", type=Path)
    parser.add_argument("--assets", help="comma separated, default: every asset with data")
    parser.add_argument("--horizons", default="1h,1d", help=f"comma separated from {', '.join(HORIZONS)}")
    parser.add_argument("--every", default="1h", help="spacing between task as_of times: 30m, 1h, 1d")
    parser.add_argument("--field", help="comma separated field models, default: every zoo model that can run")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)

    horizons = [h.strip() for h in args.horizons.split(",") if h.strip()]
    bad = [h for h in horizons if h not in HORIZONS]
    if bad:
        raise SystemExit(f"unknown horizons: {', '.join(bad)}")

    known = load_assets()
    symbols = (
        _assets(args.assets) if args.assets else sorted(s for s in known if (args.data / f"{s}.csv").exists())
    )
    histories = {s: load_history(args.data, known[s]) for s in symbols}
    histories = {s: h for s, h in histories.items() if len(h) > 1}
    if not histories:
        raise SystemExit(f"no data in {args.data}: run `stx-data pull` first")

    target = load_model(args.model)
    if args.field:
        field = dict(load_model(name.strip()) for name in args.field.split(",") if name.strip())
    else:
        field = available_zoo()
    report = run_backtest(target, field, histories, horizons, _duration(args.every))
    print(json.dumps(report.to_json(), indent=2) if args.json else format_report(report))


def serve_main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(
            "usage: stx-serve <model> [stx-miner flags]\n\n"
            "Runs the Stocktensor subnet miner with a kit model.\n"
            f"<model> is a zoo name ({', '.join(ZOO)}) or module:callable with the kit signature.\n"
            "Example: stx-serve momentum --netuid <n> --wallet.name miner --wallet.hotkey default"
        )
        return
    spec, rest = argv[0], argv[1:]
    if "--model" in rest:
        raise SystemExit("pass the model as the first argument, not --model")
    load_model(spec)  # fail fast on a bad spec
    if spec in ZOO:
        target = f"stocktensor_kit.serve:{spec}"
    else:
        os.environ[ENV_MODEL] = spec
        target = "stocktensor_kit.serve:from_env"
    from neurons import miner  # stocktensor-subnet

    miner.main(["--model", target, *rest])
