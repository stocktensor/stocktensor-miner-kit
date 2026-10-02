# stocktensor-miner-kit

Starter models, Chainlink price data and a local backtest that uses the **exact
validator scoring** of the [StockTensor subnet](https://github.com/stocktensor/stocktensor-subnet).
Go from zero to a scored StockTensor miner in three commands.

StockTensor is a Bittensor subnet where miners forecast Robinhood Chain stock
tokens (NVDA, AAPL, TSLA, SPY and more) over 1 hour, 1 day and 1 week.
Validators score every forecast against the Chainlink price on Robinhood Chain.

## Quick start

```bash
pip install "git+https://github.com/stocktensor/stocktensor-miner-kit"   # or: uv tool install …

stx-data pull --assets NVDA,AAPL,SPY --since 30d     # 1. Chainlink history → data/*.csv
stx-backtest momentum --horizons 1h,1d               # 2. score a model against the zoo
stx-serve momentum --netuid <n> \
  --wallet.name miner --wallet.hotkey default         # 3. run it as a live miner
```

`stx-serve` takes the subnet miner's flags (see `stx-miner --help` and the
subnet's `docs/MINING.md`).

## What's inside

| Command | What it does |
|---|---|
| `stx-data pull` | Walks each Chainlink proxy backwards (`getRoundData`, batched JSON-RPC, phase-aware) and writes `round_id,answer,updated_at` per asset. Incremental: re-runs only fetch new rounds. Default RPC `https://robinhood-rpc.publicnode.com`, override with `--rpc` or `STX_RPC`. |
| `stx-backtest <model>` | Replays tasks from the data with the validator's rules (no tasks over the weekend, void when the feed did not update during the horizon) and scores them with `stocktensor.scoring`. `--json` for machine output. |
| `stx-serve <model>` | Starts the subnet miner neuron with a kit model plugged in. |

### Model zoo (baselines, not alpha)

| Model | Idea |
|---|---|
| `vol_bands` | No direction view. 80% band from 14-day hourly realised volatility, scaled by the open-market hours in the horizon. |
| `momentum` | Shrunk 3-day drift plus volatility bands; `p_up` from drift / σ. |
| `session_aware` | No direction view. Separate volatility per US session (regular, pre, post, overnight); the weekend adds nothing. |
| `gbm` | scikit-learn quantile gradient boosting on simple features. Needs `pip install "stocktensor-miner-kit[ml]"`. |

These are honest starting points so you can see the plumbing work end to end.
In our own backtests on recent data the plain `vol_bands` baseline is hard to
beat. Treat every number you get as a property of the data window you
pulled, not a promise.

Your own model is any function with this signature, loaded as `module:callable`:

```python
from stocktensor.protocol import Forecast


def predict(
    asset: str, horizon_seconds: int, history: list[tuple[int, float]], reference_price: float, as_of: int
) -> Forecast:
    # history = [(updated_at, price), ...] up to as_of, oldest first
    ...
```

```bash
stx-backtest my_pkg.my_model:predict
stx-serve my_pkg.my_model:predict --netuid <n> ...
```

## What the scoring rewards

Each forecast is an 80% interval (`low`, `high`), a `point` and `p_up`, the
probability the price ends above the reference. Per task, every valid forecast
is **ranked** against the others on three losses (from the subnet's
`docs/SCORING.md`):

- **interval (50%)**: narrow intervals that still contain the realised price. A miss costs 10× the distance.
- **direction (30%)**: Brier score of `p_up`. Confident and right beats 0.5, but confident and wrong is expensive.
- **point (20%)**: absolute error of the point estimate.

Missing, late or invalid answers score 0. Because scores are ranks, a backtest
score only means something relative to the field, so `stx-backtest` always
scores your model next to the zoo (`--field` to choose).

## Data notes

- Prices are the Chainlink token price on Robinhood Chain (underlying × token
  multiplier), the same number validators score against.
- Feeds update on a 0.5% deviation or a 24 h heartbeat, 24/5, so quiet assets
  have few rounds and many 1 h tasks are void. That is how the subnet behaves too.
- Some feeds have placeholder answers from before launch, far off the real
  price. `History` drops rounds more than 2× away from their neighbours' median.

## Development

```bash
uv sync --extra ml
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

The kit imports scoring from the `stocktensor` package (stocktensor-subnet) and
checks it against the subnet's golden vectors (`tests/golden`, refresh with
`uv run python scripts/sync_golden.py`).

## Links

- Website: https://stocktensor.io
- Docs: https://docs.stocktensor.io
- dApp: https://dapp.stocktensor.io
- X: https://x.com/stocktensor
- Telegram: https://t.me/stocktensorio
- GitHub: https://github.com/stocktensor

## License

MIT © 2026 StockTensor · [stocktensor.io](https://stocktensor.io)
