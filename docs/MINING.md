# Mining with the kit

From nothing to a miner that validators can score.

## 1. Install

```bash
git clone https://github.com/stocktensor/stocktensor-miner-kit && cd stocktensor-miner-kit
uv sync                      # add --extra ml for the gbm model
```

## 2. Get data

```bash
uv run stx-data pull --since 60d              # every scored asset
uv run stx-data pull --assets NVDA,TSLA       # later: only new rounds are fetched
```

One CSV per asset lands in `data/`. Public RPCs rate-limit; raise `--interval`
if you see HTTP 429 errors, or point `--rpc` at your own node.

## 3. Find a model worth running

```bash
uv run stx-backtest vol_bands --horizons 1h,1d,1w --every 1h
uv run stx-backtest my_pkg.my_model:predict --json > report.json
```

Read the table top to bottom:

- `mean`: average task score (0–1). With N models in the field, an average model sits around 0.5.
- `rolling`: the same time-decayed average a validator uses for weights (3-day half-life, 14-day window).
- `failed`: forecasts that raised or were invalid. Each one is a 0.

Then check the breakdown by asset, horizon and session. Models often win in
one session and lose in another.

Avoid look-ahead: your model only receives rounds published at or before
`as_of`. If you load extra data yourself, apply the same rule, or the backtest
will flatter you.

## 4. Register and serve

You need a Bittensor wallet with a hotkey registered on the Stocktensor subnet
(see the subnet's `docs/MINING.md` for netuid, registration and ports).

```bash
uv run stx-serve my_pkg.my_model:predict \
  --netuid <n> --network finney \
  --wallet.name miner --wallet.hotkey default \
  --port 8091 --external-ip <your public ip>
```

This runs the subnet's miner neuron. It answers signed requests from
validators, signs each forecast with your hotkey and publishes your `ip:port`
on chain. Keep the port reachable from the internet.

At serve time the miner hands your model the last 7 days of Chainlink rounds.
A model that needs a longer window must load it itself (for example from the
CSVs `stx-data pull` keeps up to date).

## 5. Iterate

Validators publish signed epoch bundles with every forecast and score. Use
them, together with `stx-backtest`, to see where your model loses.
