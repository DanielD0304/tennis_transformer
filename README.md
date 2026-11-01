# Tennis Market Efficiency

**Question:** Can a model beat the pre-match tennis market (ATP/WTA) after fees, spreads and statistical
corrections, and where do apparent edges come from?

**Short answer:** No. Every model improves on the one before it, but none beats the margin-free sharp market
(Pinnacle / Betfair Exchange), and no Polymarket/Kalshi segment survives the go/no-go rules. The most
instructive result is how one apparent edge (+23.8 % yield, Polymarket ITF, 1 h before the start) turned out to be
an artifact of illiquid markets.

> The repository started as a learning project ("predict matches with a transformer", hence the name) and
> grew into an evaluation framework. The transformer is the earliest stage and is no longer the focus.

## Results

Walk-forward backtest 2020–2026, ATP, log loss (lower is better):

| Predictor | Log loss |
|---|---|
| Rank logistic | 0.6320 |
| Elo | 0.6283 |
| Elo2 (538-style K, surface mix) | 0.6232 |
| LightGBM (`gbdt`) | 0.6089 |
| **Market (Pinnacle, margin removed)** | **0.5891** |

No model or stack lowers the market's log loss significantly (go/no-go: NO-GO everywhere).

Timing backtest on Polymarket/Kalshi (venue × level × entry time × model, after fees and spread, Holm-corrected,
robustness rule over quarters): of 120 segments, 0 are GO, 91 NO-GO, 6 candidates, 11 with too few bets and
12 not tradable. Details: [`docs/ERGEBNISSE_PHASE45.md`](docs/ERGEBNISSE_PHASE45.md) (German).

## What the project covers

- **Data pipeline:** five sources merged into one table (~670k matches, ATP and WTA, main tour, Challenger,
  ITF): Sackmann archive, tennis-data.co.uk, ESPN, Polymarket and Kalshi resolutions. Name matching,
  de-duplication, chronological ordering (`src/sources/`, `src/names.py`).
- **Models:** Elo2, LightGBM, a transformer encoder written from scratch, and a logistic stack with the market
  (`src/elo.py`, `src/models/`, `src/model.py`).
- **Evaluation:** walk-forward with leakage asserts, log loss / Brier / ECE, paired bootstrap and
  Diebold-Mariano against the market, bet simulation per venue with taxes and fees, Kelly staking
  (`src/evaluation/`).
- **Odds collector:** timestamped snapshots from bookmakers, Polymarket (incl. order books) and Kalshi
  (`src/odds/`).
- **Paper trading:** simulated bets at the price actually available, settled afterwards, with closing line
  value (`src/paper/`).

## Method notes

- The benchmark is the market, not a rank baseline. Accuracy alone says little about betting value.
- Margin removal matters: the proportional method overstates long shots; the "sharp vs rest" strategy uses the
  power method (see `docs/DETAILS.md`, section "Sharp vs. Rest").
- Rules against false positives were tightened after each artifact: profits must hold in at least two
  quarters, and no quarter may carry more than 60 % of the bets.

## Limitations

- Closing odds from tennis-data are not obtainable prices; timestamped snapshots exist only since October 2026.
- Polymarket history has no order-book depth, so entry prices are approximated there. Paper trading uses real
  asks.
- Bootstrap intervals treat bets as independent; bets from the same day or tournament are correlated.
- Sackmann data is licensed CC BY-NC-SA 4.0 (non-commercial). Whether Polymarket or Kalshi may be used from
  your jurisdiction is up to you to check. This is not financial or legal advice.

## Quick start

```bash
pip install -r requirements.txt
python -m src.update_data                      # build data/matches.csv (not part of the repo)
python -m src.evaluation.backtest --start 2020 --end 2026 --tour atp --models market,rank,elo,elo2,gbdt
python -m pytest                               # 302 tests
```

Data and model files are created locally under `data/` and `models/` and are not committed.

## Documentation

- [`docs/DETAILS.md`](docs/DETAILS.md): full technical documentation (German): architecture, data sources,
  evaluation, venues, collector, timing backtest, paper trading
- [`docs/ROI_PLAN.md`](docs/ROI_PLAN.md): original plan and findings (German)
- [`docs/ERGEBNISSE_PHASE45.md`](docs/ERGEBNISSE_PHASE45.md): timing backtest and go/no-go results (German)
