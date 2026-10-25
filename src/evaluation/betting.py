"""
Generische Wett-Simulation: simulate(predictions, strategy, venue, staking).

Strategien und Staking sind Parameter-Objekte statt kopierter Schleifen. Die alten Strategien
aus main.py lassen sich so abbilden:
    Filtered Betting  -> Favorite(model, min_odds=1.30)
    Pure Value        -> Value(model, min_ev=0.0)
    Sniper            -> ConfidenceBand(model, 0.65, 0.80)
    Baseline (Rang)   -> BetterRank()
Neu: Value('market', ...) auf einer Soft-Venue = „Sharp vs. Rest“ (faire Pinnacle-Wahrscheinlichkeit
gegen den Preis der Venue), ganz ohne eigenes Modell.

EV und Kelly rechnen mit der Netto-Quote der Venue (nach Steuer/Gebühr).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .venues import Venue

A, NONE, B = 1, 0, -1


# -------------------------------------------------------------- Strategien

@dataclass(frozen=True)
class Favorite:
    """Auf den Spieler mit p > 0,5, aber nur ab einer Mindestquote."""
    model: str = 'blend'
    min_odds: float = 1.30

    @property
    def name(self):
        return f"Favorit({self.model}, Quote>={self.min_odds:.2f})"

    def choose(self, p_a, gross_a, gross_b, net_a, net_b, rank_a, rank_b):
        side = np.where(p_a > 0.5, A, np.where(p_a < 0.5, B, NONE))
        odds = np.where(side == A, gross_a, gross_b)
        return np.where(odds >= self.min_odds, side, NONE)


@dataclass(frozen=True)
class Value:
    """Die Seite mit dem höheren EV (p * Netto-Quote - 1), wenn er über der Schwelle liegt."""
    model: str = 'blend'
    min_ev: float = 0.0

    @property
    def name(self):
        return f"Value({self.model}, EV>{self.min_ev:+.0%})"

    def choose(self, p_a, gross_a, gross_b, net_a, net_b, rank_a, rank_b):
        ev_a, ev_b = p_a * net_a - 1, (1 - p_a) * net_b - 1
        side = np.where(ev_a >= ev_b, A, B)
        ev = np.where(side == A, ev_a, ev_b)
        return np.where(ev > self.min_ev, side, NONE)


@dataclass(frozen=True)
class ConfidenceBand:
    """„Sniper“: auf den Favoriten des Modells, nur wenn dessen p im Band [low, high] liegt."""
    model: str = 'blend'
    low: float = 0.65
    high: float = 0.80

    @property
    def name(self):
        return f"Band({self.model}, {self.low:.2f}-{self.high:.2f})"

    def choose(self, p_a, gross_a, gross_b, net_a, net_b, rank_a, rank_b):
        side = np.where(p_a > 0.5, A, np.where(p_a < 0.5, B, NONE))
        p_side = np.where(side == A, p_a, 1 - p_a)
        return np.where((p_side >= self.low) & (p_side <= self.high), side, NONE)


@dataclass(frozen=True)
class BetterRank:
    """Immer auf den besser platzierten Spieler (gleicher Rang: keine Wette)."""
    model: str = None

    @property
    def name(self):
        return "Besserer Rang"

    def choose(self, p_a, gross_a, gross_b, net_a, net_b, rank_a, rank_b):
        return np.where(rank_a < rank_b, A, np.where(rank_b < rank_a, B, NONE))


# ----------------------------------------------------------------- Staking

@dataclass(frozen=True)
class Flat:
    stake: float = 10.0

    @property
    def name(self):
        return f"flat {self.stake:g}"

    def amount(self, bankroll, p, net_odds):
        return self.stake


@dataclass(frozen=True)
class Kelly:
    """Fraktionales Kelly auf die aktuelle Bankroll, gedeckelt auf cap (Anteil der Bankroll)."""
    fraction: float = 0.25
    cap: float = 0.02

    @property
    def name(self):
        return f"Kelly x{self.fraction:g} (max {self.cap:.0%})"

    def amount(self, bankroll, p, net_odds):
        if not np.isfinite(p) or net_odds <= 1:
            return 0.0
        full = (p * net_odds - 1) / (net_odds - 1)
        return max(bankroll, 0.0) * min(max(full, 0.0) * self.fraction, self.cap)


# -------------------------------------------------------------- Simulation

def bootstrap_yield(profits, stakes, n_boot=2000, seed=0, level=0.95):
    """KI des Yields (Summe Gewinn / Summe Einsatz) per Bootstrap über die Wetten."""
    profits, stakes = np.asarray(profits, float), np.asarray(stakes, float)
    n = len(profits)
    if n == 0:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    yields = []
    for start in range(0, n_boot, 100):  # in Blöcken, damit große Stichproben nicht den Speicher sprengen
        idx = rng.integers(0, n, size=(min(100, n_boot - start), n))
        yields.append(profits[idx].sum(axis=1) / stakes[idx].sum(axis=1))
    yields = np.concatenate(yields)
    alpha = (1 - level) / 2
    low, high = np.quantile(yields, [alpha, 1 - alpha])
    return float(low), float(high)


def max_drawdown(curve):
    """Größter Rückgang vom bisherigen Höchststand, als Anteil des Höchststands."""
    curve = np.asarray(curve, float)
    if len(curve) == 0:
        return 0.0
    peak = np.maximum.accumulate(curve)
    return float(np.max(np.where(peak > 0, (peak - curve) / peak, 0.0)))


def simulate(predictions, strategy, venue: Venue, staking=Flat(), bankroll=1000.0, n_boot=2000, seed=0):
    """
    Spielt die Strategie chronologisch auf einer Venue durch.
    Gibt (Kennzahlen-Dict, DataFrame der Wetten) zurück. CLV bleibt None, bis es
    Quoten-Snapshots mit Zeitstempel gibt (Phase 2).
    """
    df = predictions.sort_values('date', kind='stable')
    n = len(df)
    p_a = df[f'p_{strategy.model}'].to_numpy(float) if strategy.model else np.full(n, np.nan)
    gross_a, gross_b = venue.gross_odds(df)
    net_a, net_b = venue.net_odds_from_gross(gross_a), venue.net_odds_from_gross(gross_b)
    has_price = np.isfinite(gross_a) & np.isfinite(gross_b) & (gross_a > 1) & (gross_b > 1)
    if strategy.model:
        has_price &= np.isfinite(p_a)

    with np.errstate(invalid='ignore'):
        side = strategy.choose(p_a, gross_a, gross_b, net_a, net_b,
                               df['rank_a'].to_numpy(float), df['rank_b'].to_numpy(float))
    side = np.where(has_price, side, NONE)
    void = venue.is_void(df['status'])
    label = df['label'].to_numpy(int)

    bank = bankroll
    curve = [bank]
    rows = []
    for i in np.flatnonzero(side != NONE):
        on_a = side[i] == A
        odds_net = net_a[i] if on_a else net_b[i]
        p_side = p_a[i] if on_a else 1 - p_a[i]
        stake = staking.amount(bank, p_side, odds_net)
        if venue.max_stake is not None:
            stake = min(stake, venue.max_stake)
        if stake <= 0:
            continue
        won = (label[i] == 1) == on_a
        if void[i]:
            profit = 0.0
        else:
            profit = stake * (odds_net - 1) if won else -stake
            bank += profit
            curve.append(bank)
        rows.append({'match_id': df['match_id'].iat[i], 'date': df['date'].iat[i], 'side': 'A' if on_a else 'B',
                     'p': p_side, 'odds': gross_a[i] if on_a else gross_b[i], 'net_odds': odds_net,
                     'stake': stake, 'void': bool(void[i]), 'won': bool(won) and not void[i],
                     'profit': profit, 'bankroll': bank, 'clv': None})
    bets = pd.DataFrame(rows, columns=['match_id', 'date', 'side', 'p', 'odds', 'net_odds', 'stake', 'void',
                                       'won', 'profit', 'bankroll', 'clv'])

    settled = bets.loc[~bets['void'].astype(bool)]
    staked = float(settled['stake'].sum())
    profit = float(settled['profit'].sum())
    ci_low, ci_high = bootstrap_yield(settled['profit'], settled['stake'], n_boot=n_boot, seed=seed)
    summary = {
        'strategy': strategy.name, 'venue': venue.label, 'staking': staking.name,
        'n_bets': len(settled), 'n_void': int(bets['void'].sum()),
        'hit_rate': float(settled['won'].mean()) if len(settled) else float('nan'),
        'avg_odds': float(settled['odds'].mean()) if len(settled) else float('nan'),
        'staked': staked, 'profit': profit,
        'yield': profit / staked if staked > 0 else float('nan'),
        'yield_ci_low': ci_low, 'yield_ci_high': ci_high,
        'roi': profit / bankroll, 'max_drawdown': max_drawdown(curve), 'final_bankroll': bank,
        'clv': None,
    }
    return summary, bets


def default_strategies(model):
    """Die alten Strategien aus main.py plus „Sharp vs. Rest“ (Markt als Wahrheit)."""
    return [BetterRank(), Favorite(model, 1.30), Value(model, 0.0), ConfidenceBand(model, 0.65, 0.80),
            Value('market', 0.02)]


def run_grid(predictions, strategies, venues, staking=Flat(), **kwargs):
    rows = []
    for venue in venues:
        for strategy in strategies:
            if strategy.model and f'p_{strategy.model}' not in predictions.columns:
                continue
            summary, _ = simulate(predictions, strategy, venue, staking, **kwargs)
            rows.append(summary)
    return pd.DataFrame(rows)
