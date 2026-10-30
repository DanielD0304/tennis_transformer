"""
Auswertung der Paper-Wetten: Kennzahlen pro Strategie/Venue/Level/Segment, Bankroll-Kurve und das Go/No-Go-Kriterium
für echtes Geld aus docs/ROI_PLAN.md (Phase 5):
  >= 300 abgerechnete Paper-Wetten, mittlerer CLV > +2 %, untere Grenze des Yield-KI > 0 nach Gebühren.
Das Kriterium gilt insgesamt und zusätzlich für jedes Segment der Modell-Strategie (go_no_go_by_segment): jedes
Segment ist eine eigene Hypothese aus dem Timing-Backtest.
"""
import numpy as np
import pandas as pd

from ..evaluation.betting import bootstrap_yield

START_BANKROLL = 1000.0
GO_MIN_BETS = 300
GO_MIN_CLV = 0.02


def settled(bets):
    return bets[bets['status'].isin(['won', 'lost'])]


def stats(bets, n_boot=2000, seed=0):
    """Kennzahlen einer Gruppe von Wetten (Kelly-Einsätze, dazu der Flat-Yield)."""
    s = settled(bets)
    staked, profit = float(s['stake'].sum()), float(s['profit'].sum())
    low, high = bootstrap_yield(s['profit'], s['stake'], n_boot=n_boot, seed=seed) if len(s) else (np.nan, np.nan)
    flat = float(s['profit_flat'].sum() / s['stake_flat'].sum()) if len(s) and s['stake_flat'].sum() > 0 else np.nan
    clv = s['clv'].dropna()
    return {'n': len(bets), 'open': int((bets['status'] == 'open').sum()), 'void': int((bets['status'] == 'void').sum()),
            'settled': len(s), 'hit_rate': float((s['status'] == 'won').mean()) if len(s) else np.nan,
            'staked': staked, 'profit': profit, 'yield': profit / staked if staked > 0 else np.nan,
            'yield_ci_low': low, 'yield_ci_high': high, 'yield_flat': flat,
            'clv': float(clv.mean()) if len(clv) else np.nan, 'n_clv': len(clv),
            'clv_sharp': float(s['clv_sharp'].dropna().mean()) if s['clv_sharp'].notna().any() else np.nan}


def summary(bets, by=('strategy', 'venue', 'level', 'segment'), n_boot=2000):
    """Eine Zeile pro Gruppe (einzeln je Spalte von by, fehlende Spalten fallen weg) plus 'gesamt'."""
    rows = [{'gruppe': 'gesamt', 'wert': 'alle', **stats(bets, n_boot)}]
    for col in (c for c in by if c in bets):
        for key, g in bets.groupby(bets[col].fillna('?')):
            rows.append({'gruppe': col, 'wert': key, **stats(g, n_boot)})
    return pd.DataFrame(rows)


def bankroll_curve(bets, start=START_BANKROLL):
    """Bankroll nach jeder abgerechneten Wette (Kelly), chronologisch nach Abrechnung: [(Zeit, Bankroll), ...]."""
    s = settled(bets).sort_values('settled_utc', kind='stable')
    values = start + s['profit'].cumsum()
    return [(None, start)] + list(zip(s['settled_utc'], values))


def bankroll(bets, strategy=None, start=START_BANKROLL):
    """Aktuelle Bankroll einer Strategie (Start + Gewinne aus abgerechneten Wetten)."""
    s = settled(bets if strategy is None else bets[bets['strategy'] == strategy])
    return start + float(s['profit'].sum())


def go_no_go(bets, min_bets=GO_MIN_BETS, min_clv=GO_MIN_CLV, n_boot=2000):
    """Kriterium für echtes Geld: alle drei Bedingungen müssen erfüllt sein."""
    st = stats(bets, n_boot)
    checks = [
        ('≥ 300 abgerechnete Paper-Wetten', st['settled'] >= min_bets, f"{st['settled']}"),
        ('mittlerer CLV > +2 %', bool(np.isfinite(st['clv']) and st['clv'] > min_clv),
         f"{st['clv']:+.1%}" if np.isfinite(st['clv']) else '–'),
        ('untere Grenze Yield-KI > 0 (nach Gebühren)', bool(np.isfinite(st['yield_ci_low']) and st['yield_ci_low'] > 0),
         f"{st['yield_ci_low']:+.1%}" if np.isfinite(st['yield_ci_low']) else '–'),
    ]
    return {'go': all(ok for _, ok, _ in checks), 'checks': checks, 'stats': st}


def go_no_go_by_segment(bets, min_bets=GO_MIN_BETS, min_clv=GO_MIN_CLV, n_boot=2000):
    """Das Kriterium je Segment der Modell-Strategie: {segment: go_no_go(...)}. Wetten ohne Segment (sharp, ältere
    Modell-Wetten) zählen hier nicht."""
    if 'segment' not in bets:
        return {}
    with_segment = bets[bets['segment'].notna()]
    return {name: go_no_go(g, min_bets, min_clv, n_boot) for name, g in with_segment.groupby('segment')}
