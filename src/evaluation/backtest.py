"""
Walk-forward-Backtest: Pro Zeitfenster (Standard: Monat) werden die Predictor nur auf Matches mit
Datum < Fensterbeginn gefittet und sagen dann das Fenster vorher.

    python -m src.evaluation.backtest --start 2020 --end 2025 --models market,rank,elo,blend
    python -m src.evaluation.backtest --start 2024 --end 2025 --models market,rank,elo,transformer,blend
    python -m src.evaluation.backtest --start 2020 --end 2026 --tour wta

Jede Tour läuft als eigener Backtest (eigene Fits, eigenes Elo); --tour all (Standard) rechnet alle
Touren nacheinander und gibt die Tabellen pro Tour aus.

Ergebnis: data/eval/predictions.parquet (eine Zeile pro Match, p je Predictor, alle Quoten aus
Sicht von Spieler A). Danach werden die Metrik-Tabelle pro Modell und Jahr, der Signifikanztest
gegen den Markt und die Wett-Tabelle pro Strategie und Venue ausgegeben.

Datum eines Matches ist das echte Matchdatum, sonst der Turnierstart (Sackmann). Bei Turnieren über
einen Monatswechsel landen dadurch alle Matches im Monat des Turnierstarts.
"""
import argparse
import os

import numpy as np
import pandas as pd

from . import betting, metrics
from .predictors import MarketPredictor, EloPredictor, TransformerPredictor, make_predictors, prepare_matches
from .venues import VENUES

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PREDICTIONS_PATH = os.path.join(REPO_ROOT, "data", "eval", "predictions.parquet")

PREDICTION_COLUMNS = [
    'match_id', 'date', 'year', 'tour', 'tourney_name', 'surface', 'round', 'player_a', 'player_b', 'label',
    'status', 'rank_a', 'rank_b', 'elo_a', 'elo_b', 'b365_a', 'b365_b', 'ps_a', 'ps_b', 'avg_a', 'avg_b',
    'market_source', 'fair_ps',
]
FREQ = {'month': 'MS', 'week': 'W-MON', 'year': 'YS', 'quarter': 'QS'}
# Predictor ohne Fit: dürfen auch für die Historie vor dem Start gerechnet werden (Warm-up für den Blend)
STATIC = (MarketPredictor, EloPredictor)


def windows(start, end, freq='month'):
    """Fenstergrenzen [von, bis) vom 1.1. des Startjahres bis zum 31.12. des Endjahres."""
    first, stop = pd.Timestamp(f"{start}-01-01"), pd.Timestamp(f"{end + 1}-01-01")
    edges = list(pd.date_range(first, stop, freq=FREQ.get(freq, freq)))
    if not edges or edges[0] > first:
        edges.insert(0, first)
    if edges[-1] < stop:
        edges.append(stop)
    return list(zip(edges[:-1], edges[1:]))


def walk_forward(matches, predictors, start, end, freq='month', log=print):
    """
    matches: Ausgabe von predictors.prepare_matches (inkl. Historie vor `start`).
    Gibt die Vorhersagen für start..end zurück (Spalten PREDICTION_COLUMNS + p_<name>).
    """
    frame = matches.copy()
    for p in predictors:
        frame[f'p_{p.name}'] = np.nan
        if isinstance(p, STATIC):
            frame[f'p_{p.name}'] = p.predict(frame)
        if isinstance(p, TransformerPredictor):
            log("Transformer: Features und Vorhersagen für alle Matches ...")
            p.prepare(matches)

    dates = frame['date']
    for lo, hi in windows(start, end, freq):
        in_window = (dates >= lo) & (dates < hi)
        if not in_window.any():
            continue
        history = frame[dates < lo]
        assert (history['date'] < lo).all(), "Leakage: Fit-Daten reichen ins Testfenster"
        window = frame[in_window]
        for p in predictors:
            if isinstance(p, STATIC):
                continue
            p.fit(history)
            frame.loc[in_window, f'p_{p.name}'] = p.predict(window)
            window = frame[in_window]  # Blend braucht die eben gerechneten p der Basismodelle

    in_range = (dates >= pd.Timestamp(f"{start}-01-01")) & (dates < pd.Timestamp(f"{end + 1}-01-01"))
    cols = [c for c in PREDICTION_COLUMNS if c in frame.columns] + [f'p_{p.name}' for p in predictors]
    return frame.loc[in_range, cols].reset_index(drop=True)


def run(df, models, start, end, freq='month', config=None, model_path=None, log=print):
    predictors = make_predictors(models, config=config, model_path=model_path)
    matches = prepare_matches(df)
    return walk_forward(matches, predictors, start, end, freq=freq, log=log), [p.name for p in predictors]


def save(predictions, path=PREDICTIONS_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    predictions.to_parquet(path, index=False)
    return path


# ------------------------------------------------------------------ Ausgabe

def _fmt(df, pct=()):
    df = df.copy()
    for c in pct:
        if c in df.columns:
            df[c] = df[c].map(lambda v: f"{v:+.1%}" if pd.notna(v) else "-")
    return df.to_string(index=False, float_format=lambda v: f"{v:.4f}")


def report(predictions, models, bet_model=None, venues=None, staking=None, n_boot=2000, seed=0, log=print):
    sample = metrics.common_sample(predictions, models)
    n_total = int((predictions['status'] != 'W/O').sum())
    log(f"\nVergleichs-Stichprobe: {len(sample)} von {n_total} Matches "
        f"(mit Pinnacle-Quote und Vorhersage von allen Modellen)")
    if sample.empty:
        log("Keine Matches mit Pinnacle-Quote – Markt-Vergleich nicht möglich.")
        others = [m for m in models if m != 'market' and predictions[f'p_{m}'].notna().any()]
        played = predictions[(predictions['status'] != 'W/O') & predictions[[f'p_{m}' for m in others]].notna().all(axis=1)]
        if others and len(played):
            log("\n=== Metriken ohne Markt (alle gespielten Matches, NICHT mit dem Markt vergleichbar) ===")
            log(_fmt(metrics.metric_table(played.assign(market_source='ps'), others)))
    else:
        log("\n=== Metriken pro Modell und Jahr (gleiche Stichprobe) ===")
        log(_fmt(metrics.metric_table(predictions, models)))
        sig = metrics.significance_vs_market(predictions, models, n_boot=n_boot, seed=seed)
        log("\n=== Log-Loss-Differenz gegen den Markt (negativ = besser als Pinnacle) ===")
        log(_fmt(sig))
        blends = [m for m in models if m.startswith('blend')]
        for b in blends:
            verdict = "GO" if metrics.go_no_go(sig, b) else "NO-GO"
            log(f"Go/No-Go {b}: {verdict} (Blend muss den Log Loss des Markts signifikant senken)")

    bet_model = bet_model or next((m for m in reversed(models) if m != 'market'), 'market')
    venues = venues or list(VENUES.values())
    grid = betting.run_grid(predictions, betting.default_strategies(bet_model), venues,
                            staking or betting.Flat(), n_boot=n_boot, seed=seed)
    log(f"\n=== Wetten pro Strategie und Venue ({(staking or betting.Flat()).name}) ===")
    shown = grid[(grid['n_bets'] > 0) | (grid['n_void'] > 0)] if len(grid) else grid
    if shown.empty:
        log("Keine Wetten (keine Quoten oder keine Signale).")
    else:
        cols = ['venue', 'strategy', 'n_bets', 'n_void', 'hit_rate', 'avg_odds', 'yield', 'yield_ci_low',
                'yield_ci_high', 'roi', 'max_drawdown']
        log(_fmt(shown[cols], pct=('hit_rate', 'yield', 'yield_ci_low', 'yield_ci_high', 'roi', 'max_drawdown')))
        log("Proxy = Preis aus fairer Pinnacle-Wahrscheinlichkeit, keine echten historischen Kurse. "
            "CLV folgt mit den Quoten-Snapshots (Phase 2).")
    return grid


def main(argv=None):
    from ..config import default_config
    from ..data_loader import load_matches

    ap = argparse.ArgumentParser(description="Walk-forward-Backtest gegen den Markt")
    ap.add_argument('--start', type=int, default=default_config.test_year - 2)
    ap.add_argument('--end', type=int, default=default_config.test_year)
    ap.add_argument('--models', default='market,rank,elo,blend',
                    help="Kommagetrennt: market, rank, elo, transformer, blend, blend_<modell>")
    ap.add_argument('--freq', default='month', help="month | week | quarter | year (oder Pandas-Alias)")
    ap.add_argument('--data-start', type=int, default=default_config.data_years_start,
                    help="Ab diesem Jahr werden Matches geladen (Historie für Elo und Fits)")
    ap.add_argument('--model-path', default=None, help="Transformer-Gewichte (Standard: config.best_model_path)")
    ap.add_argument('--bet-model', default=None, help="Modell für die Wett-Strategien (Standard: letztes in --models)")
    ap.add_argument('--venues', default=','.join(VENUES), help="Kommagetrennte Venue-Profile")
    ap.add_argument('--kelly', type=float, default=None, help="Fraktionales Kelly statt flat, z.B. 0.25")
    ap.add_argument('--kelly-cap', type=float, default=0.02)
    ap.add_argument('--bootstrap', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=PREDICTIONS_PATH)
    ap.add_argument('--tour', default='all', help="atp | wta | all (jede Tour getrennt)")
    ap.add_argument('--no-refresh', action='store_true', help="data/matches.csv nicht aktualisieren")
    args = ap.parse_args(argv)

    df = load_matches(first_year=min(args.data_start, args.start), last_year=args.end, refresh=not args.no_refresh)
    tours = sorted(df['tour'].unique()) if args.tour == 'all' else [args.tour]
    staking = betting.Kelly(args.kelly, args.kelly_cap) if args.kelly else betting.Flat()
    venues = [VENUES[v.strip()] for v in args.venues.split(',') if v.strip()]
    results = []
    for tour in tours:
        print(f"\n{'#' * 20} {tour.upper()} {'#' * 20}")
        predictions, names = run(df[df['tour'] == tour], args.models.split(','), args.start, args.end,
                                 freq=args.freq, model_path=args.model_path)
        report(predictions, names, bet_model=args.bet_model, venues=venues, staking=staking,
               n_boot=args.bootstrap, seed=args.seed)
        results.append(predictions)
    predictions = pd.concat(results, ignore_index=True)
    print(f"Vorhersagen gespeichert: {save(predictions, args.out)} ({len(predictions)} Matches)")


if __name__ == "__main__":
    main()
