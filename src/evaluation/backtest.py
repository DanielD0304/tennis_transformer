"""
Walk-forward-Backtest: Pro Zeitfenster (Standard: Monat) werden die Predictor nur auf Matches mit
Datum < Fensterbeginn gefittet und sagen dann das Fenster vorher.

    python -m src.evaluation.backtest --start 2020 --end 2025 --models market,rank,elo,blend
    python -m src.evaluation.backtest --start 2024 --end 2025 --models market,rank,elo,transformer,blend
    python -m src.evaluation.backtest --start 2020 --end 2026 --tour wta
    python -m src.evaluation.backtest --start 2024 --end 2026 --level challenger   # nur Challenger auswerten

Elo, Historien und Fits laufen immer über alle Level einer Tour (Haupttour, Challenger, ITF), --level filtert
nur die ausgewerteten Matches.

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
from .predictors import Elo2Predictor, EloPredictor, MarketPredictor, make_predictors, prepare_matches
from .venues import VENUES

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PREDICTIONS_PATH = os.path.join(REPO_ROOT, "data", "eval", "predictions.parquet")

PREDICTION_COLUMNS = [
    'match_id', 'date', 'year', 'tour', 'tourney_name', 'surface', 'round', 'player_a', 'player_b', 'label',
    'status', 'rank_a', 'rank_b', 'elo_a', 'elo_b', 'elo2_a', 'elo2_b', 'b365_a', 'b365_b', 'ps_a', 'ps_b', 'avg_a', 'avg_b',
    'max_a', 'max_b', 'bfe_a', 'bfe_b', 'market_source', 'fair_ps', 'tourney_level', 'best_of', 'level', 'source',
]
DEFAULT_VENUES = ('pinnacle', 'betfair', 'bet365_de')
FREQ = {'month': 'MS', 'week': 'W-MON', 'year': 'YS', 'quarter': 'QS'}
# Predictor ohne Fit: dürfen auch für die Historie vor dem Start gerechnet werden (Warm-up für den Blend)
STATIC = (MarketPredictor, EloPredictor, Elo2Predictor)


def windows(start, end, freq='month'):
    """Fenstergrenzen [von, bis) vom 1.1. des Startjahres bis zum 31.12. des Endjahres."""
    first, stop = pd.Timestamp(f"{start}-01-01"), pd.Timestamp(f"{end + 1}-01-01")
    edges = list(pd.date_range(first, stop, freq=FREQ.get(freq, freq)))
    if not edges or edges[0] > first:
        edges.insert(0, first)
    if edges[-1] < stop:
        edges.append(stop)
    return list(zip(edges[:-1], edges[1:]))


def walk_forward(matches, predictors, start, end, freq='month', log=print, warmup=0):
    """
    matches: Ausgabe von predictors.prepare_matches (inkl. Historie vor `start`).
    Gibt die Vorhersagen für start..end zurück (Spalten PREDICTION_COLUMNS + p_<name>).
    warmup: so viele Jahre vor `start` laufen schon mit (nicht im Ergebnis), damit Blend und Stack
    zu Beginn out-of-sample-Vorhersagen der Basismodelle zum Fitten haben.
    """
    frame = matches.copy()
    for p in predictors:
        frame[f'p_{p.name}'] = np.nan
        if isinstance(p, STATIC):
            frame[f'p_{p.name}'] = p.predict(frame)
        if hasattr(p, 'prepare'):
            log(f"{p.name}: Features für alle Matches ...")
            p.prepare(matches)

    dates = frame['date']
    for lo, hi in windows(start - warmup, end, freq):
        in_window = (dates >= lo) & (dates < hi)
        if not in_window.any():
            continue
        history = frame[dates < lo]
        assert (history['date'] < lo).all(), "Leakage: Fit-Daten reichen ins Testfenster"
        window = frame[in_window]
        for p in predictors:
            if isinstance(p, STATIC) or lo.year < getattr(p, 'first_year', lo.year):
                continue
            p.fit(history)
            frame.loc[in_window, f'p_{p.name}'] = p.predict(window)
            window = frame[in_window]  # Blend braucht die eben gerechneten p der Basismodelle

    in_range = (dates >= pd.Timestamp(f"{start}-01-01")) & (dates < pd.Timestamp(f"{end + 1}-01-01"))
    cols = [c for c in PREDICTION_COLUMNS if c in frame.columns] + [f'p_{p.name}' for p in predictors]
    return frame.loc[in_range, cols].reset_index(drop=True)


def run(df, models, start, end, freq='month', config=None, model_path=None, log=print, warmup=1,
        gbdt_refit='quarter', with_predictors=False):
    predictors = make_predictors(models, config=config, model_path=model_path, gbdt_refit=gbdt_refit)
    matches = prepare_matches(df)
    predictions = walk_forward(matches, predictors, start, end, freq=freq, log=log, warmup=warmup)
    names = [p.name for p in predictors]
    return (predictions, names, predictors) if with_predictors else (predictions, names)


def filter_levels(predictions, levels):
    """Nur Matches der Level (z.B. ['challenger']); None oder 'all' = alle."""
    if not levels or 'all' in levels:
        return predictions
    return predictions[predictions['level'].isin(levels)].reset_index(drop=True)


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


def coverage(predictions, models, sources=metrics.SHARP_SOURCES):
    """Anteil der gespielten Matches mit scharfem Marktpreis, für die das Modell ein p hat."""
    base = predictions[predictions['market_source'].isin(sources) & (predictions['status'] != 'W/O')]
    return {m: float(base[f'p_{m}'].notna().mean()) if len(base) else 0.0 for m in models}


def split_by_coverage(predictions, models, min_share=0.9):
    """
    Modelle mit (fast) voller Abdeckung und solche, die nur einen Teil der Jahre vorhersagen (z.B. der
    Transformer erst nach seinem Validierungsjahr). Die Teil-Modelle bekommen eine eigene Stichprobe,
    damit sie die Hauptstichprobe nicht verkleinern.
    """
    cov = coverage(predictions, models)
    top = max(cov.values()) if cov else 0.0
    full = [m for m in models if cov[m] >= min_share * top]
    return full, [m for m in models if m not in full]


def _compare(predictions, models, n_boot, seed, log, title):
    sample = metrics.common_sample(predictions, models)
    log(f"\n=== {title}: {len(sample)} Matches mit scharfem Marktpreis (Pinnacle, sonst Betfair Exchange) "
        f"und Vorhersage von {', '.join(models)} ===")
    if sample.empty:
        return None
    by_source = sample['market_source'].value_counts().to_dict()
    log("Marktquelle: " + ", ".join(f"{k} {v}" for k, v in by_source.items()))
    log("\n--- Metriken pro Modell und Jahr (gleiche Stichprobe) ---")
    log(_fmt(metrics.metric_table(predictions, models)))
    sig = metrics.significance_vs_market(predictions, models, n_boot=n_boot, seed=seed)
    log("\n--- Log-Loss-Differenz gegen den Markt (negativ = besser als der Markt) ---")
    log(_fmt(sig))
    for m in models:
        if m != 'market':
            log(f"Go/No-Go {m}: {'GO' if metrics.go_no_go(sig, m) else 'NO-GO'}")
    return sig


def report(predictions, models, bet_model=None, venues=None, staking=None, n_boot=2000, seed=0, log=print,
           predictors=None):
    n_total = int((predictions['status'] != 'W/O').sum())
    has_market = predictions['market_source'].isin(metrics.SHARP_SOURCES).any()
    go = []
    if not has_market:
        log(f"\nKeine Matches mit scharfem Marktpreis – Markt-Vergleich nicht möglich ({n_total} gespielte Matches).")
        others = [m for m in models if m != 'market' and predictions[f'p_{m}'].notna().any()]
        full, partial = split_by_coverage(predictions.assign(market_source='ps'), others)
        for group, title in ((full, "alle Modelle"), (full + partial, "inkl. Teil-Modelle")):
            if not group or (title != "alle Modelle" and not partial):
                continue
            sample = metrics.common_sample(predictions, group, sources=None)
            log(f"\n=== Metriken ohne Markt ({title}, {len(sample)} gespielte Matches, NICHT mit dem Markt "
                f"vergleichbar) ===")
            log(_fmt(metrics.metric_table(predictions, group, sources=None)))
            ref = 'elo2' if 'elo2' in group else group[0]
            sig = metrics.significance_vs_market(predictions, group, n_boot=n_boot, seed=seed, ref_model=ref,
                                                 sources=None)
            log(f"\n--- Log-Loss-Differenz gegen {ref} (negativ = besser als {ref}) ---")
            log(_fmt(sig))
    else:
        full, partial = split_by_coverage(predictions, models)
        sig = _compare(predictions, full, n_boot, seed, log, "Hauptstichprobe")
        if sig is not None:
            go += [m for m in full if m != 'market' and metrics.go_no_go(sig, m)]
        if partial:
            sig_p = _compare(predictions, full + partial, n_boot, seed, log,
                             f"Teilstichprobe mit {', '.join(partial)}")
            if sig_p is not None:
                go += [m for m in partial if metrics.go_no_go(sig_p, m)]
        if go:
            for m in go:
                group = full + (partial if m in partial else [])
                log(f"\n=== {m} schlägt den Markt signifikant: Segmente (Log-Loss-Differenz gegen den Markt) ===")
                log(_fmt(metrics.segment_table(predictions, m, group, n_boot=min(n_boot, 500), seed=seed)))
        else:
            log("\nKein Modell senkt den Log Loss des Markts signifikant (alle NO-GO).")

    for p in predictors or []:
        if hasattr(p, 'weights') and p.weights():
            log(f"\nGewichte {p.name} (letzter Fit, Logits; Markt allein = 1, 0, ...): "
                + ", ".join(f"{k} {v:+.3f}" for k, v in p.weights().items()))
        if hasattr(p, 'importance') and len(p.importance()):
            log(f"\n=== Feature-Importance {p.name} (mittlerer Gain-Anteil über {len(p.fits)} Fits, Top 25) ===")
            log(p.importance().head(25).to_string(float_format=lambda v: f"{v:.3f}"))

    bet_model = bet_model or next((m for m in reversed(models) if m != 'market'), 'market')
    venues = venues or [VENUES[v] for v in DEFAULT_VENUES]
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
        log("betfair = echte Exchange-Kurse (ab August 2025), 5 % Kommission auf Gewinne. "
            "Proxy = Preis aus fairer Pinnacle-Wahrscheinlichkeit. CLV folgt mit den Quoten-Snapshots (Phase 2).")
    return grid


def main(argv=None):
    from ..config import default_config
    from ..data_loader import load_matches

    ap = argparse.ArgumentParser(description="Walk-forward-Backtest gegen den Markt")
    ap.add_argument('--start', type=int, default=default_config.test_year - 2)
    ap.add_argument('--end', type=int, default=default_config.test_year)
    ap.add_argument('--models', default='market,rank,elo,elo2,gbdt,stack',
                    help="Kommagetrennt: market, rank, elo, elo2, gbdt, transformer, blend, blend_<modell>, "
                         "stack, stack_bands, stack_tf")
    ap.add_argument('--gbdt-refit', default='quarter', help="month | quarter | year")
    ap.add_argument('--warmup', type=int, default=1, help="Jahre vor --start, die schon mitlaufen (für Stack/Blend)")
    ap.add_argument('--freq', default='month', help="month | week | quarter | year (oder Pandas-Alias)")
    ap.add_argument('--data-start', type=int, default=default_config.data_years_start,
                    help="Ab diesem Jahr werden Matches geladen (Historie für Elo und Fits)")
    ap.add_argument('--model-path', default=None, help="Transformer-Gewichte (Standard: config.best_model_path)")
    ap.add_argument('--bet-model', default=None, help="Modell für die Wett-Strategien (Standard: letztes in --models)")
    ap.add_argument('--venues', default=','.join(DEFAULT_VENUES),
                    help=f"Kommagetrennte Venue-Profile (bekannt: {', '.join(VENUES)})")
    ap.add_argument('--kelly', type=float, default=None, help="Fraktionales Kelly statt flat, z.B. 0.25")
    ap.add_argument('--kelly-cap', type=float, default=0.02)
    ap.add_argument('--bootstrap', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=PREDICTIONS_PATH)
    ap.add_argument('--tour', default='all', help="atp | wta | all (jede Tour getrennt)")
    ap.add_argument('--level', default='main', help="main | challenger | itf | all (kommagetrennt), filtert die "
                                                     "Auswertung; Elo und Fits nutzen immer alle Level")
    ap.add_argument('--no-refresh', action='store_true', help="data/matches.csv nicht aktualisieren")
    args = ap.parse_args(argv)

    df = load_matches(first_year=min(args.data_start, args.start), last_year=args.end, refresh=not args.no_refresh)
    tours = sorted(df['tour'].unique()) if args.tour == 'all' else [args.tour]
    staking = betting.Kelly(args.kelly, args.kelly_cap) if args.kelly else betting.Flat()
    venues = [VENUES[v.strip()] for v in args.venues.split(',') if v.strip()]
    results = []
    for tour in tours:
        print(f"\n{'#' * 20} {tour.upper()} {'#' * 20}")
        predictions, names, predictors = run(df[df['tour'] == tour], args.models.split(','), args.start, args.end,
                                             freq=args.freq, model_path=args.model_path, warmup=args.warmup,
                                             gbdt_refit=args.gbdt_refit, with_predictors=True)
        predictions = filter_levels(predictions, args.level.split(','))
        print(f"Ausgewertet: Level {args.level}, {len(predictions)} Matches")
        report(predictions, names, bet_model=args.bet_model, venues=venues, staking=staking,
               n_boot=args.bootstrap, seed=args.seed, predictors=predictors)
        results.append(predictions)
    predictions = pd.concat(results, ignore_index=True)
    print(f"Vorhersagen gespeichert: {save(predictions, args.out)} ({len(predictions)} Matches)")


if __name__ == "__main__":
    main()
