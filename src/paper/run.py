"""
Paper-Trading-Lauf (Phase 5): Snapshots holen, neue Signale als simulierte Wetten speichern, offene Wetten
abrechnen, Kennzahlen ausgeben.

    python -m src.paper.run                        # einmal (z.B. täglich oder stündlich per cron/schtasks)
    python -m src.paper.run --loop 30              # alle 30 Minuten
    python -m src.paper.run --no-collect           # nur vorhandene Snapshots nutzen (Collector läuft separat)
    python -m src.paper.run --model gbdt --threshold 0.08

Pro Runde:
1. (Standard) Snapshots von Polymarket und Kalshi holen, die Odds API nur mit --odds-api (Kontingent!).
2. Strategie „modell“: elo2/gbdt (MatchPredictor) gegen den aktuellen Preis bei Polymarket/Kalshi.
   Strategie „sharp“: Pinnacle/Betfair ohne Marge gegen den besten Netto-Preis aller anderen Venues.
3. Abrechnung über die PM/Kalshi-Auflösung (Historie der letzten Tage wird nachgeladen) bzw. die Matchdaten,
   CLV gegen den letzten Snapshot vor dem Start.
Wetten liegen in data/paper/bets.parquet, das Dashboard unter /scraper/paper/.
"""
import argparse
import time
from datetime import timedelta

import pandas as pd

from ..odds.schema import SNAPSHOT_DIR, load_snapshots, now_utc
from . import report, settle, signals
from .store import BETS_PATH, add_bets, load_bets, save_bets

PREDICTOR_MAX_AGE = 24 * 3600  # Sekunden
HISTORY_DAYS = 10  # so weit zurück wird die PM/Kalshi-Historie für die Abrechnung nachgeladen


def make_prob(model='elo2', log=print):
    """P(player_a gewinnt) aus dem Live-Predictor; gbdt fällt ohne trainiertes Modell auf elo2 zurück."""
    from ..predict import MatchPredictor
    try:
        predictor = MatchPredictor(model=model)
    except FileNotFoundError as e:
        log(f"  {model} nicht verfügbar ({e}), nehme elo2")
        predictor = MatchPredictor(model='elo2')
    cache = {}

    def prob(row):
        start = row.get('start_time_utc')
        key = (row['player_a'], row['player_b'], row.get('tourney'))
        if key not in cache:
            try:
                cache[key] = predictor.predict_proba({'player_a': row['player_a'], 'player_b': row['player_b'],
                                                      'date': start.date() if start is not None else None,
                                                      'tournament': row.get('tourney')})
            except Exception:
                cache[key] = None
        return cache[key]
    return prob, predictor.model_name


def run_once(args, log=print, prob=None, model_name=None):
    now = now_utc()
    if args.collect:
        from ..odds.collect import collect_once
        venues = ['polymarket', 'kalshi'] + (['odds_api'] if args.odds_api else []) + (['softbooks'] if args.softbooks else [])
        collect_once(venues, args.snapshots, log=log)
    snaps = load_snapshots(now - timedelta(days=args.days), None, args.snapshots)
    bets = load_bets(args.bets)
    log(f"[{now:%Y-%m-%d %H:%M}] {len(snaps)} Snapshots, {len(bets)} Wetten bisher")

    new = []
    if 'modell' in args.strategies and not snaps.empty:
        if prob is None:
            prob, model_name = make_prob(args.model, log)
        new.append(signals.model_signals(snaps, prob, pd.Timestamp(now), report.bankroll(bets, 'modell'),
                                         args.threshold, strategy='modell', log=log))
    if 'sharp' in args.strategies and not snaps.empty:
        new.append(signals.sharp_signals(snaps, pd.Timestamp(now), report.bankroll(bets, 'sharp'), args.sharp_threshold))
    for part in new:
        bets, n = add_bets(bets, part)
        if len(part):
            log(f"  {part['strategy'].iloc[0]}: {len(part)} Signale, {n} neue Wetten")

    if args.history:
        from ..odds.history import update
        update(('polymarket', 'kalshi'), since=now - timedelta(days=HISTORY_DAYS),
               levels=('main', 'challenger', 'itf'), log=lambda *a: None)
    from ..odds.history import read_history
    from ..odds.matching import load_matches
    history = {v: read_history(v) for v in ('polymarket', 'kalshi')}
    bets, n = settle.settle(bets, pd.Timestamp(now), history, load_matches(), snaps)
    log(f"  {n} Wetten abgerechnet")
    save_bets(bets, args.bets)

    decision = report.go_no_go(bets)
    st = decision['stats']
    log(f"  Stand: {st['settled']} abgerechnet, {st['open']} offen, Yield {st['yield']:+.1%} "
        f"[{st['yield_ci_low']:+.1%}, {st['yield_ci_high']:+.1%}], CLV {st['clv']:+.1%} -> "
        f"{'GO' if decision['go'] else 'NO-GO'} für echtes Geld" if st['settled'] else
        f"  Stand: noch keine abgerechneten Wetten ({st['open']} offen)")
    return bets, prob, model_name


def main(argv=None):
    ap = argparse.ArgumentParser(description="Paper-Trading: Signale -> simulierte Wetten -> Abrechnung -> CLV")
    ap.add_argument('--loop', type=float, metavar='MINUTEN', help="dauerhaft alle N Minuten")
    ap.add_argument('--no-collect', dest='collect', action='store_false', help="keine neuen Snapshots holen")
    ap.add_argument('--odds-api', action='store_true', help="auch die Odds API abfragen (kostet Kontingent)")
    ap.add_argument('--softbooks', action='store_true', help="auch Tipico/bwin scrapen (Playwright)")
    ap.add_argument('--no-history', dest='history', action='store_false', help="PM/Kalshi-Historie nicht nachladen")
    ap.add_argument('--strategies', default='modell,sharp')
    ap.add_argument('--model', default='gbdt', help="elo2 | gbdt (ohne models/gbdt_<tour>.pkl: elo2)")
    ap.add_argument('--threshold', type=float, default=signals.MODEL_THRESHOLD, help="Mindest-Edge Modell")
    ap.add_argument('--sharp-threshold', type=float, default=0.02, help="Mindest-Edge Sharp vs. Rest")
    ap.add_argument('--days', type=int, default=14, help="Snapshots der letzten N Tage laden")
    ap.add_argument('--bets', default=BETS_PATH)
    ap.add_argument('--snapshots', default=SNAPSHOT_DIR)
    args = ap.parse_args(argv)
    args.strategies = [s for s in args.strategies.split(',') if s]

    prob, model_name, built = None, None, 0.0
    try:
        while True:
            t0 = time.time()
            if t0 - built > PREDICTOR_MAX_AGE:  # Predictor (Elo, Form) einmal am Tag neu aufbauen
                prob, built = None, t0
            _, prob, model_name = run_once(args, prob=prob, model_name=model_name)
            if not args.loop:
                break
            time.sleep(max(0.0, args.loop * 60 - (time.time() - t0)))
    except KeyboardInterrupt:
        print("Beendet.")


if __name__ == '__main__':
    main()
