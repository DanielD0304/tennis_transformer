"""
Paper-Trading-Lauf (Phase 5): Snapshots holen, neue Signale als simulierte Wetten speichern, offene Wetten
abrechnen, Kennzahlen ausgeben.

    python -m src.paper.run                        # einmal (z.B. täglich oder stündlich per cron/schtasks)
    python -m src.paper.run --loop 30              # alle 30 Minuten
    python -m src.paper.run --no-collect           # nur vorhandene Snapshots nutzen (Collector läuft separat)
    python -m src.paper.run --model gbdt --threshold 0.08
    python -m src.paper.run --segments polymarket:challenger:24h,polymarket:itf:0.5-2   # eigene Segmente

Pro Runde:
1. (Standard) Snapshots von Polymarket und Kalshi holen, die Odds API nur mit --odds-api (Kontingent!).
2. Strategie „modell“: elo2/gbdt (MatchPredictor) gegen den aktuellen Preis bei Polymarket/Kalshi, nur in den
   Segmenten (Venue x Level x Stunden bis zum Start), die der Timing-Backtest als Kandidat oder GO ausweist
   (--segments). Jede Wette trägt ihr Segment, Go/No-Go gibt es zusätzlich je Segment.
   Strategie „sharp“: Pinnacle/Betfair ohne Marge gegen den besten Netto-Preis aller anderen Venues. Verglichen wird
   nur innerhalb von 5 Minuten, also mit den Preisen, die der Odds-API-Lauf mit abfragt (odds/collect.py). Mit
   --odds-api --strategies sharp fragt dieser Lauf selbst ab und wettet sofort, ohne Predictor.
3. Abrechnung über die PM/Kalshi-Auflösung bzw. die Matchdaten, CLV gegen den letzten Snapshot vor dem Start.
   Fällige offene Wetten werden gezielt per Markt-ID (Polymarket) bzw. Event-Ticker (Kalshi) abgefragt, die volle
   Historie lädt der Nightly-Lauf (python -m src.odds.collect --history).
Wetten liegen in data/paper/bets.parquet, das Dashboard unter /scraper/paper/.

Der Predictor-Zustand kommt aus data/cache/ (predict.py) und wird nur nach einem Update von data/matches.csv neu
gebaut. Mit --loop bleibt der Prozess offen: pandas & Co. werden einmal geladen, der Predictor bleibt im Speicher und
wird neu gebaut, sobald sich data/matches.csv ändert (spätestens nach PREDICTOR_MAX_AGE); eine fehlgeschlagene Runde
beendet den Loop nicht.
"""
import argparse
import sys
import time
import traceback
from datetime import timedelta

import pandas as pd

from ..odds.schema import SNAPSHOT_DIR, load_snapshots, now_utc
from . import report, settle, signals
from .store import BETS_PATH, add_bets, load_bets, save_bets

PREDICTOR_MAX_AGE = 24 * 3600  # Sekunden


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


MATCH_DAYS = 30  # Matchdaten für die Abrechnung: nur die letzten Tage (data/matches.csv hat > 600.000 Zeilen)
LEVEL_DAYS = 730  # Turniere für die Level-Zuordnung
LEVEL_KEYS = ('tour', 'tourney_name', 'tourney_date', 'level', 'location')


def recent_matches(now, path=None):
    """
    data/matches.csv zeilenweise lesen: (Matches der letzten MATCH_DAYS für die Abrechnung, Turniere der letzten
    LEVEL_DAYS aus Quellen mit verlässlichem Level für fill_levels). Spart Speicher gegenüber dem ganzen Datensatz.
    Ohne path (data/matches.csv) gecacht: data/cache/recent_matches.pkl gilt für dieselbe Datei und dieselben Tage.
    """
    import csv
    import os

    from .. import cache
    from ..sources.build import LEVEL_SOURCES
    from ..sources.common import MATCHES_PATH
    use_cache = path is None
    path = path or MATCHES_PATH
    if not os.path.exists(path):
        return [], []
    since = f"{now - timedelta(days=MATCH_DAYS):%Y%m%d}"
    since_level = f"{now - timedelta(days=LEVEL_DAYS):%Y%m%d}"
    keep = ('winner_name', 'loser_name', 'match_date', 'tourney_date', 'status', 'source', 'tourney_name', 'level',
            'location')
    key = {'matches': cache.file_stamp(path), 'since': (since, since_level), 'keep': keep, 'level': LEVEL_KEYS,
           'sources': LEVEL_SOURCES}
    found = cache.load('recent_matches', key) if use_cache else None
    if found is not None:
        return found
    matches, tournaments = [], {}
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            day = r.get('match_date') or r.get('tourney_date') or ''
            if day >= since:
                matches.append({k: r.get(k) for k in keep})
            if day >= since_level and r.get('source') in LEVEL_SOURCES:
                t = tuple(r.get(k) for k in LEVEL_KEYS)
                tournaments.setdefault(t, dict(zip(LEVEL_KEYS, t)))
    found = matches, list(tournaments.values())
    if use_cache:
        cache.save('recent_matches', key, found)
    return found


def snapshot_levels(snaps):
    """Kalshi-Snapshots als Level-Referenz für fill_levels: dieselben Turniere in derselben Woche."""
    k = snaps[(snaps['venue'] == 'kalshi') & snaps['level'].notna() & snaps['tourney'].notna()]
    when = pd.to_datetime(k['start_time_utc'], utc=True).fillna(k['ts_utc'])
    return [{'tour': t, 'tourney_name': n, 'tourney_date': f"{d:%Y%m%d}", 'level': lv}
            for t, n, d, lv in zip(k['tour'], k['tourney'], when, k['level'])]


def fill_levels(bets, level_rows):
    """
    Fehlendes Level (Polymarket trennt Challenger und Haupttour nicht, die Odds API kennzeichnet nichts) über
    Turniername, Termin und Tour (build._level_lookup). Für Wetten und Snapshots: Termin ist die Startzeit, sonst
    die Anlage- bzw. Abfragezeit.
    """
    from ..sources.build import _level_lookup
    missing = bets['level'].isna() if len(bets) else []
    if not len(bets) or not missing.any():
        return bets
    lookup = _level_lookup(level_rows)
    todo = bets[missing]
    created = todo['created_utc'] if 'created_utc' in todo else todo['ts_utc']
    when = pd.to_datetime(todo['start_time_utc'], utc=True).fillna(pd.to_datetime(created, utc=True))
    bets.loc[missing, 'level'] = [lookup(t or '', f"{d:%Y%m%d}", tour)
                                  for t, d, tour in zip(todo['tourney'], when, todo['tour'])]
    return bets


def open_results(bets, now, history, log=print):
    """
    Auflösung der fälligen offenen Wetten bei Polymarket/Kalshi, die history noch nicht kennt, gezielt per Markt-ID
    bzw. Event-Ticker statt über die ganze Historie (die lädt der Nightly-Lauf). -> {venue: DataFrame(event_id,
    winner)} mit winner 'a', 'b' oder 'void' (50-50, Absage); noch offene Märkte fehlen.
    """
    from ..odds import kalshi, polymarket
    from ..odds.history import RATE, RETRIES
    from ..odds.http import JsonClient
    known = settle.market_results(history)
    todo = bets[settle.due(bets, now)]
    out = {}
    for venue, fetch in (('polymarket', polymarket.fetch_resolution), ('kalshi', kalshi.fetch_resolution)):
        mine = todo[todo['venue'] == venue]
        items = sorted({(str(e), a) for e, a in zip(mine['event_id'], mine['player_a']) if (venue, str(e)) not in known})
        if not items:
            continue
        client = JsonClient(max_per_sec=RATE[venue], retries=RETRIES[venue])
        rows = []
        for event_id, player_a in items:
            try:
                winner = fetch(client, event_id, player_a)
            except Exception as e:  # die Wette bleibt offen: nächste Runde, Matchdaten oder Nightly-Historie
                log(f"  {venue} {event_id}: Ergebnis nicht abrufbar ({type(e).__name__}: {e})")
                continue
            if winner is not None:
                rows.append({'event_id': event_id, 'winner': winner})
        log(f"  {venue}: {len(items)} fällige Märkte abgefragt, {len(rows)} aufgelöst")
        out[venue] = pd.DataFrame(rows, columns=['event_id', 'winner'])
    return out


def _status(decision):
    st = decision['stats']
    if not st['settled']:
        return f"noch keine abgerechneten Wetten ({st['open']} offen)"
    return (f"{st['settled']} abgerechnet, {st['open']} offen, Yield {st['yield']:+.1%} "
            f"[{st['yield_ci_low']:+.1%}, {st['yield_ci_high']:+.1%}], CLV {st['clv']:+.1%} -> "
            f"{'GO' if decision['go'] else 'NO-GO'} für echtes Geld")


def run_once(args, log=print, prob=None, model_name=None):
    # Predictor und Matchdaten vor dem Abfragen laden (dauert Minuten), damit die Preise beim Signal frisch sind
    if 'modell' in args.strategies and prob is None:
        prob, model_name = make_prob(args.model, log)
    matches, level_rows = recent_matches(now_utc())
    if args.collect:
        from ..odds.collect import collect_once
        venues = ['polymarket', 'kalshi'] + (['odds_api'] if args.odds_api else []) + (['softbooks'] if args.softbooks else [])
        collect_once(venues, args.snapshots, log=log)
    # erst nach dem Abfragen, sonst fallen die eben geholten Snapshots aus dem Fenster ts_utc <= now
    now = now_utc()
    snaps = load_snapshots(now - timedelta(days=args.days), None, args.snapshots)
    bets = load_bets(args.bets)
    log(f"[{now:%Y-%m-%d %H:%M}] {len(snaps)} Snapshots, {len(bets)} Wetten bisher")

    level_rows = level_rows + snapshot_levels(snaps)
    new = []
    if 'modell' in args.strategies and not snaps.empty:
        blend = None if args.raw else signals.load_blend()
        # die Segmente des Backtests gelten für das Modell, mit dem er sie gefunden hat (Blend oder roh)
        segments = signals.segments_from_arg(args.segments, f"blend_{model_name}" if blend else model_name, log)
        log("  Modell: " + ("alle Märkte (--segments alle)" if segments is None else
                            "Segmente " + (", ".join(s.name for s in segments) or "keine")))
        new.append(signals.model_signals(snaps, prob, pd.Timestamp(now), report.bankroll(bets, 'modell'),
                                         args.threshold, strategy='modell', log=log, blend=blend, model=model_name,
                                         segments=segments, levels=lambda df: fill_levels(df, level_rows)))
    if 'sharp' in args.strategies and not snaps.empty:
        new.append(signals.sharp_signals(snaps, pd.Timestamp(now), report.bankroll(bets, 'sharp'), args.sharp_threshold,
                                         log=log))
    for part in new:
        fill_levels(part, level_rows)
        bets, n = add_bets(bets, part)
        if len(part):
            log(f"  {part['strategy'].iloc[0]}: {len(part)} Signale, {n} neue Wetten")

    from ..odds.history import read_history
    history = {v: read_history(v) for v in ('polymarket', 'kalshi')}
    if args.history:
        for venue, found in open_results(bets, pd.Timestamp(now), history, log).items():
            history[venue] = pd.concat([history[venue][['event_id', 'winner']], found], ignore_index=True)
    bets, n = settle.settle(bets, pd.Timestamp(now), history, matches, snaps)
    log(f"  {n} Wetten abgerechnet")
    save_bets(bets, args.bets)

    log(f"  Stand: {_status(report.go_no_go(bets))}")
    for name, decision in report.go_no_go_by_segment(bets).items():
        log(f"    Segment {name}: {_status(decision)}")
    return bets, prob, model_name


def main(argv=None):
    ap = argparse.ArgumentParser(description="Paper-Trading: Signale -> simulierte Wetten -> Abrechnung -> CLV")
    ap.add_argument('--loop', type=float, metavar='MINUTEN', help="dauerhaft alle N Minuten")
    ap.add_argument('--no-collect', dest='collect', action='store_false', help="keine neuen Snapshots holen")
    ap.add_argument('--odds-api', action='store_true', help="auch die Odds API abfragen (kostet Kontingent)")
    ap.add_argument('--softbooks', action='store_true', help="auch Tipico/bwin scrapen (Playwright)")
    ap.add_argument('--no-history', dest='history', action='store_false',
                    help="fällige PM/Kalshi-Wetten nicht gezielt abfragen (nur Historie und Matchdaten)")
    ap.add_argument('--strategies', default='modell,sharp')
    ap.add_argument('--model', default='gbdt', help="elo2 | gbdt (ohne models/gbdt_<tour>.pkl: elo2)")
    ap.add_argument('--threshold', type=float, default=signals.MODEL_THRESHOLD, help="Mindest-Edge Modell")
    ap.add_argument('--segments', default='auto',
                    help="Modell nur in diesen Segmenten: auto (Kandidat/GO aus data/eval/timing.parquet), alle "
                         "oder venue:level:fenster,... (Fenster 24h, 6h, 1h, close oder von-bis in Stunden vor Start)")
    ap.add_argument('--raw', action='store_true', help="Modell ohne Blend mit dem Preis (data/eval/timing_blend.json)")
    ap.add_argument('--sharp-threshold', type=float, default=0.02, help="Mindest-Edge Sharp vs. Rest")
    ap.add_argument('--days', type=int, default=14, help="Snapshots der letzten N Tage laden")
    ap.add_argument('--bets', default=BETS_PATH)
    ap.add_argument('--snapshots', default=SNAPSHOT_DIR)
    args = ap.parse_args(argv)
    args.strategies = [s for s in args.strategies.split(',') if s]
    if args.loop:
        try:  # Dauerbetrieb mit Umleitung in eine Log-Datei: jede Zeile sofort schreiben
            sys.stdout.reconfigure(line_buffering=True)
        except AttributeError:
            pass
    if args.segments not in ('auto', 'alle'):
        try:
            signals.parse_segments(args.segments)
        except ValueError as e:
            ap.error(str(e))

    prob, model_name, built, stamp = None, None, 0.0, None
    try:
        while True:
            t0 = time.time()
            if t0 - built > PREDICTOR_MAX_AGE or _matches_stamp() != stamp:
                # Predictor (Elo, Form) neu, sobald sich data/matches.csv ändert, spätestens nach einem Tag
                # (der Aufbau aktualisiert data/matches.csv, wenn sie älter als 24 Stunden ist)
                prob, built, stamp = None, t0, _matches_stamp()
            try:
                if prob is None and 'modell' in args.strategies:
                    prob, model_name = make_prob(args.model)
                run_once(args, prob=prob, model_name=model_name)
            except Exception:
                if not args.loop:
                    raise
                print(f"Runde fehlgeschlagen, nächster Versuch in {args.loop:g} Minuten:")
                traceback.print_exc()
            if not args.loop:
                break
            time.sleep(max(0.0, args.loop * 60 - (time.time() - t0)))
    except KeyboardInterrupt:
        print("Beendet.")


def _matches_stamp():
    from .. import cache
    from ..sources.common import MATCHES_PATH
    try:
        return cache.file_stamp(MATCHES_PATH)
    except OSError:
        return None


if __name__ == '__main__':
    main()
