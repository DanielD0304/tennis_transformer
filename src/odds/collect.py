"""
Collector: alle Venues abfragen und die Snapshots an data/odds/snapshots/YYYY-MM-DD.parquet anhängen.

    python -m src.odds.collect                      # einmal
    python -m src.odds.collect --loop 20            # alle 20 Minuten (Strg+C beendet)
    python -m src.odds.collect --venues polymarket,kalshi
    python -m src.odds.collect --history            # historische Polymarket-/Kalshi-Preise laden

Fehler einer Venue stoppen die anderen nicht. Die Odds API kostet Kontingent und läuft im Loop deshalb
nur jede n-te Runde (--odds-api-every, Standard 6 = alle 2 Stunden bei --loop 20).
"""
import argparse
import time
from datetime import datetime

from . import history, kalshi, odds_api, polymarket, softbooks
from .schema import SNAPSHOT_DIR, append_snapshots, clean

ADAPTERS = {
    'polymarket': polymarket.fetch_current,
    'kalshi': kalshi.fetch_current,
    'odds_api': odds_api.fetch_current,
    'softbooks': softbooks.fetch_current,
}


def collect_once(venues=tuple(ADAPTERS), root=SNAPSHOT_DIR, adapters=None, log=print):
    """Jede Venue einmal abfragen. Ergebnis: {venue: Anzahl Snapshots} bzw. {venue: 'Fehler: ...'}."""
    adapters = adapters or ADAPTERS
    result = {}
    for name in venues:
        t0 = time.time()
        try:
            snaps = clean(adapters[name](log=log), log)
            n = append_snapshots(snaps, root)
            matches = len({s.event_id for s in snaps})
            result[name] = n
            log(f"  {name}: {n} Snapshots ({matches} Märkte) in {time.time() - t0:.1f}s")
        except Exception as e:
            result[name] = f"Fehler: {type(e).__name__}: {e}"
            log(f"  {name}: {result[name]}")
    return result


def loop(minutes, venues, odds_api_every=6, root=SNAPSHOT_DIR, log=print):
    i = 0
    while True:
        start = time.time()
        run = [v for v in venues if v != 'odds_api' or i % odds_api_every == 0]
        log(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Runde {i + 1}: {', '.join(run)}")
        collect_once(run, root, log=log)
        i += 1
        time.sleep(max(0.0, minutes * 60 - (time.time() - start)))


def main(argv=None):
    p = argparse.ArgumentParser(description="Quoten-Snapshots sammeln (Polymarket, Kalshi, Odds API, Soft-Books)")
    p.add_argument('--loop', type=float, metavar='MINUTEN', help="dauerhaft alle N Minuten abfragen")
    p.add_argument('--venues', default=','.join(ADAPTERS), help=f"Auswahl aus {','.join(ADAPTERS)}")
    p.add_argument('--odds-api-every', type=int, default=6, help="im Loop die Odds API nur jede n-te Runde")
    p.add_argument('--history', action='store_true', help="historische Polymarket-/Kalshi-Preise laden")
    p.add_argument('--since', help="--history: nur Matches ab Datum (YYYY-MM-DD)")
    args = p.parse_args(argv)

    if args.history:
        history.main(['--since', args.since] if args.since else [])
        return
    venues = [v for v in args.venues.split(',') if v]
    unknown = set(venues) - set(ADAPTERS)
    if unknown:
        p.error(f"Unbekannte Venues: {', '.join(sorted(unknown))}")
    try:
        if args.loop:
            loop(args.loop, venues, args.odds_api_every)
        else:
            collect_once(venues)
    except KeyboardInterrupt:
        print("Beendet.")


if __name__ == '__main__':
    main()
