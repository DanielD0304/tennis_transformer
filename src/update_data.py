"""
Lädt alle Matchdaten herunter und baut data/matches.csv.

    python -m src.update_data          # täglicher Refresh (nur Neues)
    python -m src.update_data --full   # alles neu laden (z.B. nach Löschen von data/)

Ohne pandas/torch lauffähig. Der MatchPredictor ruft update() automatisch auf,
wenn der letzte Refresh älter als 24 Stunden ist.
"""
import argparse
import json
import os
import time
from datetime import date, datetime, timedelta

from .sources import build as build_mod
from .sources import espn, sackmann, tuk
from .sources.common import DATA_DIR, MATCHES_PATH, write_matches

FIRST_YEAR = 2015
STATE_PATH = os.path.join(DATA_DIR, "update_state.json")
# So viele Tage vor dem Ende der Sackmann-Daten fängt ESPN an (Überlappung für laufende Turniere)
ESPN_OVERLAP_DAYS = 14
# Die letzten Tage bei ESPN neu laden, weil sich Matches dort noch ändern
ESPN_REFRESH_DAYS = 3


def last_update():
    try:
        with open(STATE_PATH, encoding="utf-8") as f:
            return datetime.fromisoformat(json.load(f)['updated'])
    except (OSError, ValueError, KeyError):
        return None


def is_stale(max_age_hours=24):
    last = last_update()
    return last is None or not os.path.exists(MATCHES_PATH) or datetime.now() - last > timedelta(hours=max_age_hours)


def update(full=False, first_year=FIRST_YEAR, log=print):
    t0 = time.time()
    today = date.today()
    years = list(range(first_year, today.year + 1))

    log("Sackmann (Mirror) ...")
    sackmann.download(years, force=full)
    sack_rows = sackmann.load(years)
    cutoff = max((r['tourney_date'] for r in sack_rows), default=f"{today.year}0101")
    log(f"  {len(sack_rows)} Matches, Daten bis Turnierstart {cutoff}")

    log("tennis-data.co.uk ...")
    tuk.download(years[:-1], max_age_hours=0 if full else None)
    tuk.download(years[-1:], max_age_hours=0 if full else 20)
    tuk_rows = tuk.load(years)
    log(f"  {len(tuk_rows)} Matches")

    log("ESPN ...")
    espn_start = datetime.strptime(cutoff, "%Y%m%d").date() - timedelta(days=ESPN_OVERLAP_DAYS)
    loaded = espn.download(espn_start, today, refresh_from=None if full else today - timedelta(days=ESPN_REFRESH_DAYS))
    espn_rows = espn.load(espn_start, today)
    log(f"  {len(loaded)} Tage geladen, {len(espn_rows)} Matches")

    rows, stats = build_mod.build(sack_rows, tuk_rows, espn_rows)
    write_matches(rows)
    last = max(r['match_date'] or r['tourney_date'] for r in rows)
    state = {
        'updated': datetime.now().isoformat(timespec='seconds'),
        'matches': len(rows),
        'last_match_date': last,
        'sackmann_until': cutoff,
        'mirror_commit': sackmann.MIRROR_COMMIT,
        **stats,
    }
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    log(f"{len(rows)} Matches gespeichert in {MATCHES_PATH} "
        f"(Sackmann {stats['sackmann']}, tennis-data {stats['tuk']}, ESPN {stats['espn']}, "
        f"Duplikate {stats['duplicates']}, mit Quoten {stats['with_odds']}), letztes Match {last}, "
        f"{time.time() - t0:.0f}s")
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", action="store_true", help="alles neu laden")
    parser.add_argument("--first-year", type=int, default=FIRST_YEAR)
    args = parser.parse_args()
    update(full=args.full, first_year=args.first_year)
