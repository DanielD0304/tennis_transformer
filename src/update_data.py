"""
Lädt alle Matchdaten (ATP und WTA) herunter und baut data/matches.csv.

    python -m src.update_data          # täglicher Refresh (nur Neues)
    python -m src.update_data --full   # alles neu laden (z.B. nach Löschen von data/)
    python -m src.update_data --levels main   # nur Haupttour (ohne Challenger/ITF)

Standard sind alle Level (Haupttour inkl. Qualifikation, Challenger, ITF): Elo, Historien und Features
laufen über alle Level einer Tour gemeinsam, weil Spieler zwischen den Leveln wechseln. Nach dem Ende des
Sackmann-Archivs kommen ATP-Haupttour, Challenger und WTA-Haupttour mit Stats von TennisMyLife
(sources/tml.py, nur nicht-kommerzielle Nutzung), ITF und WTA 125 aus den abgerechneten Polymarket-/Kalshi-Märkten
(data/odds/history, siehe sources/markets.py), sofern die Historie geladen ist.

Ohne pandas/torch lauffähig. Der MatchPredictor ruft update() automatisch auf,
wenn der letzte Refresh älter als 24 Stunden ist.
"""
import argparse
import json
import os
import time
from datetime import date, datetime, timedelta

from .sources import build as build_mod
from .sources import espn, markets, sackmann, tml, tuk
from .sources.common import DATA_DIR, LEVELS, MATCHES_PATH, TOURS, write_matches

FIRST_YEAR = 2015
STATE_PATH = os.path.join(DATA_DIR, "update_state.json")
# So viele Tage vor dem Ende der Sackmann-Daten fängt ESPN an (Überlappung für laufende Turniere)
ESPN_OVERLAP_DAYS = 14
# Die letzten Tage bei ESPN neu laden, weil sich Matches dort noch ändern
ESPN_REFRESH_DAYS = 3
# TennisMyLife-Dateien frühestens nach so vielen Stunden neu laden (die Seite pflegt das laufende Jahr täglich)
TML_MAX_AGE_HOURS = 20


def last_update():
    try:
        with open(STATE_PATH, encoding="utf-8") as f:
            return datetime.fromisoformat(json.load(f)['updated'])
    except (OSError, ValueError, KeyError):
        return None


def is_stale(max_age_hours=24):
    last = last_update()
    return last is None or not os.path.exists(MATCHES_PATH) or datetime.now() - last > timedelta(hours=max_age_hours)


def _coverage(rows):
    """Anzahl Matches je 'tour/level/source' (Daten-Abdeckung pro Level)."""
    from collections import Counter
    return dict(sorted(Counter(f"{r.get('tour')}/{r.get('level') or 'main'}/{r.get('source')}" for r in rows).items()))


def update(full=False, first_year=FIRST_YEAR, log=print, tours=TOURS, levels=LEVELS):
    t0 = time.time()
    today = date.today()
    years = list(range(first_year, today.year + 1))

    sack_rows, tml_rows, tuk_rows, espn_rows, players, cutoffs, tuk_missing = [], [], [], [], {}, {}, {}
    for tour in tours:
        log(f"[{tour.upper()}] Sackmann (Mirror) ...")
        sackmann.download(years, force=full, tour=tour, levels=levels)
        rows = sackmann.load(years, tour=tour, levels=levels)
        players[tour] = sackmann.load_players(tour)
        cutoffs[tour] = max((r['tourney_date'] for r in rows), default=f"{today.year}0101")
        sack_rows += rows
        by_level = {lv: sum(r['level'] == lv for r in rows) for lv in levels}
        log(f"  {len(rows)} Matches ({', '.join(f'{k} {v}' for k, v in by_level.items())}), "
            f"Daten bis Turnierstart {cutoffs[tour]}, {len(players[tour])} Spieler")

        # TennisMyLife: Stats für die Zeit nach dem Sackmann-Ende (ab dem Jahr, in dem Sackmann endet)
        log(f"[{tour.upper()}] TennisMyLife ...")
        tml_years = [y for y in years if y >= int(cutoffs[tour][:4])]
        tml.download(tml_years, tour=tour, levels=levels, max_age_hours=0 if full else TML_MAX_AGE_HOURS)
        rows = tml.load(tml_years, tour=tour, levels=levels)
        tml_rows += rows
        log(f"  {len(rows)} Matches ab {tml_years[0] if tml_years else '-'}, "
            f"mit Stats {sum(1 for r in rows if r.get('w_svpt') not in (None, '', '0'))}")

        log(f"[{tour.upper()}] tennis-data.co.uk ...")
        tuk.download(years[:-1], max_age_hours=0 if full else None, tour=tour)
        tuk.download(years[-1:], max_age_hours=0 if full else 20, tour=tour)
        rows = tuk.load(years, tour=tour)
        tuk_rows += rows
        tuk_missing[tour] = tuk.missing(years, tour=tour)
        log(f"  {len(rows)} Matches" + (f", fehlende Dateien: {tuk_missing[tour]} (ohne diese keine Quoten)"
                                         if tuk_missing[tour] else ""))

        log(f"[{tour.upper()}] ESPN ...")
        espn_start = datetime.strptime(cutoffs[tour], "%Y%m%d").date() - timedelta(days=ESPN_OVERLAP_DAYS)
        loaded = espn.download(espn_start, today, tour=tour,
                               refresh_from=None if full else today - timedelta(days=ESPN_REFRESH_DAYS))
        rows = espn.load(espn_start, today, tour=tour)
        espn_rows += rows
        log(f"  {len(loaded)} Tage geladen, {len(rows)} Matches")

    log("[Märkte] Polymarket/Kalshi als Ergebnisquelle ...")
    market_rows = [r for r in markets.load(levels, log=log) if r['tour'] in tours]
    rows, stats = build_mod.build(sack_rows, tuk_rows, espn_rows, players, market_rows, tml_rows)
    rows = [r for r in rows if (r.get('level') or 'main') in levels]  # Level aus dem Turniernamen (ESPN, Märkte)
    write_matches(rows)
    last = max(r['match_date'] or r['tourney_date'] for r in rows)
    state = {
        'updated': datetime.now().isoformat(timespec='seconds'),
        'matches': len(rows),
        'last_match_date': last,
        'sackmann_until': cutoffs.get('atp', ''),
        'sackmann_until_by_tour': cutoffs,
        'tuk_missing_years': tuk_missing,
        'mirror_commit': sackmann.MIRROR_COMMIT,
        'levels': list(levels),
        'by_level_source': _coverage(rows),
        **stats,
    }
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    log(f"{len(rows)} Matches gespeichert in {MATCHES_PATH} "
        f"(Sackmann {stats['sackmann']}, TennisMyLife {stats['tml']}, tennis-data {stats['tuk']}, "
        f"ESPN {stats['espn']}, "
        f"Duplikate {stats['duplicates']}, mit Quoten {stats['with_odds']}), letztes Match {last}, "
        f"bereinigt: {stats.get('renamed_players', 0)} Namen vereinheitlicht, "
        f"{stats.get('clean_duplicates', 0)} Doppel-Einträge, {stats.get('clean_odds_removed', 0)} Quoten, "
        f"{stats.get('clean_surface_filled', 0)} Untergründe; "
        f"{time.time() - t0:.0f}s")
    # Namen, die sich keinem Spieler zuordnen ließen, nie still übernehmen: Sie stehen als eigene Spieler ohne Historie
    # im Datensatz und verfälschen Elo und Features ihrer Gegner (Issue #40)
    for tour, name, n in stats.get('unresolved_names', []):
        log(f"WARNUNG: Kurzform ohne Zuordnung, steht als eigener Spieler im Datensatz: {name} ({tour}, {n} Matches). "
            f"Bitte in sources/build.MANUAL_ALIASES eintragen.")
    if stats.get('tuk_unmatched_names'):
        log(f"tennis-data-Zeilen vor dem Sackmann-Ende ohne Gegenstück (Quoten fehlen dort): {stats.get('tuk_unmatched', 0)}, "
            "am häufigsten " + ", ".join(f"{name} ({tour}, {n})" for tour, name, n in stats['tuk_unmatched_names'][:8]))
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", action="store_true", help="alles neu laden")
    parser.add_argument("--first-year", type=int, default=FIRST_YEAR)
    parser.add_argument("--tours", default=",".join(TOURS), help="Kommagetrennt: atp, wta")
    parser.add_argument("--levels", default=",".join(LEVELS), help="Kommagetrennt: main, challenger, itf")
    args = parser.parse_args()
    update(full=args.full, first_year=args.first_year, tours=tuple(t.strip() for t in args.tours.split(",") if t.strip()),
           levels=tuple(lv.strip() for lv in args.levels.split(",") if lv.strip()))
