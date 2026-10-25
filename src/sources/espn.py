"""
ESPN-Scoreboard (öffentliche, aber undokumentierte JSON-Schnittstelle): jedes beendete
ATP- bzw. WTA-Match innerhalb von Stunden, auch im laufenden Turnier. Keine Aufschlag-Stats,
kein Ranking und kein Untergrund – das ergänzt build.py aus den anderen Quellen.

Ein Abruf für ein Datum liefert alle an dem Tag laufenden Turniere mit allen Runden.
Die Rohdaten werden pro Tag gespeichert, damit man sie später neu auswerten kann.
"""
import json
import os
import re
import time
from datetime import timedelta

from .common import RAW_DIR, ROUND_ORDER, http_get, round_code, yyyymmdd

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/tennis/{tour}/scoreboard?dates={day}"
DIR = os.path.join(RAW_DIR, "espn")
DIRS = {'atp': DIR, 'wta': os.path.join(RAW_DIR, "espn_wta")}
SINGLES = {'atp': "Men's Singles", 'wta': "Women's Singles"}

FINISHED = {'Final': 'completed', 'Retired': 'RET', 'Walkover': 'W/O', 'Default': 'DEF'}
GRAND_SLAMS = ('Australian Open', 'Roland Garros', 'French Open', 'Wimbledon', 'US Open')
# Show-Events, die bei Sackmann/tennis-data nicht als offizielle Matches zählen
EXCLUDED_EVENTS = ('Laver Cup', 'Six Kings Slam', 'Exhibition')


def path(day, tour='atp'):
    return os.path.join(DIRS[tour], f"{day:%Y%m%d}.json")


def download(start, end, refresh_from=None, pause=0.5, tour='atp'):
    """
    Lädt die Scoreboards von start bis end (inklusive). Vorhandene Tage werden übersprungen,
    außer sie liegen ab refresh_from (die letzten Tage ändern sich noch).
    """
    os.makedirs(DIRS[tour], exist_ok=True)
    loaded = []
    day = start
    while day <= end:
        p = path(day, tour)
        if not os.path.exists(p) or (refresh_from and day >= refresh_from):
            r = http_get(SCOREBOARD_URL.format(tour=tour, day=f"{day:%Y%m%d}"))
            if r is not None:
                with open(p, "w", encoding="utf-8") as f:
                    f.write(r.text)
                loaded.append(day)
            time.sleep(pause)
        day += timedelta(days=1)
    return loaded


def _score(winner, loser):
    sets = []
    for w, l in zip(winner.get('linescores', []), loser.get('linescores', [])):
        s = f"{int(w['value'])}-{int(l['value'])}"
        # Sackmann-Notation: Tiebreak-Punkte des Satzverlierers in Klammern
        set_loser = l if w.get('winner') else w
        if 'tiebreak' in set_loser:
            s += f"({int(set_loser['tiebreak'])})"
        sets.append(s)
    return " ".join(sets)


def parse(scoreboard, tour='atp'):
    """ESPN-JSON -> Zeilen im gemeinsamen Schema (nur Einzel der Tour im Hauptfeld, nur beendete Matches)."""
    rows = []
    for event in scoreboard.get('events', []):
        name = event.get('name', '')
        if any(x.lower() in name.lower() for x in EXCLUDED_EVENTS):
            continue
        venue = (event.get('venue') or {}).get('displayName', '')
        start = yyyymmdd(event.get('date'))
        for grouping in event.get('groupings', []):
            if grouping.get('grouping', {}).get('displayName') != SINGLES[tour]:
                continue
            comps = [c for c in grouping.get('competitions', [])
                     if 'qualifying' not in c.get('round', {}).get('displayName', '').lower()]
            numbered = [int(m.group(1)) for c in comps
                        if (m := re.match(r"Round (\d)", c.get('round', {}).get('displayName', '')))]
            rounds_before_qf = max(numbered) if numbered else 1

            for c in comps:
                status = FINISHED.get(c.get('status', {}).get('type', {}).get('description'))
                players = c.get('competitors', [])
                if status is None or len(players) != 2:
                    continue
                winner = next((p for p in players if p.get('winner')), None)
                loser = next((p for p in players if not p.get('winner')), None)
                if winner is None or loser is None or 'athlete' not in winner or 'athlete' not in loser:
                    continue

                label = c.get('round', {}).get('displayName', '')
                m = re.match(r"Round (\d)", label)
                rnd = round_code(int(m.group(1)), rounds_before_qf) if m else \
                    {'Quarterfinal': 'QF', 'Semifinal': 'SF', 'Final': 'F'}.get(label, label)

                score = "W/O" if status == 'W/O' else _score(winner, loser) + (" RET" if status == 'RET' else "")
                rows.append({
                    'tour': tour,
                    'espn_id': c.get('id'),
                    'tourney_id': f"espn-{event.get('id')}",
                    'tourney_name': name,
                    'tourney_date': start,
                    'match_date': yyyymmdd(c.get('date')),
                    'round': rnd,
                    'best_of': 5 if tour == 'atp' and any(s in name for s in GRAND_SLAMS) else 3,
                    'winner_name': winner['athlete'].get('displayName', '').strip(),
                    'loser_name': loser['athlete'].get('displayName', '').strip(),
                    'score': score.strip(),
                    'status': status,
                    'location': venue.split(',')[0].strip(),
                    'country': venue.split(',')[-1].strip() if ',' in venue else '',
                    'source': 'espn', 'level': 'main',
                })
    return rows


def load(start, end, tour='atp'):
    """
    Alle Matches aus den gespeicherten Tagen. Ein Match taucht an mehreren Tagen auf,
    es zählt der neueste Stand.
    """
    by_id = {}
    day = start
    while day <= end:
        p = path(day, tour)
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    for row in parse(json.load(f), tour):
                        by_id[row['espn_id']] = row
            except (OSError, ValueError):
                pass
        day += timedelta(days=1)

    rows = sorted(by_id.values(), key=lambda r: (r['tourney_date'], r['match_date'], ROUND_ORDER.get(r['round'], 0)))
    for i, r in enumerate(rows):
        r['match_num'] = ROUND_ORDER.get(r['round'], 0) * 1000 + i
    return rows

