"""
Polymarket und Kalshi als Ergebnisquelle: abgerechnete Match-Märkte (data/odds/history/{venue}.parquet,
siehe src/odds/history.py) werden zu Matches im Schema von data/matches.csv.

Wozu: Das Sackmann-Archiv endet am 01.06.2026, ESPN deckt nur die Haupttour ab (keine Challenger, kein
ITF). Für Challenger und ITF reichen Elo und Form so trotzdem bis heute.

Grenzen (in den Daten gekennzeichnet über source='polymarket' bzw. 'kalshi'):
- Sieger aus der Markt-Auflösung, Spieler aus dem Markt. Kein Score, kein Ranking, keine Stats.
- Eine Aufgabe (RET) ist im Markt nicht erkennbar: Beide Venues zahlen den Spieler aus, der weiterkommt.
  status bleibt deshalb 'completed'. Märkte mit 50-50-Auflösung (Absage, W/O) fehlen ganz.
- Datum ist die geplante Startzeit der Venue (UTC).
Duplikate (dasselbe Match bei beiden Venues oder schon aus Sackmann/ESPN bekannt) erkennt build.py über
build.MatchIndex, dieses Modul liefert nur die Rohzeilen.
"""
import os

from .common import DATA_DIR, yyyymmdd

HISTORY_DIR = os.path.join(DATA_DIR, "odds", "history")
VENUES = ('kalshi', 'polymarket')  # Reihenfolge = Priorität bei Duplikaten
# Sackmann-Code für tourney_level je Level (Haupttour: aus dem Turniernamen, siehe build.py)
LEVEL_CODES = {'challenger': 'C', 'itf': 'ITF'}


def market_rows(df, venue):
    """DataFrame der Historie einer Venue -> Liste von Match-Dicts (nur Märkte mit Sieger und Tour)."""
    rows = []
    for r in df.to_dict('records'):
        if r.get('winner') not in ('a', 'b') or r.get('tour') not in ('atp', 'wta') or not r.get('start_time_utc'):
            continue
        a, b = str(r['player_a']).strip(), str(r['player_b']).strip()
        if not a or not b or a == b:
            continue
        winner, loser = (a, b) if r['winner'] == 'a' else (b, a)
        day = yyyymmdd(r['start_time_utc'])
        level = r.get('level')
        if not isinstance(level, str) or not level:
            level = None  # auch NaN aus dem Parquet: build.py bestimmt das Level über den Turniernamen
        rows.append({
            'tour': r['tour'], 'level': level, 'tourney_id': f"{venue}-{r.get('tourney') or day}",
            'tourney_name': r.get('tourney') or '', 'tourney_level': LEVEL_CODES.get(level, ''),
            'tourney_date': day, 'match_date': day, 'round': '', 'best_of': 3,
            'winner_name': winner, 'loser_name': loser, 'score': '', 'status': 'completed',
            'source': venue, 'market_id': str(r['event_id']),
        })
    return rows


def load(levels=('main', 'challenger', 'itf'), root=HISTORY_DIR, venues=VENUES, log=print):
    """Alle Markt-Matches der Level aus den Historie-Dateien. Ohne pandas oder Dateien: leere Liste."""
    try:
        import pandas as pd
    except ImportError:
        return []
    out = []
    for venue in venues:
        path = os.path.join(root, f"{venue}.parquet")
        if not os.path.exists(path):
            continue
        rows = [r for r in market_rows(pd.read_parquet(path), venue) if r['level'] is None or r['level'] in levels]
        log(f"  {venue}: {len(rows)} abgerechnete Matches mit Sieger als Ergebnisquelle")
        out += rows
    return out
