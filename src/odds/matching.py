"""
Quoten (Snapshots oder historische Märkte) auf Matches aus data/matches.csv bzw. aufeinander abbilden.

Wiederverwendet src/sources/build.MatchIndex: Zeitfenster (±2 Tage um den Matchtag, bei Sackmann das
ganze Turnier) und Namensvergleich über src/names.candidate_keys. Gefunden wird ein Match auch mit
vertauschten Seiten; same_order sagt, ob player_a der Sieger (matches.csv) bzw. player_a der anderen
Venue ist.

Achtung: data/matches.csv enthält bisher nur ATP-Haupttour (Top-Turniere, kein Challenger, keine
WTA). Die Abgleich-Quote gegen matches.csv ist deshalb nur für tour='atp', level 'main' bzw.
unbekannt aussagekräftig.
"""
import os

from ..sources.build import MatchIndex
from ..sources.common import MATCHES_PATH, yyyymmdd


def _when(row):
    for key in ('start_time_utc', 'ts_utc'):
        value = row.get(key)
        if value is not None and value == value:  # NaT/NaN != sich selbst
            return yyyymmdd(value)
    return ""


def _odds_row(row, i=None):
    day = _when(row)
    return {'source': 'odds', 'match_date': day, 'tourney_date': day, 'winner_name': row['player_a'],
            'loser_name': row['player_b'], '_i': i}


def load_matches(path=MATCHES_PATH):
    import csv
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_index(rows):
    index = MatchIndex()
    for r in rows:
        if r.get('winner_name') and r.get('loser_name') and (r.get('match_date') or r.get('tourney_date')):
            index.add(r)
    return index


def link_matches(df, matches):
    """
    Pro Zeile von df (player_a, player_b, start_time_utc/ts_utc) das Match aus matches (Liste von
    dicts wie in matches.csv). Neue Spalten: match_found, a_won (player_a ist Sieger laut matches.csv),
    match_date, match_tourney, match_status.
    """
    df = df.copy()
    index = build_index(matches)
    found, a_won, dates, tourneys, status = [], [], [], [], []
    odds = {f'{k}_{side}': [] for k in ('ps', 'avg') for side in 'ab'}
    for row in df.to_dict('records'):
        q = _odds_row(row)
        other, same = index.find(q) if q['match_date'] else (None, None)
        found.append(other is not None)
        a_won.append(same if other is not None else None)
        dates.append((other or {}).get('match_date') or (other or {}).get('tourney_date'))
        tourneys.append((other or {}).get('tourney_name'))
        status.append((other or {}).get('status'))
        for key, col in (('ps', 'PS'), ('avg', 'Avg')):
            w, l = _odd((other or {}).get(f'{col}W')), _odd((other or {}).get(f'{col}L'))
            odds[f'{key}_a'].append(w if same else l)
            odds[f'{key}_b'].append(l if same else w)
    df['match_found'], df['a_won'], df['match_date'] = found, a_won, dates
    df['match_tourney'], df['match_status'] = tourneys, status
    for col, values in odds.items():
        df[col] = values
    return df


def _odd(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value > 1 else None


def pair_venues(left, right):
    """
    Dieselben Matches in zwei Tabellen (z.B. Polymarket- und Kalshi-Historie). Liefert ein DataFrame mit
    left_event_id, right_event_id, same_order (player_a links = player_a rechts).
    """
    import pandas as pd
    index = MatchIndex()
    rights = right.reset_index(drop=True).to_dict('records')
    for i, row in enumerate(rights):
        q = _odds_row(row, i)
        if q['match_date']:
            index.add(q)
    pairs = []
    for row in left.to_dict('records'):
        q = _odds_row(row)
        if not q['match_date']:
            continue
        other, same = index.find(q)
        if other is not None:
            pairs.append({'left_event_id': row['event_id'], 'right_event_id': rights[other['_i']]['event_id'],
                          'same_order': bool(same)})
    return pd.DataFrame(pairs, columns=['left_event_id', 'right_event_id', 'same_order'])


def match_rate(df, matches, by=('venue',)):
    """Abgleich-Quote je Gruppe: Anzahl Matches (je venue/event_id einmal), gefunden, Quote."""
    import pandas as pd
    unique = df.drop_duplicates(['venue', 'event_id'])
    linked = link_matches(unique, matches)
    rows = []
    for key, g in linked.groupby(list(by), dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)), 'n': len(g), 'matched': int(g['match_found'].sum()),
                     'rate': float(g['match_found'].mean()) if len(g) else float('nan')})
    return pd.DataFrame(rows)
