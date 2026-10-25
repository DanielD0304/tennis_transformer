"""
Ablage der Paper-Wetten: data/paper/bets.parquet, eine Zeile pro Wette.

status: 'open' | 'won' | 'lost' | 'void'. Preise sind Dezimalquoten (brutto, wie bei der Venue verfügbar),
net_odds nach Gebühr bzw. Steuer. stake ist der Kelly-Einsatz, stake_flat der Vergleichseinsatz.
"""
import os

from ..sources.common import DATA_DIR

PAPER_DIR = os.path.join(DATA_DIR, "paper")
BETS_PATH = os.path.join(PAPER_DIR, "bets.parquet")

COLUMNS = ['bet_id', 'created_utc', 'strategy', 'venue', 'event_id', 'tour', 'level', 'tourney', 'player_a',
           'player_b', 'side', 'player', 'start_time_utc', 'odds', 'net_odds', 'p', 'p_market', 'edge', 'stake',
           'stake_flat', 'status', 'settled_utc', 'profit', 'profit_flat', 'close_odds', 'clv', 'clv_sharp',
           'result_source']
DATETIME = ['created_utc', 'start_time_utc', 'settled_utc']
FLOAT = ['odds', 'net_odds', 'p', 'p_market', 'edge', 'stake', 'stake_flat', 'profit', 'profit_flat', 'close_odds',
         'clv', 'clv_sharp']
STATUSES = ('open', 'won', 'lost', 'void')


def bet_id(strategy, venue, event_id, side):
    return f"{strategy}|{venue}|{event_id}|{side}"


def normalize(df):
    import pandas as pd
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = None
    for c in DATETIME:
        df[c] = pd.to_datetime(df[c], utc=True)
    for c in FLOAT:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype(float)
    for c in COLUMNS:
        if c not in DATETIME + FLOAT:
            df[c] = df[c].astype(object).where(df[c].notna(), None)
    return df[COLUMNS]


def load_bets(path=None):
    import pandas as pd
    path = path or BETS_PATH
    if not os.path.exists(path):
        return normalize(pd.DataFrame(columns=COLUMNS))
    return normalize(pd.read_parquet(path))


def save_bets(df, path=None):
    path = path or BETS_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    normalize(df.copy()).sort_values('created_utc', kind='stable').to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return path


def _query(r):
    """Wette -> Zeile für build.MatchIndex (Datum = Startzeit, sonst Anlagezeit)."""
    from ..sources.common import yyyymmdd
    when = r['start_time_utc'] if r['start_time_utc'] is not None and r['start_time_utc'] == r['start_time_utc'] \
        else r['created_utc']
    day = yyyymmdd(when)
    return {'source': 'paper', 'winner_name': r['player_a'], 'loser_name': r['player_b'], 'match_date': day,
            'tourney_date': day}


def add_bets(existing, new):
    """Neue Wetten anhängen. Übersprungen: schon vorhandene bet_id und Matches, die dieselbe Strategie schon
    (auch bei einer anderen Venue oder auf der anderen Seite) gewettet hat. -> (alle Wetten, Anzahl neu)"""
    import pandas as pd
    from ..sources.build import MatchIndex
    if new is None or len(new) == 0:
        return existing, 0
    index = {}
    for r in existing.to_dict('records'):
        index.setdefault(r['strategy'], MatchIndex()).add(_query(r))
    known = set(existing['bet_id'])
    keep = []
    for r in normalize(new.copy()).to_dict('records'):
        q = _query(r)
        idx = index.setdefault(r['strategy'], MatchIndex())
        if r['bet_id'] in known or idx.find(q)[0] is not None:
            continue
        idx.add(q)
        known.add(r['bet_id'])
        keep.append(r)
    if not keep:
        return existing, 0
    added = normalize(pd.DataFrame(keep))
    return (pd.concat([existing, added], ignore_index=True) if len(existing) else added), len(keep)
