"""
Flache Feature-Tabelle für tabellarische Modelle (Phase 3, LightGBM): eine Zeile pro Match aus Sicht
von Spieler A (A/B-Zuordnung wie im Backtest, evaluation.predictors.prepare_matches).

Alle Werte stammen ausschließlich aus Matches, die in der echten Reihenfolge (chronology.py) VOR dem
Match liegen. Die Tabelle entsteht in einem Durchlauf: erst wird die Zeile aus dem aktuellen Stand
gebaut, danach fließt das Ergebnis in den Stand ein. tests/test_features.py erzwingt das.

    from src.evaluation.predictors import prepare_matches
    from src.features import feature_table
    table = feature_table(prepare_matches(df))
"""
import math
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from .elo import is_retired
from .preprocessing import (FATIGUE_WINDOWS, MINUTES_PER_GAME, _num, _score, _text, age_years, is_home,
                            is_walkover)

FORM_WINDOW = 10      # letzte N Matches für Form-Werte
SURFACE_WINDOW = 20   # letzte N Matches auf dem Untergrund
MAX_FATIGUE_DAYS = max(FATIGUE_WINDOWS)

# Werte pro Spieler; in der Tabelle als <name>_a, <name>_b und <name>_diff (= a - b)
PLAYER_FEATURES = [
    'elo', 'log_rank', 'n_matches', 'win_rate', 'game_share', 'set_share', 'tiebreak_rate',
    'retired_rate', 'surface_win_rate', 'surface_matches', 'days_since_last', 'retired_last',
] + [f'fatigue_{k}_{d}d' for d in FATIGUE_WINDOWS for k in ('matches', 'sets', 'minutes')] \
  + ['age', 'left_handed', 'home']
MATCH_FEATURES = ['best_of_5', 'indoor', 'surf_hard', 'surf_clay', 'surf_grass', 'surf_carpet',
                  'h2h_wins_a', 'h2h_wins_b', 'h2h_pct_a', 'elo_prob_a']
KEY_COLUMNS = ['match_id', 'date', 'year', 'tour', 'player_a', 'player_b', 'label', 'status']


class _Player:
    __slots__ = ('recent', 'surface', 'days', 'n', 'last_date', 'last_retired')

    def __init__(self):
        self.recent = deque(maxlen=FORM_WINDOW)          # (won, game_share, set_share, tiebreaks, retired)
        self.surface = defaultdict(lambda: deque(maxlen=SURFACE_WINDOW))  # Untergrund -> [won]
        self.days = deque()                               # (Datum, Sätze, Minuten) für die Müdigkeit
        self.n = 0
        self.last_date = None
        self.last_retired = 0  # hat der Spieler sein letztes Match aufgegeben?


def _mean(values, default=np.nan):
    return float(np.mean(values)) if values else default


def _player_row(state, surface, date, elo, rank, dob, hand, ioc, tourney_ioc):
    recent = list(state.recent)
    on_surface = list(state.surface[surface]) if surface else []
    out = {
        'elo': elo,
        'log_rank': math.log(rank + 1) if rank and rank > 0 else np.nan,
        'n_matches': state.n,
        'win_rate': _mean([r[0] for r in recent]),
        'game_share': _mean([r[1] for r in recent]),
        'set_share': _mean([r[2] for r in recent]),
        'tiebreak_rate': _mean([r[3] for r in recent]),
        'retired_rate': _mean([r[4] for r in recent]),
        'surface_win_rate': _mean(on_surface),
        'surface_matches': len(on_surface),
        'days_since_last': (date - state.last_date).days if state.last_date is not None else np.nan,
        'retired_last': state.last_retired,
    }
    for days in FATIGUE_WINDOWS:
        threshold = date - pd.Timedelta(days=days)
        window = [d for d in state.days if threshold <= d[0] <= date]
        out[f'fatigue_matches_{days}d'] = len(window)
        out[f'fatigue_sets_{days}d'] = sum(d[1] for d in window)
        out[f'fatigue_minutes_{days}d'] = sum(d[2] for d in window)
    age = age_years(dob, date)
    out['age'] = age if age is not None else np.nan
    out['left_handed'] = 1 if hand == 'L' else 0
    out['home'] = is_home(ioc, tourney_ioc)
    return out


def _update(state, date, surface, won, score, minutes, retired_self, retired_match):
    games = score.games_w + score.games_l
    state.recent.append((int(won), score.game_share(won) if score.valid else 0.5,
                         score.set_share(won) if score.valid else 0.5,
                         score.tiebreaks / max(score.n_sets, 1), int(retired_match)))
    state.surface[surface].append(int(won))
    state.days.append((date, score.n_sets, minutes if minutes > 0 else games * MINUTES_PER_GAME))
    while state.days and (date - state.days[0][0]).days > MAX_FATIGUE_DAYS:
        state.days.popleft()
    state.n += 1
    state.last_date = date
    state.last_retired = int(retired_self)


class FeatureBuilder:
    """
    Stand aller Spieler einer Datenbasis. row(m) baut die Feature-Zeile für ein Match aus dem aktuellen
    Stand (ohne ihn zu ändern), update(m) schreibt das Ergebnis ein. feature_table ruft beides im
    Wechsel auf, predict.MatchPredictor nutzt row() für Matches, die noch nicht gespielt sind.
    m: Dict mit den Spalten von prepare_matches (player_a/b, label, date, surface, ...).
    """

    def __init__(self):
        self.players = defaultdict(_Player)  # (Tour, Name) -> Stand
        self.h2h = defaultdict(int)          # (Tour, Sieger, Verlierer) -> Siege

    def row(self, m):
        tour = _text(m.get('tour')) or 'atp'
        a, b = m['player_a'], m['player_b']
        a_won = int(m['label']) == 1
        date = m['date']
        surface = _text(m.get('surface'))
        tourney_ioc = _text(m.get('tourney_ioc'))
        side_a, side_b = ('winner', 'loser') if a_won else ('loser', 'winner')
        best_of = int(_num(m.get('best_of'), 3)) or 3

        row = {k: m.get(k) for k in KEY_COLUMNS}
        feats = {}
        for p, side, x in ((a, side_a, 'a'), (b, side_b, 'b')):
            feats[x] = _player_row(self.players[(tour, p)], surface, date, _num(m.get(f'elo_{x}'), np.nan),
                                   _num(m.get(f'rank_{x}'), 0), m.get(f'{side}_dob'), _text(m.get(f'{side}_hand')),
                                   _text(m.get(f'{side}_ioc')), tourney_ioc)
        for f in PLAYER_FEATURES:
            row[f'{f}_a'], row[f'{f}_b'] = feats['a'][f], feats['b'][f]
            row[f'{f}_diff'] = feats['a'][f] - feats['b'][f]
        wins_a, wins_b = self.h2h[(tour, a, b)], self.h2h[(tour, b, a)]
        row.update({
            'best_of_5': int(best_of == 5),
            'indoor': int(_text(m.get('court')) == 'Indoor'),
            'surf_hard': int(surface == 'Hard'), 'surf_clay': int(surface == 'Clay'),
            'surf_grass': int(surface == 'Grass'), 'surf_carpet': int(surface == 'Carpet'),
            'h2h_wins_a': wins_a, 'h2h_wins_b': wins_b,
            'h2h_pct_a': wins_a / (wins_a + wins_b) if wins_a + wins_b else 0.5,
            'elo_prob_a': 1 / (1 + 10 ** ((row['elo_b'] - row['elo_a']) / 400)),
        })
        return row

    def update(self, m):
        # W/O ist kein gespieltes Match
        if is_walkover(m):
            return
        tour = _text(m.get('tour')) or 'atp'
        a, b = m['player_a'], m['player_b']
        a_won = int(m['label']) == 1
        date = m['date']
        surface = _text(m.get('surface'))
        score = _score(_text(m.get('score')))
        retired = is_retired(m)
        winner, loser = (a, b) if a_won else (b, a)
        minutes = _num(m.get('minutes'))
        _update(self.players[(tour, winner)], date, surface, True, score, minutes, False, retired)
        # Bei RET/DEF gibt immer der Verlierer auf
        _update(self.players[(tour, loser)], date, surface, False, score, minutes, retired, retired)
        self.h2h[(tour, winner, loser)] += 1


FEATURE_COLUMNS = MATCH_FEATURES + [f'{f}_{x}' for f in PLAYER_FEATURES for x in ('a', 'b', 'diff')]


def feature_table(matches, builder=None):
    """
    matches: Ausgabe von evaluation.predictors.prepare_matches (A/B-Sicht, mit elo_a/elo_b).
    -> DataFrame mit KEY_COLUMNS, MATCH_FEATURES und <feature>_a/_b/_diff für PLAYER_FEATURES,
    in der Reihenfolge von matches (die schon chronologisch ist). builder: FeatureBuilder, der danach
    den Stand nach dem letzten Match hält (für Live-Vorhersagen).
    """
    builder = builder if builder is not None else FeatureBuilder()
    rows = []
    for m in matches.to_dict('records'):
        rows.append(builder.row(m))  # erst die Zeile aus dem Stand VOR dem Match ...
        builder.update(m)            # ... dann fließt das Ergebnis ein
    return pd.DataFrame(rows, columns=KEY_COLUMNS + FEATURE_COLUMNS)
