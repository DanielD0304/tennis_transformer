"""
Die flache Feature-Tabelle (src/features.py) darf nur Daten VOR dem Match benutzen.
Erzwungen über zwei Eigenschaften:
  - Abschneiden: Features einer Zeile ändern sich nicht, wenn alle späteren Matches fehlen.
  - Ergebnis: Das Ergebnis eines Matches (und aller späteren) ändert seine eigenen Features nicht.
"""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.predictors import prepare_matches
from src.features import KEY_COLUMNS, PLAYER_FEATURES, feature_table
from tests.test_evaluation import synthetic_matches


@pytest.fixture(scope="module")
def raw():
    df = synthetic_matches(years=(2020,), per_month=15, n_players=12, seed=3)
    rng = np.random.default_rng(0)
    df['minutes'] = rng.integers(60, 200, len(df))
    df['score'] = np.where(df['status'] == 'W/O', 'W/O', np.where(df['status'] == 'RET', '6-4 2-1 RET', '7-6(4) 3-6 6-2'))
    df['winner_ioc'] = 'GER'
    df['loser_ioc'] = 'FRA'
    df['tourney_ioc'] = np.where(np.arange(len(df)) % 2, 'GER', 'FRA')
    df['winner_dob'] = '19990101'
    df['loser_dob'] = '20010101'
    df['court'] = 'Indoor'
    df['best_of'] = 3
    return df


def _features(df):
    return feature_table(prepare_matches(df)).set_index('match_id')


def _values(t):
    return t.drop(columns=[c for c in KEY_COLUMNS if c in t.columns]).astype(float)


def test_truncation_does_not_change_features(raw):
    full = _features(raw)
    cut_day = '20200615'
    early = _features(raw[raw['match_date'] < cut_day])
    pd.testing.assert_frame_equal(_values(full.loc[early.index]), _values(early))


def test_outcome_does_not_leak(raw):
    order = prepare_matches(raw)
    k = len(order) // 2
    # Ab Match k (in echter Reihenfolge) alle Ergebnisse umdrehen: Sieger <-> Verlierer, Quoten mit
    later = set(zip(order.loc[k:, 'tourney_id'], order.loc[k:, 'match_num']))
    sel = np.array([(t, n) in later for t, n in zip(raw['tourney_id'], raw['match_num'])])
    flipped = raw.copy()
    for w, l in (('winner_name', 'loser_name'), ('winner_rank', 'loser_rank'), ('winner_ioc', 'loser_ioc'),
                 ('winner_dob', 'loser_dob'), ('PSW', 'PSL'), ('B365W', 'B365L'), ('AvgW', 'AvgL')):
        flipped.loc[sel, [w, l]] = raw.loc[sel, [l, w]].to_numpy()

    def by_match(df):
        t = feature_table(prepare_matches(df))
        m = prepare_matches(df)
        return {(tid, num): row for tid, num, row in zip(m['tourney_id'], m['match_num'], t.to_dict('records'))}

    a, b = by_match(raw), by_match(flipped)
    for key in list(zip(order['tourney_id'], order['match_num']))[:k + 1]:
        mine, other = a[key], b[key]
        # A/B-Zuordnung hängt am Sieger und kann sich mit dem Umdrehen ändern
        same = mine['player_a'] == other['player_a']
        for f in PLAYER_FEATURES:
            expected = mine[f'{f}_a'] if same else mine[f'{f}_b']
            assert other[f'{f}_a'] == pytest.approx(expected, nan_ok=True), (key, f)


def test_feature_values(raw):
    t = _features(raw)
    assert t['h2h_pct_a'].between(0, 1).all()
    assert (t['fatigue_matches_14d_a'] >= t['fatigue_matches_7d_a']).all()
    assert t['indoor'].eq(1).all()
    played = t[t['n_matches_a'] > 0]
    assert played['win_rate_a'].between(0, 1).all()
    assert set(t['home_a'].unique()) <= {0, 1}
    assert (t['age_a'] > 18).all()
    first = t.sort_values('date').iloc[0]
    assert first['n_matches_a'] == 0 and np.isnan(first['win_rate_a'])
