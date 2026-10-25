"""
Tests für Elo, Reihenfolge und die Features in preprocessing (synthetische Matches, kein Netz).
"""
import pandas as pd
import pytest

from src.chronology import chronological
from src.config import default_config
from src.elo import compute_elo_ratings
from src.preprocessing import (FEATURE_NAMES, context_features, create_training_samples, extract_match_features,
                               fatigue)


def _m(w, l, day, **kw):
    return {'tourney_id': kw.pop('tid', 't'), 'tourney_name': kw.pop('name', 'Open'), 'surface': 'Hard',
            'tourney_date': kw.pop('tdate', day), 'match_date': day, 'match_num': kw.pop('num', 0),
            'round': kw.pop('round', 'R32'), 'winner_name': w, 'loser_name': l, 'winner_rank': 10,
            'loser_rank': 20, 'score': kw.pop('score', '6-4 6-4'), 'status': kw.pop('status', 'completed'),
            'tour': kw.pop('tour', 'atp'), **kw}


def test_config_matches_features():
    assert default_config.input_dim == len(FEATURE_NAMES)
    assert default_config.max_rank == 500


# ----------------------------------------------------------------- Elo

def test_elo_ret_weight():
    df = pd.DataFrame([_m('A', 'B', '20240101'), _m('C', 'D', '20240101', score='6-4 2-1 RET', status='RET')])
    out = compute_elo_ratings(df, ret_weight=0.5).set_index('winner_name')
    full = out.loc['A', 'winner_elo_after'] - 1500
    ret = out.loc['C', 'winner_elo_after'] - 1500
    assert full == pytest.approx(16.0)
    assert ret == pytest.approx(8.0)
    # Gewicht konfigurierbar, 1.0 = wie ein normales Match
    out = compute_elo_ratings(df, ret_weight=1.0).set_index('winner_name')
    assert out.loc['C', 'winner_elo_after'] - 1500 == pytest.approx(16.0)


def test_elo_walkover_and_score_only_retirement():
    df = pd.DataFrame([_m('A', 'B', '20240101', score='W/O', status='W/O'),
                       _m('A', 'B', '20240102', score='6-1 Ret.', status='')])
    out = compute_elo_ratings(df)
    assert out.loc[0, 'winner_elo_after'] == 1500  # W/O ändert nichts
    assert out.loc[1, 'winner_elo_after'] - 1500 == pytest.approx(8.0)


def test_elo_separate_per_tour():
    df = pd.DataFrame([_m('Murray', 'X', '20240101', tour='atp'), _m('Murray', 'Y', '20240102', tour='wta'),
                       _m('Z', 'Murray', '20240103', tour='wta')])
    out = compute_elo_ratings(df)
    # WTA-Murray startet bei 1500, obwohl ATP-Murray schon gewonnen hat
    assert out.loc[1, 'winner_elo'] == 1500
    assert out.loc[2, 'loser_elo'] == pytest.approx(1516.0)


def test_sort_by_match_date():
    # gleiche Turnierwoche, Sackmann-Nummern sagen B-C zuerst, echtes Datum sagt A-B zuerst
    rows = [_m('B', 'C', '20240105', tdate='20240101', num=1, round='R32'),
            _m('A', 'B', '20240102', tdate='20240101', num=2, round='R32')]
    df = pd.DataFrame(rows)
    assert list(chronological(df)['winner_name']) == ['A', 'B']
    out = compute_elo_ratings(df)
    assert list(out['winner_name']) == ['A', 'B']
    assert out.loc[1, 'winner_elo'] == pytest.approx(1500 - 16)  # B hat vorher gegen A verloren

    # ohne match_date: Turnierdatum, dann Runde
    rows = [_m('B', 'C', '', tdate='20240101', num=1, round='QF'),
            _m('A', 'B', '', tdate='20240101', num=2, round='R16')]
    assert list(chronological(pd.DataFrame(rows))['round']) == ['R16', 'QF']


# ------------------------------------------------------------ Features

def _hist(day, sets=2, minutes=90.0):
    return {'_date_obj': pd.Timestamp(day), '_sets': sets, '_minutes': minutes}


def test_fatigue_only_past():
    history = [_hist('2024-01-01'), _hist('2024-01-05', 3, 150.0), _hist('2024-01-09')]
    assert fatigue(history, pd.Timestamp('2024-01-10'), 7) == (2, 5, 240.0)
    assert fatigue(history, pd.Timestamp('2024-01-10'), 14) == (3, 7, 330.0)
    # Matches nach dem Stichtag zählen nie (Historie ist chronologisch)
    assert fatigue(history + [_hist('2024-01-20')], pd.Timestamp('2024-01-10'), 14) == (3, 7, 330.0)
    ctx = context_features(history, pd.Timestamp('2024-01-10'), 0.5, dob='20000110', hand='L', home=1)
    assert ctx['fatigue_matches_7d'] == pytest.approx(2 / 5)
    assert ctx['age'] == pytest.approx(24.0 / 40, abs=1e-3)
    assert ctx['left_handed'] == 1 and ctx['ctx_home'] == 1


def _samples(rows, **kw):
    df = compute_elo_ratings(pd.DataFrame(rows))
    from src.data_loader import enforce_schema
    return create_training_samples(enforce_schema(df), n_matches=3, n_recent=5, **kw), df


def _season(a='A', b='B', start=1, n=8, **kw):
    """n Matches zwischen a und b mit Abstand von zwei Tagen (beide genug Historie und aktiv)."""
    return [_m(a if i % 2 else b, b if i % 2 else a, f"202401{start + 2 * i:02d}", num=i, **kw) for i in range(n)]


def test_samples_use_only_past_matches():
    rows = _season(n=8)
    samples, _ = _samples(rows)
    first = samples[0]
    # Ergebnis von heute und später steckt nicht in den Features: Historie endet vor dem Match
    later = rows + [_m('A', 'B', '20240131', num=99, minutes=300)]
    samples_later, _ = _samples(later)
    assert samples_later[0]['player_a_recent'] == first['player_a_recent']
    # Müdigkeit zum 6. Match (11.01.): Matches am 01., 03., 05., 07., 09. -> 3 in den letzten 7 Tagen
    sixth = next(s for s in samples if s['match_index'] == 5)
    assert sixth['player_a_recent'][0]['fatigue_matches_7d'] == pytest.approx(3 / 5)
    assert sixth['player_a_recent'][0]['fatigue_matches_14d'] == pytest.approx(5 / 5)


def test_home_feature():
    row = pd.Series(_m('A', 'B', '20240101', winner_ioc='GER', loser_ioc='FRA', tourney_ioc='GER',
                       court='Indoor', best_of=5, minutes=120))
    from src.data_loader import enforce_schema
    row = enforce_schema(pd.DataFrame([row])).iloc[0]
    w = extract_match_features(row, 'winner', 0.5)
    l = extract_match_features(row, 'loser', 0.5)
    assert (w['home'], l['home']) == (1, 0)
    assert w['indoor'] == 1 and w['best_of_5'] == 1
    assert w['game_share'] == pytest.approx(0.6) and l['game_share'] == pytest.approx(0.4)
    assert w['minutes'] == pytest.approx(120 / 180) and w['has_minutes'] == 1
    # ohne Land beim Turnier kein Heimvorteil
    row2 = row.copy()
    row2['tourney_ioc'] = ''
    assert extract_match_features(row2, 'winner', 0.5)['home'] == 0


def test_ctx_home_in_samples():
    rows = _season(n=8, winner_ioc='GER', loser_ioc='FRA', tourney_ioc='GER')
    # Sieger/Verlierer wechseln, Nationalität gehört zum Spieler
    for r in rows:
        r['winner_ioc'], r['loser_ioc'] = ('GER', 'FRA') if r['winner_name'] == 'A' else ('FRA', 'GER')
    samples, _ = _samples(rows)
    for s in samples:
        a_home = s['player_a_recent'][0]['ctx_home']
        b_home = s['player_b_recent'][0]['ctx_home']
        assert {a_home, b_home} == {0, 1}


def test_rank_filter_boundary():
    rows = _season(n=8)
    for r in rows:
        r['winner_rank'], r['loser_rank'] = 500, 501
    assert _samples(rows, max_rank=500)[0] == []
    assert len(_samples(rows, max_rank=501)[0]) > 0
    for r in rows:
        r['winner_rank'], r['loser_rank'] = 500, 300
    assert len(_samples(rows)[0]) > 0  # Standard 500: Grenze inklusive


def test_tours_have_separate_history():
    # gleiche Namen auf beiden Touren: WTA-Historie leer -> keine WTA-Samples
    rows = _season(n=8) + [_m('A', 'B', '20240120', num=50, tour='wta')]
    samples, df = _samples(rows)
    assert all(df.loc[s['match_index'], 'tour'] == 'atp' for s in samples)
