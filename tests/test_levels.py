"""Challenger/ITF-Matchdaten: Level aus den Sackmann-Dateien, Filter, Märkte als Ergebnisquelle (ohne Netz)."""
import csv

import pandas as pd

from src.evaluation import backtest
from src.evaluation.predictors import prepare_matches
from src.sources import build, markets, sackmann

HEADER = ['tourney_id', 'tourney_name', 'surface', 'tourney_level', 'tourney_date', 'match_num', 'round',
          'winner_name', 'loser_name', 'score']


def _write(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _row(level, name, w, l, day='20250106', num=1, rnd='R32'):
    return {'tourney_id': f'{name}-{day}', 'tourney_name': name, 'surface': 'Clay', 'tourney_level': level,
            'tourney_date': day, 'match_num': num, 'round': rnd, 'winner_name': w, 'loser_name': l, 'score': '6-3 6-4'}


def test_level_of_codes():
    assert sackmann.level_of('main', 'G') == 'main'
    assert sackmann.level_of('qual_chall', 'C') == 'challenger'
    assert sackmann.level_of('qual_chall', 'A') == 'main'  # Qualifikation der Haupttour
    assert sackmann.level_of('futures', '15') == 'itf'
    assert sackmann.level_of('qual_itf', 'C') == 'challenger'  # WTA 125
    assert sackmann.level_of('qual_itf', '35') == 'itf'
    assert sackmann.level_of('qual_itf', 'PM') == 'main'


def test_kinds_for_levels():
    assert sackmann.kinds(('main',), 'atp') == ['main', 'qual_chall']
    assert sackmann.kinds(('itf',), 'atp') == ['futures']
    assert sackmann.kinds(('challenger', 'itf'), 'wta') == ['qual_itf']


def test_load_filters_levels(tmp_path, monkeypatch):
    monkeypatch.setattr(sackmann, 'DIR', str(tmp_path))
    _write(sackmann.path(2025, 'atp', 'main'), [_row('A', 'Brisbane', 'A One', 'B Two')])
    _write(sackmann.path(2025, 'atp', 'qual_chall'), [_row('C', 'Nonthaburi', 'C Three', 'D Four'),
                                                     _row('A', 'Brisbane', 'E Five', 'F Six', rnd='Q1')])
    _write(sackmann.path(2025, 'atp', 'futures'), [_row('15', 'M15 Antalya', 'G Seven', 'H Eight')])

    all_rows = sackmann.load([2025], 'atp', levels=('main', 'challenger', 'itf'))
    assert sorted(r['level'] for r in all_rows) == ['challenger', 'itf', 'main', 'main']
    assert {r['tourney_level'] for r in all_rows} == {'A', 'C', '15'}  # Sackmann-Code bleibt erhalten
    only = sackmann.load([2025], 'atp', levels=('challenger',))
    assert [(r['level'], r['winner_name']) for r in only] == [('challenger', 'C Three')]
    main = sackmann.load([2025], 'atp', levels=('main',))
    assert {r['winner_name'] for r in main} == {'A One', 'E Five'}


def test_backtest_level_filter():
    df = pd.DataFrame({'level': ['main', 'challenger', 'itf', 'main'], 'x': range(4)})
    assert list(backtest.filter_levels(df, ['challenger', 'itf'])['x']) == [1, 2]
    assert len(backtest.filter_levels(df, ['all'])) == 4
    assert len(backtest.filter_levels(df, None)) == 4


def _history(rows):
    cols = ['event_id', 'tour', 'level', 'tourney', 'start_time_utc', 'player_a', 'player_b', 'winner']
    return pd.DataFrame(rows, columns=cols)


def test_market_rows_from_resolution():
    df = _history([
        ['1', 'atp', 'challenger', 'Lima', pd.Timestamp('2026-10-20 15:00', tz='UTC'), 'Ann Alpha', 'Bea Beta', 'b'],
        ['2', 'atp', 'itf', 'M15', pd.Timestamp('2026-10-20 15:00', tz='UTC'), 'X Y', 'Z W', None],  # 50-50
        ['3', None, 'main', 'Davis', pd.Timestamp('2026-10-20 15:00', tz='UTC'), 'Q R', 'S T', 'a'],  # Tour fehlt
    ])
    rows = markets.market_rows(df, 'polymarket')
    assert len(rows) == 1
    r = rows[0]
    assert (r['winner_name'], r['loser_name']) == ('Bea Beta', 'Ann Alpha')
    assert (r['source'], r['level'], r['score'], r['match_date'], r['tourney_level']) == \
        ('polymarket', 'challenger', '', '20261020', 'C')


def _sack(w, l, day, level='challenger', tid='T1'):
    return {'tour': 'atp', 'level': level, 'tourney_id': tid, 'tourney_name': 'Lima Challenger', 'surface': 'Clay',
            'tourney_level': 'C', 'tourney_date': day, 'match_date': day, 'match_num': 1, 'round': 'R32',
            'winner_name': w, 'loser_name': l, 'score': '6-1 6-1', 'status': 'completed', 'source': 'sackmann'}


def _market(w, l, day, source, level='challenger'):
    return {'tour': 'atp', 'level': level, 'tourney_id': f'{source}-x', 'tourney_name': 'Lima', 'tourney_level': 'C',
            'tourney_date': day, 'match_date': day, 'round': '', 'best_of': 3, 'winner_name': w, 'loser_name': l,
            'score': '', 'status': 'completed', 'source': source}


def test_build_uses_markets_after_sackmann_and_detects_duplicates():
    sack = [_sack('Ann Alpha', 'Bea Beta', '20260525'), _sack('Cid Gamma', 'Dan Delta', '20260601', tid='T2')]
    mk = [
        _market('Ann Alpha', 'Bea Beta', '20260527', 'kalshi'),       # Duplikat eines Sackmann-Matches
        _market('Eve Epsilon', 'Fay Zeta', '20261010', 'kalshi'),     # neu, nach dem Sackmann-Ende
        _market('Eve Epsilon', 'Fay Zeta', '20261011', 'polymarket'),  # dasselbe Match bei PM (Kalshi hat Vorrang)
        _market('Gus Eta', 'Hal Theta', '20261011', 'polymarket'),    # nur bei PM
        _market('Ivy Iota', 'Jon Kappa', '20260415', 'polymarket'),   # vor dem Ende, nicht bei Sackmann: verworfen
    ]
    rows, stats = build.build(sack, [], [], market_rows=mk)
    by_source = sorted((r['source'], r['winner_name']) for r in rows)
    assert by_source == [('kalshi', 'Eve Epsilon'), ('polymarket', 'Gus Eta'),
                         ('sackmann', 'Ann Alpha'), ('sackmann', 'Cid Gamma')]
    assert stats['kalshi'] == 1 and stats['polymarket'] == 1
    assert stats['polymarket_duplicates'] == 1
    # Das echte Matchdatum kommt aus dem Markt in die Sackmann-Zeile
    assert next(r for r in rows if r['winner_name'] == 'Ann Alpha')['match_date'] == '20260527'
    new = next(r for r in rows if r['winner_name'] == 'Gus Eta')
    assert new['level'] == 'challenger' and new['score'] == '' and new['surface'] == 'Clay'


def test_elo_runs_across_levels():
    """Elo2 läuft über alle Level einer Tour: ein Challenger-Sieg zählt auch für das spätere Haupttour-Match."""
    rows = [_sack('Ann Alpha', 'Bea Beta', f'202501{d:02d}', tid=f'T{d}') for d in range(1, 21)]
    rows.append({**_sack('Ann Alpha', 'Bea Beta', '20250301', level='main', tid='M1'), 'tourney_level': 'A'})
    df = pd.DataFrame(rows).assign(winner_rank=100, loser_rank=200)
    m = prepare_matches(df)
    last = m[m['level'] == 'main'].iloc[0]
    p_ann = last['elo2_prob_a'] if last['player_a'] == 'Ann Alpha' else 1 - last['elo2_prob_a']
    assert p_ann > 0.7
    assert set(m['level']) == {'main', 'challenger'}


def test_level_from_tourney_name_for_espn_and_polymarket():
    sack = [_sack('Ann Alpha', 'Bea Beta', '20260525'), {**_sack('Cid Gamma', 'Dan Delta', '20260601', tid='T2'),
                                                       'tourney_name': 'Lisboa Belem Open'}]
    espn = [{'tour': 'atp', 'level': 'main', 'tourney_id': 'espn-1', 'tourney_name': 'Lisboa Belem Open',
             'tourney_date': '20260920', 'match_date': '20260922', 'round': 'R16', 'winner_name': 'Eve Epsilon',
             'loser_name': 'Fay Zeta', 'score': '6-4 6-4', 'status': 'completed', 'source': 'espn'},
            {'tour': 'atp', 'level': 'main', 'tourney_id': 'espn-2', 'tourney_name': 'Erste Bank Open',
             'tourney_date': '20261020', 'match_date': '20261021', 'round': 'R32', 'winner_name': 'Gus Eta',
             'loser_name': 'Hal Theta', 'score': '6-4 6-4', 'status': 'completed', 'source': 'espn'}]
    mk = [{**_market('Ivy Iota', 'Jon Kappa', '20261012', 'polymarket'), 'level': None, 'tourney_level': ''}]
    rows, _ = build.build(sack, [], espn, market_rows=mk)
    level = {r['winner_name']: (r['level'], r['tourney_level']) for r in rows}
    assert level['Eve Epsilon'] == ('challenger', 'C')
    assert level['Gus Eta'][0] == 'main'
    assert level['Ivy Iota'] == ('challenger', 'C')  # 'Lima' wie 'Lima Challenger' bei Sackmann
