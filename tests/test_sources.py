"""Tests für Download-Auswertung und Zusammenführung (ohne Netz, ohne pandas/torch)."""
import os

from src.names import candidate_keys
from src.sources import build, espn, tuk
from src.sources.common import round_code, status_from_score


def _competitor(name, won, games, tiebreaks=None):
    tiebreaks = tiebreaks or {}
    return {
        'winner': won,
        'athlete': {'displayName': name},
        'linescores': [{'value': float(g), 'winner': w, **({'tiebreak': tiebreaks[i]} if i in tiebreaks else {})}
                       for i, (g, w) in enumerate(games)],
    }


def _comp(cid, rnd, status, a, b, date="2026-10-08T05:00Z"):
    return {'id': cid, 'date': date, 'round': {'displayName': rnd},
            'status': {'type': {'description': status}}, 'competitors': [a, b]}


SCOREBOARD = {'events': [
    {'id': '315-2026', 'name': 'Rolex Shanghai Masters', 'date': '2026-10-05T04:00Z',
     'venue': {'displayName': 'Shanghai, China PR'},
     'groupings': [
         {'grouping': {'displayName': "Men's Singles"}, 'competitions': [
             _comp('1', 'Qualifying 1st Round', 'Final',
                   _competitor('Q Winner', True, [(6, True), (6, True)]), _competitor('Q Loser', False, [(1, False), (1, False)])),
             _comp('2', 'Round 1', 'Final',
                   _competitor('Arthur Gea', True, [(6, True), (6, False), (7, True)], {1: 1}),
                   _competitor('Ugo Humbert', False, [(4, False), (7, True), (5, False)], {1: 7})),
             _comp('3', 'Round 2', 'Retired',
                   _competitor('Carlos Alcaraz', True, [(6, True), (2, False)]),
                   _competitor('Juan Manuel Cerundolo', False, [(3, False), (2, False)])),
             _comp('4', 'Round 2', 'Walkover',
                   _competitor('Jannik Sinner', True, []), _competitor('Ben Shelton', False, [])),
             _comp('5', 'Round 4', 'Scheduled',
                   _competitor('A', False, []), _competitor('B', False, [])),
             _comp('6', 'Quarterfinal', 'Scheduled',
                   _competitor('C', False, []), _competitor('D', False, [])),
         ]},
         {'grouping': {'displayName': "Women's Singles"}, 'competitions': [
             _comp('9', 'Round 1', 'Final', _competitor('X', True, [(6, True)]), _competitor('Y', False, [(0, False)])),
         ]},
     ]},
    {'id': '99', 'name': 'Laver Cup', 'groupings': []},
]}


def test_espn_parse():
    rows = {r['espn_id']: r for r in espn.parse(SCOREBOARD)}
    # Qualifikation, Damen, geplante Matches und Show-Events fliegen raus
    assert set(rows) == {'2', '3', '4'}

    r1 = rows['2']
    assert (r1['winner_name'], r1['loser_name']) == ('Arthur Gea', 'Ugo Humbert')
    assert r1['score'] == '6-4 6-7(1) 7-5'
    assert r1['round'] == 'R128'  # 4 Runden vor dem Viertelfinale
    assert (r1['tourney_date'], r1['match_date']) == ('20261005', '20261008')
    assert r1['location'] == 'Shanghai'

    assert rows['3']['round'] == 'R64'
    assert (rows['3']['status'], rows['3']['score']) == ('RET', '6-3 2-2 RET')
    assert (rows['4']['status'], rows['4']['score']) == ('W/O', 'W/O')


def test_round_codes():
    assert round_code(1, 4) == 'R128'
    assert round_code(4, 4) == 'R16'
    assert round_code(1, 2) == 'R32'
    assert tuk._round('2nd Round', 2) == 'R16'
    assert tuk._round('The Final', 2) == 'F'


def test_status_from_score():
    assert status_from_score('6-4 3-1 RET') == 'RET'
    assert status_from_score('W/O') == 'W/O'
    assert status_from_score('6-4 6-4') == 'completed'


def test_candidate_keys():
    assert ('cerundolo', 'j') in candidate_keys('Juan Manuel Cerundolo')
    assert ('minaur', 'a') in candidate_keys('Alex de Minaur')
    assert candidate_keys('Cerundolo J.M.') == [('cerundolo', 'j')]


def _row(source, w, l, tourney_date, match_date=None, **kw):
    return {'source': source, 'winner_name': w, 'loser_name': l, 'tourney_date': tourney_date,
            'match_date': match_date or tourney_date, 'match_num': kw.pop('match_num', 0), **kw}


def test_build_merges_and_dedupes():
    sackmann = [
        _row('sackmann', 'Carlos Alcaraz', 'Juan Manuel Cerundolo', '20260525', surface='Clay',
             tourney_name='Roland Garros', winner_rank=2, loser_rank=60, w_svpt=80),
        _row('sackmann', 'Felix Auger Aliassime', 'Daniil Medvedev', '20260511', surface='Clay',
             tourney_name='Rome Masters', winner_rank=25, loser_rank=9),
    ]
    tuk_rows = [
        # Dasselbe RG-Match (spätes Matchdatum) -> Duplikat, liefert aber Quoten (vertauschte Reihenfolge)
        _row('tuk', 'Cerundolo J.M.', 'Alcaraz C.', '20260525', '20260605',
             B365W=5.0, B365L=1.2, PSW=5.5, PSL=1.18, AvgW=5.0, AvgL=1.2, MaxW=5.8, MaxL=1.22,
             BFEW=5.9, BFEL=1.21, location='Paris'),
        # Neues Match nach dem Sackmann-Ende
        _row('tuk', 'Alcaraz C.', 'Medvedev D.', '20261001', '20261005', surface='Hard',
             tourney_name='China Open', winner_rank=1, loser_rank=12, B365W=1.3, B365L=3.5,
             location='Beijing'),
        _row('tuk', 'Medvedev D.', 'Gea A.', '20260920', '20260922', surface='Hard',
             tourney_name='Shanghai Masters', winner_rank=11, loser_rank=70, location='Shanghai'),
    ]
    espn_rows = [
        # Gleiches Match wie bei tennis-data (anderes Datum durch Zeitzone) -> Duplikat
        _row('espn', 'Carlos Alcaraz', 'Daniil Medvedev', '20260930', '20261006', tourney_name='China Open'),
        # Laufendes Turnier, gibt es nur bei ESPN; Schreibweise mit Bindestrich und Akzent
        _row('espn', 'Félix Auger-Aliassime', 'Arthur Gea', '20261005', '20261008',
             tourney_name='Rolex Shanghai Masters', location='Shanghai'),
    ]

    rows, stats = build.build(sackmann, tuk_rows, espn_rows)

    assert stats['sackmann'] == 2
    assert stats['tuk'] == 2
    assert stats['espn'] == 1
    assert stats['duplicates'] == 2

    names = {r['winner_name'] for r in rows} | {r['loser_name'] for r in rows}
    # Alle Schreibweisen auf einen Namen pro Spieler
    assert names == {'Carlos Alcaraz', 'Juan Manuel Cerundolo', 'Felix Auger Aliassime', 'Daniil Medvedev', 'Arthur Gea'}

    rg = next(r for r in rows if r['source'] == 'sackmann' and r['winner_name'] == 'Carlos Alcaraz')
    # Quoten getauscht, weil tennis-data das Match andersherum führt
    assert (rg['B365W'], rg['B365L'], rg['PSW'], rg['PSL']) == (1.2, 5.0, 1.18, 5.5)
    assert (rg['BFEW'], rg['BFEL'], rg['MaxW'], rg['MaxL']) == (1.21, 5.9, 1.22, 5.8)

    sh = next(r for r in rows if r['source'] == 'espn')
    assert sh['surface'] == 'Hard'  # über 'shanghai' im Turniernamen
    assert sh['winner_rank'] == 25  # letztes bekanntes Ranking
    assert sh['loser_rank'] == 70

    # chronologisch sortiert
    assert [r['tourney_date'] for r in rows] == sorted(r['tourney_date'] for r in rows)


def test_build_learns_aliases_but_keeps_different_players_apart():
    sackmann = [
        _row('sackmann', 'Yibing Wu', 'Arthur Fils', '20260525', surface='Clay', tourney_name='Roland Garros'),
        _row('sackmann', 'Arthur Fery', 'Novak Djokovic', '20260525', surface='Clay', tourney_name='Roland Garros', match_num=1),
    ]
    espn_rows = [
        # Überlappung mit Sackmann, chinesische Reihenfolge -> Duplikat, Alias Wu Yibing = Yibing Wu
        _row('espn', 'Wu Yibing', 'Arthur Fils', '20260525', '20260527', tourney_name='Roland Garros'),
        # Später, nur bei ESPN: muss auf den Sackmann-Namen umbenannt werden
        _row('espn', 'Wu Yibing', 'Arthur Fery', '20261005', '20261008', tourney_name='Rolex Shanghai Masters'),
    ]
    tuk_rows = [
        # 'Novak D.' (Dennis Novak) darf nicht mit Novak Djokovic verschmelzen
        _row('tuk', 'Novak D.', 'Arthur Fils', '20261001', '20261002', surface='Hard', tourney_name='China Open'),
    ]
    rows, stats = build.build(sackmann, tuk_rows, espn_rows)

    names = {n for r in rows for n in (r['winner_name'], r['loser_name'])}
    assert 'Wu Yibing' not in names and 'Yibing Wu' in names
    # verschiedene Spieler bleiben getrennt
    assert {'Arthur Fils', 'Arthur Fery', 'Novak Djokovic', 'Novak D.'} <= names
    # das RG-Match vom 27.05. liegt nach dem Sackmann-Stichtag -> zählt als Duplikat
    assert stats['duplicates'] == 1 and stats['espn'] == 1 and stats['tuk'] == 1


def test_same_player_rules():
    sp = build._same_player
    assert sp('Aleksandr Shevchenko', 'Alexander Shevchenko')
    assert sp('Coleman Wong', 'Chak Lam Coleman Wong')
    assert sp('Daniel Merida', 'Daniel Merida Aguilar')
    assert sp('SoonWoo Kwon', 'Soon Woo Kwon')
    assert sp('Mpetshi G.', 'Giovanni Mpetshi Perricard')
    assert sp('Zhang Zh.', 'Zhang Zhizhen')
    assert not sp('Arthur Fils', 'Arthur Fery')
    assert not sp('Eduardo Nava', 'Emilio Nava')
    assert not sp('Alexander Zverev', 'Alexander Bublik')
    assert not sp('Daniel Altmaier', 'Adolfo Daniel Vallejo')


def test_clean():
    from collections import Counter
    base = dict(source='sackmann', tourney_id='t1', tourney_date='20240101', round='R32', score='6-4 6-4')
    rows = [
        {**base, 'winner_name': 'A', 'loser_name': 'B', 'surface': 'Clay', 'B365W': 1.5, 'B365L': 2.6, 'PSW': 0.97, 'PSL': 29.0},
        {**base, 'winner_name': 'A', 'loser_name': 'B', 'surface': 'Clay'},          # exaktes Duplikat
        {**base, 'winner_name': 'C', 'loser_name': 'D', 'surface': '', 'AvgW': 1.2}, # Quote ohne Gegenquote
    ]
    stats = Counter()
    out = build.clean(rows, stats)
    assert len(out) == 2
    assert (out[0]['B365W'], out[0]['PSW'], out[0]['PSL']) == (1.5, None, None)
    assert out[1]['surface'] == 'Hard' and out[1]['AvgW'] is None
    assert stats == Counter(clean_duplicates=1, clean_surface_filled=1, clean_odds_removed=2)


def test_manual_alias_and_jr():
    sackmann = [_row('sackmann', 'Tomas Barrios Vera', 'Eric Jr. Olivarez', '20260101', surface='Hard', tourney_name='X')]
    tuk_rows = [_row('tuk', 'Barrios M.', 'Alexander Blockx', '20260901', '20260901', surface='Hard', tourney_name='US Open')]
    rows, _ = build.build(sackmann, tuk_rows, [])
    assert rows[-1]['winner_name'] == 'Tomas Barrios Vera'
    assert build._is_full('Eric Jr. Olivarez') and not build._is_full('Zhang Zh.')


def test_espn_parse_wta():
    rows = espn.parse(SCOREBOARD, tour='wta')
    assert [(r['winner_name'], r['tour']) for r in rows] == [('X', 'wta')]
    assert rows[0]['best_of'] == 3
    assert rows[0]['country'] == 'China PR'


def test_clean_exchange_and_max_odds():
    from collections import Counter
    base = dict(source='tuk', tourney_id='t1', tourney_date='20260301', round='R32', surface='Hard')
    rows = [
        # Exchange fast ohne Marge: plausibel; Max-Kurse knapp unter 100 % (Bestkurse über alle Bücher): plausibel
        {**base, 'winner_name': 'A', 'loser_name': 'B', 'score': '6-4', 'BFEW': 1.52, 'BFEL': 2.9,
         'MaxW': 1.58, 'MaxL': 2.85},
        # Exchange mit 54 % Überrundung (dünner Markt) und Max-Kurse mit Tippfehler: leeren
        {**base, 'winner_name': 'C', 'loser_name': 'D', 'score': '6-3', 'BFEW': 1.1, 'BFEL': 2.0,
         'MaxW': 3.0, 'MaxL': 3.0},
    ]
    stats = Counter()
    out = build.clean(rows, stats)
    assert (out[0]['BFEW'], out[0]['MaxL']) == (1.52, 2.85)
    assert (out[1]['BFEW'], out[1]['BFEL'], out[1]['MaxW'], out[1]['MaxL']) == (None, None, None, None)
    assert stats['clean_odds_removed'] == 2


def test_tuk_load_reads_betfair_and_max(tmp_path, monkeypatch):
    openpyxl = __import__('pytest').importorskip('openpyxl')
    wb = openpyxl.Workbook()
    ws = wb.active
    header = ['ATP', 'Location', 'Tournament', 'Date', 'Series', 'Court', 'Surface', 'Round', 'Best of', 'Winner',
              'Loser', 'WRank', 'LRank', 'W1', 'L1', 'W2', 'L2', 'Comment', 'B365W', 'B365L', 'PSW', 'PSL',
              'MaxW', 'MaxL', 'AvgW', 'AvgL', 'BFEW', 'BFEL']
    ws.append(header)
    from datetime import datetime
    ws.append([1, 'Doha', 'Qatar Open', datetime(2026, 2, 17), 'ATP500', 'Outdoor', 'Hard', '1st Round', 3,
               'Sinner J.', 'Fils A.', 2, 20, 6, 4, 6, 3, 'Completed', 1.2, 4.5, None, None, 1.25, 5.0, 1.2, 4.4,
               1.24, 5.1])
    monkeypatch.setitem(tuk.DIRS, 'atp', str(tmp_path))
    wb.save(tmp_path / '2026.xlsx')
    rows = tuk.load([2026])
    assert len(rows) == 1
    r = rows[0]
    assert (r['BFEW'], r['BFEL'], r['MaxW'], r['MaxL']) == (1.24, 5.1, 1.25, 5.0)
    assert r['PSW'] is None and r['AvgL'] == 4.4


def test_tuk_links_per_tour():
    html = ('<a href="hrjk/2025/2025.xlsx">ATP</a> <a href="hrjk/2025w/2025.xlsx">WTA</a>'
            '<a href="/2014/2014.xls">alt</a>')
    links = tuk.parse_links(html)
    assert links[('atp', 2025)].endswith('/hrjk/2025/2025.xlsx')
    assert links[('wta', 2025)].endswith('/hrjk/2025w/2025.xlsx')
    assert ('atp', 2014) in links
    assert tuk.path(2025, 'wta').endswith(os.path.join('tuk_wta', '2025.xlsx'))  # Windows: '\'


def test_build_keeps_tours_apart():
    """Gleicher Nachname bei Herren und Damen: weder Namen noch Matches oder Quoten werden verknüpft."""
    sackmann = [
        _row('sackmann', 'Mischa Zverev', 'Andy Murray', '20260525', tour='atp', surface='Clay',
             tourney_name='Roland Garros'),
        _row('sackmann', 'Jamie Murray', 'Ana Zverev', '20260525', tour='wta', surface='Clay',
             tourney_name='Roland Garros'),
    ]
    tuk_rows = [
        # ATP-Duplikat mit Quoten -> nur das ATP-Match bekommt sie
        _row('tuk', 'Zverev M.', 'Murray A.', '20260525', '20260527', tour='atp', B365W=1.5, B365L=2.5,
             PSW=1.55, PSL=2.6, AvgW=1.5, AvgL=2.5, court='Outdoor', location='Paris'),
        # neues WTA-Match nach dem Stichtag: 'Murray J.' darf nicht auf die ATP-Seite zeigen
        _row('tuk', 'Murray J.', 'Zverev A.', '20261001', '20261002', tour='wta', surface='Hard',
             tourney_name='China Open', location='Beijing'),
    ]
    rows, stats = build.build(sackmann, tuk_rows, [])

    assert stats['atp_matches'] == 1 and stats['wta_matches'] == 2
    by_tour = {t: [r for r in rows if r['tour'] == t] for t in ('atp', 'wta')}
    atp_names = {n for r in by_tour['atp'] for n in (r['winner_name'], r['loser_name'])}
    wta_names = {n for r in by_tour['wta'] for n in (r['winner_name'], r['loser_name'])}
    assert atp_names == {'Mischa Zverev', 'Andy Murray'}
    assert wta_names == {'Jamie Murray', 'Ana Zverev'}

    atp = by_tour['atp'][0]
    assert (atp['B365W'], atp['PSL']) == (1.5, 2.6)
    # echtes Datum, Indoor/Outdoor und Ort kommen aus dem tennis-data-Duplikat
    assert (atp['match_date'], atp['court'], atp['location']) == ('20260527', 'Outdoor', 'Paris')
    assert atp['tourney_ioc'] == 'FRA'
    wta_rg = next(r for r in by_tour['wta'] if r['source'] == 'sackmann')
    assert wta_rg.get('B365W') in (None, "") and wta_rg['match_date'] == '20260525'


def test_build_attaches_player_metadata():
    sackmann = [_row('sackmann', 'Carlos Alcaraz', 'Jannik Sinner', '20260525', surface='Clay',
                     tourney_name='Roland Garros', winner_id='207989', loser_id='206173', winner_ioc='ESP')]
    espn_rows = [_row('espn', 'Jannik Sinner', 'Carlos Alcaraz', '20260901', '20260905', tourney_name='US Open',
                      country='USA')]
    players = {'atp': [
        {'player_id': '207989', 'name': 'Carlos Alcaraz', 'hand': 'R', 'dob': '20030505', 'ioc': 'ESP'},
        {'player_id': '206173', 'name': 'Jannik Sinner', 'hand': 'R', 'dob': '20010816', 'ioc': 'ITA'},
        # Namensvetter auf der Liste: über die ID trotzdem eindeutig
        {'player_id': '999', 'name': 'Jannik Sinner', 'hand': 'L', 'dob': '19800101', 'ioc': 'AUT'},
    ]}
    rows, _ = build.build(sackmann, [], espn_rows, players)
    us = next(r for r in rows if r['source'] == 'espn')
    assert (us['winner_ioc'], us['winner_dob'], us['loser_ioc']) == ('ITA', '20010816', 'ESP')
    assert us['tourney_ioc'] == 'USA'
    rg = next(r for r in rows if r['source'] == 'sackmann')
    assert rg['tourney_ioc'] == 'FRA' and rg['loser_hand'] == 'R'


def test_tourney_country():
    from src.sources.places import tourney_country
    assert tourney_country('Rolex Shanghai Masters', 'Shanghai', 'China PR') == 'CHN'
    assert tourney_country('Indian Wells Masters') == 'USA'
    assert tourney_country('Monte Carlo Masters') == 'MON'
    assert tourney_country('BNP Paribas Masters', 'Paris') == 'FRA'
    assert tourney_country('Davis Cup WG R1: CZE vs AUS') == ''
