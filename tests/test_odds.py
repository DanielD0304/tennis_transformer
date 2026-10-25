"""
Tests für src/odds (ohne Netzwerk).

Die Fixtures in tests/fixtures/odds/ sind echte, gekürzte Antworten von Polymarket (Gamma/CLOB) und
Kalshi vom 25.10.2026. Ausnahme: odds_api_odds_doku_format.json folgt dem dokumentierten Format der
Odds API v4, weil in der Cloud-Session kein API-Key vorhanden war.
"""
import json
import math
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.odds import collect, history, kalshi, matching, odds_api, polymarket, report, softbooks
from src.odds.schema import (Snapshot, append_snapshots, load_snapshots, mid, odds_to_price, price_to_odds,
                             to_decimal, utc, validate)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "odds")
TS = datetime(2026, 10, 25, 14, 30, tzinfo=timezone.utc)


def fixture(name):
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as f:
        return json.load(f)


def snap(**kw):
    base = dict(ts_utc=TS, venue='polymarket', event_id='1', player_a='Daniil Medvedev', player_b='Jannik Sinner',
                price_a=0.4, price_b=0.6, price_type='probability', tour='atp')
    return Snapshot(**{**base, **kw})


# ------------------------------------------------------------ Preis <-> Quote

def test_price_odds_conversion():
    assert price_to_odds(0.5) == 2.0
    assert price_to_odds(0.25) == 4.0
    assert price_to_odds(0) is None and price_to_odds(None) is None and price_to_odds(1.5) is None
    assert odds_to_price(2.0) == 0.5
    assert odds_to_price(1.0) is None
    assert to_decimal(1.85, 'decimal') == 1.85 and to_decimal(0.8, 'probability') == 1.25
    assert mid(0.62, 0.63) == pytest.approx(0.625)
    assert mid(0.0, 1.0) is None and mid(None, 0.5) is None


def test_snapshot_derives_odds():
    s = snap(price_a=0.8, price_b=0.25)
    assert s.odds_a == pytest.approx(1.25) and s.odds_b == pytest.approx(4.0)
    s = snap(price_a=1.6, price_b=2.45, price_type='decimal')
    assert (s.odds_a, s.odds_b) == (1.6, 2.45)


def test_utc_parsing():
    assert utc("2026-10-25 13:00:00+00") == datetime(2026, 10, 25, 13, tzinfo=timezone.utc)
    assert utc("2026-10-25T13:00:00Z") == datetime(2026, 10, 25, 13, tzinfo=timezone.utc)
    assert utc(0) == datetime(1970, 1, 1, tzinfo=timezone.utc)
    assert utc(None) is None


# ------------------------------------------------------------- Validierung

@pytest.mark.parametrize("change", [
    {'ts_utc': datetime(2026, 10, 25)},  # ohne Zeitzone
    {'player_b': 'Daniil Medvedev'},
    {'player_a': ''},
    {'price_a': 1.4},  # Wahrscheinlichkeit > 1
    {'price_type': 'decimal', 'price_a': 0.9, 'price_b': 1.9},  # Quote <= 1
    {'price_type': 'american'},
    {'tour': 'itf'},
    {'price_a': None, 'price_b': None},
])
def test_validate_rejects(change):
    with pytest.raises(ValueError):
        validate(snap(**change))


def test_validate_accepts_missing_side():
    assert validate(snap(price_b=None)).price_a == 0.4


# --------------------------------------------------------- Append / Laden

def test_append_and_load(tmp_path):
    root = str(tmp_path)
    day2 = TS + timedelta(days=1)
    assert append_snapshots([snap(), snap(event_id='2', venue='kalshi')], root) == 2
    assert append_snapshots([snap(ts_utc=TS + timedelta(minutes=20)), snap(ts_utc=day2)], root) == 2
    assert sorted(os.listdir(root)) == ['2026-10-25.parquet', '2026-10-26.parquet']

    df = load_snapshots(root=root)
    assert len(df) == 4 and df['ts_utc'].is_monotonic_increasing
    assert str(df['ts_utc'].dt.tz) == 'UTC'
    assert df['odds_a'].iloc[0] == pytest.approx(2.5)
    assert len(load_snapshots('2026-10-25', '2026-10-25', root)) == 3
    assert len(load_snapshots(start='2026-10-26', root=root)) == 1
    assert list(load_snapshots(root=root, venues=['kalshi'])['event_id']) == ['2']
    assert load_snapshots(root=str(tmp_path / 'leer')).empty


def test_append_rejects_invalid(tmp_path):
    with pytest.raises(ValueError):
        append_snapshots([snap(price_a=2.0)], str(tmp_path))
    assert append_snapshots([], str(tmp_path)) == 0


# ---------------------------------------------------------------- Polymarket

def test_polymarket_parse_open_events():
    events = fixture('polymarket_events_open.json')
    snaps = polymarket.parse_events(events, TS, fixture('polymarket_midpoints.json'))
    # zwei offene Sieger-Märkte; der Zusatzmarkt 'Completed Match' und das geschlossene Match fallen weg
    assert len(snaps) == 2
    s = snaps[0]
    assert (s.player_a, s.player_b) == ('Leylah Fernandez', 'Himeno Sakatsume')
    assert s.tour == 'wta' and s.tourney == 'Pan Pacific Open' and s.price_type == 'probability'
    assert s.price_a == pytest.approx(0.795) and s.odds_a == pytest.approx(1 / 0.795)
    assert s.best_bid_b == pytest.approx(1 - s.best_ask_a)
    assert s.start_time_utc == datetime(2026, 10, 27, 2, tzinfo=timezone.utc)
    assert s.source_url.startswith("https://polymarket.com/event/")
    for s in snaps:
        validate(s)


def test_polymarket_without_midpoints_uses_gamma_prices():
    snaps = polymarket.parse_events(fixture('polymarket_events_open.json'), TS)
    assert snaps[0].price_a == pytest.approx(mid(snaps[0].best_bid_a, snaps[0].best_ask_a))
    assert snaps[0].price_a + snaps[0].price_b == pytest.approx(1)


def test_polymarket_closed_markets():
    markets = fixture('polymarket_markets_closed.json')['markets']
    events = [{**m['events'][0], 'markets': [m]} for m in markets]
    infos = [polymarket.match_info(e) for e in events]
    by_q = {m['question']: i for m, i in zip(markets, infos)}
    assert by_q['Hangzhou Open: Francisco Cabral / Lucas Miedler vs Ariel Behar / Joran Vliegen'] is None  # Doppel
    cazaux = by_q['Hangzhou Open: Arthur Cazaux vs Matteo Arnaldi']
    # alte Märkte haben nur Nachnamen als Outcomes, die vollen Namen kommen aus dem Titel
    assert (cazaux['player_a'], cazaux['player_b']) == ('Arthur Cazaux', 'Matteo Arnaldi')
    assert cazaux['winner'] == 'a' and cazaux['tour'] == 'atp' and cazaux['closed']
    assert by_q['Hangzhou Open: Adam Walton vs Sebastian Korda']['winner'] == 'b'


def test_polymarket_winner_void():
    market = {'closed': True, 'outcomePrices': '["0.5", "0.5"]'}
    assert polymarket.winner_of(market) is None
    assert polymarket.winner_of({'closed': False, 'outcomePrices': '["1", "0"]'}) is None


def test_polymarket_skips_futures():
    event = {'title': 'US Open Winner (M)', 'markets': [
        {'id': '1', 'question': 'Will Sinner win the US Open?', 'outcomes': '["Yes", "No"]'}]}
    assert polymarket.match_market(event) is None


# -------------------------------------------------------------------- Kalshi

def test_kalshi_parse_open_event():
    event = fixture('kalshi_events_open.json')['events'][0]
    milestone = fixture('kalshi_milestones.json')['milestones'][0]
    s = kalshi.parse_event(event, TS, milestone)
    assert (s.player_a, s.player_b) == ('Matteo Berrettini', 'Tomas Martin Etcheverry')
    assert s.price_a == pytest.approx(0.625) and s.price_b == pytest.approx(0.375)
    assert (s.best_bid_a, s.best_ask_a) == (0.62, 0.63)
    assert s.tour == 'atp' and s.level == 'main' and s.tourney == 'ATP Vienna'
    assert s.start_time_utc == datetime(2026, 10, 26, 12, tzinfo=timezone.utc)
    assert s.volume == pytest.approx(260.05 + 402.13)
    validate(s)


def test_kalshi_order_follows_first_competitor():
    event = fixture('kalshi_events_open.json')['events'][0]
    milestone = fixture('kalshi_milestones.json')['milestones'][0]
    swapped = {**event, 'markets': event['markets'][::-1]}
    assert kalshi.parse_event(swapped, TS, milestone).player_a == 'Matteo Berrettini'
    # ohne Milestone: Reihenfolge wie geliefert, Tour aus der Serie
    s = kalshi.parse_event(swapped, TS)
    assert s.player_a == 'Tomas Martin Etcheverry' and s.tour == 'atp' and s.start_time_utc is None


def test_kalshi_tour_level():
    assert kalshi.tour_level({'details': {'tour': 'WTA Challenger'}}) == ('wta', 'challenger')
    assert kalshi.tour_level({'details': {'tour': 'ITF', 'gender': 'women'}}) == ('wta', 'itf')
    assert kalshi.tour_level({'details': {'tour': 'Davis Cup'}}) == (None, None)


def test_kalshi_winner():
    markets = fixture('kalshi_markets_historical.json')['markets']
    assert kalshi.winner_of(markets[0]) == 'b'  # Van Assche verliert gegen Kovacevic
    assert kalshi.winner_of(markets[1]) == 'a'
    assert kalshi.winner_of({'result': 'scalar'}) is None


# ------------------------------------------------------------------ Odds API

def test_odds_api_parse():
    snaps = odds_api.parse_odds(fixture('odds_api_odds_doku_format.json'), TS)
    venues = [(s.venue, s.event_id[:4]) for s in snaps]
    # Betfair EU gewinnt gegen UK, Unibet hat nur Handicap -> kein Snapshot
    assert venues == [('betfair', 'e912'), ('pinnacle', 'e912'), ('bet365', 'e912'), ('pinnacle', 'c2a7')]
    bf = snaps[0]
    assert (bf.price_a, bf.price_b, bf.price_type) == (1.64, 2.58, 'decimal')
    assert bf.tour == 'atp' and bf.start_time_utc == datetime(2026, 10, 26, 12, tzinfo=timezone.utc)
    # Outcome-Reihenfolge egal, Zuordnung über den Namen
    wta = snaps[-1]
    assert wta.tour == 'wta' and (wta.player_a, wta.price_a, wta.price_b) == ('Panna Udvardy', 2.55, 1.55)
    for s in snaps:
        validate(s)


def test_odds_api_without_key_does_not_crash(monkeypatch):
    monkeypatch.delenv('ODDS_API_KEY', raising=False)
    messages = []

    class NoNetwork:
        def get_json(self, *a, **k):
            raise AssertionError("ohne Key darf nichts abgefragt werden")

    assert odds_api.fetch_current(client=NoNetwork(), log=messages.append) == []
    assert 'ODDS_API_KEY' in messages[0]


# ----------------------------------------------------------------- Soft-Books

def test_softbooks_parse():
    events = [(['Tabilo, A.', 'Carreno-Busta, P.'], [1.85, 1.95], 'tipico'),
              (['D. Medvedev RUS', 'J.-L. Struff GER'], [1.3, 3.4], 'bwin'),
              (['Krawietz / Puetz', 'Ram / Salisbury'], [2.0, 1.8], 'bwin')]
    snaps = softbooks.parse_events(events, TS)
    assert [s.venue for s in snaps] == ['tipico', 'bwin']
    assert snaps[0].event_id == 'tipico:tabilo-a-vs-carreno-busta-p'
    assert snaps[1].odds_b == 3.4 and snaps[1].price_type == 'decimal'


# ----------------------------------------------------------- Closing-Ableitung

def test_opening_closing_uses_last_price_before_start():
    start = datetime(2026, 10, 25, 12, tzinfo=timezone.utc)
    pts = [(start - timedelta(hours=5), 0.40, None, None), (start - timedelta(minutes=10), 0.55, 0.54, 0.56),
           (start, 0.57, None, None), (start + timedelta(minutes=10), 0.90, None, None),
           (start - timedelta(hours=2), 0.45, None, None), (start - timedelta(hours=1), None, None, None)]
    oc = history.opening_closing(pts, start)
    assert oc['open_price_a'] == 0.40 and oc['open_ts'] == start - timedelta(hours=5)
    assert oc['close_price_a'] == 0.57 and oc['close_ts'] == start  # genau zum Start zählt noch
    assert oc['close_1h_price_a'] == 0.45  # letzter Preis mindestens 1 h vorher
    assert oc['n_points'] == 4
    assert history.opening_closing([(start + timedelta(minutes=1), 0.5, None, None)], start) is None


def test_polymarket_history_row():
    h = fixture('polymarket_prices_history.json')
    market = next(m for m in fixture('polymarket_markets_closed.json')['markets'] if m['id'] == h['_market'])
    info = polymarket.match_info({**market['events'][0], 'markets': [market]})
    row = history.polymarket_row(info, h, TS)
    start = utc(h['_gameStartTime'])
    last_before = max((p for p in h['history'] if p['t'] <= start.timestamp()), key=lambda p: p['t'])
    assert row['close_price_a'] == last_before['p'] and row['close_ts'] <= start
    assert row['winner'] == 'a' and row['venue'] == 'polymarket'


@pytest.mark.parametrize("name", ['kalshi_candles_historical.json', 'kalshi_candles_live.json'])
def test_kalshi_candles_closing(name):
    c = fixture(name)
    start = utc(c['_start'])
    points = history.kalshi_points(c)
    assert all(p[1] is None or 0 <= p[1] <= 1 for p in points)
    oc = history.opening_closing([p[:4] for p in points], start)
    last = max((p for p in points if p[0] <= start), key=lambda p: p[0])
    assert oc['close_ts'] == last[0] <= start
    assert oc['close_price_a'] == pytest.approx(mid(oc['close_bid_a'], oc['close_ask_a']))


def test_kalshi_history_row():
    a, b = fixture('kalshi_markets_historical.json')['markets']
    candles = fixture('kalshi_candles_historical.json')
    milestone = {'start_date': candles['_start'], 'details': {'tour': 'ATP', 'tournament_name': 'US Open'}}
    row = history.kalshi_row(a['event_ticker'], milestone, a, b, candles, TS)
    assert row['player_a'] == 'Luca Van Assche' and row['winner'] == 'b'
    assert row['level'] == 'main' and row['close_ts'] <= utc(candles['_start'])
    assert row['volume'] == pytest.approx(1467993.09 + 639984.01)


def test_save_history_dedupes(tmp_path):
    row = {c: None for c in history.COLUMNS}
    row.update(venue='kalshi', event_id='E1', start_time_utc=TS, close_price_a=0.4, n_points=3, winner='a')
    history.save_history('kalshi', [row], str(tmp_path))
    df = history.save_history('kalshi', [{**row, 'close_price_a': 0.5}], str(tmp_path))
    assert len(df) == 1 and df['close_price_a'].iloc[0] == 0.5


# ------------------------------------------------------------------- Matching

MATCHES = [
    {'source': 'sackmann', 'tourney_date': '20261019', 'match_date': '', 'tourney_name': 'Vienna',
     'winner_name': 'Matteo Berrettini', 'loser_name': 'Tomas Martin Etcheverry', 'status': 'completed'},
    {'source': 'espn', 'tourney_date': '20261019', 'match_date': '20261025', 'tourney_name': 'Basel',
     'winner_name': 'Arthur Rinderknech', 'loser_name': 'Jerome Kym', 'status': 'completed',
     'PSW': '1.50', 'PSL': '2.70'},
    {'source': 'espn', 'tourney_date': '20261019', 'match_date': '20261025', 'tourney_name': 'Basel',
     'winner_name': 'Felix Auger-Aliassime', 'loser_name': 'Juan Manuel Cerundolo', 'status': 'RET'},
]


def test_link_matches_name_variants_and_swap():
    df = pd.DataFrame([
        # Kalshi-Schreibweise, gleiche Reihenfolge
        {'venue': 'kalshi', 'event_id': 'K1', 'player_a': 'Matteo Berrettini', 'player_b': 'Tomas Martin Etcheverry',
         'start_time_utc': utc('2026-10-21T12:00:00Z')},
        # vertauschte Seiten, Soft-Book-Schreibweise ohne Startzeit -> ts_utc
        {'venue': 'tipico', 'event_id': 'T1', 'player_a': 'Kym, J.', 'player_b': 'Rinderknech, A.',
         'start_time_utc': None, 'ts_utc': utc('2026-10-25T09:00:00Z')},
        # Kurzform mit Bindestrich und Doppelnachname
        {'venue': 'bwin', 'event_id': 'B1', 'player_a': 'F. Auger-Aliassime CAN', 'player_b': 'J.M. Cerundolo ARG',
         'start_time_utc': utc('2026-10-24T18:00:00Z')},
        # anderer Tag (außerhalb des Fensters) und unbekannter Spieler
        {'venue': 'kalshi', 'event_id': 'K2', 'player_a': 'Arthur Rinderknech', 'player_b': 'Jerome Kym',
         'start_time_utc': utc('2026-10-10T12:00:00Z')},
        {'venue': 'kalshi', 'event_id': 'K3', 'player_a': 'Nobody Known', 'player_b': 'Jerome Kym',
         'start_time_utc': utc('2026-10-25T12:00:00Z')},
    ])
    linked = matching.link_matches(df, MATCHES)
    assert list(linked['match_found']) == [True, True, True, False, False]
    assert list(linked['a_won'][:3]) == [True, False, True]
    assert linked['match_status'].iloc[2] == 'RET'
    # Pinnacle-Quote aus Sicht von A: Tipico hat Kym (Verlierer) als A
    assert (linked['ps_a'].iloc[1], linked['ps_b'].iloc[1]) == (2.70, 1.50)
    assert pd.isna(linked['ps_a'].iloc[0])

    rate = matching.match_rate(df, MATCHES).set_index('venue')
    assert rate.loc['kalshi', 'n'] == 3 and rate.loc['kalshi', 'matched'] == 1
    assert rate.loc['bwin', 'rate'] == 1.0


def test_pair_venues_across_markets():
    pm = pd.DataFrame([{'event_id': 'P1', 'player_a': 'Jerome Kym', 'player_b': 'Arthur Rinderknech',
                        'start_time_utc': utc('2026-10-25T11:05:00Z')},
                       {'event_id': 'P2', 'player_a': 'Leylah Fernandez', 'player_b': 'Himeno Sakatsume',
                        'start_time_utc': utc('2026-10-27T02:00:00Z')}])
    ks = pd.DataFrame([{'event_id': 'K1', 'player_a': 'Arthur Rinderknech', 'player_b': 'Jerome Kym',
                        'start_time_utc': utc('2026-10-25T11:00:00Z')}])
    pairs = matching.pair_venues(pm, ks)
    assert pairs.to_dict('records') == [{'left_event_id': 'P1', 'right_event_id': 'K1', 'same_order': False}]


# ------------------------------------------------------------ Collector / Report

def test_collect_once_isolates_failures(tmp_path):
    def ok(log=print):
        return [snap(), snap(event_id='2', price_a=7.0)]  # der zweite ist ungültig und wird verworfen

    def broken(log=print):
        raise RuntimeError("Venue down")

    result = collect.collect_once(['ok', 'broken'], str(tmp_path), adapters={'ok': ok, 'broken': broken},
                                  log=lambda *_: None)
    assert result['ok'] == 1 and result['broken'].startswith('Fehler: RuntimeError')
    assert len(load_snapshots(root=str(tmp_path))) == 1


def test_report_on_small_history(tmp_path):
    rows = []
    for i in range(40):
        p = 0.2 + 0.6 * (i % 10) / 9
        tag = chr(97 + i // 26) + chr(97 + i % 26)  # Ziffern zählen im Namensvergleich nicht
        for venue in ('polymarket', 'kalshi'):
            rows.append({c: None for c in history.COLUMNS} | {
                'venue': venue, 'event_id': f'{venue}{i}', 'player_a': f'Anton Alpha{tag}',
                'player_b': f'Bert Beta{tag}', 'start_time_utc': TS + timedelta(days=i), 'tour': 'atp',
                'close_price_a': p, 'open_price_a': 0.5, 'close_1h_price_a': p, 'volume': 100.0,
                'n_points': 5, 'winner': 'a' if i % 3 else 'b'})
    for venue in ('polymarket', 'kalshi'):
        history.save_history(venue, [r for r in rows if r['venue'] == venue], str(tmp_path))
    out = report.report(history_root=str(tmp_path), snapshot_root=str(tmp_path / 's'), matches=[],
                        log=lambda *_: None)
    closing = out['polymarket'].set_index(['gruppe', 'preis']).loc[('polymarket gesamt', 'closing')]
    assert closing['n'] == 40 and 0 < closing['log_loss'] < 2
    assert out['head_to_head'].attrs['ergebnis_gleich'] == 1.0


def test_vs_pinnacle():
    rows = []
    for i, (price, winner) in enumerate([(0.30, 'b'), (0.40, 'a'), (0.35, 'b')]):
        rows.append({c: None for c in history.COLUMNS} | {
            'venue': 'kalshi', 'event_id': f'K{i}', 'player_a': 'Jerome Kym', 'player_b': 'Arthur Rinderknech',
            'start_time_utc': utc('2026-10-25T11:00:00Z'), 'close_price_a': price, 'close_1h_price_a': price,
            'close_bid_a': price - 0.01, 'close_ask_a': price + 0.01, 'winner': winner})
    df = pd.DataFrame(rows)
    df['close_bid_a'] = df['close_bid_a'].astype(float)
    df['close_ask_a'] = df['close_ask_a'].astype(float)
    r = report.vs_pinnacle(df, 'kalshi', MATCHES)
    # Zeile K1 (Kym gewinnt laut Markt, laut matches.csv aber Rinderknech) fällt raus
    assert r['n'] == 2
    p_pin = (1 / 2.7) / (1 / 2.7 + 1 / 1.5)
    assert r['mittel_abs_dp'] == pytest.approx((abs(0.30 - p_pin) + abs(0.35 - p_pin)) / 2)
    assert r['ll_pinnacle'] == pytest.approx(-math.log(1 - p_pin))
    # Pinnacle fair für Kym: 0.357. K0: Ask 0.31 + Fee 0.02 = 0.33 -> Edge +8 %, Kym verliert -> -100 %.
    # K2: Ask 0.36 + 0.02 = 0.38 zu teuer, Rinderknech 1 - 0.34 + 0.02 = 0.68 > 0.643 auch -> keine Wette
    assert r['wetten'] == 1 and r['yield_flat'] == -1.0
    assert r['edge_mittel'] == pytest.approx(p_pin / 0.33 - 1)
    assert report.vs_pinnacle(df, 'kalshi', []) is None
