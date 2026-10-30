"""Timing-Backtest: Preis zum Zeitpunkt t, Gebühren/Spread, Walk-forward-Schwellen, Leakage (ohne Netz)."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.evaluation import timing
from src.evaluation.predictors import prepare_matches
from src.evaluation.venues import kalshi_fee
from src.odds import history

START = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)


# ------------------------------------------------------------ Preis zum Zeitpunkt t

def test_price_at_takes_last_point_not_after_t():
    pts = [(START - timedelta(hours=30), 0.40, 0.39, 0.41, None, 5.0),
           (START - timedelta(hours=25), 0.45, 0.44, 0.46, None, 7.0),
           (START - timedelta(hours=23), 0.70, 0.69, 0.71, None, 9.0)]  # nach t = -24 h
    at = history.price_at(pts, START - timedelta(hours=24))
    assert at['price_a'] == 0.45 and at['ts'] == START - timedelta(hours=25)
    assert (at['bid_a'], at['ask_a']) == (0.44, 0.46)
    assert at['volume'] == 12.0  # nur Volumen bis t
    assert history.price_at(pts, START - timedelta(hours=31)) is None  # vorher kein Preis
    # genau auf t zählt noch
    assert history.price_at(pts, START - timedelta(hours=25))['price_a'] == 0.45


def test_timing_points_columns():
    pts = [(START - timedelta(hours=48), 0.30, None, None),
           (START - timedelta(hours=7), 0.35, None, None),
           (START - timedelta(minutes=30), 0.50, None, None),
           (START + timedelta(minutes=30), 0.99, None, None)]  # nach dem Start: nie verwendet
    out = history.timing_points(pts, START)
    assert out['t24h_price_a'] == 0.30
    assert out['t6h_price_a'] == 0.35
    assert out['t1h_price_a'] == 0.35
    assert out['move_price_a'] == 0.35 and out['move_ts'] == START - timedelta(hours=7)  # erster echter Kurs
    assert set(history.TIMING_COLUMNS) == set(out)


# ------------------------------------------------------------ Gebühren und Spread

def _market_frame(**kw):
    base = {'event_id': ['e1'], 'start_time_utc': [pd.Timestamp(START)], 'close_ts': [pd.Timestamp(START - timedelta(minutes=10))],
            'winner': ['a'], 'level': ['challenger'], 'volume': [1000.0], 'close_price_a': [0.55],
            't24h_ts': [pd.Timestamp(START - timedelta(hours=25))], 't24h_price_a': [0.50], 't24h_bid_a': [0.48],
            't24h_ask_a': [0.52], 't24h_volume': [500.0], 'p_elo2': [0.60], 'p_market': [0.56]}
    base.update(kw)
    return pd.DataFrame(base)


def test_kalshi_entry_at_ask_with_fee():
    df = _market_frame()
    mid, buy_a, buy_b = timing.entry_prices(df, '24h', 'kalshi')
    assert mid[0] == 0.50 and buy_a[0] == 0.52 and buy_b[0] == pytest.approx(0.52)  # B = 1 - Bid von A
    net = timing.net_odds(buy_a, timing.venue_profile('kalshi'))
    assert net[0] == pytest.approx(1 / (0.52 + float(kalshi_fee(0.52))))


def test_polymarket_entry_with_spread_and_taker_fee():
    df = _market_frame(t24h_price_a=[0.48])
    _, buy_a, buy_b = timing.entry_prices(df, '24h', 'polymarket', pm_spread=0.04)
    assert buy_a[0] == pytest.approx(0.50) and buy_b[0] == pytest.approx(0.54)
    net = timing.net_odds(buy_a, timing.venue_profile('polymarket', pm_fee=0.01))
    assert net[0] == pytest.approx(1 / 0.50 * 0.99)


def test_polymarket_placeholder_and_first_real_price():
    # 0,5 = leeres Orderbuch: kein Handel. Eröffnung bei PM = erster echter Kurs (move_*)
    df = _market_frame(open_ts=[pd.Timestamp(START - timedelta(days=2))], open_price_a=[0.5],
                       move_ts=[pd.Timestamp(START - timedelta(hours=40))], move_price_a=[0.45])
    assert timing.candidate_bets(df, '24h', 'polymarket', 'elo2').empty  # t24h_price_a = 0,5
    bets = timing.candidate_bets(df, 'open', 'polymarket', 'elo2')
    assert bets['price'].iloc[0] == pytest.approx(0.46)  # 0,45 + halber Spread


def test_candidate_bet_profit_and_clv():
    df = _market_frame()
    bets = timing.candidate_bets(df, '24h', 'kalshi', 'elo2')
    assert len(bets) == 1
    b = bets.iloc[0]
    net = 1 / (0.52 + float(kalshi_fee(0.52)))
    assert b['side'] == 'a' and b['won']
    assert b['ev'] == pytest.approx(0.60 * net - 1)
    assert b['profit'] == pytest.approx(net - 1)
    assert b['clv'] == pytest.approx(0.55 / 0.52 - 1)  # gegen Closing der Venue
    assert b['clv_sharp'] == pytest.approx(0.56 / 0.52 - 1)  # gegen Pinnacle/Betfair ohne Marge


def test_no_bet_without_orderbook_or_extreme_or_stale():
    assert timing.candidate_bets(_market_frame(t24h_ask_a=[np.nan], t24h_bid_a=[np.nan]), '24h', 'kalshi', 'elo2').empty
    assert timing.candidate_bets(_market_frame(t24h_price_a=[0.99]), '24h', 'polymarket', 'elo2').empty
    stale = _market_frame(close_ts=[pd.Timestamp(START - timedelta(hours=5))])  # Startzeit verschoben
    assert timing.candidate_bets(stale, '24h', 'polymarket', 'elo2').empty
    low_volume = _market_frame(t24h_volume=[10.0])
    assert timing.candidate_bets(low_volume, '24h', 'kalshi', 'elo2', min_volume=100).empty


# ------------------------------------------------------------ Schwellen und Leakage

def _cands(periods):
    rows = []
    for i, (per, ev, profit) in enumerate(periods):
        rows.append({'period': pd.Period(per, 'Q'), 'ev': ev, 'profit': profit,
                     'start_time_utc': pd.Period(per, 'Q').start_time.tz_localize('UTC') + pd.Timedelta(days=i % 5)})
    return pd.DataFrame(rows)


def test_threshold_chosen_only_from_past():
    # 2025Q1: hohe Edges gewinnen, niedrige verlieren -> 2025Q2 nutzt eine hohe Schwelle
    c = _cands([('2025Q1', 0.01, -1.0), ('2025Q1', 0.01, -1.0), ('2025Q1', 0.10, 1.0),
                ('2025Q2', 0.01, 5.0), ('2025Q2', 0.10, -1.0)])
    bets = timing.walk_forward_bets(c)
    q1 = bets[bets['period'] == pd.Period('2025Q1', 'Q')]
    q2 = bets[bets['period'] == pd.Period('2025Q2', 'Q')]
    assert (q1['threshold'] == timing.DEFAULT_THRESHOLD).all()  # ohne Vergangenheit
    # aus Q1 (kleinste Schwelle mit dem höchsten Gewinn), der Gewinn in Q2 (ev 0.01) spielt keine Rolle
    assert (q2['threshold'] == 0.02).all()
    assert list(q2['ev']) == [0.10]


def test_holm_adjustment():
    adj = timing.holm([0.01, 0.04, np.nan, 0.03])
    assert adj[0] == pytest.approx(0.03) and adj[3] == pytest.approx(0.06) and adj[1] == pytest.approx(0.06)
    assert np.isnan(adj[2])


def test_model_probability_uses_only_past():
    """p (elo2) eines Matches ändert sich nicht, wenn ein späteres Ergebnis anders ausgeht."""
    rows = []
    for d in range(1, 29):
        w, l = ('Ann Alpha', 'Bea Beta') if d % 3 else ('Bea Beta', 'Ann Alpha')
        rows.append({'tour': 'atp', 'level': 'challenger', 'tourney_id': f'T{d}', 'tourney_name': 'X',
                     'surface': 'Hard', 'tourney_date': f'202502{d:02d}', 'match_date': f'202502{d:02d}',
                     'match_num': d, 'round': 'R32', 'winner_name': w, 'loser_name': l, 'score': '6-1 6-1',
                     'status': 'completed', 'winner_rank': 100, 'loser_rank': 120})
    df = pd.DataFrame(rows)
    flipped = df.copy()
    last = flipped.index[-1]
    flipped.loc[last, ['winner_name', 'loser_name']] = flipped.loc[last, ['loser_name', 'winner_name']].to_numpy()

    def p_ann(frame):
        m = prepare_matches(frame)
        m = m[m['date'] < pd.Timestamp('2025-02-28')]
        return np.where(m['player_a'] == 'Ann Alpha', m['elo2_prob_a'], 1 - m['elo2_prob_a'])
    assert np.allclose(p_ann(df), p_ann(flipped))


def test_link_predictions_orients_to_market():
    preds = pd.DataFrame({'match_id': ['m1'], 'date': [pd.Timestamp('2026-10-20')], 'player_a': ['Bea Beta'],
                          'player_b': ['Ann Alpha'], 'p_elo2': [0.3], 'p_market': [0.35], 'market_source': ['ps'],
                          'level': ['challenger']})
    hist = pd.DataFrame({'event_id': ['e1'], 'player_a': ['Ann Alpha'], 'player_b': ['Bea Beta'],
                         'start_time_utc': [pd.Timestamp(START)], 'level': [None]})
    out = timing.link_predictions(hist, preds)
    assert out['match_id'].iloc[0] == 'm1' and bool(out['flip'].iloc[0])
    assert out['p_elo2'].iloc[0] == pytest.approx(0.7) and out['p_market'].iloc[0] == pytest.approx(0.65)
    assert out['level'].iloc[0] == 'challenger'  # Polymarket ohne Level: aus den Matchdaten


def test_run_segments_end_to_end():
    rng = np.random.default_rng(0)
    n = 400
    start = pd.date_range('2025-01-01', periods=n, freq='D', tz='UTC')
    true_p = rng.uniform(0.2, 0.8, n)
    winner = np.where(rng.random(n) < true_p, 'a', 'b')
    df = pd.DataFrame({
        'event_id': [f'e{i}' for i in range(n)], 'match_id': [f'm{i}' for i in range(n)], 'start_time_utc': start,
        'close_ts': start - pd.Timedelta(minutes=5), 'winner': winner, 'level': 'itf', 'volume': 1000.0,
        'close_price_a': true_p, 'open_ts': start - pd.Timedelta(days=2), 'open_price_a': 0.5,
        'move_ts': start - pd.Timedelta(days=1), 'move_price_a': 0.48,
        'open_bid_a': np.nan, 'open_ask_a': np.nan, 'p_elo2': true_p, 'p_market': np.nan,
    })
    table, bets, weights = timing.run_segments({'polymarket': df}, ['polymarket'], ['itf'], ['open'], ['elo2'],
                                               n_boot=200, with_blend=True)
    row = table[(table['level'] == 'itf')].iloc[0]
    # Wahres p gegen einen Eröffnungspreis von 0,5: klarer Vorteil, Log Loss besser als der Preis
    assert row['n_bets'] > 50 and row['yield'] > 0 and row['ll_diff'] < 0
    assert set(table['decision']) <= {'GO', 'Kandidat', 'NO-GO', 'zu wenig', 'nicht handelbar'}
    assert 0 < row['max_period_share'] <= 1 and row['n_periods_pos'] <= row['n_periods']
    w_price, w_model, n_fit = weights['polymarket']['open']['elo2']
    assert w_model > w_price and n_fit == 400  # Eröffnungspreis konstant 0,48: nur das Modell trägt Information


def _row(**kw):
    base = {'venue': 'kalshi', 'level': 'itf', 'point': '6h', 'model': 'elo2', 'n_bets': 500, 'p_yield': 1e-6,
            'yield_ci_low': 0.05, 'clv': 0.03, 'yield': 0.1, 'n_periods': 3, 'n_periods_pos': 3, 'max_period_share': 0.4}
    return {**base, **kw}


def test_decision_rules():
    t = pd.DataFrame([_row(), _row(level='challenger', n_periods=1, n_periods_pos=1),
                      _row(level='main', venue='polymarket', point='open'), _row(level='all'),
                      _row(level='main', point='1h', n_bets=10), _row(level='itf', point='1h', clv=-0.01)])
    d = timing.add_decision(t)['decision'].tolist()
    assert d == ['GO', 'Kandidat', 'nicht handelbar', 'zu wenig', 'zu wenig', 'Kandidat']


def _bets(quarters):
    """Wetten zur Quote 2,0 (Gewinn +1 bzw. -1), pro Quartal (Quartal, Anzahl, Trefferquote)."""
    rows = []
    for per, n, hit in quarters:
        wins = round(n * hit)
        rows += [{'period': pd.Period(per, 'Q'), 'won': i < wins, 'profit': 1.0 if i < wins else -1.0, 'price': 0.5,
                  'clv': 0.03, 'clv_sharp': np.nan} for i in range(n)]
    return pd.DataFrame(rows)


def _decide(bets):
    row = {'venue': 'polymarket', 'level': 'itf', 'point': '1h', 'model': 'blend_gbdt',
           **timing.summarize_bets(bets, n_boot=500)}
    return timing.add_decision(pd.DataFrame([row])).iloc[0]


def test_no_go_when_one_quarter_carries_the_bets():
    # Wie Polymarket ITF -1 h, blend_gbdt im Lauf vom 27.10.2026: 47 Wetten im Q3, 1.618 im Oktober. Zwei Quartale
    # mit je >= 30 Wetten (alte Regel: GO), aber 97 % der Wetten in einem Quartal
    r = _decide(_bets([('2026Q3', 47, 0.8), ('2026Q4', 1618, 0.6)]))
    assert r['decision'] == 'Kandidat'
    assert r['p_holm'] < 0.05 and r['yield_ci_low'] > 0 and r['n_periods'] == 2
    assert r['max_period_share'] == pytest.approx(1618 / 1665)


def test_go_needs_profit_confirmed_in_two_quarters():
    # Gleich viele Wetten in zwei Quartalen, der Gewinn steckt aber nur im ersten (zweites Quartal Yield 0)
    r = _decide(_bets([('2026Q2', 500, 0.65), ('2026Q3', 500, 0.5)]))
    assert r['decision'] == 'Kandidat'
    assert r['p_holm'] < 0.05 and r['yield_ci_low'] > 0
    assert (r['n_periods'], r['n_periods_pos'], r['max_period_share']) == (2, 1, 0.5)
    # Gegenprobe: in beiden Quartalen einzeln bestätigt -> GO
    r = _decide(_bets([('2026Q2', 500, 0.62), ('2026Q3', 500, 0.62)]))
    assert r['decision'] == 'GO' and r['n_periods_pos'] == 2
