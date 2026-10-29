"""Sharp vs. Rest auf Snapshots, Kelly-Deckel und Paper-Trading (Signal -> Wette -> Abrechnung -> CLV), ohne Netz."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.evaluation import sharp
from src.evaluation.betting import Kelly
from src.odds.schema import Snapshot, to_frame
from src.paper import report, settle, signals
from src.paper.store import add_bets, load_bets, save_bets

NOW = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)
START = NOW + timedelta(hours=5)


def snap(venue, ts, a='Jannik Sinner', b='Carlos Alcaraz', price_a=None, price_b=None, event_id=None, **kw):
    ptype = 'probability' if venue in ('polymarket', 'kalshi') else 'decimal'
    return Snapshot(ts_utc=ts, venue=venue, event_id=event_id or f'{venue}-1', player_a=a, player_b=b,
                    price_a=price_a, price_b=price_b, price_type=ptype, tour='atp', start_time_utc=START, **kw)


def frame(*snaps):
    return to_frame(list(snaps))


# ------------------------------------------------------------- Sharp vs. Rest

def test_link_events_across_venues_and_flip():
    df = frame(snap('pinnacle', NOW, price_a=1.8, price_b=2.1),
               snap('kalshi', NOW, a='Carlos Alcaraz', b='Jannik Sinner', price_a=0.45, price_b=0.55))
    linked = sharp.link_events(df)
    assert linked['match_key'].nunique() == 1
    assert list(linked.sort_values('venue')['flip']) == [True, False]  # kalshi vertauscht, pinnacle Referenz


def test_align_within_five_minutes_only():
    df = frame(snap('pinnacle', NOW, price_a=1.8, price_b=2.1),
               snap('tipico', NOW + timedelta(minutes=4), price_a=1.95, price_b=1.9),
               snap('bwin', NOW + timedelta(minutes=6), price_a=2.2, price_b=1.7))
    aligned = sharp.align(df)
    assert list(aligned['venue']) == ['tipico']  # bwin liegt 6 Minuten daneben


def test_line_shopping_net_of_tax_and_fees():
    p_fair = (1 / 1.8) / (1 / 1.8 + 1 / 2.1)
    df = frame(snap('pinnacle', NOW, price_a=1.8, price_b=2.1),
               snap('tipico', NOW, price_a=2.00, price_b=1.80),  # 2,00 brutto, nach 5,3 % Steuer 1,894
               snap('polymarket', NOW, price_a=0.50, price_b=0.50, best_ask_a=0.51, best_ask_b=0.51))  # 1,96
    sig = sharp.signals(df, threshold=0.0)
    a = sig[sig['side'] == 'a'].iloc[0]
    assert a['venue'] == 'polymarket'
    assert a['net_odds'] == pytest.approx(1 / 0.51)
    assert a['edge'] == pytest.approx(p_fair / 0.51 - 1)
    # Tipico allein: Steuer frisst den Vorteil
    only_tipico = sharp.signals(frame(snap('pinnacle', NOW, price_a=1.8, price_b=2.1),
                                      snap('tipico', NOW, price_a=2.00, price_b=1.80)), threshold=0.0)
    assert only_tipico.empty or only_tipico['edge'].max() < p_fair * 2.0 - 1


def test_buy_odds_ignore_ask_below_mid():
    """Polymarket: Mitte live aus dem CLOB, Bid/Ask aus Gamma (teils veraltet). Ein Ask unter der Mitte zählt nicht."""
    df = frame(snap('polymarket', NOW, price_a=0.625, price_b=0.375, best_bid_a=0.72, best_ask_a=0.73,
                    best_bid_b=0.27, best_ask_b=0.28),
               snap('kalshi', NOW, price_a=0.40, price_b=0.60, best_bid_a=0.39, best_ask_a=0.41,
                    best_bid_b=0.59, best_ask_b=0.61))
    buy_a, buy_b = sharp.buy_odds(df, pm_spread=0.02)
    assert buy_a[0] == pytest.approx(1 / 0.73)  # über der Mitte: der Ask gilt
    assert buy_b[0] == pytest.approx(1 / (0.375 + 0.01))  # 0,28 < 0,375: Preis + halber Spread
    assert (buy_a[1], buy_b[1]) == (pytest.approx(1 / 0.41), pytest.approx(1 / 0.61))


def test_implausible_sharp_book_is_no_fair_price():
    """Betfair 1,06/1,10 (Buch 185 %) ist kein fairer Preis: kein Signal, Closing aus dem letzten plausiblen Kurs."""
    df = frame(snap('betfair', NOW, price_a=1.06, price_b=1.10), snap('tipico', NOW, price_a=1.25, price_b=4.0))
    assert sharp.signals(df, threshold=0.0).empty
    close = sharp.closing_fair(frame(snap('betfair', NOW, price_a=1.27, price_b=4.4),
                                     snap('betfair', NOW + timedelta(minutes=30), price_a=1.06, price_b=1.10)))
    assert close.iloc[0] == pytest.approx((1 / 1.27) / (1 / 1.27 + 1 / 4.4))


def test_kelly_cap_and_shrink():
    k = Kelly(0.25, 0.02)
    assert k.amount(1000, 0.9, 3.0) == pytest.approx(20.0)  # Deckel 2 %
    assert k.amount(1000, 0.52, 2.0) == pytest.approx(1000 * 0.25 * 0.04)
    assert k.amount(1000, 0.4, 2.0) == 0.0
    bets = pd.DataFrame({'p_fair': [0.54], 'odds': [2.0], 'net_odds': [2.0]})
    out = sharp.stake_sizes(bets, 1000, k, shrink=0.5)
    assert out['p_shrunk'].iloc[0] == pytest.approx(0.52)  # halb Richtung 1/2,0
    assert out['stake_kelly'].iloc[0] == pytest.approx(1000 * 0.25 * 0.04)


def test_evaluate_ret_rules_and_clv():
    bets = pd.DataFrame({'match_key': ['m1', 'm2'], 'side': ['a', 'a'], 'venue': ['tipico', 'kalshi'],
                         'odds': [2.0, 2.0], 'net_odds': [1.894, 1.95], 'stake_flat': [10.0, 10.0],
                         'stake_kelly': [5.0, 5.0], 'p_fair': [0.55, 0.55]})
    outcomes = pd.DataFrame({'a_won': [1.0, 1.0], 'status': ['RET', 'RET']}, index=['m1', 'm2'])
    out = sharp.evaluate(bets, outcomes, close=pd.Series({'m1': 0.6, 'm2': 0.5}))
    assert bool(out['void'].iloc[0]) and out['profit_flat'].iloc[0] == 0  # Tipico storniert bei Aufgabe (Annahme)
    assert not bool(out['void'].iloc[1]) and out['profit_flat'].iloc[1] == pytest.approx(9.5)  # Kalshi zahlt aus
    assert out['clv'].iloc[0] == pytest.approx(0.6 * 2.0 - 1)


# ------------------------------------------------------------------ Paper

def _snaps_for_paper():
    return frame(
        snap('kalshi', NOW - timedelta(minutes=20), price_a=0.40, price_b=0.60, best_bid_a=0.39, best_ask_a=0.41,
             best_bid_b=0.59, best_ask_b=0.61, event_id='KX-1', level='challenger'),
        snap('kalshi', START - timedelta(minutes=30), price_a=0.50, price_b=0.50, best_bid_a=0.49, best_ask_a=0.51,
             best_bid_b=0.49, best_ask_b=0.51, event_id='KX-1', level='challenger'),  # letzter Preis vor Start
        snap('kalshi', START + timedelta(minutes=30), price_a=0.90, price_b=0.10, event_id='KX-1', level='challenger'),
    )


def test_paper_signal_bet_settle_clv(tmp_path):
    snaps = _snaps_for_paper()
    current = snaps[snaps['ts_utc'] <= pd.Timestamp(NOW)]
    new = signals.model_signals(current, lambda r: 0.55, pd.Timestamp(NOW), bankroll=1000, threshold=0.05)
    assert len(new) == 1
    b = new.iloc[0]
    assert (b['venue'], b['side'], b['status'], b['level']) == ('kalshi', 'a', 'open', 'challenger')
    assert b['odds'] == pytest.approx(1 / 0.41)
    assert 0 < b['stake'] <= 20.0 and b['stake_flat'] == signals.FLAT_STAKE

    bets, n = add_bets(load_bets(str(tmp_path / 'bets.parquet')), new)
    assert n == 1
    bets, n = add_bets(bets, new)  # derselbe Lauf noch einmal: keine doppelte Wette
    assert n == 0

    # Abrechnung: noch nicht fällig, dann über die Kalshi-Auflösung
    later = pd.Timestamp(START + timedelta(hours=1))
    assert settle.settle(bets, later, {}, [], snaps)[1] == 0
    history = {'kalshi': pd.DataFrame({'event_id': ['KX-1'], 'winner': ['a']})}
    done, n = settle.settle(bets, pd.Timestamp(START + timedelta(hours=4)), history, [], snaps)
    assert n == 1
    r = done.iloc[0]
    assert r['status'] == 'won' and r['result_source'] == 'kalshi'
    assert r['profit'] == pytest.approx(r['stake'] * (r['net_odds'] - 1))
    assert r['close_odds'] == pytest.approx(1 / 0.50)  # letzter Snapshot VOR dem Start, nicht der danach
    assert r['clv'] == pytest.approx((1 / 0.41) / 2.0 - 1)

    path = str(tmp_path / 'bets.parquet')
    save_bets(done, path)
    assert load_bets(path)['status'].iloc[0] == 'won'


def test_paper_settle_from_matches_ret_rules():
    bets = pd.DataFrame([
        {'bet_id': 'x1', 'created_utc': pd.Timestamp(NOW), 'strategy': 'sharp', 'venue': 'tipico', 'event_id': 't1',
         'player_a': 'Jannik Sinner', 'player_b': 'Carlos Alcaraz', 'side': 'a', 'start_time_utc': pd.Timestamp(START),
         'odds': 2.0, 'net_odds': 1.894, 'stake': 10.0, 'stake_flat': 10.0, 'status': 'open'},
        {'bet_id': 'x2', 'created_utc': pd.Timestamp(NOW), 'strategy': 'modell', 'venue': 'polymarket', 'event_id': 'p9',
         'player_a': 'Jannik Sinner', 'player_b': 'Carlos Alcaraz', 'side': 'b', 'start_time_utc': pd.Timestamp(START),
         'odds': 2.0, 'net_odds': 2.0, 'stake': 10.0, 'stake_flat': 10.0, 'status': 'open'},
    ])
    from src.paper.store import normalize
    bets = normalize(bets)
    matches = [{'winner_name': 'Jannik Sinner', 'loser_name': 'Carlos Alcaraz', 'match_date': START.strftime('%Y%m%d'),
                'tourney_date': START.strftime('%Y%m%d'), 'status': 'RET', 'source': 'espn'}]
    done, n = settle.settle(bets, pd.Timestamp(START + timedelta(hours=5)), {}, matches, None)
    assert n == 2
    assert done.loc[0, 'status'] == 'void' and done.loc[0, 'profit'] == 0  # Tipico: RET void
    assert done.loc[1, 'status'] == 'lost' and done.loc[1, 'profit'] == -10  # PM: Sieger nach Aufgabe zählt


def test_paper_report_go_criteria():
    n = 320
    bets = pd.DataFrame({'strategy': 'sharp', 'venue': 'kalshi', 'level': 'itf', 'status': ['won'] * n,
                         'stake': 10.0, 'stake_flat': 10.0, 'profit': 1.0, 'profit_flat': 1.0, 'clv': 0.03,
                         'clv_sharp': np.nan, 'settled_utc': pd.date_range('2026-01-01', periods=n, tz='UTC')})
    d = report.go_no_go(bets, n_boot=200)
    assert d['go'] and all(ok for _, ok, _ in d['checks'])
    few = report.go_no_go(bets.head(100), n_boot=200)
    assert not few['go'] and not few['checks'][0][1]
    curve = report.bankroll_curve(bets)
    assert curve[0][1] == report.START_BANKROLL and curve[-1][1] == pytest.approx(report.START_BANKROLL + n)
    table = report.summary(bets, n_boot=200)
    assert set(table['gruppe']) == {'gesamt', 'strategy', 'venue', 'level'}


def test_sharp_paper_signals_side_in_venue_orientation():
    df = frame(snap('pinnacle', NOW, price_a=1.8, price_b=2.1),
               snap('kalshi', NOW, a='Carlos Alcaraz', b='Jannik Sinner', price_a=0.40, price_b=0.60,
                    best_bid_a=0.39, best_ask_a=0.41, best_bid_b=0.59, best_ask_b=0.61))
    out = signals.sharp_signals(df, pd.Timestamp(NOW + timedelta(minutes=1)), threshold=0.0)
    # fair Alcaraz ~ 0,46 > 0,41 + Gebühr: Wette auf Alcaraz = Seite A im Kalshi-Event
    assert len(out) == 1
    assert (out['player'].iloc[0], out['side'].iloc[0]) == ('Carlos Alcaraz', 'a')


def test_recent_matches_and_level_fill(tmp_path):
    import csv
    from src.paper import run
    path = tmp_path / 'matches.csv'
    rows = [{'winner_name': 'A', 'loser_name': 'B', 'match_date': '20261020', 'tourney_date': '20261019',
             'status': 'RET', 'source': 'espn', 'tourney_name': 'Paris', 'level': 'main'},
            {'winner_name': 'C', 'loser_name': 'D', 'match_date': '20250301', 'tourney_date': '20250301',
             'status': 'completed', 'source': 'sackmann', 'tourney_name': 'Lima Challenger', 'level': 'challenger'},
            {'winner_name': 'E', 'loser_name': 'F', 'match_date': '20200101', 'tourney_date': '20200101',
             'status': 'completed', 'source': 'sackmann', 'tourney_name': 'Old Challenger', 'level': 'challenger'}]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    matches, level_rows = run.recent_matches(NOW, str(path))
    assert [m['winner_name'] for m in matches] == ['A']  # nur die letzten 30 Tage
    assert [r['tourney_name'] for r in level_rows] == ['Lima Challenger']
    bets = pd.DataFrame({'level': [None, 'itf'], 'tourney': ['Lima', 'M15 X'], 'tour': ['atp', 'atp'],
                         'start_time_utc': [pd.Timestamp(START)] * 2, 'created_utc': [pd.Timestamp(NOW)] * 2})
    run.fill_levels(bets, level_rows)
    assert list(bets['level']) == ['challenger', 'itf']


def test_fill_levels_marker_tour_and_calendar_week():
    """Haupttour wird nicht mehr zu Challenger/ITF, nur weil die Stadt auch ein kleines Turnier hat."""
    from src.paper import run
    rows = [{'tour': 'atp', 'tourney_name': 'M15 Vienna', 'tourney_date': '20250901', 'level': 'itf'},
            {'tour': 'atp', 'tourney_name': 'Shanghai CH', 'tourney_date': '20250908', 'level': 'challenger'},
            {'tour': 'atp', 'tourney_name': 'Shanghai Masters', 'tourney_date': '20251001', 'level': 'main'},
            {'tour': 'atp', 'tourney_name': 'Ningbo CH', 'tourney_date': '20251020', 'level': 'challenger'},
            {'tour': 'wta', 'tourney_name': 'Ningbo', 'tourney_date': '20251013', 'level': 'main'}]
    bets = pd.DataFrame({'level': None, 'tour': ['atp', 'atp', 'atp', 'atp'],
                         'tourney': ['ATP Vienna Open', 'Shanghai', 'Shanghai', 'Ningbo'],
                         'start_time_utc': pd.to_datetime(['2026-10-27', '2026-09-10', '2026-10-05', '2026-10-28'],
                                                          utc=True),
                         'created_utc': pd.Timestamp(NOW)})
    run.fill_levels(bets, rows)
    # Marker 'ATP ...' (Odds API), Challenger im September, Masters im Oktober, Ningbo der Herren ist ein Challenger
    assert list(bets['level']) == ['main', 'challenger', 'main', 'challenger']


def test_fill_levels_uses_kalshi_snapshots_of_the_same_week():
    from src.paper import run
    rows = [{'tour': 'atp', 'tourney_name': 'Shanghai CH', 'tourney_date': '20250908', 'level': 'challenger'}]
    bets = pd.DataFrame({'level': [None], 'tour': ['atp'], 'tourney': ['Shanghai'],
                         'start_time_utc': [pd.Timestamp(START)], 'created_utc': [pd.Timestamp(NOW)]})
    run.fill_levels(bets, rows)
    assert bets['level'].iloc[0] == 'challenger'  # nur Sackmann: der Challenger im September
    snaps = frame(snap('kalshi', NOW, price_a=0.5, price_b=0.5, event_id='KX-SH', tourney='ATP Shanghai', level='main'))
    bets['level'] = None
    run.fill_levels(bets, rows + run.snapshot_levels(snaps))
    assert bets['level'].iloc[0] == 'main'  # Kalshi führt das Turnier in dieser Woche als Haupttour


def test_run_once_uses_snapshots_collected_in_same_round(tmp_path, monkeypatch):
    """Die eben geholten Snapshots müssen in derselben Runde zu Signalen werden (nicht erst 30 Minuten später)."""
    from types import SimpleNamespace
    from src.odds import collect, history
    from src.odds.schema import append_snapshots
    from src.paper import run
    clock = [NOW]  # die Abfrage dauert ein paar Sekunden

    def fake_collect(venues, root, log=print):
        ts = clock[0] + timedelta(seconds=2)
        clock[0] = ts + timedelta(seconds=2)
        append_snapshots([Snapshot(ts_utc=ts, venue='polymarket', event_id='pm-1', player_a='Jannik Sinner',
                                   player_b='Carlos Alcaraz', price_a=0.5, price_b=0.5, price_type='probability',
                                   tour='atp', start_time_utc=ts + timedelta(hours=5))], root)

    monkeypatch.setattr(collect, 'collect_once', fake_collect)
    monkeypatch.setattr(run, 'now_utc', lambda: clock[0])
    monkeypatch.setattr(history, 'read_history', lambda venue: pd.DataFrame())
    monkeypatch.setattr(run, 'recent_matches', lambda now: ([], []))
    args = SimpleNamespace(collect=True, odds_api=False, softbooks=False, snapshots=str(tmp_path / 'snaps'), days=1,
                           bets=str(tmp_path / 'bets.parquet'), strategies=['modell'], model='elo2', threshold=0.05,
                           raw=True, sharp_threshold=0.02, history=False)
    bets, _, _ = run.run_once(args, log=lambda *a: None, prob=lambda row: 0.8, model_name='elo2')
    assert len(bets) == 1 and bets['player'].iloc[0] == 'Jannik Sinner'
