"""Sharp vs. Rest auf Snapshots, Kelly-Deckel und Paper-Trading (Signal -> Wette -> Abrechnung -> CLV), ohne Netz."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.evaluation import sharp
from src.evaluation.betting import Kelly
from src.evaluation.predictors import remove_margin
from src.odds.schema import Snapshot, average_price, dump_levels, to_frame
from src.paper import report, settle, signals
from src.paper.store import add_bets, load_bets, normalize, save_bets

NOW = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)
START = NOW + timedelta(hours=5)


def after_fee(gross):
    """Netto-Quote bei Polymarket: 5 % Taker-Gebühr auf den möglichen Gewinn."""
    return 1 + (gross - 1) * 0.95


def deep(price):
    """Orderbuch einer Kaufseite, das jeden Einsatz zum besten Ask füllt."""
    return dump_levels([(price, 100000.0)])


def snap(venue, ts, a='Jannik Sinner', b='Carlos Alcaraz', price_a=None, price_b=None, event_id=None, start=START,
         **kw):
    ptype = 'probability' if venue in ('polymarket', 'kalshi') else 'decimal'
    return Snapshot(ts_utc=ts, venue=venue, event_id=event_id or f'{venue}-1', player_a=a, player_b=b,
                    price_a=price_a, price_b=price_b, price_type=ptype, tour='atp', start_time_utc=start, **kw)


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
    p_fair = sharp.power_fair(1.8, 2.1)
    df = frame(snap('pinnacle', NOW, price_a=1.8, price_b=2.1),
               snap('tipico', NOW, price_a=2.00, price_b=1.80),  # 2,00 brutto, nach 5,3 % Steuer 1,894
               snap('polymarket', NOW, price_a=0.505, price_b=0.495, best_ask_a=0.51, best_ask_b=0.50))  # 1,96
    sig = sharp.signals(df, threshold=0.0)
    a = sig[sig['side'] == 'a'].iloc[0]
    assert a['venue'] == 'polymarket'
    assert a['net_odds'] == pytest.approx(after_fee(1 / 0.51))  # auch nach Gebühr über den 1,894 von Tipico
    assert a['edge'] == pytest.approx(p_fair * after_fee(1 / 0.51) - 1)
    # Tipico allein: Steuer frisst den Vorteil
    only_tipico = sharp.signals(frame(snap('pinnacle', NOW, price_a=1.8, price_b=2.1),
                                      snap('tipico', NOW, price_a=2.00, price_b=1.80)), threshold=0.0)
    assert only_tipico.empty or only_tipico['edge'].max() < p_fair * 2.0 - 1


def test_buy_odds_ignore_ask_below_mid():
    """Polymarket: Ein Ask unter der Mitte zählt nicht und bekommt auch keinen Ersatzpreis, die Gegenseite bleibt."""
    df = frame(snap('polymarket', NOW, price_a=0.625, price_b=0.375, best_bid_a=0.72, best_ask_a=0.73,
                    best_bid_b=0.27, best_ask_b=0.28),
               snap('kalshi', NOW, price_a=0.40, price_b=0.60, best_bid_a=0.39, best_ask_a=0.41,
                    best_bid_b=0.59, best_ask_b=0.61))
    buy_a, buy_b = sharp.buy_odds(df, pm_spread=0.02)
    assert buy_a[0] == pytest.approx(1 / 0.73)  # über der Mitte: der Ask gilt
    assert np.isnan(buy_b[0])  # 0,28 < 0,375: kein Kaufpreis (früher Preis + halber Spread)
    assert (buy_a[1], buy_b[1]) == (pytest.approx(1 / 0.41), pytest.approx(1 / 0.61))


EMPTY_BOOK = dict(best_bid_a=0.0, best_ask_a=1.0, best_bid_b=0.0, best_ask_b=1.0)  # so meldet Gamma ein leeres Orderbuch


def test_buy_odds_polymarket_needs_real_ask_and_no_placeholder():
    """Polymarket: kein Kaufpreis ohne brauchbares Ask und bei einem Preis von genau 0,5. Kalshi behält den Rückfall."""
    df = frame(snap('polymarket', NOW, price_a=0.5, price_b=0.5, **EMPTY_BOOK),
               snap('polymarket', NOW, price_a=0.24, price_b=0.76, **EMPTY_BOOK),  # leer, der Preis ist nicht 0,5
               snap('polymarket', NOW, price_a=0.40, price_b=0.60),  # kein Bid/Ask im Snapshot
               snap('polymarket', NOW, price_a=0.5, price_b=0.5, best_bid_a=0.49, best_ask_a=0.51,  # wie der Backtest
                    best_bid_b=0.49, best_ask_b=0.51),
               snap('polymarket', NOW, price_a=0.495, price_b=0.505, best_bid_a=0.49, best_ask_a=0.50,  # echtes Buch
                    best_bid_b=0.50, best_ask_b=0.51),
               snap('kalshi', NOW, price_a=0.40, price_b=0.60),
               snap('tipico', NOW, price_a=2.0, price_b=1.8))
    buy_a, buy_b = sharp.buy_odds(df, pm_spread=0.02)
    assert np.isnan(buy_a[:4]).all() and np.isnan(buy_b[:4]).all()
    assert (buy_a[4], buy_b[4]) == (pytest.approx(1 / 0.50), pytest.approx(1 / 0.51))
    assert (buy_a[5], buy_b[5]) == (pytest.approx(1 / 0.41), pytest.approx(1 / 0.61))
    assert (buy_a[6], buy_b[6]) == (2.0, 1.8)


def test_sharp_signals_ignore_polymarket_without_real_ask():
    """Line Shopping nimmt die höchste Quote: Ein leeres Polymarket-Buch (0,5 + halber Spread wären 1,96) darf kein
    Signal auslösen und die echte Quote eines anderen Anbieters nicht verdrängen."""
    fair = snap('pinnacle', NOW, price_a=1.4, price_b=3.2)  # A fair ~ 0,70
    empty = snap('polymarket', NOW, price_a=0.5, price_b=0.5, **EMPTY_BOOK)
    assert sharp.signals(frame(fair, empty), threshold=0.02).empty
    assert signals.sharp_signals(frame(fair, empty), pd.Timestamp(NOW + timedelta(minutes=1))).empty
    sig = sharp.signals(frame(fair, empty, snap('tipico', NOW, price_a=1.6, price_b=2.4)), threshold=0.02)
    assert list(sig['venue']) == ['tipico']


@pytest.mark.parametrize('odds_a, odds_b', [(1.09, 8.57), (1.9, 1.95), (1.01, 26.0), (1.27, 4.4), (1.03, 15.0),
                                            (2.04, 2.08), (1.52, 2.9), (1.002, 400.0), (1.05, 8.0)])
def test_power_fair_sums_to_one_and_solves_k(odds_a, odds_b):
    """Beide Seiten ergeben 1 (signals rechnet Seite B als 1 - p), auch bei Buch unter 1 und langen Außenseitern."""
    p_a, p_b = sharp.power_fair(odds_a, odds_b), sharp.power_fair(odds_b, odds_a)
    assert p_a + p_b == pytest.approx(1.0, abs=1e-12)
    k = np.log(p_a) / np.log(1 / odds_a)  # p = (1/o)^k mit demselben k auf beiden Seiten
    assert p_b == pytest.approx((1 / odds_b) ** k, abs=1e-12)


def test_power_fair_takes_margin_from_longshot():
    """Landaluce am 27.10.: Pinnacle 1,09/8,57. Proportional ergibt 0,113, über dem Betfair-Back von 10,5 (0,095)."""
    p_dog = sharp.power_fair(8.57, 1.09)
    assert p_dog == pytest.approx(0.09148, abs=1e-5)  # mit scipy.optimize.brentq nachgerechnet, k = 1,1133
    assert remove_margin(8.57, 1.09) == pytest.approx(0.1129, abs=5e-4)
    bid, ask = 1 - 1 / 1.09, 1 / 10.5  # Betfair 1,09/10,5: Back auf den Favoriten spiegelt das Bid des Außenseiters
    assert bid < p_dog < ask < remove_margin(8.57, 1.09)
    assert sharp.power_fair(2.0, 2.0) == pytest.approx(0.5)  # gleiche Quoten: nichts zu verteilen
    assert sharp.power_fair(1 / 0.6, 1 / 0.4) == pytest.approx(0.6)  # ohne Marge bleibt der Preis


@pytest.mark.parametrize('odds_a, odds_b', [(1.09, 10.5), (1.27, 4.4), (2.0, 1.98), (1.02, 34.0), (10.5, 1.09)])
def test_power_fair_stays_between_betfair_bid_and_ask(odds_a, odds_b):
    """Nur Back-Preise: Der faire Preis liegt zwischen dem Back auf A (Ask) und dem gespiegelten Back auf B (Bid)."""
    assert 1 - 1 / odds_b < sharp.power_fair(odds_a, odds_b) < 1 / odds_a


def test_longshot_signal_comes_from_margin_method():
    """Die Exchange-Quote 10,5 gegen Pinnacle 8,57: proportional 12 % Edge nach Steuer, mit Power keine Wette."""
    df = frame(snap('pinnacle', NOW, price_a=1.09, price_b=8.57), snap('matchbook', NOW, price_a=1.06, price_b=10.5))
    assert remove_margin(8.57, 1.09) * 10.5 * (1 - sharp.DE_BET_TAX) - 1 > 0.12
    assert sharp.signals(df, threshold=0.02).empty
    aligned = sharp.align(df)
    assert aligned['p_fair'].iloc[0] == pytest.approx(sharp.power_fair(1.09, 8.57))


def test_implausible_sharp_book_is_no_fair_price():
    """Betfair 1,06/1,10 (Buch 185 %) ist kein fairer Preis: kein Signal, Closing aus dem letzten plausiblen Kurs."""
    df = frame(snap('betfair', NOW, price_a=1.06, price_b=1.10), snap('tipico', NOW, price_a=1.25, price_b=4.0))
    assert sharp.signals(df, threshold=0.0).empty
    close = sharp.closing_fair(frame(snap('betfair', NOW, price_a=1.27, price_b=4.4),
                                     snap('betfair', NOW + timedelta(minutes=30), price_a=1.06, price_b=1.10)))
    assert close.iloc[0] == pytest.approx(sharp.power_fair(1.27, 4.4))


FAIR_B20 = snap('pinnacle', NOW, price_a=1.25, price_b=5.0)  # Buch 1,0, also B fair 0,20 und faire Quote 5,0
STALE = ('williamhill', (1.05, 10.0))


def market(*quotes, normal=(1.2, 4.4), n=6):
    """n Bücher mit derselben Quote (normal) und dazu quotes als (Venue, (Quote A, Quote B))."""
    books = [snap(f'book{i}', NOW, price_a=normal[0], price_b=normal[1]) for i in range(n)]
    return books + [snap(venue, NOW, price_a=a, price_b=b) for venue, (a, b) in quotes]


def test_outlier_loses_line_shopping_with_its_clone():
    """Wie Putintseva am 27.10.: 10,0 bei William Hill und 888 (selber Feed) gegen Median 5,75. Die nächste Quote gewinnt."""
    clones = (STALE, ('sport888', STALE[1]))
    assert sharp.signals(frame(FAIR_B20, *market(*clones)), threshold=0.02).empty
    sig = sharp.signals(frame(FAIR_B20, *market(*clones, ('matchbook', (1.21, 5.6)))), threshold=0.02)
    assert list(sig['venue']) == ['matchbook'] and list(sig['odds']) == [5.6]  # 4,9 Punkte über dem Median, Edge 6 %


@pytest.mark.parametrize('fair, normal, quote', [
    ((1.5, 2.6), (1.5, 2.3), (1.75, 2.0)),  # Favorit: 9,5 Punkte, aber nur das 1,17-Fache
    ((1.06, 11.0), (1.05, 12.0), (1.03, 19.0)),  # Außenseiter: nur 3,1 Punkte, aber das 1,58-Fache
])
def test_outlier_in_points_or_as_ratio(fair, normal, quote):
    """Ein Maß allein reicht nicht: Der Quotient übersieht Favoriten, die Punkte übersehen lange Außenseiter."""
    pin = snap('pinnacle', NOW, price_a=fair[0], price_b=fair[1])
    assert sharp.signals(frame(pin, *market(('stale', quote), normal=normal)), threshold=0.02).empty
    alone = market(('stale', quote), n=0)  # ohne andere Bücher greift die Regel nicht, die Quote wäre ein Signal
    assert list(sharp.signals(frame(pin, *alone), threshold=0.02)['venue']) == ['stale']


def test_normal_spread_of_exchanges_stays():
    """Landaluce am 27.10.: Matchbook 10,5 gegen Median 7,7 (3,5 Punkte, 1,36-fach) ist bei einer Exchange ohne Marge normal.
    Die Quote bleibt im Line Shopping, ist mit der Power-Marge aber kein Signal mehr (proportional waren es 12 % Edge)."""
    pin = snap('pinnacle', NOW, price_a=1.09, price_b=8.57)
    sig = sharp.signals(frame(pin, *market(('matchbook', (1.09, 10.5)), normal=(1.08, 7.7))), threshold=-1.0)
    best = sig[sig['side'] == 'b']
    assert list(best['venue']) == ['matchbook']
    assert best['edge'].iloc[0] == pytest.approx(sharp.power_fair(8.57, 1.09) * 10.5 * (1 - sharp.DE_BET_TAX) - 1)


@pytest.mark.parametrize('gap, out', [(sharp.MAX_GAP - 0.005, False), (sharp.MAX_GAP + 0.005, True)])
def test_outlier_limit_in_points(gap, out):
    """Knapp unter und über MAX_GAP, Median 2,0 (der Quotient bleibt unter MAX_RATIO)."""
    pin = snap('pinnacle', NOW, price_a=2.0, price_b=2.0)
    quote = 1 / (0.5 - gap)
    assert sharp.signals(frame(pin, *market(('stale', (1.7, quote)), normal=(1.9, 2.0))), threshold=0.02).empty == out


def test_outlier_needs_enough_books():
    """Unter MIN_BOOKS Quoten ist der Median nicht belastbar, weil sich Bücher Feeds teilen: Die Quote bleibt."""
    few = market(STALE, n=sharp.MIN_BOOKS - 2)
    assert list(sharp.signals(frame(FAIR_B20, *few), threshold=0.02)['venue']) == ['williamhill']
    assert sharp.signals(frame(FAIR_B20, *market(STALE, n=sharp.MIN_BOOKS - 1)), threshold=0.02).empty


def test_outlier_with_swapped_players_and_on_prediction_markets():
    """Das Maß gilt aus Sicht des Sharp-Events, auch wenn das Buch A und B tauscht, und ebenso für den Ask der Märkte."""
    swapped = snap('williamhill', NOW, a='Carlos Alcaraz', b='Jannik Sinner', price_a=10.0, price_b=1.05)
    assert sharp.signals(frame(FAIR_B20, *market(), swapped), threshold=0.02).empty
    cheap = snap('polymarket', NOW, price_a=0.9, price_b=0.09, best_ask_a=0.91, best_ask_b=0.1)  # Ask 0,10 = Quote 10,0
    assert sharp.signals(frame(FAIR_B20, *market(), cheap), threshold=0.02).empty


def test_paper_bet_skips_outlier():
    """Der Weg bis zur Wette: Die Paper-Runde bucht nicht den Ausreißer, sondern die nächste Quote mit Edge."""
    now = pd.Timestamp(NOW + timedelta(minutes=1))
    assert signals.sharp_signals(frame(FAIR_B20, *market(STALE)), now).empty
    bets = signals.sharp_signals(frame(FAIR_B20, *market(STALE, ('matchbook', (1.21, 5.6)))), now)
    assert list(bets['venue']) == ['matchbook'] and list(bets['odds']) == [5.6]


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


PLAYERS = [('Jannik Sinner', 'Carlos Alcaraz'), ('Daniil Medvedev', 'Alexander Zverev'), ('Casper Ruud', 'Holger Rune'),
           ('Taylor Fritz', 'Tommy Paul'), ('Lan Mi', 'Elza Tomase'), ('Ben Shelton', 'Jack Draper')]
BOOK = dict(price_a=0.55, price_b=0.45, best_bid_a=0.54, best_ask_a=0.56, best_bid_b=0.44, best_ask_b=0.46,
            asks_a=deep(0.56), asks_b=deep(0.46))


def test_model_signals_polymarket_only_at_real_ask():
    """Das Modell sieht überall einen großen Vorteil auf A, gewettet wird nur, wo es dafür einen echten Preis gibt."""
    def market(event_id, venue='polymarket', **prices):
        return snap(venue, NOW, event_id=event_id, **prices)

    snaps = frame(
        market('leer', price_a=0.5, price_b=0.5, **EMPTY_BOOK),
        market('leer-anderer-preis', price_a=0.24, price_b=0.76, **EMPTY_BOOK),
        market('ohne-bid-ask', price_a=0.40, price_b=0.60),
        market('platzhalter-mit-ask', price_a=0.5, price_b=0.5, best_bid_a=0.63, best_ask_a=0.68, best_bid_b=0.32,
               best_ask_b=0.37, asks_a=deep(0.68), asks_b=deep(0.37)),
        # A nur zu einem Ask unter der Mitte, B hat ein Ask, aber keinen Vorteil: auch hier keine Wette
        market('ask-unter-mitte', price_a=0.60, price_b=0.40, best_bid_a=0.45, best_ask_a=0.50, best_bid_b=0.50,
               best_ask_b=0.55, asks_a=deep(0.50), asks_b=deep(0.55)),
        market('echtes-ask', price_a=0.40, price_b=0.60, best_bid_a=0.39, best_ask_a=0.41, best_bid_b=0.59,
               best_ask_b=0.61, asks_a=deep(0.41), asks_b=deep(0.61)),
        market('ask-genau-0.5', price_a=0.495, price_b=0.505, best_bid_a=0.49, best_ask_a=0.50, best_bid_b=0.50,
               best_ask_b=0.51, asks_a=deep(0.50), asks_b=deep(0.51)),
        market('nur-ask-auf-a', price_a=0.40, price_b=0.60, best_ask_a=0.41, best_bid_b=0.59, asks_a=deep(0.41)),
        market('kalshi-ohne-ask', venue='kalshi', price_a=0.40, price_b=0.60))
    logs = []
    out = signals.model_signals(snaps, lambda r: 0.8, pd.Timestamp(NOW), threshold=0.05, log=logs.append)
    assert dict(zip(out['event_id'], 1 / out['odds'])) == pytest.approx(
        {'echtes-ask': 0.41, 'ask-genau-0.5': 0.50, 'nur-ask-auf-a': 0.41, 'kalshi-ohne-ask': 0.41})
    assert (out['side'] == 'a').all()
    assert out.set_index('event_id')['p_market'].to_dict()['echtes-ask'] == 0.40  # Mitte der gewetteten Seite
    assert any(m.startswith('  Modell: 4 von 9 Märkten ohne Kaufpreis') for m in logs)


def polymarket_market(asks_a, price_a=0.39):
    """Polymarket-Markt mit diesem Orderbuch auf der Kaufseite A, auf B teuer genug, dass dort kein Vorteil liegt."""
    return frame(snap('polymarket', NOW, price_a=price_a, price_b=1 - price_a, best_bid_a=price_a - 0.01,
                      best_ask_a=asks_a[0][0], best_bid_b=1 - asks_a[0][0], best_ask_b=1 - price_a + 0.01,
                      asks_a=dump_levels(asks_a), asks_b=deep(1 - price_a + 0.01)))


def model_bets(snaps, p, logs=None, bankroll=1000.0, threshold=0.05):
    return signals.model_signals(snaps, lambda r: p, pd.Timestamp(NOW), bankroll=bankroll, threshold=threshold,
                                 log=(logs.append if logs is not None else lambda *a: None))


def test_model_signals_polymarket_pays_average_price_of_each_stake():
    """Am besten Ask liegen nur 2 USD: Kelly (20 USD, Deckel 2 %) und flat (10 USD) zahlen je ihren Durchschnittspreis."""
    out = model_bets(polymarket_market([(0.40, 5.0), (0.42, 100.0), (0.70, 1000.0)]), 0.8)
    assert len(out) == 1
    b = out.iloc[0]
    kelly_shares, flat_shares = 5 + 18 / 0.42, 5 + 8 / 0.42
    assert b['stake'] == pytest.approx(20.0) and b['stake_flat'] == signals.FLAT_STAKE
    assert b['odds'] == pytest.approx(kelly_shares / 20)  # 1 / 0,4184, nicht 1 / 0,40
    assert b['net_odds'] == pytest.approx(after_fee(kelly_shares / 20))
    assert b['net_odds_flat'] == pytest.approx(after_fee(flat_shares / 10))
    assert b['edge'] == pytest.approx(0.8 * after_fee(kelly_shares / 20) - 1)
    assert b['p_market'] == pytest.approx(0.39)  # weiter die Mitte


def test_model_signals_deep_book_pays_best_ask():
    """Reicht die Tiefe am besten Ask, ändert sich nichts gegenüber dem Kauf zum besten Ask."""
    b = model_bets(polymarket_market([(0.40, 100.0), (0.45, 1000.0)]), 0.8).iloc[0]
    assert (b['odds'], b['net_odds_flat']) == (pytest.approx(1 / 0.40), pytest.approx(after_fee(1 / 0.40)))
    assert b['stake'] == pytest.approx(20.0)


def test_model_signals_edge_is_after_taker_fee():
    """6 % Vorteil zum Preis, nach 5 % Gebühr auf den Gewinn bleiben 3,35 %: unter der Schwelle keine Wette."""
    assert model_bets(polymarket_market([(0.50, 1000.0)], price_a=0.49), 0.53).empty
    b = model_bets(polymarket_market([(0.50, 1000.0)], price_a=0.49), 0.56).iloc[0]
    assert (b['odds'], b['net_odds']) == (pytest.approx(2.0), pytest.approx(1.95))
    assert b['edge'] == pytest.approx(0.56 * 1.95 - 1)
    # Kelly rechnet mit der Netto-Quote, der Preis, zu dem p geschrumpft wird, bleibt der Marktpreis
    assert b['stake'] == pytest.approx(signals.KELLY.amount(1000.0, sharp.shrink_to_market(0.56, 0.50), 1.95))


def test_kelly_stake_fits_its_own_average_price():
    """Der Einsatz bestimmt den Preis und der Preis den Einsatz: Am Ende passen beide zusammen, unter dem Deckel."""
    asks = [(0.50, 6.0), (0.52, 1000.0)]
    p = 0.56
    at_best_ask = signals.KELLY.amount(1000.0, sharp.shrink_to_market(p, 0.50), after_fee(1 / 0.50))
    assert at_best_ask == pytest.approx(8.82, abs=0.01)  # zum besten Ask will Kelly 8,82 USD, Deckel 20 USD
    b = model_bets(polymarket_market(asks, price_a=0.49), p).iloc[0]
    price = average_price(asks, b['stake'])
    assert 1 / b['odds'] == pytest.approx(price)
    wanted = signals.KELLY.amount(1000.0, sharp.shrink_to_market(p, price), after_fee(1 / price))
    assert b['stake'] == pytest.approx(wanted, abs=1e-6)
    assert 3.0 < b['stake'] < at_best_ask  # über dem ersten Level, teurer als der beste Ask, deshalb weniger Einsatz
    assert price > 0.50
    assert b['net_odds_flat'] == pytest.approx(after_fee(1 / average_price(asks, 10.0)))


@pytest.mark.parametrize('case, asks', [
    ('flat passt nicht', [(0.40, 10.0)]),  # 4 USD im ganzen Buch
    ('nur Kelly passt nicht', [(0.40, 30.0)]),  # 12 USD im ganzen Buch, der Deckel erlaubt Kelly 20 USD
])
def test_model_signals_no_bet_when_book_does_not_fill_the_stake(case, asks):
    """Kein Teil-Kauf: Die Wette gibt es nur, wenn das Buch beide Einsätze füllt, auch wenn am besten Ask Vorteil liegt."""
    logs = []
    assert model_bets(polymarket_market(asks), 0.8, logs).empty
    assert any('1 Orderbuch füllt den Einsatz nicht' in m for m in logs)


def test_model_signals_book_that_just_fills_the_stake_counts():
    b = model_bets(polymarket_market([(0.40, 50.0)]), 0.8).iloc[0]  # genau 20 USD im Buch
    assert b['stake'] == pytest.approx(20.0) and b['odds'] == pytest.approx(1 / 0.40)


def test_model_signals_polymarket_without_depth_is_no_bet():
    """Snapshot von vor der Tiefe oder ohne erreichbares Buch: kein Rückfall auf den besten Ask."""
    snaps = frame(snap('polymarket', NOW, price_a=0.39, price_b=0.61, best_bid_a=0.38, best_ask_a=0.40, best_bid_b=0.60,
                       best_ask_b=0.62))
    assert model_bets(snaps, 0.8).empty


def test_model_signals_average_price_can_eat_the_edge():
    """20 % Vorteil am besten Ask, aber dort liegt nur 1 USD und dahinter 0,70: Der teure Rest frisst den Vorteil."""
    logs = []
    assert model_bets(polymarket_market([(0.50, 2.0), (0.70, 1000.0)], price_a=0.49), 0.6, logs).empty
    assert any('1 Vorteil beim Durchschnittspreis zu klein' in m for m in logs)


def test_model_signals_threshold_counts_for_the_more_expensive_stake():
    """Kelly (6,25 USD) passt ans beste Level und hätte 7 % Vorteil, flat (10 USD) läuft in 0,60 und schafft 5 % nicht."""
    asks = [(0.50, 16.0), (0.60, 1000.0)]
    kelly_edge, flat_edge = (0.55 * after_fee(1 / average_price(asks, stake)) - 1 for stake in (6.25, 10.0))
    assert kelly_edge > 0.05 > flat_edge
    logs = []
    assert model_bets(polymarket_market(asks, price_a=0.49), 0.55, logs).empty
    assert any('1 Vorteil beim Durchschnittspreis zu klein' in m for m in logs)
    assert len(model_bets(polymarket_market(asks, price_a=0.49), 0.55, threshold=0.02)) == 1


def test_model_signals_kalshi_keeps_best_ask():
    """Kalshi hat keine Tiefe im Snapshot und bleibt beim besten Ask, ein Preis für beide Einsätze."""
    snaps = frame(snap('kalshi', NOW, price_a=0.40, price_b=0.60, best_bid_a=0.39, best_ask_a=0.41, best_bid_b=0.59,
                       best_ask_b=0.61))
    b = model_bets(snaps, 0.8).iloc[0]
    assert b['odds'] == pytest.approx(1 / 0.41) and np.isnan(b['net_odds_flat'])


def test_snapshot_book_to_bet_on_side_b(tmp_path):
    """Vom CLOB-Buch bis zur Wette: Seite B kauft die gespiegelten Bids von Token A, auch über die Parquet-Datei."""
    from src.odds import polymarket
    from src.odds.schema import append_snapshots, load_snapshots
    title = 'M15 Bari: Jannik Sinner vs Carlos Alcaraz'
    event = {'slug': 'itf-sinner-alcaraz', 'title': title, 'startTime': START.isoformat(),
             'markets': [{'id': '7', 'question': title, 'sportsMarketType': 'moneyline',
                          'outcomes': '["Jannik Sinner", "Carlos Alcaraz"]', 'clobTokenIds': '["111", "222"]',
                          'gameStartTime': START.isoformat()}]}
    bids_a, asks_a = [(0.59, 5.0), (0.50, 1000.0)], [(0.61, 5.0), (0.70, 1000.0)]
    snaps = polymarket.parse_events([event], NOW, {'111': '0.6', '222': '0.4'}, {'111': (bids_a, asks_a)})
    root = str(tmp_path / 'snaps')
    append_snapshots(snaps, root)
    b = model_bets(load_snapshots(root=root), 0.05).iloc[0]
    assert (b['side'], b['player']) == ('b', 'Carlos Alcaraz')
    # 5 Anteile zu 0,41 (2,05 USD), der Rest zu 0,50
    assert b['net_odds_flat'] == pytest.approx(after_fee((5 + (10 - 2.05) / 0.50) / 10))
    assert b['odds'] == pytest.approx((5 + (20 - 2.05) / 0.50) / 20)


def test_settle_kelly_and_flat_each_at_their_own_price():
    """Gewinn je Einsatz zu dessen Preis. Ältere Wetten und andere Venues haben einen Preis für beide."""
    def bet(event_id, side='a', **prices):
        return {'bet_id': event_id, 'created_utc': pd.Timestamp(NOW), 'strategy': 'modell', 'venue': 'polymarket',
                'event_id': event_id, 'player_a': 'Jannik Sinner', 'player_b': 'Carlos Alcaraz', 'side': side,
                'start_time_utc': pd.Timestamp(START), 'stake': 20.0, 'stake_flat': 10.0, 'status': 'open', **prices}

    bets = normalize(pd.DataFrame([bet('tiefe', odds=2.4, net_odds=2.4, net_odds_flat=2.45),
                                   bet('tiefe-verloren', side='b', odds=2.4, net_odds=2.4, net_odds_flat=2.45),
                                   bet('alt', odds=2.0, net_odds=2.0)]))
    history = {'polymarket': pd.DataFrame({'event_id': ['tiefe', 'tiefe-verloren', 'alt'], 'winner': ['a', 'a', 'a']})}
    done, n = settle.settle(bets, pd.Timestamp(START + timedelta(hours=4)), history, [], None)
    assert n == 3
    assert list(done['profit']) == pytest.approx([20 * 1.4, -20.0, 20 * 1.0])
    assert list(done['profit_flat']) == pytest.approx([10 * 1.45, -10.0, 10 * 1.0])
    assert done.loc[2, 'net_odds_flat'] != done.loc[2, 'net_odds_flat']  # die alte Wette bleibt, wie sie war


def test_model_signals_only_in_segments():
    """Gewettet wird nur, wo Venue, Level und Stunden bis zum Start in ein Segment fallen; die Wette trägt es."""
    segments = [signals.Segment('polymarket', 'challenger', 18, 30), signals.Segment('polymarket', 'itf', 0.5, 1.5)]
    cases = [('pm-ch-24', 'polymarket', 'challenger', 24), ('pm-ch-5', 'polymarket', 'challenger', 5),
             ('pm-main-24', 'polymarket', 'main', 24), ('kx-ch-24', 'kalshi', 'challenger', 24),
             ('pm-itf-1', 'polymarket', 'itf', 1), ('pm-itf-30.5', 'polymarket', 'itf', 30.5)]
    snaps = frame(*[snap(venue, NOW, a, b, event_id=e, level=level, start=NOW + timedelta(hours=hours), **BOOK)
                    for (e, venue, level, hours), (a, b) in zip(cases, PLAYERS)])
    out = signals.model_signals(snaps, lambda r: 0.8, pd.Timestamp(NOW), threshold=0.05, segments=segments,
                                log=lambda *a: None)
    assert dict(zip(out['event_id'], out['segment'])) == {'pm-ch-24': 'polymarket challenger 18-30h',
                                                          'pm-itf-1': 'polymarket itf 0.5-1.5h'}
    assert all(b.endswith(f'|{s}') for b, s in zip(out['bet_id'], out['segment']))
    # ohne Segmente wie bisher: alles, was innerhalb von HORIZON beginnt
    assert len(signals.model_signals(snaps, lambda r: 0.8, pd.Timestamp(NOW), threshold=0.05,
                                     log=lambda *a: None)) == len(cases)


def test_load_segments_from_timing_table_and_fallback(tmp_path):
    path = str(tmp_path / 'timing.parquet')
    pd.DataFrame([('polymarket', 'challenger', '24h', 'blend_gbdt', 'Kandidat'),
                  ('polymarket', 'challenger', '24h', 'gbdt', 'Kandidat'),  # dasselbe Segment mit anderem Modell
                  ('polymarket', 'itf', '1h', 'blend_gbdt', 'GO'),
                  ('polymarket', 'itf', '6h', 'gbdt', 'Kandidat'),  # nur das rohe gbdt
                  ('polymarket', 'main', '6h', 'blend_gbdt', 'NO-GO'),
                  ('polymarket', 'all', 'close', 'blend_gbdt', 'Kandidat'),  # Zusammenfassung über die Level
                  ('kalshi', 'challenger', 'open', 'blend_gbdt', 'Kandidat')],  # Eröffnung: kein festes Fenster
                 columns=['venue', 'level', 'point', 'model', 'decision']).to_parquet(path)

    def names(segments):
        return [s.name for s in segments]

    quiet = {'log': lambda *a: None}
    assert names(signals.load_segments(path, model='blend_gbdt', **quiet)) == [
        'polymarket challenger 18-30h', 'polymarket itf 0.5-1.5h']
    assert names(signals.load_segments(path, **quiet)) == [
        'polymarket challenger 18-30h', 'polymarket itf 4.5-7.5h', 'polymarket itf 0.5-1.5h']
    # ohne Datei: die Kandidaten aus docs/ERGEBNISSE_PHASE45.md, ITF nur für blend_gbdt
    missing = str(tmp_path / 'fehlt.parquet')
    assert names(signals.load_segments(missing, model='blend_gbdt', **quiet)) == [
        'polymarket challenger 18-30h', 'polymarket itf 4.5-7.5h', 'polymarket itf 0.5-1.5h', 'polymarket itf 0-0.5h']
    assert names(signals.load_segments(missing, model='gbdt', **quiet)) == ['polymarket challenger 18-30h']
    # --segments: Zeitpunkt des Backtests oder Fenster in Stunden, 'alle' ohne Einschränkung
    assert names(signals.parse_segments('kalshi:itf:6h, polymarket:main:2-3.5')) == [
        'kalshi itf 4.5-7.5h', 'polymarket main 2-3.5h']
    assert signals.segments_from_arg('alle') is None
    for bad in ('polymarket:itf', 'polymarket:itf:2h', 'polymarket:itf:3-1'):
        with pytest.raises(ValueError):
            signals.parse_segments(bad)


def test_add_bets_once_per_match_and_segment(tmp_path):
    """Jedes Segment ist eine eigene Hypothese: dasselbe Match darf in zwei Segmenten gewettet werden, im selben nur
    einmal (auch nicht bei einer anderen Venue oder auf der anderen Seite)."""
    from src.paper.store import bet_id

    def bet(segment, venue='polymarket', event_id='pm-1', side='a'):
        return {'bet_id': bet_id('modell', venue, event_id, side, segment), 'created_utc': pd.Timestamp(NOW),
                'strategy': 'modell', 'segment': segment, 'venue': venue, 'event_id': event_id,
                'player_a': 'Jannik Sinner', 'player_b': 'Carlos Alcaraz', 'side': side,
                'start_time_utc': pd.Timestamp(START), 'status': 'open'}

    bets, n = add_bets(load_bets(str(tmp_path / 'bets.parquet')), pd.DataFrame([bet('polymarket itf 4.5-7.5h')]))
    assert n == 1
    bets, n = add_bets(bets, pd.DataFrame([bet('polymarket itf 0.5-1.5h'),
                                           bet('polymarket itf 4.5-7.5h', 'kalshi', 'KX-1', 'b')]))
    assert n == 1 and list(bets['segment']) == ['polymarket itf 4.5-7.5h', 'polymarket itf 0.5-1.5h']
    assert bets['bet_id'].is_unique


def test_paper_settle_from_matches_ret_rules():
    bets = pd.DataFrame([
        {'bet_id': 'x1', 'created_utc': pd.Timestamp(NOW), 'strategy': 'sharp', 'venue': 'tipico', 'event_id': 't1',
         'player_a': 'Jannik Sinner', 'player_b': 'Carlos Alcaraz', 'side': 'a', 'start_time_utc': pd.Timestamp(START),
         'odds': 2.0, 'net_odds': 1.894, 'stake': 10.0, 'stake_flat': 10.0, 'status': 'open'},
        {'bet_id': 'x2', 'created_utc': pd.Timestamp(NOW), 'strategy': 'modell', 'venue': 'polymarket', 'event_id': 'p9',
         'player_a': 'Jannik Sinner', 'player_b': 'Carlos Alcaraz', 'side': 'b', 'start_time_utc': pd.Timestamp(START),
         'odds': 2.0, 'net_odds': 2.0, 'stake': 10.0, 'stake_flat': 10.0, 'status': 'open'},
    ])
    bets = normalize(bets)
    matches = [{'winner_name': 'Jannik Sinner', 'loser_name': 'Carlos Alcaraz', 'match_date': START.strftime('%Y%m%d'),
                'tourney_date': START.strftime('%Y%m%d'), 'status': 'RET', 'source': 'espn'}]
    done, n = settle.settle(bets, pd.Timestamp(START + timedelta(hours=5)), {}, matches, None)
    assert n == 2
    assert done.loc[0, 'status'] == 'void' and done.loc[0, 'profit'] == 0  # Tipico: RET void
    assert done.loc[1, 'status'] == 'lost' and done.loc[1, 'profit'] == -10  # PM: Sieger nach Aufgabe zählt


def test_old_bets_file_stays_readable_and_unchanged(tmp_path):
    """bets.parquet von vor net_odds_flat: lesbar, die Spalte ist leer, alle anderen Werte bleiben wie sie sind."""
    path = str(tmp_path / 'bets.parquet')
    old = normalize(pd.DataFrame([{'bet_id': 'alt', 'created_utc': pd.Timestamp(NOW), 'strategy': 'modell',
                                   'venue': 'polymarket', 'event_id': 'p1', 'side': 'a', 'odds': 2.0, 'net_odds': 2.0,
                                   'stake': 5.0, 'stake_flat': 10.0, 'status': 'open'}]))
    old.drop(columns='net_odds_flat').to_parquet(path, index=False)
    bets = load_bets(path)
    assert np.isnan(bets['net_odds_flat'].iloc[0])
    columns = [c for c in old.columns if c != 'net_odds_flat']
    pd.testing.assert_frame_equal(bets[columns], old[columns], check_dtype=False)  # nur die Zeit-Auflösung ändert sich


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


def test_paper_report_go_no_go_per_segment():
    n = 320
    good = pd.DataFrame({'strategy': 'modell', 'segment': 'polymarket challenger 18-30h', 'venue': 'polymarket',
                         'level': 'challenger', 'status': ['won'] * n, 'stake': 10.0, 'stake_flat': 10.0,
                         'profit': 1.0, 'profit_flat': 1.0, 'clv': 0.03, 'clv_sharp': np.nan,
                         'settled_utc': pd.date_range('2026-01-01', periods=n, tz='UTC')})
    few = good.head(40).assign(segment='polymarket itf 0.5-1.5h', level='itf')
    without = good.head(10).assign(segment=None, strategy='sharp')  # sharp und Modell-Wetten vor den Segmenten
    bets = pd.concat([good, few, without], ignore_index=True)
    by = report.go_no_go_by_segment(bets, n_boot=200)
    assert list(by) == ['polymarket challenger 18-30h', 'polymarket itf 0.5-1.5h']
    assert by['polymarket challenger 18-30h']['go']
    assert not by['polymarket itf 0.5-1.5h']['go'] and not by['polymarket itf 0.5-1.5h']['checks'][0][1]
    table = report.summary(bets, n_boot=200)
    assert set(table.loc[table['gruppe'] == 'segment', 'wert']) == {'polymarket challenger 18-30h',
                                                                    'polymarket itf 0.5-1.5h', '?'}


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
                                   player_b='Carlos Alcaraz', price_type='probability', tour='atp',
                                   start_time_utc=ts + timedelta(hours=5), **BOOK)], root)

    monkeypatch.setattr(collect, 'collect_once', fake_collect)
    monkeypatch.setattr(run, 'now_utc', lambda: clock[0])
    monkeypatch.setattr(history, 'read_history', lambda venue: pd.DataFrame())
    monkeypatch.setattr(run, 'recent_matches', lambda now: ([], []))
    args = SimpleNamespace(collect=True, odds_api=False, softbooks=False, snapshots=str(tmp_path / 'snaps'), days=1,
                           bets=str(tmp_path / 'bets.parquet'), strategies=['modell'], model='elo2', threshold=0.05,
                           raw=True, sharp_threshold=0.02, history=False, segments='alle')
    bets, _, _ = run.run_once(args, log=lambda *a: None, prob=lambda row: 0.8, model_name='elo2')
    assert len(bets) == 1 and bets['player'].iloc[0] == 'Jannik Sinner'


CHEAP_A = dict(price_a=0.495, price_b=0.505, best_bid_a=0.49, best_ask_a=0.50, best_bid_b=0.50, best_ask_b=0.51)


def _collector_adapters(calls, ts=NOW):
    """Collector ohne Netz: Jeder Adapter merkt sich den Aufruf. Pinnacle sieht A fair bei ~ 0,54, der Ask ist 0,50."""
    def adapter(name, snapshot):
        def fetch(log=print):
            calls.append(name)
            return [snapshot]
        return fetch

    return {'polymarket': adapter('polymarket', snap('polymarket', ts, **CHEAP_A)),
            'kalshi': adapter('kalshi', snap('kalshi', ts, **CHEAP_A)),
            'odds_api': adapter('odds_api', snap('pinnacle', ts, price_a=1.8, price_b=2.1))}


def _sharp_round_args(tmp_path, root, **kw):
    from types import SimpleNamespace
    return SimpleNamespace(**{**dict(collect=False, odds_api=False, softbooks=False, snapshots=root, days=1,
                                     bets=str(tmp_path / 'bets.parquet'), strategies=['sharp'], model='elo2',
                                     threshold=0.05, raw=True, sharp_threshold=0.02, history=False, segments='auto'),
                              **kw})


def test_odds_api_run_also_queries_the_prediction_markets(tmp_path):
    """--venues odds_api fragt Polymarket und Kalshi mit ab, sonst läge nie einer ihrer Preise im Fenster von sharp. Die
    Odds API bleibt bei einer Abfrage je Lauf, und andere Läufe fragen sie gar nicht ab (Kontingent)."""
    from src.odds import collect
    from src.odds.schema import load_snapshots
    calls, root = [], str(tmp_path)
    adapters = _collector_adapters(calls)
    result = collect.collect_once(['odds_api'], root, adapters=adapters, log=lambda *a: None)
    assert calls == ['polymarket', 'kalshi', 'odds_api']
    assert result == {'polymarket': 1, 'kalshi': 1, 'odds_api': 1}
    assert sorted(sharp.align(load_snapshots(root=root))['venue']) == ['kalshi', 'polymarket']

    for venues, expected in ((['polymarket', 'kalshi'], ['polymarket', 'kalshi']),  # Paper-Runde
                             (['kalshi'], ['kalshi']),
                             (['polymarket', 'kalshi', 'odds_api'], ['polymarket', 'kalshi', 'odds_api']),  # --odds-api
                             (['odds_api', 'kalshi'], ['polymarket', 'odds_api', 'kalshi'])):
        calls.clear()
        collect.collect_once(venues, root, adapters=adapters, log=lambda *a: None)
        assert calls == expected


def test_failed_prediction_market_does_not_stop_the_odds_api_run(tmp_path):
    """Fällt Polymarket aus, fehlt nur dessen Preis: Die Credits der Odds API sind damit nicht verloren."""
    from src.odds import collect
    from src.odds.schema import load_snapshots

    def broken(log=print):
        raise RuntimeError("Polymarket down")

    calls, root = [], str(tmp_path)
    adapters = {**_collector_adapters(calls), 'polymarket': broken}
    result = collect.collect_once(['odds_api'], root, adapters=adapters, log=lambda *a: None)
    assert result['polymarket'].startswith('Fehler: RuntimeError')
    assert calls == ['kalshi', 'odds_api']
    assert list(sharp.align(load_snapshots(root=root))['venue']) == ['kalshi']


def test_sharp_compares_the_price_of_the_same_run_not_of_the_paper_round():
    """Zeitplan: Odds API um :05, Paper-Runde um :18. Deren Preis liegt 13 Minuten neben dem Sharp-Preis und zählt
    nicht, auch wenn er besser aussieht. Verglichen wird mit dem Preis, den der Odds-API-Lauf mit abgefragt hat."""
    fair = snap('pinnacle', NOW, price_a=1.8, price_b=2.1)
    same_run = snap('polymarket', NOW - timedelta(seconds=3), **CHEAP_A)
    paper_round = snap('polymarket', NOW + timedelta(minutes=13), price_a=0.395, price_b=0.605, best_bid_a=0.39,
                       best_ask_a=0.40, best_bid_b=0.60, best_ask_b=0.61)
    now = pd.Timestamp(NOW + timedelta(minutes=13, seconds=5))
    assert sharp.align(frame(fair, paper_round)).empty
    assert signals.sharp_signals(frame(fair, paper_round), now).empty
    out = signals.sharp_signals(frame(fair, same_run, paper_round), now)
    assert list(out['venue']) == ['polymarket']
    assert out['odds'].iloc[0] == pytest.approx(1 / 0.50)


def test_sharp_signals_ignore_venue_price_after_the_start():
    """Polymarket und Kalshi handeln im Match weiter: Liegt nur der Sharp-Preis vor dem Start, zählt der Preis nicht."""
    fair = snap('pinnacle', START - timedelta(minutes=2), price_a=1.8, price_b=2.1)
    before = snap('polymarket', START - timedelta(minutes=1), **CHEAP_A)
    live = snap('polymarket', START + timedelta(minutes=1), **CHEAP_A)
    assert sharp.signals(frame(fair, live), threshold=0.0).empty
    assert sharp.coverage(frame(fair, live)) == {}
    assert len(sharp.signals(frame(fair, before), threshold=0.0)) == 1
    assert sharp.coverage(frame(fair, before)) == {'polymarket': 1}


def test_paper_round_bets_on_the_prices_of_the_odds_api_run(tmp_path, monkeypatch):
    """Der geplante Ablauf ohne Änderung der Tasks: Der Odds-API-Lauf schreibt alle drei Venues, die nächste Paper-Runde
    wettet bei der besten davon. Das Log zeigt, womit sharp verglichen hat."""
    from src.odds import collect, history
    from src.paper import run
    root = str(tmp_path / 'snaps')
    collect.collect_once(['odds_api'], root, adapters=_collector_adapters([]), log=lambda *a: None)
    monkeypatch.setattr(run, 'now_utc', lambda: NOW + timedelta(minutes=13))
    monkeypatch.setattr(run, 'recent_matches', lambda now: ([], []))
    monkeypatch.setattr(history, 'read_history', lambda venue: pd.DataFrame())
    logs = []
    bets, _, _ = run.run_once(_sharp_round_args(tmp_path, root), log=logs.append)
    assert list(zip(bets['strategy'], bets['venue'], bets['side'])) == [('sharp', 'polymarket', 'a')]
    assert bets['odds'].iloc[0] == pytest.approx(2.0)
    assert '  Sharp: 2 Vergleiche vor dem Start, davon Polymarket 1, Kalshi 1' in logs


def test_paper_run_with_odds_api_bets_in_the_same_round(tmp_path, monkeypatch):
    """python -m src.paper.run --odds-api --strategies sharp: abfragen und wetten in einem Lauf, ohne Predictor."""
    from src.odds import collect, history
    from src.paper import run
    calls = []
    monkeypatch.setattr(collect, 'ADAPTERS', _collector_adapters(calls))
    monkeypatch.setattr(run, 'now_utc', lambda: NOW + timedelta(seconds=5))
    monkeypatch.setattr(run, 'recent_matches', lambda now: ([], []))
    monkeypatch.setattr(run, 'make_prob', lambda *a: pytest.fail("sharp braucht keinen Predictor"))
    monkeypatch.setattr(history, 'read_history', lambda venue: pd.DataFrame())
    args = _sharp_round_args(tmp_path, str(tmp_path / 'snaps'), collect=True, odds_api=True)
    bets, _, _ = run.run_once(args, log=lambda *a: None)
    assert calls == ['polymarket', 'kalshi', 'odds_api']
    assert list(bets['venue']) == ['polymarket']


def test_sharp_log_tells_when_prediction_markets_are_missing():
    """Ohne Vergleich bei Polymarket und Kalshi wettet sharp nur bei Buchmachern: Das steht in jeder Runde im Log."""
    now = pd.Timestamp(NOW + timedelta(minutes=1))
    logs = []
    signals.sharp_signals(frame(snap('polymarket', NOW, **BOOK)), now, log=logs.append)
    assert logs == ['  Sharp: keine Sharp-Preise aus den letzten 90 Minuten']
    logs.clear()
    only_books = frame(snap('pinnacle', NOW, price_a=1.8, price_b=2.1), snap('tipico', NOW, price_a=1.9, price_b=1.9))
    signals.sharp_signals(only_books, now, log=logs.append)
    assert logs == ['  Sharp: 1 Vergleiche vor dem Start, davon Polymarket 0, Kalshi 0']


def test_open_results_queries_only_due_unknown_markets(monkeypatch):
    """Abrechnung fragt nur fällige offene PM/Kalshi-Wetten gezielt ab, die die Historie noch nicht kennt."""
    from src.odds import kalshi, polymarket
    from src.paper import run
    calls = []
    monkeypatch.setattr(polymarket, 'fetch_resolution',
                        lambda client, e, a: calls.append(('polymarket', e)) or {'pm-1': 'b', 'pm-2': None}[e])
    monkeypatch.setattr(kalshi, 'fetch_resolution', lambda client, e, a: calls.append(('kalshi', e, a)) or 'void')

    def bet(venue, event_id, start, strategy='modell', segment=None):
        return {'bet_id': f'{strategy}|{venue}|{event_id}|a|{segment}', 'created_utc': pd.Timestamp(NOW),
                'strategy': strategy, 'segment': segment, 'venue': venue, 'event_id': event_id,
                'player_a': 'Jannik Sinner', 'player_b': 'Carlos Alcaraz', 'side': 'a',
                'start_time_utc': pd.Timestamp(start), 'odds': 2.0, 'net_odds': 2.0, 'stake': 10.0, 'stake_flat': 10.0,
                'status': 'open'}

    bets = normalize(pd.DataFrame([
        # je Segment und Strategie eine Wette auf denselben ITF-Markt: einmal abfragen, alle abrechnen
        bet('polymarket', 'pm-1', START, segment='polymarket itf 18-30h'),
        bet('polymarket', 'pm-1', START, segment='polymarket itf 0.5-1.5h'),
        bet('polymarket', 'pm-1', START, strategy='sharp'),
        bet('polymarket', 'pm-2', START),                  # Markt noch nicht aufgelöst
        bet('polymarket', 'pm-3', START),                  # steht schon in der Historie
        bet('kalshi', 'KXITFMATCH-1', START),
        bet('kalshi', 'KXITFWMATCH-2', START + timedelta(days=1)),  # noch nicht fällig
        bet('tipico', 't-1', START),                       # Buchmacher: nur über die Matchdaten
    ]))
    history = {'polymarket': pd.DataFrame({'event_id': ['pm-3'], 'winner': ['a']}),
               'kalshi': pd.DataFrame(columns=['event_id', 'winner'])}
    now = pd.Timestamp(START + timedelta(hours=4))
    found = run.open_results(bets, now, history, log=lambda *a: None)
    assert sorted(calls) == [('kalshi', 'KXITFMATCH-1', 'Jannik Sinner'), ('polymarket', 'pm-1'), ('polymarket', 'pm-2')]
    assert found['polymarket'].to_dict('records') == [{'event_id': 'pm-1', 'winner': 'b'}]
    assert found['kalshi'].to_dict('records') == [{'event_id': 'KXITFMATCH-1', 'winner': 'void'}]

    merged = {v: pd.concat([history[v], found[v]], ignore_index=True) for v in history}
    done, n = settle.settle(bets, now, merged, [], None)
    assert n == 5
    assert list(done['status']) == ['lost', 'lost', 'lost', 'open', 'won', 'void', 'open', 'open']


def test_recent_matches_cached_for_data_matches(tmp_path, monkeypatch):
    import csv

    from src import cache
    from src.paper import run
    from src.sources import common
    path = tmp_path / 'matches.csv'

    def write(rows):
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    rows = [{'winner_name': 'A', 'loser_name': 'B', 'match_date': '20261020', 'tourney_date': '20261019',
             'status': 'RET', 'source': 'espn', 'tourney_name': 'Paris', 'level': 'main'}]
    write(rows)
    monkeypatch.setattr(common, 'MATCHES_PATH', str(path))
    monkeypatch.setattr(cache, 'CACHE_DIR', str(tmp_path / 'cache'))
    first = run.recent_matches(NOW)
    assert [m['winner_name'] for m in first[0]] == ['A']
    assert (tmp_path / 'cache' / 'recent_matches.pkl').exists()

    with monkeypatch.context() as m:
        m.setattr(csv, 'DictReader', lambda *a, **k: pytest.fail("data/matches.csv trotz Cache gelesen"))
        assert run.recent_matches(NOW) == first
    write(rows + [{**rows[0], 'winner_name': 'C'}])  # neue Daten: neu lesen
    assert [m['winner_name'] for m in run.recent_matches(NOW)[0]] == ['A', 'C']


def test_loop_survives_failed_round(monkeypatch):
    """Im Dauerbetrieb kostet ein Fehler nur die Runde, ohne --loop bricht er den Lauf ab (Exit-Code)."""
    from src.paper import run
    rounds = []

    def fake_run_once(args, log=print, prob=None, model_name=None):
        rounds.append(1)
        if len(rounds) == 1:
            raise RuntimeError("Netz weg")
        raise KeyboardInterrupt

    monkeypatch.setattr(run, 'run_once', fake_run_once)
    monkeypatch.setattr(run.time, 'sleep', lambda s: None)
    run.main(['--loop', '30', '--strategies', 'sharp'])
    assert len(rounds) == 2
    rounds.clear()
    with pytest.raises(RuntimeError):
        run.main(['--strategies', 'sharp'])


def test_run_once_bets_only_in_segments_of_the_timing_backtest(tmp_path, monkeypatch):
    """Standard: die Segmente mit Kandidat/GO aus data/eval/timing.parquet für das genutzte Modell. Das fehlende Level
    der Polymarket-Märkte kommt vorher aus fill_levels (Turniername und Kalenderwoche)."""
    from types import SimpleNamespace
    from src.evaluation import timing
    from src.odds import history
    from src.odds.schema import append_snapshots
    from src.paper import run
    table = str(tmp_path / 'timing.parquet')
    pd.DataFrame([('polymarket', 'challenger', '24h', 'elo2', 'Kandidat'), ('polymarket', 'itf', '1h', 'elo2', 'GO'),
                  ('polymarket', 'itf', '6h', 'gbdt', 'Kandidat'), ('polymarket', 'main', '24h', 'elo2', 'NO-GO')],
                 columns=['venue', 'level', 'point', 'model', 'decision']).to_parquet(table)
    monkeypatch.setattr(timing, 'TABLE_PATH', table)
    reference = [{'tour': 'atp', 'tourney_name': 'Charlottesville Challenger', 'tourney_date': f'{NOW:%Y%m%d}',
                  'level': 'challenger'}]
    monkeypatch.setattr(run, 'recent_matches', lambda now: ([], reference))
    monkeypatch.setattr(run, 'now_utc', lambda: NOW)
    monkeypatch.setattr(history, 'read_history', lambda venue: pd.DataFrame())
    cases = [('ch-24', 'Charlottesville', None, 24), ('ch-5', 'Charlottesville', None, 5),
             ('main-24', 'ATP Paris', None, 24), ('itf-1', 'M15 Monastir', 'itf', 1), ('itf-6', 'M15 Monastir', 'itf', 6)]
    root = str(tmp_path / 'snaps')
    append_snapshots([Snapshot(ts_utc=NOW - timedelta(minutes=5), venue='polymarket', event_id=e, player_a=a,
                               player_b=b, price_type='probability', tour='atp', level=level, tourney=tourney,
                               start_time_utc=NOW + timedelta(hours=hours), **BOOK)
                      for (e, tourney, level, hours), (a, b) in zip(cases, PLAYERS)], root)
    args = SimpleNamespace(collect=False, odds_api=False, softbooks=False, snapshots=root, days=1,
                           bets=str(tmp_path / 'bets.parquet'), strategies=['modell'], model='elo2', threshold=0.05,
                           raw=True, sharp_threshold=0.02, history=False, segments='auto')
    bets, _, _ = run.run_once(args, log=lambda *a: None, prob=lambda row: 0.8, model_name='elo2')
    # itf-6 fällt weg: 6 h ist nur für gbdt ein Kandidat, gewettet wird mit elo2
    assert sorted(zip(bets['event_id'], bets['level'], bets['segment'])) == [
        ('ch-24', 'challenger', 'polymarket challenger 18-30h'), ('itf-1', 'itf', 'polymarket itf 0.5-1.5h')]
