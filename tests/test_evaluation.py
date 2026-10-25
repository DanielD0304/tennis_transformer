"""
Tests für src/evaluation mit synthetischen Matches (kein Netzwerk, kein Training).
Spieler haben eine feste Stärke, Quoten entstehen aus der wahren Wahrscheinlichkeit plus Marge.
"""
import math
import random

import numpy as np
import pandas as pd
import pytest

from src.evaluation import backtest, betting, metrics
from src.evaluation.predictors import (BlendPredictor, Predictor, RankPredictor, make_predictors,
                                       prepare_matches, remove_margin)
from src.evaluation.venues import VENUES, Venue, ceil_to_cent, kalshi_fee


def synthetic_matches(years=(2019, 2020, 2021), per_month=40, n_players=40, seed=0, margin=1.05):
    rng = random.Random(seed)
    skill = {f"Player {i:02d}": rng.gauss(0, 1) for i in range(n_players)}
    rank_of = {p: i + 1 for i, p in enumerate(sorted(skill, key=skill.get, reverse=True))}
    rows = []
    for year in years:
        for month in range(1, 13):
            for i in range(per_month):
                a, b = rng.sample(list(skill), 2)
                p_a = 1 / (1 + math.exp(-(skill[a] - skill[b])))
                w, l = (a, b) if rng.random() < p_a else (b, a)
                p_w = min(max(p_a if w == a else 1 - p_a, 0.08), 0.92)  # Quoten > 1 auch mit Marge
                odds_w, odds_l = 1 / (p_w * margin), 1 / ((1 - p_w) * margin)
                status = 'completed'
                r = rng.random()
                if r < 0.03:
                    status = 'RET'
                elif r < 0.04:
                    status = 'W/O'
                day = f"{year}{month:02d}{1 + i % 28:02d}"
                rows.append({
                    'tourney_id': f"{year}-{month}", 'tourney_name': f"Open {month}", 'surface': 'Hard',
                    'tourney_date': day, 'match_date': day, 'match_num': i, 'round': 'R32',
                    'winner_name': w, 'loser_name': l, 'winner_rank': rank_of[w], 'loser_rank': rank_of[l],
                    'score': 'W/O' if status == 'W/O' else '6-4 6-4', 'status': status,
                    'B365W': odds_w * 0.98, 'B365L': odds_l * 0.98, 'PSW': odds_w, 'PSL': odds_l,
                    'AvgW': odds_w * 0.99, 'AvgL': odds_l * 0.99,
                })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def matches():
    return prepare_matches(synthetic_matches())


@pytest.fixture(scope="module")
def predictions(matches):
    preds = make_predictors(['market', 'rank', 'elo', 'blend'])
    return backtest.walk_forward(matches, preds, 2020, 2021, log=lambda *a: None)


# ------------------------------------------------------------ Quoten & A/B

def test_remove_margin_sums_to_one():
    oa, ob = np.array([1.5, 1.9, 1.05]), np.array([2.6, 1.95, 9.0])
    pa, pb = remove_margin(oa, ob), remove_margin(ob, oa)
    np.testing.assert_allclose(pa + pb, 1.0)
    assert pa[0] == pytest.approx((1 / 1.5) / (1 / 1.5 + 1 / 2.6))


def test_orientation_consistent(matches):
    # A ist mal Sieger, mal Verlierer
    assert 0.3 < matches['label'].mean() < 0.7
    won = matches['label'] == 1
    np.testing.assert_array_equal(np.where(won, matches['winner_name'], matches['loser_name']), matches['player_a'])
    np.testing.assert_allclose(np.where(won, matches['PSW'], matches['PSL']), matches['ps_a'])
    np.testing.assert_allclose(np.where(won, matches['B365L'], matches['B365W']), matches['b365_b'])
    np.testing.assert_array_equal(np.where(won, matches['winner_rank'], matches['loser_rank']), matches['rank_a'])


def test_orientation_swap_keeps_odds_and_label():
    """Gleiches Match mit vertauschtem Seed: Label und Quoten tauschen gemeinsam."""
    raw = synthetic_matches(years=(2020,), per_month=5)
    m1, m2 = prepare_matches(raw, seed=1), prepare_matches(raw, seed=2)
    m2 = m2.set_index('match_id').loc[m1['match_id']].reset_index()
    flipped = m1['player_a'] != m2['player_a']
    assert flipped.any() and (~flipped).any()
    f = flipped.to_numpy()
    np.testing.assert_array_equal(m1['label'][f], 1 - m2['label'][f])
    np.testing.assert_allclose(m1['ps_a'][f], m2['ps_b'][f])
    np.testing.assert_allclose(m1['avg_b'][f], m2['avg_a'][f])
    np.testing.assert_allclose(m1['fair_ps'][f], 1 - m2['fair_ps'][f])
    # Pinnacle-Favorit gewinnt aus beiden Sichten gleich oft
    assert ((m1['fair_ps'] > 0.5) == (m1['label'] == 1)).mean() == pytest.approx(
        ((m2['fair_ps'] > 0.5) == (m2['label'] == 1)).mean())


def test_market_fallback_to_avg():
    raw = synthetic_matches(years=(2020,), per_month=5)
    raw.loc[:9, ['PSW', 'PSL']] = np.nan
    m = prepare_matches(raw, with_elo=False)
    assert (m['market_source'] == 'avg').sum() == 10
    p = make_predictors(['market'])[0].predict(m)
    assert np.isfinite(p).all()


def test_market_fallback_chain_pinnacle_betfair_avg():
    raw = synthetic_matches(years=(2020,), per_month=5).head(30).copy()
    raw['BFEW'], raw['BFEL'] = raw['PSW'] * 1.04, raw['PSL'] * 1.04
    raw.loc[10:19, ['PSW', 'PSL']] = np.nan                      # nur Betfair + Avg
    raw.loc[20:, ['PSW', 'PSL', 'BFEW', 'BFEL']] = np.nan         # nur Avg
    raw.loc[20:, ['AvgW', 'AvgL']] = [[1.4, 3.0]] * 10
    m = prepare_matches(raw, with_elo=False).set_index('match_id')
    src = m['market_source']
    assert (src == 'ps').sum() == 10 and (src == 'bfe').sum() == 10 and (src == 'avg').sum() == 10
    p = pd.Series(make_predictors(['market'])[0].predict(m), index=m.index)
    for book in ('ps', 'bfe', 'avg'):
        rows = m[src == book]
        np.testing.assert_allclose(p[rows.index], remove_margin(rows[f'{book}_a'], rows[f'{book}_b']))
    # Betfair-Quoten werden mit A/B getauscht wie alle anderen
    won = m['label'] == 1
    np.testing.assert_allclose(np.where(won, m['BFEW'], m['BFEL'])[src == 'bfe'], m.loc[src == 'bfe', 'bfe_a'])
    # Vergleichs-Stichprobe: Pinnacle und Betfair sind scharf, Avg nicht
    preds = m.reset_index().assign(p_market=p.to_numpy())
    assert len(metrics.common_sample(preds, ['market'])) == int((m['status'] != 'W/O')[src != 'avg'].sum())


# ----------------------------------------------------------------- Metriken

def test_metrics_by_hand():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.8, 0.3, 0.6, 0.5])
    expected_ll = -(math.log(0.8) + math.log(0.7) + math.log(0.6) + math.log(0.5)) / 4
    assert metrics.log_loss(y, p) == pytest.approx(expected_ll)
    assert metrics.brier(y, p) == pytest.approx((0.04 + 0.09 + 0.16 + 0.25) / 4)
    # 0.5 zählt halb
    assert metrics.accuracy(y, p) == pytest.approx(3.5 / 4)


def test_ece_by_hand():
    y = np.array([1, 1, 0, 0, 1])
    p = np.array([0.9, 0.9, 0.9, 0.1, 0.1])
    # Bin 0.9: 3 Matches, mean_p 0.9, Siegquote 2/3; Bin 0.1: 2 Matches, mean_p 0.1, Siegquote 0.5
    expected = (3 * abs(2 / 3 - 0.9) + 2 * abs(0.5 - 0.1)) / 5
    assert metrics.ece(y, p) == pytest.approx(expected)
    table = metrics.calibration_table(y, p)
    assert table['count'].sum() == 5
    assert metrics.ece(np.array([1, 0]), np.array([0.5, 0.5])) == pytest.approx(0.0)


def test_paired_bootstrap_deterministic():
    rng = np.random.default_rng(1)
    a, b = rng.random(500), rng.random(500) + 0.05
    r1 = metrics.paired_bootstrap(a, b, n_boot=500, seed=7)
    r2 = metrics.paired_bootstrap(a, b, n_boot=500, seed=7)
    assert r1 == r2
    assert r1['ci_low'] < r1['mean_diff'] < r1['ci_high'] < 0
    assert r1['p_value'] < 0.05
    dm = metrics.diebold_mariano(a, b)
    assert dm['stat'] < 0 and dm['p_value'] < 0.05


def test_bootstrap_yield_deterministic():
    rng = np.random.default_rng(0)
    profits, stakes = np.where(rng.random(200) < 0.5, 9.0, -10.0), np.full(200, 10.0)
    assert betting.bootstrap_yield(profits, stakes, seed=3) == betting.bootstrap_yield(profits, stakes, seed=3)
    assert betting.bootstrap_yield(profits, stakes, seed=3) != betting.bootstrap_yield(profits, stakes, seed=4)


# --------------------------------------------------------------- Walk-forward

class SpyPredictor(Predictor):
    name = "spy"

    def __init__(self):
        super().__init__()
        self.calls = []

    def fit(self, history):
        super().fit(history)
        self._last_max = history['date'].max()

    def predict(self, window):
        self._check_no_leakage(window)
        self.calls.append((self._last_max, window['date'].min(), window['date'].max()))
        return np.full(len(window), 0.5)


def test_walk_forward_no_leakage(matches):
    spy = SpyPredictor()
    backtest.walk_forward(matches, [spy], 2020, 2021, log=lambda *a: None)
    assert len(spy.calls) == 24  # monatlich
    for fit_max, win_min, _ in spy.calls:
        assert fit_max < win_min
    # Fenster grenzen lückenlos aneinander
    assert all(spy.calls[i][2] < spy.calls[i + 1][1] for i in range(len(spy.calls) - 1))


def test_leakage_assert_fires(matches):
    rank = RankPredictor()
    rank.fit(matches[matches['year'] <= 2020])
    with pytest.raises(AssertionError):
        rank.predict(matches[matches['year'] == 2020])


def test_walk_forward_yearly_windows():
    assert len(backtest.windows(2020, 2021, 'year')) == 2
    assert len(backtest.windows(2020, 2020, 'month')) == 12


def test_predictions_frame(predictions):
    for col in ['match_id', 'date', 'label', 'status', 'p_market', 'p_rank', 'p_elo', 'p_blend', 'ps_a', 'b365_b']:
        assert col in predictions.columns
    assert predictions['date'].dt.year.between(2020, 2021).all()
    assert predictions[['p_market', 'p_rank', 'p_elo', 'p_blend']].notna().all().all()
    # Synthetischer Markt kennt die wahre Stärke: muss Rang und Elo schlagen
    table = metrics.metric_table(predictions, ['market', 'rank', 'elo', 'blend'])
    total = table[table['year'] == 'Gesamt'].set_index('model')['log_loss']
    assert total['market'] < total['rank'] and total['market'] < total['elo']
    assert total['blend'] <= total['elo']


def test_save_parquet(predictions, tmp_path):
    path = backtest.save(predictions, tmp_path / "eval" / "predictions.parquet")
    loaded = pd.read_parquet(path)
    assert len(loaded) == len(predictions)


def test_blend_falls_back_to_market(matches):
    blend = BlendPredictor('elo')
    window = matches.head(20).assign(p_market=0.6, p_elo=0.3)
    blend.fit(window.iloc[:0])
    np.testing.assert_allclose(blend.predict(window), 0.6)


def test_unknown_model():
    with pytest.raises(ValueError):
        make_predictors(['market', 'xgboost'])
    with pytest.raises(ValueError):
        make_predictors(['market', 'blend_transformer'])
    with pytest.raises(ValueError):
        make_predictors(['market', 'stack'])  # ohne Basismodell
    with pytest.raises(ValueError):
        make_predictors(['market', 'elo2', 'stack_tf'])  # Transformer fehlt


def test_report_runs(predictions):
    grid = backtest.report(predictions, ['market', 'rank', 'elo', 'blend'], n_boot=100, log=lambda *a: None)
    # Standard: Pinnacle, Betfair (echte Kurse, in den synthetischen Daten keine) und Bet365 mit Steuer
    assert set(grid['venue']) == {'pinnacle', 'betfair', 'bet365_de'}
    grid = backtest.report(predictions, ['market', 'rank', 'elo', 'blend'], n_boot=100, log=lambda *a: None,
                           venues=list(VENUES.values()))
    assert set(grid['venue']) >= {'pinnacle', 'bet365_de', 'kalshi (Proxy)', 'betfair_proxy (Proxy)'}


# ------------------------------------------------------------------- Venues

def _one(status='completed', label=1, ps=(2.0, 2.0), b365=(2.0, 2.0), bfe=(2.1, 1.95)):
    return pd.DataFrame([{
        'match_id': 'x', 'date': pd.Timestamp('2021-01-01'), 'label': label, 'status': status,
        'rank_a': 1, 'rank_b': 2, 'ps_a': ps[0], 'ps_b': ps[1], 'b365_a': b365[0], 'b365_b': b365[1],
        'bfe_a': bfe[0], 'bfe_b': bfe[1],
        'avg_a': b365[0], 'avg_b': b365[1], 'fair_ps': remove_margin(ps[0], ps[1]), 'p_m': 0.9,
    }])


def test_german_tax():
    venue = VENUES['bet365_de']
    assert venue.net_odds_from_gross(2.0) == pytest.approx(2.0 * (1 - 0.053))
    summary, _ = betting.simulate(_one(), betting.Favorite('m', 1.0), venue, betting.Flat(10))
    assert summary['profit'] == pytest.approx(10 * (2.0 * 0.947 - 1))
    summary, _ = betting.simulate(_one(label=0), betting.Favorite('m', 1.0), venue, betting.Flat(10))
    assert summary['profit'] == pytest.approx(-10)
    custom = venue.with_(stake_tax=0.0)
    assert custom.net_odds_from_gross(2.0) == pytest.approx(2.0)


def test_betfair_commission():
    venue = VENUES['betfair_proxy']
    # Preis = faire Pinnacle-Wahrscheinlichkeit 0.5 -> Quote 2.0; 5 % auf den Nettogewinn
    summary, _ = betting.simulate(_one(), betting.Favorite('m', 1.0), venue, betting.Flat(10))
    assert summary['profit'] == pytest.approx(10 * 1.0 * 0.95)
    assert venue.with_(commission=0.02).net_odds_from_gross(3.0) == pytest.approx(1 + 2 * 0.98)
    assert venue.proxy and venue.label == 'betfair_proxy (Proxy)'


def test_betfair_real_odds():
    venue = VENUES['betfair']
    assert not venue.proxy and venue.price == 'bfe'
    # Echter Exchange-Kurs 2.1 für A, 5 % Kommission nur auf den Gewinn
    summary, _ = betting.simulate(_one(), betting.Favorite('m', 1.0), venue, betting.Flat(10))
    assert summary['profit'] == pytest.approx(10 * 1.1 * 0.95)
    lost, _ = betting.simulate(_one(label=0), betting.Favorite('m', 1.0), venue, betting.Flat(10))
    assert lost['profit'] == pytest.approx(-10)
    # Ohne BFE-Kurs (vor August 2025) keine Wette
    none, _ = betting.simulate(_one(bfe=(np.nan, np.nan)), betting.Favorite('m', 1.0), venue, betting.Flat(10))
    assert none['n_bets'] == 0


def test_kalshi_fee():
    assert kalshi_fee(0.5) == pytest.approx(0.02)  # 0.07*0.25 = 0.0175 -> 0.02
    assert kalshi_fee(0.9) == pytest.approx(0.01)  # 0.0063 -> 0.01
    assert kalshi_fee(0.5, contracts=100) == pytest.approx(1.75)
    assert ceil_to_cent(0.07 * 100 * 0.5 * 0.5) == pytest.approx(1.75)  # kein Float-Aufrunden auf 1.76
    venue = VENUES['kalshi']
    assert venue.net_odds_from_gross(2.0) == pytest.approx(1 / 0.52)


def test_polymarket_spread_and_fee():
    venue = VENUES['polymarket'].with_(spread=0.02, taker_fee=0.01)
    ga, gb = venue.gross_odds(_one())
    assert ga[0] == pytest.approx(1 / 0.51) and gb[0] == pytest.approx(1 / 0.51)
    assert venue.net_odds_from_gross(2.0) == pytest.approx(2.0 * 0.99)


def test_ret_void_vs_settled_and_wo_always_void():
    settled = Venue('s', price='ps', ret_rule='settled')
    void = Venue('v', price='ps', ret_rule='void')
    strat = betting.Favorite('m', 1.0)
    s, _ = betting.simulate(_one(status='RET'), strat, settled, betting.Flat(10))
    assert s['n_bets'] == 1 and s['profit'] == pytest.approx(10)
    s, bets = betting.simulate(_one(status='RET'), strat, void, betting.Flat(10))
    assert s['n_bets'] == 0 and s['n_void'] == 1 and s['profit'] == 0 and bets['void'].all()
    for venue in (settled, void):
        s, _ = betting.simulate(_one(status='W/O'), strat, venue, betting.Flat(10))
        assert s['n_void'] == 1 and s['profit'] == 0
    with pytest.raises(ValueError):
        Venue('x', price='ps', ret_rule='maybe')


# ------------------------------------------------------------ Staking & Strategien

def test_kelly_cap():
    kelly = betting.Kelly(fraction=0.5, cap=0.02)
    # Voll-Kelly bei p=0.9, Quote 2.0: 0.8 -> halb 0.4 -> gedeckelt auf 2 %
    assert kelly.amount(1000, 0.9, 2.0) == pytest.approx(20)
    # Kleiner Edge: p=0.51, Quote 2.0 -> Voll-Kelly 0.02, halb 1 % der Bankroll
    assert kelly.amount(1000, 0.51, 2.0) == pytest.approx(10)
    assert kelly.amount(1000, 0.4, 2.0) == 0
    venue = VENUES['pinnacle'].with_(max_stake=5.0)
    s, _ = betting.simulate(_one(), betting.Favorite('m', 1.0), venue, kelly)
    assert s['staked'] == pytest.approx(5.0)


def test_strategies_choose_sides():
    p = np.array([0.7, 0.3, 0.55, 0.5])
    ga, gb = np.array([1.2, 3.0, 1.5, 2.0]), np.array([4.0, 1.4, 2.5, 2.0])
    ra, rb = np.array([1, 5, 3, 4]), np.array([2, 1, 3, 9])
    fav = betting.Favorite('m', 1.30).choose(p, ga, gb, ga, gb, ra, rb)
    np.testing.assert_array_equal(fav, [0, -1, 1, 0])
    val = betting.Value('m', 0.0).choose(p, ga, gb, ga, gb, ra, rb)
    # EV: (-0.16 | +0.2), (-0.1 | -0.02), (-0.175 | +0.125), (0 | 0)
    np.testing.assert_array_equal(val, [-1, 0, -1, 0])
    band = betting.ConfidenceBand('m', 0.65, 0.80).choose(p, ga, gb, ga, gb, ra, rb)
    np.testing.assert_array_equal(band, [1, -1, 0, 0])
    rank = betting.BetterRank().choose(p, ga, gb, ga, gb, ra, rb)
    np.testing.assert_array_equal(rank, [1, -1, 0, 1])


# --------------------------------------------------------------- Transformer

def test_transformer_adapter_symmetric(tmp_path):
    """Zufällig initialisiertes Modell: p liegt in (0, 1) und passt zur A/B-Orientierung."""
    import torch

    from src.config import TrainingConfig
    from src.model import Model
    from src.evaluation.predictors import TransformerPredictor

    from src.model import save_checkpoint

    cfg = TrainingConfig(train_years_end=2018, val_year=2019)
    torch.manual_seed(0)
    model_path = tmp_path / "best_model.pt"
    save_checkpoint(Model(cfg.d_model, cfg.num_heads, cfg.num_layers, cfg.input_dim, cfg.max_len,
                          cfg.output_dim, cfg.extra_dim), str(model_path), temperature=1.7, extra_dim=cfg.extra_dim)

    raw = synthetic_matches(years=(2019, 2020), per_month=30)
    results = []
    for seed in (1, 2):
        m = prepare_matches(raw, seed=seed)
        tf = TransformerPredictor(cfg, str(model_path))
        tf.prepare(m)
        window = m[m['year'] == 2020]
        with pytest.raises(AssertionError):
            tf.predict(m[m['year'] == 2019])  # Trainings- und Validierungsjahre sind tabu
        assert tf.temperature == pytest.approx(1.7)
        results.append(window.assign(p=tf.predict(window)).set_index('match_id'))
    a, b = results[0], results[1].loc[results[0].index]
    ok = a['p'].notna()
    assert ok.sum() > 100
    assert ((a['p'][ok] > 0) & (a['p'][ok] < 1)).all()
    flipped = a['player_a'] != b['player_a']
    np.testing.assert_allclose(a['p'][ok & flipped], 1 - b['p'][ok & flipped], atol=1e-6)
    np.testing.assert_allclose(a['p'][ok & ~flipped], b['p'][ok & ~flipped], atol=1e-6)
