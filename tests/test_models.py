"""
Tests für die Modell-Leiter (Phase 3): Elo2, LightGBM-Predictor, Kalibrierung/Temperature Scaling,
Stacking mit dem Markt. Synthetische Daten, kein Netzwerk, keine trainierten Modelle.
"""
import math

import numpy as np
import pandas as pd
import pytest

from src.elo import Elo2, Elo2Params, best_of_5, elo2_features, expected, k_538
from src.evaluation import backtest, metrics
from src.evaluation.predictors import make_predictors, prepare_matches, sigmoid
from src.models.calibration import Isotonic, Platt, apply_temperature, fit_temperature
from tests.test_evaluation import synthetic_matches


# ------------------------------------------------------------------- Elo2

def test_k_factor_538():
    assert k_538(0) == pytest.approx(250 / 5 ** 0.4)
    assert k_538(100, c=250, o=5, s=0.4) == pytest.approx(250 / 105 ** 0.4)
    # neue Spieler bewegen sich schneller als etablierte
    assert k_538(0) > k_538(10) > k_538(500)


def test_best_of_5_stretches_probabilities():
    p3 = np.array([0.5, 0.6, 0.8, 0.3])
    p5 = best_of_5(p3)
    assert p5[0] == pytest.approx(0.5)
    assert p5[1] > 0.6 and p5[2] > 0.8 and p5[3] < 0.3
    # Satz-Wahrscheinlichkeit 0.6 -> p3 = 0.648, p5 = 0.68256
    assert best_of_5(0.6 ** 2 * (3 - 2 * 0.6)) == pytest.approx(0.68256, abs=1e-6)


def test_elo2_update_and_surface_mix():
    par = Elo2Params(surface_weight=0.25)
    elo = Elo2(par)
    elo.update('atp', 'A', 'B', 'Clay')
    k0 = k_538(0)
    overall, surf, mix, n = elo.rating('atp', 'A', 'Clay')
    assert overall == pytest.approx(1500 + k0 * 0.5)
    assert surf == pytest.approx(1500 + k0 * 0.5)
    assert n == 1
    # Auf Hartplatz hat A noch kein Untergrund-Elo: Mix = 0.75 * Gesamt + 0.25 * Start
    overall_h, surf_h, mix_h, _ = elo.rating('atp', 'A', 'Hard')
    assert surf_h == 1500 and mix_h == pytest.approx(0.75 * overall + 0.25 * 1500)
    assert elo.prob('atp', 'A', 'B', 'Clay') > elo.prob('atp', 'A', 'B', 'Hard') > 0.5
    assert elo.prob('atp', 'A', 'B', 'Clay') + elo.prob('atp', 'B', 'A', 'Clay') == pytest.approx(1.0)
    # zweites Match: kleineres K für beide
    elo.update('atp', 'A', 'B', 'Clay')
    assert elo.rating('atp', 'A', 'Clay')[0] - overall < k0 * 0.5


def test_elo2_ret_weight_and_slam():
    full = Elo2(Elo2Params(ret_weight=1.0, slam_k=1.2))
    half = Elo2(Elo2Params(ret_weight=0.5, slam_k=1.2))
    for elo in (full, half):
        elo.update('atp', 'A', 'B', 'Hard', retired=True)
    assert half.rating('atp', 'A', 'Hard')[0] - 1500 == pytest.approx((full.rating('atp', 'A', 'Hard')[0] - 1500) / 2)
    slam = Elo2(Elo2Params(slam_k=1.2))
    slam.update('atp', 'A', 'B', 'Grass', slam=True)
    assert slam.rating('atp', 'A', 'Grass')[0] - 1500 == pytest.approx(1.2 * k_538(0) * 0.5)


def test_elo2_tours_apart():
    """Gleicher Name auf ATP und WTA: getrennte Ratings, eigene Parameter pro Tour."""
    elo = Elo2({'atp': Elo2Params(c=250), 'wta': Elo2Params(c=100)})
    elo.update('atp', 'Murray', 'X', 'Hard')
    assert elo.rating('wta', 'Murray', 'Hard')[0] == 1500
    elo.update('wta', 'Murray', 'Y', 'Hard')
    assert elo.rating('atp', 'Murray', 'Hard')[0] - 1500 == pytest.approx(k_538(0, c=250) * 0.5)
    assert elo.rating('wta', 'Murray', 'Hard')[0] - 1500 == pytest.approx(k_538(0, c=100) * 0.5)


def test_elo2_features_before_match_and_walkover():
    df = pd.DataFrame([
        {'tour': 'atp', 'tourney_date': '20200101', 'match_num': 1, 'round': 'R32', 'surface': 'Hard',
         'winner_name': 'A', 'loser_name': 'B', 'score': '6-4 6-4', 'status': 'completed'},
        {'tour': 'atp', 'tourney_date': '20200101', 'match_num': 2, 'round': 'R16', 'surface': 'Hard',
         'winner_name': 'A', 'loser_name': 'C', 'score': 'W/O', 'status': 'W/O'},
        {'tour': 'atp', 'tourney_date': '20200101', 'match_num': 3, 'round': 'QF', 'surface': 'Hard',
         'winner_name': 'B', 'loser_name': 'A', 'score': '6-4 6-4', 'status': 'completed'},
    ]).iloc[::-1]  # Reihenfolge der Zeilen egal, gerechnet wird chronologisch
    f = elo2_features(df, Elo2Params())
    assert list(f.index) == list(df.index)
    first, wo, third = f.loc[0], f.loc[1], f.loc[2]
    assert first['winner_elo2'] == first['loser_elo2'] == 1500 and first['elo2_p_winner'] == 0.5
    # W/O ändert nichts: A hat nach Match 1 und dem W/O nur ein gespieltes Match
    assert third['loser_elo2'] == wo['winner_elo2'] and third['loser_elo2_n'] == 1
    assert third['elo2_p_winner'] == pytest.approx(expected(third['winner_elo2_mix'], third['loser_elo2_mix']))


def test_elo2_predictor_in_backtest():
    m = prepare_matches(synthetic_matches(years=(2019, 2020), per_month=30))
    preds = backtest.walk_forward(m, make_predictors(['market', 'elo', 'elo2']), 2020, 2020, log=lambda *a: None)
    assert preds['p_elo2'].notna().all()
    ll = metrics.log_loss(preds['label'], preds['p_elo2'])
    assert ll < math.log(2)  # lernt aus den festen Spielerstärken etwas


# ------------------------------------------------------------ Kalibrierung

def test_temperature_scaling_recovers_temperature():
    rng = np.random.default_rng(0)
    z_true = rng.normal(0, 1.5, 20000)
    y = (rng.random(len(z_true)) < sigmoid(z_true)).astype(float)
    # Modell übertreibt um Faktor 2 -> Temperatur ~ 2
    t = fit_temperature(2 * z_true, y)
    assert t == pytest.approx(2.0, rel=0.05)
    assert metrics.log_loss(y, apply_temperature(2 * z_true, t)) < metrics.log_loss(y, sigmoid(2 * z_true))
    # Symmetrisch: z -> -z ergibt 1 - p
    np.testing.assert_allclose(apply_temperature(-z_true, t), 1 - apply_temperature(z_true, t))


@pytest.mark.parametrize('cls', [Platt, Isotonic])
def test_calibrators_symmetric_and_helpful(cls):
    rng = np.random.default_rng(1)
    p_true = rng.uniform(0.05, 0.95, 5000)
    y = (rng.random(5000) < p_true).astype(float)
    p_bad = sigmoid(0.4 * np.log(p_true / (1 - p_true)))  # untertreibt (wie der alte Transformer)
    cal = cls().fit(p_bad, y)
    q = np.linspace(0.01, 0.99, 50)
    np.testing.assert_allclose(cal.predict(q) + cal.predict(1 - q), 1.0, atol=1e-9)
    assert metrics.log_loss(y, cal.predict(p_bad)) < metrics.log_loss(y, p_bad)
    assert np.all(np.diff(cal.predict(q)) >= -1e-12)  # monoton


# ------------------------------------------------------------------ LightGBM

@pytest.fixture(scope="module")
def gbdt_run():
    pytest.importorskip('lightgbm')
    raw = synthetic_matches(years=(2017, 2018, 2019, 2020), per_month=40, n_players=30)
    m = prepare_matches(raw)
    preds = make_predictors(['market', 'elo2', 'gbdt', 'stack', 'stack_bands'], gbdt_refit='year')
    gbdt = next(p for p in preds if p.name == 'gbdt')
    gbdt.min_train = 500
    gbdt.params.max_rounds = 200
    out = backtest.walk_forward(m, preds, 2020, 2020, log=lambda *a: None, warmup=1)
    return m, preds, out


def test_gbdt_no_leakage(gbdt_run):
    m, preds, out = gbdt_run
    gbdt = next(p for p in preds if p.name == 'gbdt')
    # Jährlich neu gefittet (2019 Warm-up, 2020), Training immer vor dem Jahr
    assert [f['period'] for f in gbdt.fits] == ['2019', '2020']
    for f in gbdt.fits:
        assert f['train_until'] < pd.Timestamp(f"{f['period']}-01-01")
    assert out['p_gbdt'].notna().mean() > 0.95
    # Direkter Aufruf mit einem Fenster, das in die Historie reicht, scheitert
    with pytest.raises(AssertionError):
        gbdt.fit(m[m['year'] <= 2020])
        gbdt.predict(m[m['year'] == 2020])


def test_gbdt_predictions_symmetric():
    """Gleiche Matches mit anderer A/B-Zuordnung: p_gbdt dreht sich exakt um."""
    pytest.importorskip('lightgbm')
    from src.models.gbdt import GBDTParams, GBDTPredictor

    raw = synthetic_matches(years=(2018, 2019, 2020), per_month=30, n_players=30)
    results = []
    for seed in (1, 2):
        m = prepare_matches(raw, seed=seed)
        p = GBDTPredictor(refit='year', params=GBDTParams(max_rounds=100), min_train=300)
        p.prepare(m)
        p.fit(m[m['year'] < 2020])
        window = m[m['year'] == 2020]
        results.append(window.assign(p=p.predict(window)).set_index('match_id'))
    a, b = results[0], results[1].loc[results[0].index]
    flipped = (a['player_a'] != b['player_a']).to_numpy()
    assert flipped.any() and (~flipped).any()
    # Das Modell selbst unterscheidet sich je Seed (andere Trainingszeilen); exakt prüfen wir die Symmetrie
    # eines Modells: Spiegeln der Features dreht p um
    from src.models.gbdt import swap
    model = p.model
    X = p.table.loc[window['match_id']]
    np.testing.assert_allclose(model.predict(swap(X)), 1 - model.predict(X), atol=1e-9)
    assert np.corrcoef(a['p'][flipped], 1 - b['p'][flipped])[0, 1] > 0.9


def test_gbdt_feature_importance(gbdt_run):
    _, preds, _ = gbdt_run
    imp = next(p for p in preds if p.name == 'gbdt').importance()
    assert imp.sum() == pytest.approx(1.0)
    assert imp.index[0].startswith(('elo', 'win_rate', 'log_rank', 'surface_win_rate'))


def test_gbdt_save_load(tmp_path, gbdt_run):
    from src.models.gbdt import GBDTModel
    _, preds, _ = gbdt_run
    gbdt = next(p for p in preds if p.name == 'gbdt')
    path = tmp_path / 'gbdt_atp.pkl'
    gbdt.model.save(str(path))
    loaded = GBDTModel.load(str(path))
    X = gbdt.table.head(50)
    np.testing.assert_allclose(loaded.predict(X), gbdt.model.predict(X))


# ------------------------------------------------------------------ Stacking

def test_stack_learns_weights_and_is_symmetric():
    from src.models.stack import StackPredictor
    rng = np.random.default_rng(0)
    n = 4000
    z_true = rng.normal(0, 1.2, n)
    y = (rng.random(n) < sigmoid(z_true)).astype(int)
    # Markt kennt nur einen Teil der Information, das Modell den Rest
    df = pd.DataFrame({'date': pd.Timestamp('2020-01-01'), 'label': y, 'status': 'completed',
                       'p_market': sigmoid(0.7 * z_true + rng.normal(0, 0.3, n)),
                       'p_gbdt': sigmoid(z_true + rng.normal(0, 0.6, n))})
    s = StackPredictor(bases=('gbdt',))
    s.fit(df)
    assert s.coef[1] > 0.1  # Modell bekommt Gewicht
    window = df.head(200).assign(date=pd.Timestamp('2020-02-01'))
    p = s.predict(window)
    flipped = window.assign(p_market=1 - window['p_market'], p_gbdt=1 - window['p_gbdt'])
    np.testing.assert_allclose(s.predict(flipped), 1 - p, atol=1e-12)
    # mit Quotenbereichen ebenfalls symmetrisch
    sb = StackPredictor(bases=('gbdt',), bands=(0.6, 0.75))
    sb.fit(df)
    assert len(sb.coef) == 6
    np.testing.assert_allclose(sb.predict(flipped), 1 - sb.predict(window), atol=1e-12)
    # fehlt ein Basismodell, gibt es kein p
    assert np.isnan(s.predict(window.assign(p_gbdt=np.nan))).all()


def test_stack_falls_back_to_market_and_no_leakage(gbdt_run):
    from src.models.stack import StackPredictor
    m, preds, out = gbdt_run
    stack = next(p for p in preds if p.name == 'stack')
    assert stack.bases == ('elo2', 'gbdt') and stack.coef is not None
    assert out['p_stack'].notna().mean() > 0.95
    s = StackPredictor()
    with pytest.raises(AssertionError):
        s.fit(out[out['year'] == 2020])
        s.predict(out[out['year'] == 2020])
    empty = StackPredictor()
    empty.fit(out.iloc[:0])
    window = out.head(20)
    np.testing.assert_allclose(empty.predict(window), window['p_market'])


def test_report_with_ladder(gbdt_run):
    _, preds, out = gbdt_run
    lines = []
    grid = backtest.report(out, [p.name for p in preds], n_boot=50, log=lines.append, predictors=preds)
    text = "\n".join(lines)
    assert 'Go/No-Go stack' in text and 'Feature-Importance gbdt' in text and 'Gewichte stack' in text
    assert set(grid['venue']) <= {'pinnacle', 'betfair', 'bet365_de'}
    seg = metrics.segment_table(out, 'stack', ['market', 'elo2', 'gbdt', 'stack'], n_boot=50, min_n=50)
    assert set(seg['segment']) >= {'Quotenbereich', 'Untergrund'}


def test_coverage_split():
    df = pd.DataFrame({'market_source': ['ps'] * 10, 'status': 'completed', 'p_market': 0.5,
                       'p_elo2': 0.5, 'p_transformer': [np.nan] * 8 + [0.5, 0.5]})
    full, partial = backtest.split_by_coverage(df, ['market', 'elo2', 'transformer'])
    assert full == ['market', 'elo2'] and partial == ['transformer']
