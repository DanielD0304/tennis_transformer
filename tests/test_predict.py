"""
Smoke-Test für MatchPredictor mit synthetischen Matchdaten (Schema von data/matches.csv)
und zufällig initialisiertem Modell. Prüft, dass die Pipeline (Historie -> Features -> Modell)
durchläuft und konsistente Werte liefert.
"""
import random

import pandas as pd
import pytest
import torch

from src.config import default_config
from src.model import Model
from src.predict import MatchPredictor

PLAYERS = ["Daniil Medvedev", "Jan-Lennard Struff", "Jiri Lehecka", "Nuno Borges",
           "Pablo Carreno Busta", "Juan Manuel Cerundolo"]


def fake_matches(years=(2024, 2025), with_stats_until=2024):
    rng = random.Random(0)
    rows = []
    for year in years:
        for i in range(40):
            w, l = rng.sample(PLAYERS, 2)
            stats = {}
            if year <= with_stats_until:
                stats = {'w_ace': 8, 'w_df': 2, 'w_svpt': 70, 'w_1stIn': 45, 'w_1stWon': 35, 'w_2ndWon': 14,
                         'w_SvGms': 10, 'w_bpSaved': 3, 'w_bpFaced': 4, 'l_ace': 4, 'l_df': 3, 'l_svpt': 72,
                         'l_1stIn': 40, 'l_1stWon': 28, 'l_2ndWon': 12, 'l_SvGms': 10, 'l_bpSaved': 4, 'l_bpFaced': 7}
            rows.append({
                'tourney_name': rng.choice(['Shanghai Masters', 'Roland Garros', 'Wimbledon']),
                'winner_name': w, 'loser_name': l,
                'winner_rank': rng.randint(1, 150), 'loser_rank': rng.randint(1, 150),
                'surface': rng.choice(['Hard', 'Clay', 'Grass']),
                'tourney_date': f"{year}{1 + i // 4:02d}{1 + i % 28:02d}",
                'match_date': f"{year}{1 + i // 4:02d}{1 + i % 28:02d}",
                'match_num': i, 'round': 'R32', 'score': '6-4 6-7(1) 7-5',
                'status': 'completed', 'B365W': 1.5, 'B365L': 2.5, **stats,
            })
    # Ein Walkover: darf weder in die Historie noch ins H2H
    rows.append({**rows[-1], 'match_num': 999, 'score': 'W/O', 'status': 'W/O',
                 'winner_name': PLAYERS[0], 'loser_name': PLAYERS[1]})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def predictor(tmp_path_factory):
    torch.manual_seed(0)
    cfg = default_config
    model = Model(cfg.d_model, cfg.num_heads, cfg.num_layers, cfg.input_dim, cfg.max_len, cfg.output_dim)
    model_path = tmp_path_factory.mktemp("model") / "best_model.pt"
    torch.save(model.state_dict(), model_path)
    return MatchPredictor(matches=fake_matches(), model_path=str(model_path), model="transformer")


def test_history_built(predictor):
    assert set(predictor.history) == set(PLAYERS)
    # 80 gespielte Matches, der Walkover zählt nicht
    assert sum(len(h) for h in predictor.history.values()) == 2 * 80


def test_has_stats_feature(predictor):
    hist = predictor.history["Daniil Medvedev"]
    with_stats = [m['has_stats'] for m in hist if m['_date_obj'].year == 2024]
    without = [m['has_stats'] for m in hist if m['_date_obj'].year == 2025]
    assert with_stats and set(with_stats) == {1}
    assert without and set(without) == {0}
    assert all(m['aces'] == 0 for m in hist if m['_date_obj'].year == 2025)


@pytest.mark.parametrize("name, expected", [
    ("D. Medvedev RUS", "Daniil Medvedev"),
    ("Medvedev D.", "Daniil Medvedev"),
    ("Carreno-Busta, P.", "Pablo Carreno Busta"),
    ("Struff J.L.", "Jan-Lennard Struff"),
    ("Cerundolo, J. M.", "Juan Manuel Cerundolo"),
    ("Cerundolo J.M.", "Juan Manuel Cerundolo"),
    ("Daniil Medvedev", "Daniil Medvedev"),
    ("Sramkova, R.", None),
])
def test_resolves_names(predictor, name, expected):
    assert predictor.resolve(name) == expected


def test_prediction_is_valid_and_symmetric(predictor):
    ab = predictor.predict("D. Medvedev RUS", "J.-L. Struff GER", surface="Hard")
    ba = predictor.predict("J.-L. Struff GER", "D. Medvedev RUS", surface="Hard")

    assert 0.0 < ab.prob_a < 1.0
    assert ab.prob_a + ab.prob_b == pytest.approx(1.0)
    # A/B-Reihenfolge wird gemittelt -> Ergebnis unabhängig von der Reihenfolge
    assert ab.prob_a == pytest.approx(ba.prob_b, abs=1e-6)


def test_unknown_player_gives_none(predictor):
    assert predictor.predict("Daniil Medvedev", "Unbekannt X.") is None
    assert predictor.predict("Daniil Medvedev", "Medvedev D.") is None


def test_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        MatchPredictor(matches=fake_matches(), model_path=str(tmp_path / "fehlt.pt"), model="transformer")


def test_player_info(predictor):
    info = predictor.player_info("Daniil Medvedev", surface="Hard", n_last=5)

    assert info['name'] == "Daniil Medvedev"
    assert info['n_matches'] == len(predictor.history["Daniil Medvedev"])
    assert len(info['last']) == 5
    # neuestes Match zuerst
    assert info['last'][0]['date'] >= info['last'][-1]['date']
    assert info['last_wins'] == sum(m['won'] for m in info['last'])
    assert 0 <= info['surface_wins'] <= info['surface_matches']
    assert isinstance(info['elo'], int)
    assert info['last'][0]['tournament'] in ('Shanghai Masters', 'Roland Garros', 'Wimbledon')
    # Ergebnis aus Sicht des Spielers
    for m in info['last']:
        assert m['score'] == ('6-4 6-7(1) 7-5' if m['won'] else '4-6 7-6(1) 5-7')


def test_h2h_record_is_consistent(predictor):
    a, b = "Daniil Medvedev", "Jan-Lennard Struff"
    wins_a, wins_b = predictor.h2h_record(a, b)
    assert predictor.h2h_record(b, a) == (wins_b, wins_a)

    played = [m for m in predictor.matches[a] if m['opponent'] == b]
    assert wins_a == sum(m['won'] for m in played)
    assert wins_a + wins_b == len(played)
    assert predictor.h2h_record(a, "Unbekannt X.") == (0, 0)


def test_search(predictor):
    assert predictor.search("medv")[0] == "Daniil Medvedev"
    assert "Jan-Lennard Struff" in predictor.search("Jan-Lennard Strufff")


def test_stats_dropout(predictor):
    import pandas as pd
    from src.dataset import DataSet
    from src.preprocessing import FEATURE_NAMES, STAT_FEATURES

    sample = predictor._sample("Daniil Medvedev", "Jan-Lennard Struff", "Hard", pd.Timestamp("2026-01-01"))
    cols = [FEATURE_NAMES.index(f) for f in STAT_FEATURES + ['has_stats']]
    won = FEATURE_NAMES.index('won')

    full = DataSet([sample])[0]['player_a_recent']
    dropped = DataSet([sample], stats_dropout=1.0)[0]['player_a_recent']
    assert full[:, cols].abs().sum() > 0
    assert dropped[:, cols].abs().sum() == 0
    # nur die Stats verschwinden, Ergebnis/ELO usw. bleiben
    assert torch.equal(full[:, won], dropped[:, won])


def test_surface_lookup_from_tournament(predictor):
    # Turniernamen aus den Daten: Wimbledon/Roland Garros/Shanghai mit dem Untergrund des neuesten Eintrags
    surface, source = predictor.surface_for("Wimbledon")
    assert source == 'tournament' and surface in ('Hard', 'Clay', 'Grass')
    expected = [m['surface'] for ms in predictor.matches.values() for m in ms if m['tournament'] == 'Wimbledon']
    assert surface in expected


def test_surface_auto_and_override(predictor):
    pred = predictor.predict("Daniil Medvedev", "Jan-Lennard Struff", tournament="Rolex Shanghai Masters")
    assert pred.surface_source == 'tournament'
    manual = predictor.predict("Daniil Medvedev", "Jan-Lennard Struff", surface="Clay", tournament="Shanghai Masters")
    assert (manual.surface, manual.surface_source) == ("Clay", "manual")
    # Unbekanntes Turnier, kein laufendes Turnier (Daten enden 2025): Standard
    unknown = predictor.predict("Daniil Medvedev", "Jan-Lennard Struff", tournament="Nirgendwo Open",
                                match_date="2026-10-01")
    assert (unknown.surface, unknown.surface_source) == ("Hard", "default")


def test_surface_from_current_tournament(predictor):
    last = max(m['date'] for m in predictor.matches["Daniil Medvedev"])
    surface, source = predictor.surface_for(None, ("Daniil Medvedev",), match_date=last)
    assert source == 'current'
    assert surface == predictor.matches["Daniil Medvedev"][-1]['surface']


def test_predict_proba_interface(predictor):
    p = predictor.predict_proba({'player_a': "Daniil Medvedev", 'player_b': "Jan-Lennard Struff", 'surface': "Hard"})
    q = predictor.predict_proba({'player_a': "Jan-Lennard Struff", 'player_b': "Daniil Medvedev", 'surface': "Hard"})
    assert 0 < p < 1 and p == pytest.approx(1 - q, abs=1e-6)
    assert predictor.predict_proba({'player_a': "Daniil Medvedev", 'player_b': "Unbekannt X."}) is None


def test_transformer_checkpoint_with_temperature(tmp_path):
    """Neuer Checkpoint (Elo2-Input + Temperatur): höhere Temperatur zieht p Richtung 0,5."""
    from src.model import save_checkpoint
    cfg = default_config
    torch.manual_seed(1)
    model = Model(cfg.d_model, cfg.num_heads, cfg.num_layers, cfg.input_dim, cfg.max_len, cfg.output_dim,
                  cfg.extra_dim)
    probs = []
    for t in (1.0, 4.0):
        path = tmp_path / f"m{t}.pt"
        save_checkpoint(model, str(path), temperature=t, extra_dim=cfg.extra_dim)
        pred = MatchPredictor(matches=fake_matches(), model_path=str(path), model="transformer")
        assert pred.temperature == t
        probs.append(pred.predict("Daniil Medvedev", "Jan-Lennard Struff", surface="Hard").prob_a)
    assert abs(probs[1] - 0.5) == pytest.approx(abs(probs[0] - 0.5) / 4, rel=0.05)
    ba = pred.predict("Jan-Lennard Struff", "Daniil Medvedev", surface="Hard")
    assert ba.prob_a == pytest.approx(1 - probs[1], abs=1e-6)


def test_live_model_elo2():
    pred = MatchPredictor(matches=fake_matches(), model='elo2')
    out = pred.predict("Daniil Medvedev", "Jan-Lennard Struff", surface="Clay")
    assert out.model == 'elo2'
    tour = pred.tour["Daniil Medvedev"]
    assert out.prob_a == pytest.approx(pred.elo2.prob(tour, "Daniil Medvedev", "Jan-Lennard Struff", "Clay"))
    assert out.prob_a + pred.predict("Jan-Lennard Struff", "Daniil Medvedev", surface="Clay").prob_a == pytest.approx(1)
    with pytest.raises(ValueError):
        MatchPredictor(matches=fake_matches(), model='xgboost')


def test_live_model_gbdt(tmp_path):
    pytest.importorskip('lightgbm')
    from src.evaluation.predictors import prepare_matches
    from src.models.gbdt import GBDTParams, train_live

    df = fake_matches(years=(2022, 2023, 2024, 2025))
    model = train_live(prepare_matches(df), params=GBDTParams(max_rounds=30, min_data_in_leaf=5), val_months=6)
    model.save(str(tmp_path / "gbdt_atp.pkl"))
    with pytest.raises(FileNotFoundError):
        MatchPredictor(matches=df, model='gbdt', gbdt_dir=str(tmp_path / "leer"))
    pred = MatchPredictor(matches=df, model='gbdt', gbdt_dir=str(tmp_path))
    p = pred.predict_proba({'player_a': "Daniil Medvedev", 'player_b': "Nuno Borges", 'surface': "Hard",
                            'date': "2026-01-10"})
    q = pred.predict_proba({'player_a': "Nuno Borges", 'player_b': "Daniil Medvedev", 'surface': "Hard",
                            'date': "2026-01-10"})
    assert 0 < p < 1 and p == pytest.approx(1 - q, abs=1e-9)


def test_live_gbdt_unknown_rank_like_training(tmp_path, monkeypatch):
    """Spieler ohne Rang: live wie im Training MISSING_RANK, nicht 0 (-> NaN, das LightGBM wie Rang 0 behandelt)."""
    import math

    pytest.importorskip('lightgbm')
    from src.evaluation.predictors import MISSING_RANK, prepare_matches
    from src.models.gbdt import GBDTParams, train_live

    df = fake_matches(years=(2022, 2023, 2024, 2025))
    df.loc[df['winner_name'] == "Nuno Borges", 'winner_rank'] = None
    df.loc[df['loser_name'] == "Nuno Borges", 'loser_rank'] = None
    train_live(prepare_matches(df), params=GBDTParams(max_rounds=30, min_data_in_leaf=5),
               val_months=6).save(str(tmp_path / "gbdt_atp.pkl"))
    pred = MatchPredictor(matches=df, model='gbdt', gbdt_dir=str(tmp_path))
    rows = []
    model = pred.gbdt['atp']
    monkeypatch.setattr(model, 'predict', lambda row, f=model.predict: rows.append(row) or f(row))
    pred.predict_proba({'player_a': "Daniil Medvedev", 'player_b': "Nuno Borges", 'surface': "Hard",
                        'date': "2026-01-10"})
    row = rows[0].iloc[0]
    assert row['log_rank_b'] == pytest.approx(math.log(MISSING_RANK + 1))
    assert math.isfinite(row['log_rank_diff'])


def test_records_same_values_as_iterrows():
    """records liefert dieselben Python-Werte wie iterrows, gleiche Texte und Daten sind dabei ein Objekt."""
    from src.data_loader import records
    df = pd.DataFrame({'text': ['a', None, 'a'], 'num': [1.5, float('nan'), 3.0], 'n': [1, 2, 3],
                       'day': pd.to_datetime(['2026-01-01', '2026-01-01', None])})
    df['text'] = df['text'].astype('str')
    columns = ['text', 'num', 'n', 'day', 'fehlt']
    rows = list(records(df, columns))
    for row, (_, full) in zip(rows, df.iterrows()):
        assert set(row) == set(df.columns)
        for c in df.columns:
            a, b = row[c], full[c]
            assert type(a) is type(b) and (a == b or (a != a and b != b))
    assert rows[0]['text'] is rows[2]['text'] and rows[0]['day'] is rows[1]['day']
    assert list(records(df, ['fehlt'])) == [{}, {}, {}]


def _feature_state(builder):
    """FeatureBuilder-Stand ohne leere Einträge (row() legt für Abfragen leere Defaults an)."""
    players = {k: (list(s.recent), {sf: list(d) for sf, d in s.surface.items() if d}, list(s.days), s.n, s.last_date,
                   s.last_retired) for k, s in builder.players.items() if s.n}
    return players, {k: v for k, v in builder.h2h.items() if v}


def test_live_gbdt_state_like_feature_table(tmp_path):
    """Der Feature-Stand aus dem einen Durchlauf von _build_history ist der von feature_table(prepare_matches(...))."""
    pytest.importorskip('lightgbm')
    from src.evaluation.predictors import prepare_matches
    from src.features import FeatureBuilder, feature_table
    from src.models.gbdt import GBDTParams, train_live

    df = fake_matches(years=(2022, 2023, 2024, 2025))
    df.loc[3, ['status', 'score']] = ['RET', '6-4 2-1 RET']
    train_live(prepare_matches(df), params=GBDTParams(max_rounds=30, min_data_in_leaf=5),
               val_months=6).save(str(tmp_path / "gbdt_atp.pkl"))
    pred = MatchPredictor(matches=df, model='gbdt', gbdt_dir=str(tmp_path))
    ref = FeatureBuilder()
    feature_table(prepare_matches(df), ref)
    assert _feature_state(pred.features) == _feature_state(ref)
    assert not pred.history  # Transformer-Historie nur für das Transformer-Modell


@pytest.fixture
def matches_csv(tmp_path, monkeypatch):
    """Standard-Datenpfad auf eine eigene matches.csv, Cache in tmp_path/cache, kein Daten-Update."""
    from src import cache, data_loader, update_data
    path = tmp_path / "matches.csv"
    fake_matches().to_csv(path, index=False)
    monkeypatch.setattr(data_loader, 'MATCHES_PATH', str(path))
    monkeypatch.setattr(update_data, 'is_stale', lambda *a, **k: False)
    monkeypatch.setattr(cache, 'CACHE_DIR', str(tmp_path / "cache"))
    return path


@pytest.mark.parametrize("model", ['elo2', 'gbdt'])
def test_state_cache_until_matches_change(matches_csv, tmp_path, monkeypatch, model):
    """Zweiter Aufbau kommt aus data/cache (gleiche Vorhersagen), eine neue matches.csv baut neu."""
    kwargs = {}
    if model == 'gbdt':
        pytest.importorskip('lightgbm')
        from src.evaluation.predictors import prepare_matches
        from src.models.gbdt import GBDTParams, train_live
        train_live(prepare_matches(fake_matches(years=(2022, 2023, 2024, 2025))),
                   params=GBDTParams(max_rounds=30, min_data_in_leaf=5),
                   val_months=6).save(str(tmp_path / "gbdt_atp.pkl"))
        kwargs['gbdt_dir'] = str(tmp_path)
    built = []
    build = MatchPredictor._build_history
    monkeypatch.setattr(MatchPredictor, '_build_history', lambda self, df: built.append(len(df)) or build(self, df))
    match = {'player_a': "Daniil Medvedev", 'player_b': "Nuno Borges", 'surface': "Clay", 'date': "2026-01-10"}

    first = MatchPredictor(model=model, **kwargs)
    assert (tmp_path / "cache" / f"predictor_{model}.pkl").exists() and len(built) == 1
    second = MatchPredictor(model=model, **kwargs)
    assert len(built) == 1  # aus dem Cache
    assert second.predict_proba(match) == first.predict_proba(match)
    assert second.player_info("Daniil Medvedev", "Clay") == first.player_info("Daniil Medvedev", "Clay")
    assert second.surface_for("Wimbledon") == first.surface_for("Wimbledon")
    assert MatchPredictor(model=model, use_cache=False, **kwargs).predict_proba(match) == first.predict_proba(match)
    assert len(built) == 2

    df = fake_matches()
    df = pd.concat([df, df.iloc[[0]].assign(match_num=1000, winner_name="Nuno Borges", loser_name="Daniil Medvedev")])
    df.to_csv(matches_csv, index=False)
    third = MatchPredictor(model=model, **kwargs)
    assert len(built) == 3 and third.h2h_record("Nuno Borges", "Daniil Medvedev") != first.h2h_record(
        "Nuno Borges", "Daniil Medvedev")
