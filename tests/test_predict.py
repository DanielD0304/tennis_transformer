"""
Smoke-Test für MatchPredictor mit synthetischen TUK-Daten und zufällig initialisiertem Modell.
Prüft, dass die Pipeline (Historie -> Features -> Modell) durchläuft und konsistente Werte liefert.
"""
import random

import pandas as pd
import pytest
import torch

from src import data_loader as dl
from src.config import default_config
from src.model import Model
from src.predict import MatchPredictor

PLAYERS = ["Medvedev D.", "Struff J.L.", "Lehecka J.", "Borges N.", "Carreno Busta P.", "Tabilo A."]


def fake_tuk_year(year):
    rng = random.Random(year)
    rows = []
    for i in range(40):
        w, l = rng.sample(PLAYERS, 2)
        rows.append({
            'winner_name': w, 'loser_name': l,
            'winner_rank': rng.randint(1, 150), 'loser_rank': rng.randint(1, 150),
            'surface': rng.choice(['Hard', 'Clay', 'Grass']),
            'tourney_date': f"{year}{1 + i // 4:02d}{1 + i % 28:02d}",
            'B365W': 1.5, 'B365L': 2.5,
        })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def predictor(tmp_path_factory):
    torch.manual_seed(0)
    cfg = default_config
    model = Model(cfg.d_model, cfg.num_heads, cfg.num_layers, cfg.input_dim, cfg.max_len, cfg.output_dim)
    model_path = tmp_path_factory.mktemp("model") / "best_model.pt"
    torch.save(model.state_dict(), model_path)

    mp = pytest.MonkeyPatch()
    mp.setattr(dl, "load_tuk_year", fake_tuk_year)
    try:
        yield MatchPredictor(years=[2024, 2025], model_path=str(model_path))
    finally:
        mp.undo()


def test_history_built(predictor):
    assert set(predictor.players.values()) == set(PLAYERS)
    assert sum(len(h) for h in predictor.history.values()) == 2 * 80


def test_resolves_bookie_names(predictor):
    assert predictor.resolve("D. Medvedev RUS") == "Medvedev D."
    assert predictor.resolve("Carreno-Busta, P.") == "Carreno Busta P."
    assert predictor.resolve("Sramkova, R.") is None


def test_prediction_is_valid_and_symmetric(predictor):
    ab = predictor.predict("D. Medvedev RUS", "J.-L. Struff GER", surface="Hard")
    ba = predictor.predict("J.-L. Struff GER", "D. Medvedev RUS", surface="Hard")

    assert 0.0 < ab.prob_a < 1.0
    assert ab.prob_a + ab.prob_b == pytest.approx(1.0)
    # A/B-Reihenfolge wird gemittelt -> Ergebnis unabhängig von der Reihenfolge
    assert ab.prob_a == pytest.approx(ba.prob_b, abs=1e-6)


def test_unknown_player_gives_none(predictor):
    assert predictor.predict("Medvedev D.", "Unbekannt X.") is None
    assert predictor.predict("Medvedev D.", "Medvedev D.") is None


def test_missing_model_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "load_tuk_year", fake_tuk_year)
    with pytest.raises(FileNotFoundError):
        MatchPredictor(years=[2025], model_path=str(tmp_path / "fehlt.pt"))
