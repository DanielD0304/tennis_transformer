"""Django-Seiten mit gemocktem Scraper (keine Live-Requests an die Buchmacher)."""
import django
import pytest
from django.conf import settings

django.setup()

from django.test import Client  # noqa: E402
from django.test.utils import setup_test_environment  # noqa: E402

# Sonst ist response.context leer
setup_test_environment()

from scraper import views  # noqa: E402
from src.predict import Prediction  # noqa: E402


class FakePredictor:
    def predict(self, a, b, surface):
        return Prediction(a, b, 0.6, 0.4, 20, 20)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "ALLOWED_HOSTS", ["testserver"])
    return Client()


def test_tennis_page_shows_value_bets(client, monkeypatch):
    events = [(["Tabilo, A.", "Carreno-Busta, P."], [2.00, 1.90], "tipico")]
    monkeypatch.setattr(views, "scrape_all_sites", lambda sites, n_outcomes: events)
    monkeypatch.setattr(views, "get_predictor", lambda: FakePredictor())

    r = client.get("/scraper/tennis/?surface=Clay&min_ev=1.1")

    assert r.status_code == 200
    assert r.context['surface'] == "Clay"
    assert r.context['error'] is None
    # 0.6 * 2.00 = 1.2 > 1.1
    assert [side['name'] for _, side, *_ in r.context['bets']] == ["Tabilo, A."]


def test_tennis_page_without_model_still_shows_odds(client, monkeypatch):
    events = [(["Bu, Y.", "Ruud, C."], [1.85, 1.85], "bwin")]
    monkeypatch.setattr(views, "scrape_all_sites", lambda sites, n_outcomes: events)

    def no_model():
        raise FileNotFoundError("kein Modell")
    monkeypatch.setattr(views, "get_predictor", no_model)

    r = client.get("/scraper/tennis/?surface=Rasen&min_ev=abc")

    assert r.status_code == 200
    assert r.context['surface'] == "Hard"
    assert r.context['min_ev'] == 1.05
    assert "kein Modell" in r.context['error']
    assert len(r.context['matches']) == 1


def test_football_page(client, monkeypatch):
    events = [(["Bayern", "Dortmund"], [1.5, 4.0, 5.0], "tipico")]
    monkeypatch.setattr(views, "scrape_all_sites", lambda sites, n_outcomes: events)

    r = client.get("/scraper/scrape-odds/")

    assert r.status_code == 200
    assert "Bayern - Dortmund" in r.context['results']
