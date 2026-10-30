"""Django-Seiten mit gemocktem Scraper (keine Live-Requests an die Buchmacher)."""
import django
import pytest
from django.conf import settings

django.setup()

from django.test import Client  # noqa: E402
from django.test.utils import setup_test_environment  # noqa: E402

# Sonst ist response.context leer
setup_test_environment()

from predictor import views as predictor_views  # noqa: E402
from scraper import views  # noqa: E402
from src.predict import Prediction  # noqa: E402


class FakePredictor:
    def __init__(self):
        self.surfaces = []

    def predict(self, a, b, surface):
        self.surfaces.append(surface)
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


def test_tennis_page_auto_surface(client, monkeypatch):
    events = [(["Tabilo, A.", "Carreno-Busta, P."], [2.00, 1.90], "tipico")]
    monkeypatch.setattr(views, "scrape_all_sites", lambda sites, n_outcomes: events)
    fake = FakePredictor()
    monkeypatch.setattr(views, "get_predictor", lambda: fake)

    r = client.get("/scraper/tennis/")
    assert r.context['surface'] == "Auto"
    # None = der Predictor bestimmt den Untergrund selbst
    assert fake.surfaces == [None]


def test_tennis_page_without_model_still_shows_odds(client, monkeypatch):
    events = [(["Bu, Y.", "Ruud, C."], [1.85, 1.85], "bwin")]
    monkeypatch.setattr(views, "scrape_all_sites", lambda sites, n_outcomes: events)

    def no_model():
        raise FileNotFoundError("kein Modell")
    monkeypatch.setattr(views, "get_predictor", no_model)

    r = client.get("/scraper/tennis/?surface=Rasen&min_ev=abc")

    assert r.status_code == 200
    # unbekannter Untergrund -> automatisch
    assert r.context['surface'] == "Auto"
    assert r.context['min_ev'] == 1.05
    assert "kein Modell" in r.context['error']
    assert len(r.context['matches']) == 1


def test_football_page(client, monkeypatch):
    events = [(["Bayern", "Dortmund"], [1.5, 4.0, 5.0], "tipico")]
    monkeypatch.setattr(views, "scrape_all_sites", lambda sites, n_outcomes: events)

    r = client.get("/scraper/scrape-odds/")

    assert r.status_code == 200
    assert "Bayern - Dortmund" in r.context['results']


class FakeMatchPredictor:
    players = {("medvedev", "d"): "Medvedev D.", ("struff", "j"): "Struff J.L."}

    def __init__(self):
        self.calls = []

    class last_match_date:
        @staticmethod
        def date():
            return __import__("datetime").date(2026, 10, 6)

    def resolve(self, name):
        from src.names import player_key
        return self.players.get(player_key(name))

    def search(self, query, n=8):
        return ["Struff J.L."]

    def predict(self, a, b, surface=None, match_date=None, tournament=None):
        self.calls.append((surface, tournament))
        if surface is None:
            return Prediction(a, b, 0.75, 0.25, 30, 3, [f"{b}: nur 3 Matches in der Historie"], 'Clay', 'tournament')
        return Prediction(a, b, 0.75, 0.25, 30, 3, [f"{b}: nur 3 Matches in der Historie"], surface, 'manual')

    def h2h_record(self, a, b):
        return (2, 1)

    def player_info(self, name, surface=None, n_last=10):
        return {'name': name, 'rank': 5, 'elo': 2000, 'n_matches': 30, 'last': [],
                'last_wins': 0, 'surface_wins': 0, 'surface_matches': 0}


def test_predict_page_empty_form(client, monkeypatch):
    monkeypatch.setattr(predictor_views, "get_predictor", lambda: FakeMatchPredictor())
    r = client.get("/")
    assert r.status_code == 200
    assert r.context['result'] is None
    assert r.context['players'] == ["Medvedev D.", "Struff J.L."]


def test_predict_page_shows_prediction(client, monkeypatch):
    monkeypatch.setattr(predictor_views, "get_predictor", lambda: FakeMatchPredictor())
    r = client.get("/?a=D.+Medvedev+RUS&b=Struff%2C+J.&surface=Clay&date=2026-10-12")

    assert r.status_code == 200
    res = r.context['result']
    assert (res['pct_a'], res['pct_b']) == (75.0, 25.0)
    assert res['fair_a'] == 1 / 0.75
    assert res['h2h'] == (2, 1)
    html = r.content.decode()
    assert "Medvedev D." in html and "12.10.2026" in html and "nur 3 Matches" in html


def test_predict_page_auto_surface_from_tournament(client, monkeypatch):
    fake = FakeMatchPredictor()
    monkeypatch.setattr(predictor_views, "get_predictor", lambda: fake)
    r = client.get("/?a=Medvedev+D.&b=Struff+J.L.&t=Internazionali+BNL+d%27Italia")

    assert r.context['surface'] == "Auto"
    assert fake.calls == [(None, "Internazionali BNL d'Italia")]
    assert "Clay (aus Turnier)" in r.content.decode()


def test_predict_page_unknown_player_suggests(client, monkeypatch):
    monkeypatch.setattr(predictor_views, "get_predictor", lambda: FakeMatchPredictor())
    r = client.get("/?a=Strufff&b=Medvedev+D.")

    assert r.context['result'] is None
    assert r.context['errors'] == [{'name': "Strufff", 'suggestions': ["Struff J.L."]}]


def test_predict_page_same_player(client, monkeypatch):
    monkeypatch.setattr(predictor_views, "get_predictor", lambda: FakeMatchPredictor())
    r = client.get("/?a=Medvedev+D.&b=D.+Medvedev")
    assert r.context['result'] is None
    assert r.context['errors'][0]['same'] is True


def test_predict_page_without_model(client, monkeypatch):
    def no_model():
        raise FileNotFoundError("kein Modell")
    monkeypatch.setattr(predictor_views, "get_predictor", no_model)

    r = client.get("/?a=Medvedev+D.&b=Struff+J.L.&date=kaputt")

    assert r.status_code == 200
    assert "kein Modell" in r.context['model_error']
    assert "python main.py" in r.content.decode()


def test_paper_page_renders_empty(client, monkeypatch, tmp_path):
    from src.paper import store
    monkeypatch.setattr(store, "BETS_PATH", str(tmp_path / "bets.parquet"))
    r = client.get("/scraper/paper/")
    assert r.status_code == 200
    assert r.context['decision']['go'] is False
    assert "Noch keine Paper-Wetten" in r.content.decode()


def test_paper_page_renders_bets(client, monkeypatch, tmp_path):
    import pandas as pd
    from src.paper import store
    path = str(tmp_path / "bets.parquet")
    monkeypatch.setattr(store, "BETS_PATH", path)
    now = pd.Timestamp("2026-10-25 12:00", tz="UTC")
    rows = []
    for i, status in enumerate(['won', 'lost', 'void', 'open']):
        rows.append({'bet_id': f'b{i}', 'created_utc': now, 'strategy': 'sharp' if i % 2 else 'modell',
                     'venue': 'kalshi', 'event_id': f'e{i}', 'level': 'challenger', 'player_a': 'A A', 'player_b': 'B B',
                     'side': 'a', 'player': 'A A', 'start_time_utc': now if i < 3 else pd.NaT, 'odds': 2.1,
                     'net_odds': 2.05, 'p': 0.55, 'edge': 0.12, 'stake': 10.0, 'stake_flat': 10.0, 'status': status,
                     'settled_utc': now if status != 'open' else pd.NaT,
                     'profit': {'won': 10.5, 'lost': -10.0}.get(status, 0.0), 'clv': 0.03 if status != 'open' else None})
    store.save_bets(pd.DataFrame(rows))
    r = client.get("/scraper/paper/")
    html = r.content.decode()
    assert r.status_code == 200
    assert "<polyline" in html and "NO-GO" in html
    assert len(r.context['open_bets']) == 1 and len(r.context['settled_bets']) == 3
    r = client.get("/scraper/paper/?strategy=sharp")
    assert r.context['strategy'] == 'sharp' and len(r.context['settled_bets']) == 1


def test_paper_page_go_no_go_per_segment(client, monkeypatch, tmp_path):
    import pandas as pd
    from src.paper import store
    monkeypatch.setattr(store, "BETS_PATH", str(tmp_path / "bets.parquet"))
    now = pd.Timestamp("2026-10-25 12:00", tz="UTC")
    rows = [{'bet_id': f'b{i}', 'created_utc': now, 'strategy': 'modell', 'segment': segment, 'venue': 'polymarket',
             'event_id': f'e{i}', 'level': 'itf', 'player_a': 'A A', 'player_b': 'B B', 'side': 'a', 'player': 'A A',
             'start_time_utc': now, 'odds': 2.1, 'net_odds': 2.05, 'p': 0.55, 'edge': 0.12, 'stake': 10.0,
             'stake_flat': 10.0, 'status': 'won', 'settled_utc': now, 'profit': 10.5, 'profit_flat': 10.5, 'clv': 0.03}
            for i, segment in enumerate(['polymarket itf 0.5-1.5h', 'polymarket itf 0.5-1.5h', None])]
    store.save_bets(pd.DataFrame(rows))
    r = client.get("/scraper/paper/")
    html = r.content.decode()
    assert [s['name'] for s in r.context['segments']] == ['polymarket itf 0.5-1.5h']
    assert r.context['segments'][0]['stats']['settled'] == 2
    assert "Je Segment der Modell-Strategie" in html and html.count("polymarket itf 0.5-1.5h") >= 3
