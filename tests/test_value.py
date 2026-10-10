from dataProcessing.value import add_predictions, aggregate_tennis, value_bets
from src.predict import Prediction


EVENTS = [
    (["Tabilo, A.", "Carreno-Busta, P."], [1.47, 2.50], "tipico"),
    # gleiches Match bei bwin, andere Schreibweise und umgekehrte Reihenfolge
    (["P. Carreno Busta ESP", "A. Tabilo CHI"], [2.70, 1.40], "bwin"),
    (["Bu, Y.", "Ruud, C."], [1.85, 1.85], "tipico"),
    (["A. Pavlasek/P. Rikl", "M. Arevalo/M. Pavic"], [2.75, 1.42], "bwin"),
]


def test_aggregate_merges_sites_and_keeps_best_odds():
    matches = aggregate_tennis(EVENTS)
    assert len(matches) == 2  # Doppel fliegt raus, Tabilo-Match nur einmal

    tabilo = next(m for m in matches if m.player1['name'] == "Tabilo, A.")
    assert tabilo.player1 == {'name': "Tabilo, A.", 'odd': 1.47, 'source': 'tipico'}
    assert tabilo.player2 == {'name': "Carreno-Busta, P.", 'odd': 2.70, 'source': 'bwin'}


class FakePredictor:
    def predict(self, a, b, surface):
        if "Ruud" in (a + b):
            return None
        return Prediction(a, b, 0.5, 0.5, 20, 20)


def test_value_bets_uses_ev_threshold():
    matches = add_predictions(aggregate_tennis(EVENTS), FakePredictor(), "Hard")
    bets = value_bets(matches, min_ev=1.05)

    # 0.5 * 2.70 = 1.35 ist Value, 0.5 * 1.47 nicht; Ruud-Match ohne Vorhersage
    assert [(side['name'], round(ev, 2)) for _, side, _, _, ev in bets] == [("Carreno-Busta, P.", 1.35)]


def test_bet_share():
    match = aggregate_tennis(EVENTS[2:3])[0]
    assert abs(match.bet_share() - 2 / 1.85) < 1e-9
