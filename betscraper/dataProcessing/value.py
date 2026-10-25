from betObjekts.tennismatch import tennis_match
from src.names import display_name, player_key


def aggregate_tennis(events):
    """
    Führt die Quoten aller Seiten pro Match zusammen und behält jeweils die beste.
    Die Seiten schreiben Namen unterschiedlich ('Tabilo, A.' vs. 'A. Tabilo CHI'),
    deshalb wird über player_key gematcht. Doppel werden übersprungen.
    """
    matches = {}
    for (p1, p2), (odd1, odd2), source in events:
        if '/' in p1 or '/' in p2:
            continue
        k1, k2 = player_key(p1), player_key(p2)
        if not k1 or not k2:
            continue

        key = frozenset((k1, k2))
        if key not in matches:
            matches[key] = (k1, tennis_match(display_name(p1), odd1, display_name(p2), odd2, source))
            continue

        first_key, match = matches[key]
        if k1 == first_key:
            match.update_odds(odd1, odd2, source)
        else:
            match.update_odds(odd2, odd1, source)

    return [match for _, match in matches.values()]


def add_predictions(matches, predictor, surface=None):
    """surface=None: der Predictor bestimmt den Untergrund pro Match selbst."""
    for match in matches:
        match.prediction = predictor.predict(match.player1['name'], match.player2['name'], surface=surface)
    return matches


def value_bets(matches, min_ev=1.05):
    """
    Value Bet = Modellwahrscheinlichkeit * Quote > min_ev.
    Liefert (match, spieler, quote, wahrscheinlichkeit, ev), bester EV zuerst.
    """
    bets = []
    for match in matches:
        pred = match.prediction
        if pred is None:
            continue
        for side, prob in ((match.player1, pred.prob_a), (match.player2, pred.prob_b)):
            ev = prob * side['odd']
            if ev > min_ev:
                bets.append((match, side, side['odd'], prob, ev))
    return sorted(bets, key=lambda b: b[4], reverse=True)
