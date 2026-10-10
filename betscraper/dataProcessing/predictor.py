from functools import lru_cache


@lru_cache(maxsize=1)
def get_predictor():
    # Import hier, damit die Fußball-Seite auch ohne torch läuft.
    # Laden dauert (Download + Historie aufbauen), deshalb nur einmal pro Prozess.
    from src.predict import MatchPredictor
    return MatchPredictor()
