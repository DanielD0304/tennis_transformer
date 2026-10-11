import threading
import time

# Nach dieser Zeit wird der Predictor neu gebaut, dabei werden auch die Daten aktualisiert
# (MatchPredictor lädt sie neu, wenn sie älter als 24 Stunden sind).
MAX_AGE_SECONDS = 24 * 3600

_lock = threading.Lock()
_cached = None
_loaded_at = 0.0


def get_predictor():
    global _cached, _loaded_at
    with _lock:
        if _cached is None or time.time() - _loaded_at > MAX_AGE_SECONDS:
            # Import hier, damit die Fußball-Seite auch ohne torch läuft.
            from src.predict import MatchPredictor
            _cached = MatchPredictor()
            _loaded_at = time.time()
        return _cached
