"""
Datei-Cache für teure Zwischenstände unter data/cache/ (data/ steht in .gitignore, der Ordner darf jederzeit
gelöscht werden). Ein Eintrag gilt nur für genau den Schlüssel, mit dem er geschrieben wurde, z.B. Pfad,
Änderungszeit und Größe von data/matches.csv plus ein Hash des Codes, der den Zustand aufbaut.

    key = {'matches': file_stamp(MATCHES_PATH), 'code': code_hash(['predict.py', 'elo.py'])}
    value = load('predictor_gbdt', key)
    if value is None:
        value = teuer_berechnen()
        save('predictor_gbdt', key, value)
"""
import gc
import hashlib
import os
import pickle

from .sources.common import DATA_DIR

CACHE_DIR = os.path.join(DATA_DIR, "cache")
SRC_DIR = os.path.dirname(os.path.abspath(__file__))


def file_stamp(path):
    """(absoluter Pfad, Änderungszeit in ns, Größe): ändert sich bei jedem Neuschreiben der Datei."""
    st = os.stat(path)
    return os.path.abspath(path), st.st_mtime_ns, st.st_size


def code_hash(files):
    """SHA-1 über Quelldateien (Pfade relativ zu src/)."""
    h = hashlib.sha1()
    for name in files:
        with open(os.path.join(SRC_DIR, name), 'rb') as f:
            h.update(f.read())
    return h.hexdigest()


def _path(name, root):
    return os.path.join(root or CACHE_DIR, f"{name}.pkl")


def load(name, key, root=None, log=print):
    """Gespeicherter Wert, wenn er mit genau diesem key geschrieben wurde, sonst None."""
    path = _path(name, root)
    gc_was_on = gc.isenabled()
    try:
        with open(path, 'rb') as f:
            if pickle.load(f) != key:
                return None
            # Millionen kleiner Objekte: ohne zyklische GC lädt der Zustand ein Mehrfaches schneller
            gc.disable()
            return pickle.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:  # abgebrochenes Schreiben, andere Paketversion, ...
        log(f"  Cache {path} unbrauchbar ({type(e).__name__}: {e}), wird neu gebaut")
        return None
    finally:
        if gc_was_on:
            gc.enable()


def save(name, key, value, root=None, log=print):
    """key und value nach <root>/<name>.pkl (atomar über eine temporäre Datei). Fehler brechen nichts ab."""
    path = _path(name, root)
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp, 'wb') as f:
            pickle.dump(key, f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except Exception as e:  # z.B. liest ein anderer Prozess die Datei gerade (Windows sperrt sie dann)
        log(f"  Cache {path} nicht geschrieben ({type(e).__name__}: {e})")
        try:
            os.remove(tmp)
        except OSError:
            pass
