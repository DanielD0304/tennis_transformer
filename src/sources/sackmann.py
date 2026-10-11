"""
Jeff Sackmanns ATP-Daten (Ergebnisse + Aufschlag-Stats). Das Original-Repo ist seit
September 2026 offline, wir nutzen den Archiv-Mirror auf einem festen Commit, damit die
Daten reproduzierbar bleiben. Lizenz: CC BY-NC-SA 4.0 (nur nicht-kommerziell), Daten von Jeff Sackmann.
"""
import os

from .common import RAW_DIR, http_get, read_csv, status_from_score

MIRROR_COMMIT = "83733587353df8a41f2fd4f516147d5aa83f5a8d"
BASE_URL = f"https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/{MIRROR_COMMIT}/atp"
DIR = os.path.join(RAW_DIR, "sackmann")


def path(year):
    return os.path.join(DIR, f"atp_matches_{year}.csv")


def download(years, force=False):
    """Lädt fehlende Jahre (der Mirror ändert sich nicht, einmal laden reicht)."""
    os.makedirs(DIR, exist_ok=True)
    loaded = []
    for year in years:
        if os.path.exists(path(year)) and not force:
            continue
        r = http_get(f"{BASE_URL}/atp_matches_{year}.csv")
        if r is None:
            continue
        with open(path(year), "wb") as f:
            f.write(r.content)
        loaded.append(year)
    return loaded


def load(years):
    rows = []
    for year in years:
        if not os.path.exists(path(year)):
            continue
        for r in read_csv(path(year)):
            r['match_date'] = r['tourney_date']
            r['status'] = status_from_score(r.get('score'))
            r['source'] = 'sackmann'
            rows.append(r)
    return rows
