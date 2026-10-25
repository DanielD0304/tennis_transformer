"""
Jeff Sackmanns ATP- und WTA-Daten (Ergebnisse + Aufschlag-Stats, dazu Spieler-Metadaten aus
atp_players.csv/wta_players.csv). Das Original-Repo ist seit September 2026 offline, wir nutzen den
Archiv-Mirror auf einem festen Commit, damit die Daten reproduzierbar bleiben.
Lizenz: CC BY-NC-SA 4.0 (nur nicht-kommerziell), Daten von Jeff Sackmann.
"""
import os

from .common import RAW_DIR, http_get, read_csv, status_from_score

MIRROR_COMMIT = "83733587353df8a41f2fd4f516147d5aa83f5a8d"
MIRROR_URL = f"https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/{MIRROR_COMMIT}"
BASE_URL = f"{MIRROR_URL}/atp"
DIR = os.path.join(RAW_DIR, "sackmann")


def path(year, tour='atp'):
    return os.path.join(DIR, f"{tour}_matches_{year}.csv")


def players_path(tour='atp'):
    return os.path.join(DIR, f"{tour}_players.csv")


def _fetch(url, target):
    r = http_get(url)
    if r is None:
        return False
    with open(target, "wb") as f:
        f.write(r.content)
    return True


def download(years, force=False, tour='atp'):
    """Lädt fehlende Jahre und die Spielerliste (der Mirror ändert sich nicht, einmal laden reicht)."""
    os.makedirs(DIR, exist_ok=True)
    loaded = []
    for year in years:
        if os.path.exists(path(year, tour)) and not force:
            continue
        if _fetch(f"{MIRROR_URL}/{tour}/{tour}_matches_{year}.csv", path(year, tour)):
            loaded.append(year)
    if force or not os.path.exists(players_path(tour)):
        _fetch(f"{MIRROR_URL}/{tour}/{tour}_players.csv", players_path(tour))
    return loaded


def load(years, tour='atp'):
    rows = []
    for year in years:
        if not os.path.exists(path(year, tour)):
            continue
        for r in read_csv(path(year, tour)):
            r['tour'] = tour
            r['match_date'] = r['tourney_date']
            r['status'] = status_from_score(r.get('score'))
            r['source'] = 'sackmann'
            rows.append(r)
    return rows


def load_players(tour='atp'):
    """Spielerliste: [{'player_id', 'name', 'hand', 'dob', 'ioc'}] (leer, wenn nicht geladen)."""
    if not os.path.exists(players_path(tour)):
        return []
    out = []
    for r in read_csv(players_path(tour)):
        name = f"{r.get('name_first', '')} {r.get('name_last', '')}".strip()
        if not name:
            continue
        out.append({'player_id': r.get('player_id'), 'name': name, 'hand': r.get('hand') or '',
                    'dob': (r.get('dob') or '')[:8], 'ioc': r.get('ioc') or ''})
    return out
