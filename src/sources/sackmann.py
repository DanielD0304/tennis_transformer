"""
Jeff Sackmanns ATP- und WTA-Daten (Ergebnisse + Aufschlag-Stats, dazu Spieler-Metadaten aus
atp_players.csv/wta_players.csv). Das Original-Repo ist seit September 2026 offline, wir nutzen den
Archiv-Mirror auf einem festen Commit, damit die Daten reproduzierbar bleiben.
Lizenz: CC BY-NC-SA 4.0 (nur nicht-kommerziell), Daten von Jeff Sackmann.

Neben der Haupttour liegen im Mirror (Stand bis Turnierstart 01.06.2026) auch:
  ATP: atp_matches_qual_chall_{jahr}.csv (Challenger + Qualifikation der Haupttour),
       atp_matches_futures_{jahr}.csv (ITF M15/M25)
  WTA: wta_matches_qual_itf_{jahr}.csv (WTA 125, ITF W15–W100 + Qualifikation der Haupttour)
Die Spalte 'level' ordnet jedes Match 'main', 'challenger' oder 'itf' zu (LEVEL_OF), tourney_level behält
den Sackmann-Code. Qualifikationen der Haupttour zählen als 'main' (gleiche Turniere, gleiche Märkte).
"""
import os

from .common import RAW_DIR, http_get, read_csv, status_from_score

MIRROR_COMMIT = "83733587353df8a41f2fd4f516147d5aa83f5a8d"
MIRROR_URL = f"https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/{MIRROR_COMMIT}"
BASE_URL = f"{MIRROR_URL}/atp"
DIR = os.path.join(RAW_DIR, "sackmann")


LEVELS = ('main', 'challenger', 'itf')
# Datei-Art je Tour -> Dateiname ohne Jahr; welche Datei-Arten für welche Level geladen werden
FILES = {
    'atp': {'main': 'atp_matches', 'qual_chall': 'atp_matches_qual_chall', 'futures': 'atp_matches_futures'},
    'wta': {'main': 'wta_matches', 'qual_itf': 'wta_matches_qual_itf'},
}
KINDS_FOR_LEVEL = {
    'atp': {'main': ('main', 'qual_chall'), 'challenger': ('qual_chall',), 'itf': ('futures',)},
    'wta': {'main': ('main', 'qual_itf'), 'challenger': ('qual_itf',), 'itf': ('qual_itf',)},
}
MAIN_CODES = ('G', 'M', 'A', 'F', 'D', 'O', 'PM', 'P', 'I', 'W')


def level_of(kind, tourney_level):
    """Datei-Art + Sackmann-Code -> 'main' | 'challenger' | 'itf'."""
    if kind == 'main':
        return 'main'
    if kind == 'futures':
        return 'itf'
    code = str(tourney_level or '').strip()
    if code == 'C':
        return 'challenger'
    if code in MAIN_CODES:
        return 'main'  # Qualifikation der Haupttour
    return 'itf'  # Preisgeld-Codes 15, 25, 35, 50, 75, 100


def kinds(levels, tour='atp'):
    """Datei-Arten, die für die Level gebraucht werden (Reihenfolge wie in FILES)."""
    need = {k for lv in levels for k in KINDS_FOR_LEVEL[tour].get(lv, ())}
    return [k for k in FILES[tour] if k in need]


def path(year, tour='atp', kind='main'):
    return os.path.join(DIR, f"{FILES[tour][kind]}_{year}.csv")


def players_path(tour='atp'):
    return os.path.join(DIR, f"{tour}_players.csv")


def _fetch(url, target):
    r = http_get(url)
    if r is None:
        return False
    with open(target, "wb") as f:
        f.write(r.content)
    return True


def download(years, force=False, tour='atp', levels=('main',)):
    """Lädt fehlende Jahre und die Spielerliste (der Mirror ändert sich nicht, einmal laden reicht)."""
    os.makedirs(DIR, exist_ok=True)
    loaded = []
    for kind in kinds(levels, tour):
        for year in years:
            target = path(year, tour, kind)
            if os.path.exists(target) and not force:
                continue
            if _fetch(f"{MIRROR_URL}/{tour}/{os.path.basename(target)}", target):
                loaded.append((kind, year))
    if force or not os.path.exists(players_path(tour)):
        _fetch(f"{MIRROR_URL}/{tour}/{tour}_players.csv", players_path(tour))
    return loaded


def load(years, tour='atp', levels=('main',)):
    """Matches der gewünschten Level (Spalte 'level'). Fehlende Dateien werden übersprungen."""
    rows = []
    for kind in kinds(levels, tour):
        for year in years:
            if not os.path.exists(path(year, tour, kind)):
                continue
            for r in read_csv(path(year, tour, kind)):
                r['level'] = level_of(kind, r.get('tourney_level'))
                if r['level'] not in levels:
                    continue
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
