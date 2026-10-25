"""
tennis-data.co.uk: ATP- und WTA-Ergebnisse mit Rankings, Untergrund, Indoor/Outdoor und Quoten
(Bet365, Pinnacle, Durchschnitt, Maximum, ab August 2025 Betfair Exchange). Wird etwa einmal pro Woche nach Turnierende aktualisiert. Keine Aufschlag-Stats.

Dateien: ATP unter data/raw/tuk/{jahr}.xlsx, WTA unter data/raw/tuk_wta/{jahr}.xlsx. Auf der Seite
liegen die WTA-Dateien in einem eigenen Ordner mit 'w' ({jahr}w/{jahr}.xlsx), sonst gleiches Format
(statt 'Series' heißt die Turnierstufe 'Tier'). Fehlt eine Datei, wird sie übersprungen.
"""
import io
import os
import re
import time
from collections import defaultdict

import requests

from .common import ODDS_COLUMNS, RAW_DIR, ROUND_ORDER, USER_AGENT, http_get, round_code, yyyymmdd

BASE_URL = "https://www.tennis-data.co.uk"
DIR = os.path.join(RAW_DIR, "tuk")
DIRS = {'atp': DIR, 'wta': os.path.join(RAW_DIR, "tuk_wta")}

_links = None


def parse_links(html):
    """alldata.php -> {(tour, jahr): Pfad}. WTA-Ordner enden auf 'w' (z.B. /2026w/2026.xlsx)."""
    links = {}
    for p, y, w in re.findall(r'href="([^"]*?/(\d{4})(w?)/\d{4}\.xlsx?)"', html, flags=re.I):
        links.setdefault(('wta' if w else 'atp', int(y)), f"{BASE_URL}/{p.lstrip('/')}")
    return links


def url(year, tour='atp'):
    """
    Die Dateien liegen unter einem wechselnden Pfad (z.B. /hrjk-.../2026/2026.xlsx),
    der aktuelle Link steht auf alldata.php.
    """
    global _links
    if _links is None:
        _links = {}
        try:
            r = requests.get(f"{BASE_URL}/alldata.php", headers={'User-Agent': USER_AGENT}, timeout=15)
            _links = parse_links(r.text)
        except requests.RequestException:
            pass
    folder = f"{year}w" if tour == 'wta' else f"{year}"
    return _links.get((tour, year), f"{BASE_URL}/{folder}/{year}.xlsx")


def path(year, tour='atp'):
    ext = ".xls" if year <= 2012 else ".xlsx"
    return os.path.join(DIRS[tour], f"{year}{ext}")


def missing(years, tour='atp'):
    """Jahre ohne lokale Datei (z.B. weil tennis-data Cloud-IPs blockt und sie hochgeladen werden müssen)."""
    return [y for y in years if not os.path.exists(path(y, tour))]


def download(years, max_age_hours=None, tour='atp'):
    """
    Lädt fehlende Jahre. Mit max_age_hours werden ältere Dateien neu geladen
    (für das laufende Jahr, das wöchentlich ergänzt wird).
    """
    os.makedirs(DIRS[tour], exist_ok=True)
    loaded = []
    for year in years:
        p = path(year, tour)
        if os.path.exists(p):
            if max_age_hours is None or time.time() - os.path.getmtime(p) < max_age_hours * 3600:
                continue
        r = http_get(url(year, tour), retries=1 if tour == "wta" else 3)
        if r is None or len(r.content) < 10_000:  # Fehlerseiten sind klein
            continue
        with open(p, "wb") as f:
            f.write(r.content)
        loaded.append(year)
    return loaded


def _num(v):
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _int(v):
    n = _num(v)
    return int(n) if n is not None else None


STATUS = {'Completed': 'completed', 'Retired': 'RET', 'Walkover': 'W/O', 'Awarded': 'DEF', 'Disqualified': 'DEF'}
GRAND_SLAMS = ('Australian Open', 'French Open', 'Wimbledon', 'US Open')
# tennis-data 'Series' -> Sackmann tourney_level
LEVELS = {'Grand Slam': 'G', 'Masters 1000': 'M', 'Masters': 'M', 'Masters Cup': 'F', 'ATP500': 'A', 'ATP250': 'A',
          'International Gold': 'A', 'International': 'A'}
# WTA: 'Tier' -> Sackmann-WTA-Stufen (PM = 1000er, P = 500er, I = 250er)
WTA_LEVELS = {'Grand Slam': 'G', 'WTA1000': 'PM', 'Premier Mandatory': 'PM', 'Premier 5': 'PM', 'WTA500': 'P',
              'Premier': 'P', 'WTA250': 'I', 'International': 'I', 'Tour Championships': 'F', 'Elite': 'F'}


def _round(tuk_round, max_round):
    m = re.match(r"(\d)\w\w Round", tuk_round or "")
    if m:
        # max_round = höchste "n-th Round" im Turnier = letzte Runde vor dem Viertelfinale
        return round_code(int(m.group(1)), max_round)
    return {'Quarterfinals': 'QF', 'Semifinals': 'SF', 'The Final': 'F', 'Round Robin': 'RR'}.get(tuk_round, tuk_round or "")


def load(years, tour='atp'):
    """TUK-Zeilen im gemeinsamen Schema (Namen noch im TUK-Format 'Alcaraz C.')."""
    try:
        import openpyxl
    except ImportError:
        raise RuntimeError("openpyxl wird für tennis-data.co.uk benötigt")

    rows = []
    for year in years:
        p = path(year, tour)
        if not os.path.exists(p) or p.endswith(".xls"):
            continue  # .xls (bis 2012) bräuchte xlrd, wird ab 2015 nicht gebraucht
        ws = openpyxl.load_workbook(p, read_only=True).active
        data = list(ws.iter_rows(values_only=True))
        header = [str(h).strip().replace(" ", "") if h else "" for h in data[0]]
        records = [dict(zip(header, r)) for r in data[1:] if r and r[header.index('Winner')]]

        # Turniere gruppieren: Startdatum und Runden-Anzahl
        by_tourney = defaultdict(list)
        for r in records:
            by_tourney[(r.get('Location'), r.get('Tournament'))].append(r)

        for (location, tournament), matches in by_tourney.items():
            start = min(yyyymmdd(r['Date']) for r in matches)
            numbered = [int(m.group(1)) for r in matches if (m := re.match(r"(\d)\w\w Round", r.get('Round') or ""))]
            max_round = max(numbered) if numbered else 1
            for i, r in enumerate(matches):
                sets = []
                for n in range(1, 6):
                    w, l = _int(r.get(f'W{n}')), _int(r.get(f'L{n}'))
                    if w is not None and l is not None:
                        sets.append(f"{w}-{l}")
                status = STATUS.get(r.get('Comment'), 'completed')
                score = " ".join(sets) + (" RET" if status == 'RET' else "") if status != 'W/O' else "W/O"
                rnd = _round(r.get('Round'), max_round)
                if tour == 'wta':
                    level = WTA_LEVELS.get(r.get('Tier') or r.get('Series'), 'I')
                else:
                    level = LEVELS.get(r.get('Series'), 'A')
                rows.append({
                    'tour': tour,
                    'tourney_id': f"tuk-{year}-{location}" if tour == 'atp' else f"tuk-wta-{year}-{location}",
                    'tourney_name': tournament,
                    'surface': r.get('Surface') or "",
                    'court': r.get('Court') or "",
                    'tourney_level': level,
                    'tourney_date': start,
                    'match_date': yyyymmdd(r['Date']),
                    'match_num': ROUND_ORDER.get(rnd, 0) * 1000 + i,
                    'round': rnd,
                    'best_of': _int(r.get('Bestof')) or (5 if tour == 'atp' and tournament in GRAND_SLAMS else 3),
                    'winner_name': str(r['Winner']).strip(),
                    'loser_name': str(r['Loser']).strip(),
                    'winner_rank': _int(r.get('WRank')),
                    'loser_rank': _int(r.get('LRank')),
                    'score': score.strip(),
                    'status': status,
                    **{c: _num(r.get(c)) for c in ODDS_COLUMNS},
                    'location': location,
                    'source': 'tuk', 'level': 'main',
                })
    return rows
