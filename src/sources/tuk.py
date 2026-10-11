"""
tennis-data.co.uk: ATP-Ergebnisse mit Rankings, Untergrund und Quoten (u.a. Bet365, Pinnacle).
Wird etwa einmal pro Woche nach Turnierende aktualisiert. Keine Aufschlag-Stats.
"""
import io
import os
import re
import time
from collections import defaultdict

import requests

from .common import RAW_DIR, ROUND_ORDER, USER_AGENT, http_get, round_code, yyyymmdd

BASE_URL = "https://www.tennis-data.co.uk"
DIR = os.path.join(RAW_DIR, "tuk")

_links = None


def url(year):
    """
    Die Dateien liegen unter einem wechselnden Pfad (z.B. /hrjk-.../2026/2026.xlsx),
    der aktuelle Link steht auf alldata.php.
    """
    global _links
    if _links is None:
        _links = {}
        try:
            r = requests.get(f"{BASE_URL}/alldata.php", headers={'User-Agent': USER_AGENT}, timeout=15)
            for p, y in re.findall(r'href="([^"]*?/(\d{4})/\d{4}\.xlsx?)"', r.text, flags=re.I):
                _links.setdefault(int(y), f"{BASE_URL}/{p.lstrip('/')}")
        except requests.RequestException:
            pass
    return _links.get(year, f"{BASE_URL}/{year}/{year}.xlsx")


def path(year):
    ext = ".xls" if year <= 2012 else ".xlsx"
    return os.path.join(DIR, f"{year}{ext}")


def download(years, max_age_hours=None):
    """
    Lädt fehlende Jahre. Mit max_age_hours werden ältere Dateien neu geladen
    (für das laufende Jahr, das wöchentlich ergänzt wird).
    """
    os.makedirs(DIR, exist_ok=True)
    loaded = []
    for year in years:
        p = path(year)
        if os.path.exists(p):
            if max_age_hours is None or time.time() - os.path.getmtime(p) < max_age_hours * 3600:
                continue
        r = http_get(url(year))
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


def _round(tuk_round, max_round):
    m = re.match(r"(\d)\w\w Round", tuk_round or "")
    if m:
        # max_round = höchste "n-th Round" im Turnier = letzte Runde vor dem Viertelfinale
        return round_code(int(m.group(1)), max_round)
    return {'Quarterfinals': 'QF', 'Semifinals': 'SF', 'The Final': 'F', 'Round Robin': 'RR'}.get(tuk_round, tuk_round or "")


def load(years):
    """TUK-Zeilen im gemeinsamen Schema (Namen noch im TUK-Format 'Alcaraz C.')."""
    try:
        import openpyxl
    except ImportError:
        raise RuntimeError("openpyxl wird für tennis-data.co.uk benötigt")

    rows = []
    for year in years:
        p = path(year)
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
                rows.append({
                    'tourney_id': f"tuk-{year}-{location}",
                    'tourney_name': tournament,
                    'surface': r.get('Surface') or "",
                    'tourney_level': LEVELS.get(r.get('Series'), 'A'),
                    'tourney_date': start,
                    'match_date': yyyymmdd(r['Date']),
                    'match_num': ROUND_ORDER.get(rnd, 0) * 1000 + i,
                    'round': rnd,
                    'best_of': _int(r.get('Bestof')) or (5 if tournament in GRAND_SLAMS else 3),
                    'winner_name': str(r['Winner']).strip(),
                    'loser_name': str(r['Loser']).strip(),
                    'winner_rank': _int(r.get('WRank')),
                    'loser_rank': _int(r.get('LRank')),
                    'score': score.strip(),
                    'status': status,
                    **{c: _num(r.get(c)) for c in ('B365W', 'B365L', 'PSW', 'PSL', 'AvgW', 'AvgL')},
                    'location': location,
                    'source': 'tuk',
                })
    return rows
