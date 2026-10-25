import csv
import os
import re
import time

import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
MATCHES_PATH = os.path.join(DATA_DIR, "matches.csv")

USER_AGENT = "Mozilla/5.0 (tennis_transformer; private research)"

STAT_COLUMNS = [
    f"{p}_{s}" for p in ("w", "l")
    for s in ("ace", "df", "svpt", "1stIn", "1stWon", "2ndWon", "SvGms", "bpSaved", "bpFaced")
]
# Quoten im Sieger/Verlierer-Format (tennis-data): Bet365, Pinnacle, Durchschnitt, Maximum und
# Betfair Exchange (BFE, ab August 2025; Pinnacle fehlt dort seit Februar 2026)
ODDS_PAIRS = [('B365W', 'B365L'), ('PSW', 'PSL'), ('AvgW', 'AvgL'), ('MaxW', 'MaxL'), ('BFEW', 'BFEL')]
ODDS_COLUMNS = [c for pair in ODDS_PAIRS for c in pair]

TOURS = ('atp', 'wta')
# Spieler-Metadaten (aus Sackmann *_players.csv), pro Match für Sieger und Verlierer
PLAYER_COLUMNS = [f"{p}_{s}" for p in ("winner", "loser") for s in ("ioc", "hand", "dob")]

LEVELS = ('main', 'challenger', 'itf')
# Sackmann-Schema plus tour, level ('main' | 'challenger' | 'itf'), match_date (echtes Matchdatum, falls bekannt), status und source.
# court (Indoor/Outdoor), location und tourney_ioc (Land des Turniers) ergänzt build.py.
COLUMNS = [
    'tour', 'level', 'tourney_id', 'tourney_name', 'surface', 'court', 'tourney_level', 'tourney_date', 'match_date',
    'match_num', 'round', 'best_of', 'winner_name', 'loser_name', 'winner_rank', 'loser_rank',
    'score', 'minutes', 'status', 'location', 'tourney_ioc',
] + PLAYER_COLUMNS + STAT_COLUMNS + ODDS_COLUMNS + ['source']

# Reihenfolge der Runden innerhalb eines Turniers (für match_num)
ROUND_ORDER = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'R128': 10, 'R64': 20, 'R32': 30, 'R16': 40,
               'RR': 45, 'QF': 50, 'SF': 60, 'BR': 65, 'F': 70, 'ER': 5}


def round_code(number, rounds_before_qf):
    """Runde N (1-basiert) eines Turniers mit k Runden vor dem Viertelfinale -> 'R32' usw."""
    return f"R{2 ** (rounds_before_qf - number + 4)}"


def status_from_score(score):
    s = (score or "").upper()
    if "W/O" in s or "WALKOVER" in s:
        return "W/O"
    if "RET" in s:
        return "RET"
    if "DEF" in s:
        return "DEF"
    return "completed"



def yyyymmdd(value):
    """'2026-10-05T04:00Z', datetime oder 20261005 -> '20261005'."""
    if value is None or value == "":
        return ""
    if hasattr(value, "strftime"):
        return value.strftime("%Y%m%d")
    digits = re.sub(r"\D", "", str(value))
    return digits[:8]


def http_get(url, retries=3, timeout=30, **kwargs):
    headers = {'User-Agent': USER_AGENT, **kwargs.pop('headers', {})}
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout, **kwargs)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(2 * (attempt + 1))
    return None


def write_matches(rows, path=MATCHES_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({c: ("" if row.get(c) is None else row.get(c)) for c in COLUMNS})
    os.replace(tmp, path)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
