"""
Führt Sackmann, tennis-data.co.uk und ESPN zu einem Datensatz zusammen.

Priorität pro Match: Sackmann (mit Stats) > tennis-data > ESPN. tennis-data und ESPN werden nur
für die Zeit nach dem Ende der Sackmann-Daten als eigene Matches übernommen, davor liefern sie
Quoten und helfen beim Namensabgleich.

Gleiche Matches werden über ein Zeitfenster (Sackmann kennt nur das Turnier-Startdatum, die
anderen das echte Matchdatum) und gemeinsame Namensbestandteile beider Spieler erkannt. Aus jedem
erkannten Duplikat lernen wir, dass zwei Schreibweisen derselbe Spieler sind ('Wu Yibing' =
'Yibing Wu', 'Mpetshi G.' = 'Giovanni Mpetshi Perricard'), und vereinheitlichen am Ende alle Namen.
"""
import re
import unicodedata
from bisect import bisect_right
from difflib import SequenceMatcher
from collections import Counter, defaultdict
from datetime import datetime, timedelta

from functools import lru_cache

from ..names import candidate_keys
from .common import ODDS_COLUMNS

# Fälle, die sich automatisch nicht sicher zuordnen lassen (andere Vornamen/Abkürzungen je Quelle).
# Schreibweise links -> Name im Datensatz rechts. Bei neuen Funden einfach ergänzen.
MANUAL_ALIASES = {
    'Barrios M.': 'Tomas Barrios Vera',          # Marcelo Tomas Barrios Vera
    'Schwaerzler J.J.': 'Joel Schwaerzler',      # Joel Josef Schwaerzler
}

VALID_SURFACES = ('Hard', 'Clay', 'Grass', 'Carpet')
VALID_LEVELS = ('G', 'M', 'A', 'F', 'D', 'O')
GRAND_SLAMS = ('australian open', 'roland garros', 'french open', 'wimbledon', 'us open')

STOPWORDS = {'open', 'masters', 'championships', 'championship', 'tennis', 'international', 'atp',
             'cup', 'rolex', 'bnp', 'paribas', 'the', 'and', 'tour', 'classic', 'trophy', 'mens', 'men'}


def _norm(name):
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z]", "", s)


def _date(s):
    return datetime.strptime(str(s)[:8], "%Y%m%d")


def _is_full(name):
    """Voller Name ('Daniil Medvedev') statt Kurzform ('Medvedev D.')."""
    return not re.search(r"(^|\s)(?!Jr\.|Sr\.)[A-Z][a-z]?\.", str(name))


class NameRegistry:
    """Bildet Schreibweisen über die Namens-Schlüssel auf einen vollen Namen ab."""

    def __init__(self):
        self.index = defaultdict(dict)  # key -> {normalisierter Name: Originalname}

    def add_full(self, name):
        if not name:
            return
        for key in candidate_keys(name, reverse=False):
            self.index[key].setdefault(_norm(name), name)

    def canonical(self, name):
        if not name:
            return name
        norm = _norm(name)
        for key in candidate_keys(name, reverse=False):
            variants = self.index.get(key)
            if not variants:
                continue
            if norm in variants:
                return variants[norm]
            if len(variants) == 1:
                return next(iter(variants.values()))
        return name


class Aliases:
    """Union-Find über Spielernamen."""

    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def groups(self):
        out = defaultdict(set)
        for x in list(self.parent):
            out[self.find(x)].add(x)
        return [g for g in out.values() if len(g) > 1]


def _window(row):
    if row['source'] == 'sackmann':
        start = _date(row['tourney_date'])
        return start - timedelta(days=1), start + timedelta(days=16)
    d = _date(row['match_date'] or row['tourney_date'])
    return d - timedelta(days=2), d + timedelta(days=2)


@lru_cache(maxsize=None)
def _keys(name):
    return frozenset(candidate_keys(name))


@lru_cache(maxsize=None)
def _surnames(name):
    """Nachnamen-Varianten ohne Initial, zum Vorsortieren im Index ('vallejo', 'danielvallejo', ...)."""
    return frozenset(k[0] for k in candidate_keys(name)) | frozenset(_tokens(name)[-1:])


@lru_cache(maxsize=None)
def _tokens(name):
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return tuple(t for t in re.sub(r"[^a-z ]", " ", s.replace("-", " ")).split())


@lru_cache(maxsize=None)
def _same_player(a, b):
    """
    Können zwei Schreibweisen derselbe Spieler sein?

    Kurzform ('Mpetshi G.') vs. voller Name: ihr Schlüssel muss einer der Kandidaten-Schlüssel
    des vollen Namens sein. Zwei volle Namen: gleiche Bestandteile in anderer Reihenfolge
    ('Wu Yibing'), Teilname ('Coleman Wong' in 'Chak Lam Coleman Wong') oder gleicher Nachname
    mit ähnlichem Vorname ('Aleksandr'/'Alexander'). Nur gleicher Schlüssel reicht bei vollen
    Namen nicht ('Arthur Fils' ist nicht 'Arthur Fery', 'Eduardo Nava' nicht 'Emilio Nava').
    """
    if a == b:
        return True
    full_a, full_b = _is_full(a), _is_full(b)
    if not (full_a and full_b):
        if full_a or full_b:
            abbr, full = (b, a) if full_a else (a, b)
            return candidate_keys(abbr)[:1] and candidate_keys(abbr)[0] in _keys(full)
        return candidate_keys(a)[:1] == candidate_keys(b)[:1] != []
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    if sorted(ta) == sorted(tb) or "".join(ta) == "".join(tb):
        return True
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    # Teilname: gleicher Nachname am Ende oder gleicher Vorname (spanische Doppelnamen)
    if len(short) >= 2 and set(short) <= set(long_) and (short[-1] == long_[-1] or short[0] == long_[0]):
        return True
    if candidate_keys(a)[0] == candidate_keys(b)[0] and ta[-1] == tb[-1]:
        return SequenceMatcher(None, "".join(ta[:-1]), "".join(tb[:-1])).ratio() >= 0.75
    return False


def _weeks(lo, hi):
    return range(lo.toordinal() // 7, hi.toordinal() // 7 + 1)


class MatchIndex:
    """Findet dasselbe Match aus einer anderen Quelle (Zeitfenster + beide Spieler passen)."""

    def __init__(self):
        self.buckets = defaultdict(list)  # (Nachname, Woche) -> Matches

    def add(self, row):
        lo, hi = _window(row)
        for key in _surnames(row['winner_name']) | _surnames(row['loser_name']):
            for week in _weeks(lo, hi):
                self.buckets[(key, week)].append((row, lo, hi))

    def find(self, row):
        """-> (anderes Match, gleiche Reihenfolge?) oder (None, None)."""
        lo, hi = _window(row)
        w, l = row['winner_name'], row['loser_name']
        seen = set()
        candidates = (c for key in _surnames(w) for week in _weeks(lo, hi) for c in self.buckets.get((key, week), []))
        for other, olo, ohi in candidates:
            if id(other) in seen or not (lo <= ohi and olo <= hi):
                continue
            seen.add(id(other))
            if _same_player(w, other['winner_name']) and _same_player(l, other['loser_name']):
                return other, True
            if _same_player(w, other['loser_name']) and _same_player(l, other['winner_name']):
                return other, False
        return None, None


def _tourney_lookup(rows, field, valid, default):
    """Turniername-Wörter und Orte -> Wert von field, z.B. Untergrund (der neueste Eintrag gewinnt)."""
    by_word, by_location = {}, {}
    for r in sorted(rows, key=lambda r: r['tourney_date']):
        value = r.get(field)
        if value not in valid:
            continue
        for w in re.findall(r"[a-z]+", _norm_words(r.get('tourney_name', ''))):
            if len(w) > 3 and w not in STOPWORDS:
                by_word[w] = value
        if r.get('location'):
            by_location[_norm(r['location'])] = value

    def lookup(name, location):
        for w in re.findall(r"[a-z]+", _norm_words(name)):
            if w in by_word:
                return by_word[w]
        return by_location.get(_norm(location or ''), default)
    return lookup


def _norm_words(text):
    s = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def _rank_lookup(rows):
    timeline = defaultdict(list)
    for r in rows:
        for name, rank in ((r['winner_name'], r.get('winner_rank')), (r['loser_name'], r.get('loser_rank'))):
            try:
                rank = int(float(rank))
            except (TypeError, ValueError):
                continue
            timeline[name].append((r['match_date'] or r['tourney_date'], rank))
    for v in timeline.values():
        v.sort()

    def lookup(name, day):
        v = timeline.get(name)
        if not v:
            return None
        i = bisect_right(v, (day, float('inf')))
        return v[i - 1][1] if i else v[0][1]
    return lookup


def _copy_odds(src, dst, same_order):
    for c in ODDS_COLUMNS:
        # Quoten beziehen sich auf Sieger/Verlierer -> bei vertauschter Reihenfolge tauschen
        s = c if same_order else c[:-1] + ('L' if c.endswith('W') else 'W')
        dst[c] = src.get(s)


def build(sackmann_rows, tuk_rows, espn_rows):
    registry = NameRegistry()
    for r in sackmann_rows + espn_rows:
        registry.add_full(r['winner_name'])
        registry.add_full(r['loser_name'])
    for r in sackmann_rows + tuk_rows + espn_rows:
        r['winner_name'] = registry.canonical(MANUAL_ALIASES.get(r['winner_name'], r['winner_name']))
        r['loser_name'] = registry.canonical(MANUAL_ALIASES.get(r['loser_name'], r['loser_name']))

    cutoff = max((r['tourney_date'] for r in sackmann_rows), default="00000000")

    out = list(sackmann_rows)
    index = MatchIndex()
    for r in out:
        index.add(r)

    aliases = Aliases()
    abbr_votes = defaultdict(Counter)  # Kurzform -> {voller Name: Anzahl Duplikate}
    stats = Counter(sackmann=len(out), tuk=0, espn=0, duplicates=0)
    for source, rows in (('tuk', tuk_rows), ('espn', espn_rows)):
        for r in rows:
            other, same_order = index.find(r)
            if other is not None:
                # gleiches Match: Namen verknüpfen, Quoten übernehmen
                pairs = ((r['winner_name'], other['winner_name']), (r['loser_name'], other['loser_name'])) if same_order \
                    else ((r['winner_name'], other['loser_name']), (r['loser_name'], other['winner_name']))
                for a, b in pairs:
                    if a == b:
                        continue
                    if _is_full(a) and _is_full(b):
                        aliases.union(b, a)
                    elif _is_full(b):
                        abbr_votes[a][b] += 1
                    elif _is_full(a):
                        abbr_votes[b][a] += 1
                if source == 'tuk' and other.get('B365W') in (None, ""):
                    _copy_odds(r, other, same_order)
                if (r['match_date'] or r['tourney_date']) > cutoff:
                    stats['duplicates'] += 1
                continue
            if (r['match_date'] or r['tourney_date']) <= cutoff:
                continue
            out.append(r)
            index.add(r)
            stats[source] += 1

    # Pro Spieler ein Name: bevorzugt voller Name, dann der mit den meisten Sackmann-Matches
    sack_count = Counter(n for r in sackmann_rows for n in (r['winner_name'], r['loser_name']))
    all_count = Counter(n for r in out for n in (r['winner_name'], r['loser_name']))
    rename = {}
    for group in aliases.groups():
        best = max(group, key=lambda n: (sack_count[n], all_count[n], n))
        for n in group:
            if n != best:
                rename[n] = best
    # Kurzformen nur, wenn sie eindeutig auf einen Spieler zeigen ('Martin A.' kann mehrere sein)
    for abbr, votes in abbr_votes.items():
        merged = Counter()
        for full, n in votes.items():
            merged[rename.get(full, full)] += n
        target, n = merged.most_common(1)[0]
        if n >= 0.8 * sum(merged.values()):
            rename[abbr] = target
    for r in out:
        r['winner_name'] = rename.get(r['winner_name'], r['winner_name'])
        r['loser_name'] = rename.get(r['loser_name'], r['loser_name'])
    stats['renamed_players'] = len(rename)

    # ESPN: Untergrund, Turnierstufe und Ranking ergänzen
    surface = _tourney_lookup(sackmann_rows + tuk_rows, 'surface', VALID_SURFACES, 'Hard')
    level = _tourney_lookup(sackmann_rows + tuk_rows, 'tourney_level', VALID_LEVELS, 'A')
    rank = _rank_lookup(out)
    for r in out:
        if r['source'] != 'espn':
            continue
        r['surface'] = surface(r['tourney_name'], r.get('location'))
        r['tourney_level'] = 'G' if any(g in _norm_words(r['tourney_name']) for g in GRAND_SLAMS)             else level(r['tourney_name'], r.get('location'))
        r['winner_rank'] = rank(r['winner_name'], r['match_date'])
        r['loser_rank'] = rank(r['loser_name'], r['match_date'])

    out = clean(out, stats)
    stats['with_odds'] = sum(1 for r in out if r.get('B365W') not in (None, ""))
    out.sort(key=lambda r: (str(r['tourney_date']), int(float(r.get('match_num') or 0))))
    return out, dict(stats)


def _odd(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def clean(rows, stats):
    """
    Letzte Bereinigung, Zähler landen in stats (und in data/update_state.json):
      - exakte Doppel-Einträge (gleiches Turnier, Paar, Runde, Ergebnis) entfernen
      - fehlender Untergrund -> Hard (so behandelt das Modell unbekannten Untergrund ohnehin)
      - unplausible Quoten leeren (<= 1.0, ohne Gegenquote oder Buchmacher-Marge außerhalb 95-135 %)
    """
    seen, out = set(), []
    for r in rows:
        key = (r.get('tourney_id') or r.get('tourney_name'), r['tourney_date'], frozenset((r['winner_name'], r['loser_name'])), r.get('round'), r.get('score'))
        if key in seen:
            stats['clean_duplicates'] += 1
            continue
        seen.add(key)
        out.append(r)

    for r in out:
        if r.get('surface') not in VALID_SURFACES:
            r['surface'] = 'Hard'
            stats['clean_surface_filled'] += 1
        for w, l in (('B365W', 'B365L'), ('PSW', 'PSL'), ('AvgW', 'AvgL')):
            ow, ol = _odd(r.get(w)), _odd(r.get(l))
            if ow is None and ol is None:
                continue
            if ow is None or ol is None or min(ow, ol) <= 1.0 or not 0.95 <= 1 / ow + 1 / ol <= 1.35:
                r[w] = r[l] = None
                stats['clean_odds_removed'] += 1
    return out
