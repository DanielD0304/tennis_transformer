"""
Führt Sackmann, TennisMyLife, tennis-data.co.uk und ESPN zu einem Datensatz zusammen.

Priorität pro Match: Sackmann (mit Stats) > TennisMyLife (mit Stats) > tennis-data > ESPN > Kalshi > Polymarket.
Alle Quellen außer Sackmann werden nur für die Zeit nach dem Ende der Sackmann-Daten als eigene Matches
übernommen, davor liefern sie Quoten, fehlende Stats bzw. das echte Matchdatum und helfen beim Namensabgleich.
TennisMyLife (sources/tml.py) deckt danach ATP-Haupttour, Challenger und WTA-Haupttour mit Stats ab. Für ITF und
WTA 125 sind die Prediction Markets (sources/markets.py, nur Sieger ohne Score) nach dem Sackmann-Ende die einzige
Quelle (ESPN zeigt nur die Haupttour).

ATP und WTA werden getrennt zusammengeführt: Matches und Namen werden nie über Touren hinweg
verknüpft (gleicher Nachname bei Herren und Damen ist nicht derselbe Spieler).

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

from ..names import MAX_ABBREVIATION, candidate_keys, given_prefix
from .common import LEVELS, ODDS_COLUMNS, ODDS_PAIRS, STAT_COLUMNS
from .places import tourney_country

# Fälle, die sich automatisch nicht sicher zuordnen lassen (andere Vornamen/Abkürzungen je Quelle).
# Schreibweise links -> Name im Datensatz rechts. Bei neuen Funden einfach ergänzen.
MANUAL_ALIASES = {
    'Barrios M.': 'Tomas Barrios Vera',          # Marcelo Tomas Barrios Vera
    'Schwaerzler J.J.': 'Joel Schwaerzler',      # Joel Josef Schwaerzler
    # Namenswechsel: tennis-data führt die Jahre davor unter dem alten Namen, Sackmann alles unter dem neuen
    'Gavrilova D.': 'Daria Saville',
    'Kozlova K.': 'Kateryna Baindl',
    'Kuzmova V.': 'Viktoria Hruncakova',
    'Sanders S.': 'Storm Hunter',                # sonst landet sie über den Schlüssel bei 'Sophia Sanders'
}

VALID_SURFACES = ('Hard', 'Clay', 'Grass', 'Carpet')
# ATP: G, M, A, F, D, O; WTA zusätzlich PM (1000er), P (500er), I (250er), W
VALID_LEVELS = ('G', 'M', 'A', 'F', 'D', 'O', 'PM', 'P', 'I', 'W')
VALID_COURTS = ('Indoor', 'Outdoor')
GRAND_SLAMS = ('australian open', 'roland garros', 'french open', 'wimbledon', 'us open')

STOPWORDS = {'open', 'masters', 'championships', 'championship', 'tennis', 'international', 'atp',
             'cup', 'rolex', 'bnp', 'paribas', 'the', 'and', 'tour', 'classic', 'trophy', 'mens', 'men'}


def _norm(name):
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z]", "", s)


def _date(s):
    return datetime.strptime(str(s)[:8], "%Y%m%d")


_ABBREVIATION = re.compile(r"(^|\s)(?!Jr\.|Sr\.)[A-Z][a-z]{0,%d}\." % (MAX_ABBREVIATION - 1))


def _is_full(name):
    """
    Voller Name ('Daniil Medvedev') statt Kurzform ('Medvedev D.'). tennis-data kürzt bei gleichem Nachnamen und
    Initial weiter aus ('Wang Xin.' und 'Wang Xiy.', 'Blanch Dar.'), auch das sind Kurzformen.
    """
    return not _ABBREVIATION.search(str(name))


def _rotations(tokens):
    """Namensteile ab jeder Stelle gelesen und zusammengezogen: ('wang', 'xinyu') -> 'wangxinyu', 'xinyuwang'."""
    return ["".join(tokens[i:] + tokens[:i]) for i in range(len(tokens))]


def _fits_prefix(abbr, full):
    """
    Passt das Kürzel einer Kurzform zum Vornamen des vollen Namens? 'Wang Xin.' passt zu 'Xin Yu Wang' und 'Wang Xinyu',
    nicht zu 'Xiyu Wang'. Ein Kürzel aus einem Buchstaben prüft schon der Namens-Schlüssel.
    """
    prefix = given_prefix(abbr)
    if len(prefix) <= 1:
        return True
    tokens = _tokens(full)
    # Vornamen stehen vorn oder hinten und sind teils getrennt geschrieben ('Xin Yu'), deshalb ab jedem Namensteil
    return any("".join(tokens[i:]).startswith(prefix) for i in range(len(tokens)))


class NameRegistry:
    """
    Bildet Schreibweisen auf einen vollen Namen ab: Kurzformen über den Namens-Schlüssel (und ihr Kürzel), volle Namen
    mit denselben Namensteilen in anderer Reihenfolge oder Trennung auf die Schreibweise der Referenzquelle.
    """

    def __init__(self):
        self.index = defaultdict(dict)      # key -> {normalisierter Name: Originalname}
        self.reference = {}                 # normalisierter Name -> Originalname, nur Referenzquelle (Sackmann)
        self.rotations = defaultdict(set)   # rotierter Name -> normalisierte Referenz-Namen

    def add_full(self, name, reference=False):
        if not name:
            return
        norm = _norm(name)
        for key in candidate_keys(name, reverse=False):
            self.index[key].setdefault(norm, name)
        if reference and norm not in self.reference:
            self.reference[norm] = name
            for rotated in _rotations(_tokens(name)):
                self.rotations[rotated].add(norm)

    def _reordered(self, name, norm):
        """
        Referenz-Name mit denselben Namensteilen in anderer Reihenfolge oder Trennung: 'Wang Xinyu' (ESPN) ->
        'Xin Yu Wang', 'Ma YeXin' -> 'Yexin Ma'. Nur für volle Namen, die die Referenzquelle so nicht kennt, und nur
        wenn genau ein Spieler passt.
        """
        if norm in self.reference or not _is_full(name):
            return None
        found = self.rotations.get(norm, ())
        return self.reference[next(iter(found))] if len(found) == 1 else None

    def canonical(self, name):
        if not name:
            return name
        norm = _norm(name)
        reordered = self._reordered(name, norm)
        if reordered:
            return reordered
        abbreviated = len(given_prefix(name)) > 1 and not _is_full(name)
        for key in candidate_keys(name, reverse=False):
            variants = self.index.get(key)
            if not variants:
                continue
            if norm in variants:
                return variants[norm]
            if abbreviated:
                # 'Blanch Dar.' meint Darwin, nicht Dali Blanch
                variants = {n: v for n, v in variants.items() if _fits_prefix(name, v)}
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


def _start_date_only(row):
    """
    Kennt die Zeile nur den Turnierstart? Sackmann immer. TennisMyLife bei Challenger und WTA: dort gleicht das
    Matchdatum dem Turnierstart. Das trifft auch Matches vom ersten Turniertag mit echtem Datum, für sie ist das
    weite Fenster nur großzügiger als nötig.
    """
    if row['source'] == 'sackmann':
        return True
    return row['source'] == 'tml' and row.get('match_date') in (None, "", row['tourney_date'])


# Qualifikationsrunden ohne eigenes Datum tragen den Start des Hauptfelds, gespielt werden sie bis zu 10 Tage davor
# (Grand Slams: Qualifikation in der Woche vor dem Turnier)
QUALI_ROUNDS = ('Q1', 'Q2', 'Q3', 'Q4')
QUALI_LEAD_DAYS = 10


def _window(row):
    if _start_date_only(row):
        start = _date(row['tourney_date'])
        lead = QUALI_LEAD_DAYS if row.get('round') in QUALI_ROUNDS else 1
        return start - timedelta(days=lead), start + timedelta(days=16)
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
    des vollen Namens sein, und ein Kürzel aus mehreren Buchstaben muss zum Vornamen passen ('Wang Xin.' ist
    'Xin Yu Wang', nicht 'Xiyu Wang'). Zwei Kurzformen: gleicher Schlüssel und verträgliche Kürzel.
    Zwei volle Namen: gleiche Bestandteile in anderer Reihenfolge oder Trennung ('Wu Yibing', 'Wang Xinyu' und
    'Xin Yu Wang'), Teilname ('Coleman Wong' in 'Chak Lam Coleman Wong') oder gleicher Nachname
    mit ähnlichem Vorname ('Aleksandr'/'Alexander'). Nur gleicher Schlüssel reicht bei vollen
    Namen nicht ('Arthur Fils' ist nicht 'Arthur Fery', 'Eduardo Nava' nicht 'Emilio Nava').
    """
    if a == b:
        return True
    full_a, full_b = _is_full(a), _is_full(b)
    if not (full_a and full_b):
        if full_a or full_b:
            abbr, full = (b, a) if full_a else (a, b)
            return bool(candidate_keys(abbr)[:1]) and candidate_keys(abbr)[0] in _keys(full) and _fits_prefix(abbr, full)
        pa, pb = given_prefix(a), given_prefix(b)
        return candidate_keys(a)[:1] == candidate_keys(b)[:1] != [] and (pa.startswith(pb) or pb.startswith(pa))
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    if sorted(ta) == sorted(tb) or "".join(ta) in _rotations(tb) or "".join(tb) in _rotations(ta):
        return True
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    # Teilname: gleicher Nachname am Ende oder gleicher Vorname (spanische Doppelnamen)
    # Jeder Namensteil zählt so oft, wie er vorkommt: 'Jing Jing Lu' ist kein Teilname von 'Jia Jing Lu'
    if len(short) >= 2 and not Counter(short) - Counter(long_) and (short[-1] == long_[-1] or short[0] == long_[0]):
        return True
    if candidate_keys(a)[0] == candidate_keys(b)[0] and ta[-1] == tb[-1]:
        return SequenceMatcher(None, "".join(ta[:-1]), "".join(tb[:-1])).ratio() >= 0.75
    return False


@lru_cache(maxsize=None)
def _same_surname(a, b):
    """
    Schwächer als _same_player, nur zusammen mit gleichem Gegner, Zeitfenster und Ergebnis (MatchIndex.find_by_score):
    gleicher Nachname. Zwei volle Namen: 'Catherine McNally' und 'Caty Mcnally', 'Kate Makarova' und 'Ekaterina
    Makarova'. Kurzform und voller Name: 'Osorio M.' für Maria Camila Osorio (im Datensatz 'Camila Osorio'), auch wenn
    nur ein Teil des Doppelnamens geführt wird ('Ramos-Vinolas A.' und 'Albert Ramos').
    """
    full_a, full_b = _is_full(a), _is_full(b)
    if full_a and full_b:
        ta, tb = _tokens(a), _tokens(b)
        return len(ta) > 1 and len(tb) > 1 and ta[-1] == tb[-1]
    if full_a or full_b:
        abbr, full = (b, a) if full_a else (a, b)
        if not candidate_keys(abbr)[:1]:
            return False
        surname = candidate_keys(abbr)[0][0]
        return surname in _surnames(full) or any(len(t) >= 4 and (surname.startswith(t) or surname.endswith(t))
                                                 for t in _tokens(full))
    return False


@lru_cache(maxsize=200_000)
def _set_games(score):
    """Games je Satz ohne Tiebreak-Punkte: '7-6(5) 6-1' -> ((7, 6), (6, 1)). Leer bei W/O oder ohne Spielstand."""
    text = re.sub(r"\([^)]*\)", "", str(score or ""))
    return tuple((int(w), int(l)) for w, l in re.findall(r"(\d+)-(\d+)", text))


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

    def find_by_score(self, row):
        """
        Dasselbe Match, obwohl ein Name nicht passt: gleiches Zeitfenster, gleiches Ergebnis (mindestens zwei Sätze,
        gleiche Sieger/Verlierer-Reihenfolge), ein Spieler stimmt nach _same_player, der andere trägt denselben
        Nachnamen (_same_surname). So finden Quellen zusammen, die einen Spieler unter anderem Vornamen führen
        ('Osorio M.' und 'Camila Osorio', 'Catherine McNally' und 'Caty Mcnally'). -> anderes Match oder None.
        """
        sets = _set_games(row.get('score'))
        if len(sets) < 2:
            return None
        lo, hi = _window(row)
        w, l = row['winner_name'], row['loser_name']
        seen = set()
        for key in _surnames(w) | _surnames(l):
            for week in _weeks(lo, hi):
                for other, olo, ohi in self.buckets.get((key, week), []):
                    if id(other) in seen or not (lo <= ohi and olo <= hi):
                        continue
                    seen.add(id(other))
                    if _set_games(other.get('score')) != sets:
                        continue
                    ow, ol = other['winner_name'], other['loser_name']
                    if (_same_player(w, ow) and _same_surname(l, ol)) or (_same_surname(w, ow) and _same_player(l, ol)):
                        return other
        return None


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

    def lookup(name, location=None):
        words = re.findall(r"[a-z]+", _norm_words(name))
        for w in words:
            if w in by_word:
                return by_word[w]
        if _norm(location or '') in by_location:
            return by_location[_norm(location or '')]
        # Ort steckt im Turniernamen ('Paris Masters' bei Sackmann, 'Paris' als Ort bei tennis-data)
        for w in words:
            if w in by_location:
                return by_location[w]
        return default
    return lookup


def _norm_words(text):
    s = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


# Eindeutige Marker im Turniernamen, in dieser Reihenfolge geprüft: 'ATP Challenger Ningbo', 'Poznan CH',
# 'WTA 125K Queretaro', 'W35 Monastir', 'ATP Vienna Open' (Kalshi, Odds API)
LEVEL_MARKERS = (
    (re.compile(r"\bchallenger\b|\bch\b|\b125k?\b"), 'challenger'),
    (re.compile(r"\bitf\b|\b[mw]\d{2,3}\b"), 'itf'),
    (re.compile(r"^(atp|wta)\b"), 'main'),
)
# Quellen mit verlässlichem Level (ESPN setzt immer 'main', Polymarket kennzeichnet nur ITF)
LEVEL_SOURCES = ('sackmann', 'tml', 'tuk', 'kalshi')
# Quellen mit Stats, Untergrund und Ranking im Sackmann-Format
STAT_SOURCES = ('sackmann', 'tml')
LEVEL_WINDOW = 21  # Tage im Kalenderjahr, bis zu denen ein gleichnamiges Turnier als dasselbe gilt


def level_from_name(name):
    """'main', 'challenger' oder 'itf' aus eindeutigen Markern im Turniernamen, sonst None."""
    text = _norm_words(name or '')
    for pattern, level in LEVEL_MARKERS:
        if pattern.search(text):
            return level
    return None


@lru_cache(maxsize=None)
def _level_words(name, location=''):
    return frozenset(w for text in (name, location) if text for w in re.findall(r"[a-z]+", _norm_words(text))
                     if len(w) > 3 and w not in STOPWORDS)


def _days_apart(a, b):
    """Abstand im Kalenderjahr ohne das Jahr: der 3. Januar und der 29. Dezember liegen 5 Tage auseinander."""
    d = abs(a.timetuple().tm_yday - b.timetuple().tm_yday)
    return min(d, 365 - d)


def _level_lookup(rows, window=LEVEL_WINDOW):
    """
    Level eines Turniers aus Name, Datum ('YYYYMMDD') und Tour. Zuerst Marker im Namen (level_from_name), sonst das
    Turnier aus rows (Quellen aus LEVEL_SOURCES) mit gemeinsamem Namenswort, dessen Termin im Kalenderjahr am
    nächsten liegt: Turniere kehren jährlich in derselben Woche wieder, so trennt sich 'Shanghai CH' (September)
    vom Masters (Oktober), und Kalshi-Zeilen derselben Woche geben das Level direkt vor. Ist keins näher als window
    Tage, hat die Haupttour Vorrang, ohne Treffer gilt 'main'.
    Geprüft gegen die Kalshi-Level derselben Matches (10/2026, das eigene Match nie als Referenz): 99 % richtig bei
    ESPN und Polymarket, der frühere Abgleich nur über Challenger/ITF-Namen 46 % (ESPN) bzw. 63 % (Polymarket).
    """
    tournaments = {(r.get('tour') or 'atp', r.get('tourney_name') or '', r.get('location') or '',
                    str(r.get('tourney_date') or r.get('match_date') or '')[:8], r.get('level')) for r in rows}
    by_word = defaultdict(set)  # (Tour, Wort) -> {(Datum, Level)}
    for tour, name, location, day, level in tournaments:
        if level in LEVELS and len(day) == 8:
            for w in _level_words(name, location):
                by_word[(tour, w)].add((_date(day), level))

    def lookup(name, day, tour=None, location=None):
        found = level_from_name(name)
        if found:
            return found
        when = _date(day)
        tours = (tour,) if isinstance(tour, str) and tour else ('atp', 'wta')
        words = _level_words(name or '', location or '')
        cands = {c for t in tours for w in words for c in by_word.get((t, w), ())}
        if not cands:
            return 'main'
        best = min(cands, key=lambda c: (_days_apart(c[0], when), abs((c[0] - when).days), c[1]))
        if _days_apart(best[0], when) <= window:
            return best[1]
        return 'main' if any(level == 'main' for _, level in cands) else max(cands)[1]
    return lookup


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


def _copy_stats(src, dst, same_order):
    """Aufschlag-Stats und Minuten aus src, wenn dst keine hat (nur bei gleicher Sieger/Verlierer-Reihenfolge)."""
    if not same_order or _odd(dst.get('w_svpt')) or not _odd(src.get('w_svpt')):
        return False
    for c in STAT_COLUMNS:
        dst[c] = src.get(c)
    if not _odd(dst.get('minutes')):
        dst['minutes'] = src.get('minutes')
    return True


def _copy_details(src, dst):
    """
    Details aus einer anderen Quelle in die Sackmann- bzw. TennisMyLife-Zeile desselben Matches: echtes Matchdatum
    (wo die Zeile nur den Turnierstart kennt), Indoor/Outdoor und Ort.
    """
    day = src.get('match_date')
    start = dst.get('tourney_date')
    if day and start and dst.get('match_date') in (None, "", start):
        try:
            if 0 <= (_date(day) - _date(start)).days <= 20:
                dst['match_date'] = day
        except ValueError:
            pass
    for c in ('court', 'location', 'country'):
        if src.get(c) and not dst.get(c):
            dst[c] = src[c]


def _fill_by_tourney(rows, field):
    """Fehlende Werte (court, location) aus anderen Matches desselben Turniers übernehmen."""
    known = {}
    for r in rows:
        if r.get(field):
            known.setdefault((r.get('tourney_id'), r.get('tourney_date')), r[field])
    for r in rows:
        if not r.get(field):
            r[field] = known.get((r.get('tourney_id'), r.get('tourney_date')), r.get(field) or "")


def _attach_players(rows, players):
    """
    Nationalität, Händigkeit und Geburtsdatum für Sieger und Verlierer. Sackmann-Zeilen über die
    Spieler-ID, sonst über den Namen (nur wenn er in der Spielerliste eindeutig ist).
    """
    by_id = {p['player_id']: p for p in players if p.get('player_id')}
    by_name = defaultdict(list)
    for p in players:
        by_name[_norm(p['name'])].append(p)
    known = {}
    for r in rows:
        if r.get('source') != 'sackmann':
            continue
        for side in ('winner', 'loser'):
            p = by_id.get(r.get(f'{side}_id'))
            if p is not None:
                known.setdefault(r[f'{side}_name'], p)

    def meta(name):
        if name in known:
            return known[name]
        candidates = by_name.get(_norm(name), [])
        return candidates[0] if len(candidates) == 1 else None

    for r in rows:
        for side in ('winner', 'loser'):
            p = meta(r[f'{side}_name']) or {}
            r[f'{side}_ioc'] = p.get('ioc') or r.get(f'{side}_ioc') or ""
            r[f'{side}_hand'] = p.get('hand') or r.get(f'{side}_hand') or ""
            r[f'{side}_dob'] = p.get('dob') or r.get(f'{side}_dob') or ""


def build(sackmann_rows, tuk_rows, espn_rows, players=None, market_rows=(), tml_rows=()):
    """
    Führt alle Quellen zusammen, getrennt pro Tour (Spalte 'tour', fehlend = 'atp').
    players: {tour: Spielerliste aus sackmann.load_players} für Nationalität, Hand und Alter.
    market_rows: Matches aus Polymarket/Kalshi (sources/markets.load), Reihenfolge = Priorität.
    tml_rows: Matches aus TennisMyLife (sources/tml.load), mit Stats.
    """
    market_rows, tml_rows = list(market_rows), list(tml_rows)
    def tour_of(r):
        return r.get('tour') or 'atp'

    out, stats, lists = [], Counter(), {key: [] for key in LIST_STATS}
    tours = sorted({tour_of(r) for r in sackmann_rows + tml_rows + tuk_rows + espn_rows})
    for tour in tours:
        rows, tour_stats = _build_tour([r for r in sackmann_rows if tour_of(r) == tour],
                                       [r for r in tuk_rows if tour_of(r) == tour],
                                       [r for r in espn_rows if tour_of(r) == tour],
                                       (players or {}).get(tour, []),
                                       [r for r in market_rows if tour_of(r) == tour],
                                       [r for r in tml_rows if tour_of(r) == tour])
        for r in rows:
            r['tour'] = tour
        out += rows
        for key in LIST_STATS:
            lists[key] += [[tour, *item] for item in tour_stats.pop(key, [])]
        stats.update(tour_stats)
        stats[f'{tour}_matches'] = len(rows)
    out.sort(key=lambda r: (str(r['tourney_date']), int(float(r.get('match_num') or 0))))
    return out, {**stats, **lists}


MARKET_SOURCES = ('kalshi', 'polymarket')
# Listen in den Zählern von _build_tour, pro Tour als [Tour, Name, Anzahl] gesammelt
LIST_STATS = ('unresolved_names', 'tuk_unmatched_names')
# Quellen ohne Untergrund, Turnierstufe und Ranking: ergänzt aus den anderen Quellen
NO_DETAILS = ('espn',) + MARKET_SOURCES


def _build_tour(sackmann_rows, tuk_rows, espn_rows, players, market_rows=(), tml_rows=()):
    tml_rows = list(tml_rows)
    registry = NameRegistry()
    for r in sackmann_rows:  # zuerst: die Sackmann-Schreibweise ist die Referenz
        registry.add_full(r['winner_name'], reference=True)
        registry.add_full(r['loser_name'], reference=True)
    for r in tml_rows + espn_rows:
        registry.add_full(r['winner_name'])
        registry.add_full(r['loser_name'])
    for r in sackmann_rows + tml_rows + tuk_rows + espn_rows + list(market_rows):
        r['winner_name'] = registry.canonical(MANUAL_ALIASES.get(r['winner_name'], r['winner_name']))
        r['loser_name'] = registry.canonical(MANUAL_ALIASES.get(r['loser_name'], r['loser_name']))

    cutoff = max((r['tourney_date'] for r in sackmann_rows), default="00000000")
    sack_count = Counter(n for r in sackmann_rows for n in (r['winner_name'], r['loser_name']))

    out = list(sackmann_rows)
    index = MatchIndex()
    for r in out:
        index.add(r)

    aliases = Aliases()
    abbr_votes = defaultdict(Counter)  # Kurzform -> {voller Name: Anzahl Duplikate}
    stats = Counter(sackmann=len(out), tml=0, tuk=0, espn=0, duplicates=0, score_matches=0, tuk_unmatched=0)
    tuk_unmatched = Counter()  # Name -> tennis-data-Zeilen vor dem Stichtag ohne Gegenstück (ihre Quoten gehen verloren)
    for kind, rows in (('tml', tml_rows), ('tuk', tuk_rows), ('espn', espn_rows), ('market', market_rows)):
        for r in rows:
            source = r['source'] if kind == 'market' else kind
            other, same_order = index.find(r)
            by_score = False
            if other is None and kind != 'market':
                # Märkte haben keinen Spielstand; die anderen Quellen dürfen über das Ergebnis zusammenfinden
                other = index.find_by_score(r)
                if other is not None:
                    same_order = by_score = True
                    stats['score_matches'] += 1
            if other is not None:
                # gleiches Match: Namen verknüpfen, Quoten übernehmen
                pairs = ((r['winner_name'], other['winner_name']), (r['loser_name'], other['loser_name'])) if same_order \
                    else ((r['winner_name'], other['loser_name']), (r['loser_name'], other['winner_name']))
                for a, b in pairs:
                    if a == b:
                        continue
                    if by_score and sack_count[a] and sack_count[b]:
                        # Zwei Namen, die Sackmann als verschiedene Spieler führt, bleiben verschieden (Geschwister,
                        # 'Jia Jing Lu' und 'Jing Jing Lu'): Dann irrt eine Quelle beim Namen, das Match ist dasselbe
                        continue
                    if _is_full(a) and _is_full(b):
                        aliases.union(b, a)
                    elif _is_full(b):
                        abbr_votes[a][b] += 1
                    elif _is_full(a):
                        abbr_votes[b][a] += 1
                if source == 'tuk' and other.get('B365W') in (None, ""):
                    _copy_odds(r, other, same_order)
                if other['source'] in STAT_SOURCES:
                    _copy_details(r, other)
                if source == 'tml' and _copy_stats(r, other, same_order):
                    stats['tml_stats_filled'] += 1
                if (r['match_date'] or r['tourney_date']) > cutoff:
                    stats['duplicates'] += 1
                    if source in MARKET_SOURCES:
                        stats[f'{source}_duplicates'] += 1
                continue
            if (r['match_date'] or r['tourney_date']) <= cutoff:
                if source == 'tuk':
                    stats['tuk_unmatched'] += 1
                    tuk_unmatched.update((r['winner_name'], r['loser_name']))
                continue
            out.append(r)
            index.add(r)
            stats[source] += 1

    # Pro Spieler ein Name: bevorzugt voller Name, dann der mit den meisten Sackmann-Matches
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
    # Kurzformen, die sich keinem Spieler zuordnen ließen, stehen jetzt als eigene Spieler ohne Historie im Datensatz.
    # Das darf nicht still passieren: update_data meldet sie und schreibt sie nach data/update_state.json.
    unresolved = Counter(n for r in out for n in (r['winner_name'], r['loser_name']) if not _is_full(n))
    stats['unresolved_names'] = [[n, c] for n, c in unresolved.most_common()]
    # Die Namen mit den meisten Zeilen ohne Gegenstück zeigen, wo die Zuordnung systematisch scheitert
    stats['tuk_unmatched_names'] = [[n, c] for n, c in tuk_unmatched.most_common(15) if c >= 5]

    # ESPN und Märkte: Untergrund, Turnierstufe und Ranking ergänzen
    surface = _tourney_lookup(sackmann_rows + tml_rows + tuk_rows, 'surface', VALID_SURFACES, 'Hard')
    level = _tourney_lookup(sackmann_rows + tml_rows + tuk_rows, 'tourney_level', VALID_LEVELS, 'A')
    rank = _rank_lookup(out)
    # ESPN zeigt auch einzelne Challenger/WTA-125-Turniere ohne Kennzeichnung (Level dort immer 'main'), Polymarket
    # trennt Challenger und Haupttour nicht: Level über Turniername und Termin. Kalshi kennt das Level selbst und dient
    # mit allen Märkten als Referenz, auch denen, die als Duplikat eines ESPN-Matches weggefallen sind.
    tour_level = _level_lookup(sackmann_rows + tml_rows + tuk_rows
                               + [r for r in market_rows if r['source'] == 'kalshi'])
    for r in out:
        if r['source'] not in NO_DETAILS:
            continue
        if not r.get('level') or (r['level'] == 'main' and r['source'] != 'kalshi'):
            found = tour_level(r['tourney_name'], r['match_date'] or r['tourney_date'], r.get('tour'), r.get('location'))
            r['level'] = found
            if found != 'main' and not r.get('tourney_level'):
                r['tourney_level'] = 'C' if found == 'challenger' else 'ITF'
        r['surface'] = surface(r['tourney_name'], r.get('location'))
        if not r.get('tourney_level'):
            r['tourney_level'] = 'G' if any(g in _norm_words(r['tourney_name']) for g in GRAND_SLAMS) \
                else level(r['tourney_name'], r.get('location'))
        r['winner_rank'] = rank(r['winner_name'], r['match_date'])
        r['loser_rank'] = rank(r['loser_name'], r['match_date'])

    # Indoor/Outdoor und Land des Turniers (Heimvorteil), Spieler-Metadaten
    _fill_by_tourney(out, 'court')
    _fill_by_tourney(out, 'location')
    court = _tourney_lookup(tml_rows + tuk_rows, 'court', VALID_COURTS, "")
    for r in out:
        if r.get('court') not in VALID_COURTS:
            r['court'] = court(r.get('tourney_name', ''), r.get('location'))
        r['tourney_ioc'] = tourney_country(r.get('tourney_name', ''), r.get('location'), r.get('country'))
    _attach_players(out, players)

    out = clean(out, stats)
    stats['with_odds'] = sum(1 for r in out if r.get('B365W') not in (None, ""))
    stats['with_match_date'] = sum(1 for r in out if r.get('match_date') and r['match_date'] != r['tourney_date'])
    out.sort(key=lambda r: (str(r['tourney_date']), int(float(r.get('match_num') or 0))))
    return out, stats


def _odd(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# Plausible Summe der Kehrwerte (1/Quote Sieger + 1/Quote Verlierer) je Quelle
ODDS_RANGE = {
    'default': (0.95, 1.35),
    'BFEW': (0.98, 1.10),  # Exchange: Back-Kurse fast ohne Marge, darüber dünner Markt
    'MaxW': (0.90, 1.35),  # Bestkurs über alle Bücher, kann knapp unter 100 % liegen
}


def clean(rows, stats):
    """
    Letzte Bereinigung, Zähler landen in stats (und in data/update_state.json):
      - exakte Doppel-Einträge (gleiches Turnier, Paar, Runde, Ergebnis) entfernen
      - fehlender Untergrund -> Hard (so behandelt das Modell unbekannten Untergrund ohnehin)
      - unplausible Quoten leeren (<= 1.0, ohne Gegenquote oder Summe der Kehrwerte außerhalb von
        ODDS_RANGE: Buchmacher 95-135 %, Exchange nahe 100 %, Maximum über alle Bücher auch knapp unter 100 %)
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
        for w, l in ODDS_PAIRS:
            ow, ol = _odd(r.get(w)), _odd(r.get(l))
            if ow is None and ol is None:
                continue
            low, high = ODDS_RANGE.get(w, ODDS_RANGE['default'])
            if ow is None or ol is None or min(ow, ol) <= 1.0 or not low <= 1 / ow + 1 / ol <= high:
                r[w] = r[l] = None
                stats['clean_odds_removed'] += 1
    return out
