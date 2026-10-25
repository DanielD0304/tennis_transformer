"""
Parser für Spielstände aller Quellen (Sackmann, tennis-data, ESPN):

    '6-4 6-7(1) 7-5'        normale Sätze, Tiebreak-Punkte des Satzverlierers in Klammern
    '7-6(7-4) 6-3'          ältere Notation mit beiden Tiebreak-Ständen
    '6-3 3-6 [10-8]'        Match-Tiebreak statt drittem Satz (auch '1-0(8)')
    '6-4 2-1 RET'           Aufgabe (RET, Ret., DEF, Def., ABD, Unfinished)
    'W/O', 'Walkover'       kampflos

Alle Zahlen sind aus Sicht des Siegers (wie in allen Quellen).
"""
import re
from dataclasses import dataclass, field

_SET = re.compile(r"^\[?(\d{1,2})-(\d{1,2})\]?(?:\((\d+)(?:-(\d+))?\))?$")
_RET = ('RET', 'DEF', 'ABD', 'ABN', 'UNFINISHED', 'DISQ')
_WO = ('W/O', 'WALKOVER', 'WO', 'W.O.')


@dataclass
class Score:
    sets: list = field(default_factory=list)  # [(Games Sieger, Games Verlierer, Tiebreak?)]
    games_w: int = 0
    games_l: int = 0
    sets_w: int = 0
    sets_l: int = 0
    tiebreaks: int = 0
    match_tiebreak: bool = False
    retired: bool = False   # RET oder DEF: Match nicht regulär zu Ende gespielt
    walkover: bool = False

    @property
    def n_sets(self):
        return len(self.sets)

    @property
    def valid(self):
        """Mindestens ein Satz geparst (bei W/O oder leerem Score nicht)."""
        return bool(self.sets)

    def game_share(self, winner=True):
        total = self.games_w + self.games_l
        if not total:
            return 0.5
        return (self.games_w if winner else self.games_l) / total

    def set_share(self, winner=True):
        total = self.sets_w + self.sets_l
        if not total:
            return 0.5
        return (self.sets_w if winner else self.sets_l) / total


def parse_score(score):
    """Spielstand-Text -> Score. Unbekannte Tokens werden ignoriert, kaputte Sätze übersprungen."""
    out = Score()
    if score is None:
        return out
    text = str(score).strip()
    if not text or text.lower() == 'nan':
        return out
    upper = text.upper()
    if any(w in upper for w in _WO) and not re.search(r"\d-\d", text):
        out.walkover = True
        return out
    if any(r in upper for r in _RET) or 'RET.' in upper:
        out.retired = True

    # '7-6 (4)' -> '7-6(4)', '[10 - 8]' -> '[10-8]'
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s+\(", "(", text)
    tokens = text.split()
    for token in tokens:
        m = _SET.match(token)
        if not m:
            continue
        w, l = int(m.group(1)), int(m.group(2))
        if w == l and not out.retired:
            continue  # unvollständiger Satz ohne Aufgabe (z.B. Abbruch-Stand) zählt nicht
        tb_points = m.group(3) is not None
        # Match-Tiebreak nur in eckigen Klammern: ein '10-8' ohne Klammern ist ein langer Satz (alter 5. Satz)
        if token.startswith('['):
            # Match-Tiebreak zählt als Satz mit 1:0 Games
            out.match_tiebreak = True
            out.tiebreaks += 1
            gw, gl = (1, 0) if w > l else (0, 1)
            out.sets.append((gw, gl, True))
            out.games_w += gw
            out.games_l += gl
            out.sets_w += gw
            out.sets_l += gl
            continue
        if tb_points and {w, l} == {0, 1}:
            # tennis-data/ESPN-Notation für den Match-Tiebreak: 1-0(8)
            out.match_tiebreak = True
            out.tiebreaks += 1
            out.sets.append((w, l, True))
            out.games_w += w
            out.games_l += l
            out.sets_w += w
            out.sets_l += l
            continue
        is_tb = tb_points or {w, l} == {7, 6}
        out.sets.append((w, l, is_tb))
        out.games_w += w
        out.games_l += l
        out.tiebreaks += int(is_tb)
        if _set_finished(w, l):
            out.sets_w += int(w > l)
            out.sets_l += int(l > w)
    return out


def _set_finished(w, l):
    """Ein Satz ist entschieden bei >= 6 Games und 2 Vorsprung oder 7-6 (lange Sätze wie 13-11 inklusive)."""
    hi, lo = max(w, l), min(w, l)
    return (hi >= 6 and hi - lo >= 2) or (hi == 7 and lo == 6)
