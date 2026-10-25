"""Tests für den Score-Parser (Formate aus Sackmann, tennis-data und ESPN)."""
import pytest

from src.score import parse_score


@pytest.mark.parametrize("score, games, sets, tiebreaks, n_sets, retired", [
    ("6-4 6-4", (12, 8), (2, 0), 0, 2, False),
    ("7-6(4) 3-6", (10, 12), (1, 1), 1, 2, False),
    ("6-4 6-7(1) 7-5", (19, 16), (2, 1), 1, 3, False),
    ("7-6(4) 3-6 7-6(10)", (17, 18), (2, 1), 2, 3, False),
    # ältere Notation mit beiden Tiebreak-Ständen, Leerzeichen vor der Klammer
    ("7-6(7-4) 6-3", (13, 9), (2, 0), 1, 2, False),
    ("7-6 (4) 6-3", (13, 9), (2, 0), 1, 2, False),
    # Best of 5, langer fünfter Satz ohne Tiebreak
    ("6-7(5) 6-4 4-6 7-6(2) 13-11", (36, 34), (3, 2), 2, 5, False),
    # Match-Tiebreak in eckigen Klammern zählt als Satz mit 1:0 Games
    ("6-3 3-6 [10-8]", (10, 9), (2, 1), 1, 3, False),
    ("4-6 6-4 [11-13]", (10, 11), (1, 2), 1, 3, False),
    # Aufgabe: angefangener Satz zählt bei den Games, aber nicht bei den Sätzen
    ("6-4 2-1 RET", (8, 5), (1, 0), 0, 2, True),
    ("6-4 Ret.", (6, 4), (1, 0), 0, 1, True),
    ("3-6 7-6(3) 2-0 DEF", (12, 12), (1, 1), 1, 3, True),
    ("6-7(5) 3-3 ABD", (9, 10), (0, 1), 1, 2, True),
])
def test_parse_formats(score, games, sets, tiebreaks, n_sets, retired):
    s = parse_score(score)
    assert (s.games_w, s.games_l) == games
    assert (s.sets_w, s.sets_l) == sets
    assert s.tiebreaks == tiebreaks
    assert s.n_sets == n_sets
    assert s.retired == retired
    assert s.valid and not s.walkover


@pytest.mark.parametrize("score", ["W/O", "w/o", "Walkover", "", None, float('nan'), "nan"])
def test_no_played_sets(score):
    s = parse_score(score)
    assert not s.valid
    assert s.n_sets == 0
    assert s.game_share() == 0.5 and s.set_share() == 0.5
    assert s.walkover == (str(score).upper() in ("W/O", "WALKOVER"))


def test_retired_flag_and_shares():
    s = parse_score("6-4 2-1 RET")
    assert s.retired
    assert s.game_share(winner=True) == pytest.approx(8 / 13)
    assert s.game_share(winner=False) == pytest.approx(5 / 13)
    assert s.set_share(winner=True) == 1.0


def test_tuk_match_tiebreak_notation():
    # tennis-data/ESPN schreiben den Match-Tiebreak als Satz 1-0 mit Punkten
    s = parse_score("6-7(5) 6-4 1-0(8)")
    assert s.match_tiebreak and s.tiebreaks == 2
    assert (s.sets_w, s.sets_l) == (2, 1)
    assert (s.games_w, s.games_l) == (13, 11)


def test_garbage_tokens_are_ignored():
    s = parse_score("6-4 6-6 unfinished")
    # 6-6 ohne Ende zählt bei Abbruch als angefangener Satz
    assert s.retired and s.n_sets == 2
    assert parse_score("Played and abandoned").n_sets == 0
