from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .chronology import chronological, match_dates
from .score import parse_score

# Gewicht eines Matches mit Aufgabe (RET/DEF) im Elo: K wird damit skaliert
RET_WEIGHT = 0.5


def is_retired(row):
    """Aufgabe oder Disqualifikation: Match nicht regulär zu Ende gespielt."""
    status = str(row.get('status', '') or '')
    if status in ('RET', 'DEF'):
        return True
    return status in ('', 'nan', 'completed') and parse_score(row.get('score')).retired


def is_walkover(row):
    return str(row.get('status', '')) == 'W/O' or 'W/O' in str(row.get('score', ''))


def expected(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def compute_elo_ratings(df, k_factor=32, start_elo=1500, ret_weight=RET_WEIGHT):
    """
    Berechnet ELO-Ratings für alle Matches im DataFrame, getrennt pro Tour (Spalte 'tour').
    Fügt 'winner_elo'/'loser_elo' (Stand VOR dem Match) und 'winner_elo_after'/'loser_elo_after' hinzu.
    Reihenfolge: echtes Matchdatum, sonst Turnierstart + Runde (siehe chronology.py).
    Walkover zählt nicht, Aufgabe (RET/DEF) mit K * ret_weight.
    """
    print("Berechne ELO-Ratings...")
    df = chronological(df).reset_index(drop=True)

    elo_ratings = {}  # (Tour, Spieler) -> aktuelles ELO
    winner_elos, loser_elos, winner_after, loser_after = [], [], [], []
    tours = df['tour'].fillna('atp') if 'tour' in df.columns else ['atp'] * len(df)

    for row, tour in zip(df.to_dict('records'), tours):
        w_key, l_key = (tour, row['winner_name']), (tour, row['loser_name'])
        w_elo = elo_ratings.get(w_key, start_elo)
        l_elo = elo_ratings.get(l_key, start_elo)
        winner_elos.append(w_elo)
        loser_elos.append(l_elo)

        if is_walkover(row):
            # kein gespieltes Match, ELO bleibt unverändert
            winner_after.append(w_elo)
            loser_after.append(l_elo)
            continue

        k = k_factor * (ret_weight if is_retired(row) else 1.0)
        delta = k * (1 - expected(w_elo, l_elo))
        elo_ratings[w_key] = w_elo + delta
        elo_ratings[l_key] = l_elo - delta
        winner_after.append(w_elo + delta)
        loser_after.append(l_elo - delta)

    df['winner_elo'] = winner_elos
    df['loser_elo'] = loser_elos
    df['winner_elo_after'] = winner_after
    df['loser_elo_after'] = loser_after
    return df


# ------------------------------------------------------------------ Elo2 (Phase 3, Stufe 1)
#
# K im 538-Stil: K = c / (n + o)^s, n = bisher gespielte Matches des Spielers (neue Spieler bewegen sich
# schnell, etablierte langsam). Daneben ein Elo pro Untergrund (eigene Zählung n), vorhergesagt wird mit
# dem Mix (1 - w) * Gesamt + w * Untergrund. Optional: Grand Slams mit größerem K und Best-of-5 über die
# Satz-Wahrscheinlichkeit. Alles getrennt pro Tour.

@dataclass(frozen=True)
class Elo2Params:
    c: float = 250.0
    o: float = 5.0
    s: float = 0.4
    surface_weight: float = 0.5  # Anteil des Untergrund-Elo im Mix
    ret_weight: float = 1.0      # K-Faktor bei Aufgabe (RET/DEF)
    slam_k: float = 1.0          # K-Multiplikator für Grand-Slam-Matches
    bo5: bool = False            # Best-of-5: p über die Satz-Wahrscheinlichkeit umrechnen
    start: float = 1500.0


# Gewählt mit `python -m src.elo` nur auf Daten vor 2020 (Daten ab 2015, Log Loss auf 2017-2019, TUNE_GRID
# plus c = 100/125). Der Bereich um das Optimum ist flach (+-0,001 Log Loss); RET-Gewicht 0,5 vs. 1,0 und
# der Grand-Slam-Bonus machen fast nichts aus. ATP 0,6226 (Standard c=250, s=0,4: 0,6271), WTA 0,6346 (0,6375).
ELO2_PARAMS = {
    'atp': Elo2Params(c=100.0, s=0.3, surface_weight=0.5, ret_weight=1.0, slam_k=1.0, bo5=True),
    'wta': Elo2Params(c=125.0, s=0.3, surface_weight=0.25, ret_weight=1.0, slam_k=1.0, bo5=False),
}


def k_538(n, c=250.0, o=5.0, s=0.4):
    """K-Faktor im 538-Stil für einen Spieler mit n bisherigen Matches."""
    return c / (n + o) ** s


def best_of_5(p3):
    """
    Siegwahrscheinlichkeit im Best-of-5 aus der im Best-of-3: Satz-Wahrscheinlichkeit q aus
    p3 = q^2 (3 - 2q) bestimmen (Bisektion), dann p5 = q^3 (6q^2 - 15q + 10).
    """
    p3 = np.asarray(p3, float)
    lo, hi = np.zeros_like(p3), np.ones_like(p3)
    for _ in range(40):
        mid = (lo + hi) / 2
        too_low = mid ** 2 * (3 - 2 * mid) < p3
        lo, hi = np.where(too_low, mid, lo), np.where(too_low, hi, mid)
    q = (lo + hi) / 2
    return q ** 3 * (6 * q ** 2 - 15 * q + 10)


def is_slam(row):
    return str(row.get('tourney_level', '')) == 'G'


def _best_of(row):
    try:
        return int(float(row.get('best_of')))
    except (TypeError, ValueError):
        return 3


class Elo2:
    """
    Zustand aller Ratings, pro Tour getrennt. params: Elo2Params oder {tour: Elo2Params}.
    rating() und prob() lesen den Stand, update() schreibt ein gespieltes Match ein.
    """

    def __init__(self, params=None):
        self.params = params if params is not None else ELO2_PARAMS
        self.overall, self.surface = {}, {}  # (Tour, Spieler) / (Tour, Untergrund, Spieler) -> Elo
        self.n, self.n_surface = defaultdict(int), defaultdict(int)

    def p(self, tour):
        if isinstance(self.params, Elo2Params):
            return self.params
        return self.params.get(tour) or ELO2_PARAMS.get(tour) or Elo2Params()

    def rating(self, tour, player, surface):
        """(Gesamt, Untergrund, Mix, Anzahl Matches) vor dem nächsten Match."""
        par = self.p(tour)
        overall = self.overall.get((tour, player), par.start)
        surf = self.surface.get((tour, surface, player), par.start)
        mix = (1 - par.surface_weight) * overall + par.surface_weight * surf
        return overall, surf, mix, self.n[(tour, player)]

    def prob(self, tour, a, b, surface, best_of=3):
        par = self.p(tour)
        p = expected(self.rating(tour, a, surface)[2], self.rating(tour, b, surface)[2])
        return float(best_of_5(p)) if par.bo5 and best_of == 5 else p

    def update(self, tour, winner, loser, surface, retired=False, slam=False):
        par = self.p(tour)
        mult = (par.ret_weight if retired else 1.0) * (par.slam_k if slam else 1.0)
        w, l = (tour, winner), (tour, loser)
        ew, el = self.overall.get(w, par.start), self.overall.get(l, par.start)
        delta = 1 - expected(ew, el)
        self.overall[w] = ew + mult * k_538(self.n[w], par.c, par.o, par.s) * delta
        self.overall[l] = el - mult * k_538(self.n[l], par.c, par.o, par.s) * delta
        self.n[w] += 1
        self.n[l] += 1
        if surface:
            sw, sl = (tour, surface, winner), (tour, surface, loser)
            rw, rl = self.surface.get(sw, par.start), self.surface.get(sl, par.start)
            delta = 1 - expected(rw, rl)
            self.surface[sw] = rw + mult * k_538(self.n_surface[sw], par.c, par.o, par.s) * delta
            self.surface[sl] = rl - mult * k_538(self.n_surface[sl], par.c, par.o, par.s) * delta
            self.n_surface[sw] += 1
            self.n_surface[sl] += 1

    def observe(self, row, tour=None):
        """Ein Match im Schema von data/matches.csv einschreiben (W/O zählt nicht)."""
        if is_walkover(row):
            return
        tour = tour or (row.get('tour') if isinstance(row.get('tour'), str) else 'atp')
        self.update(tour, row['winner_name'], row['loser_name'], _surface(row), is_retired(row), is_slam(row))


def _surface(row):
    s = row.get('surface')
    return s if isinstance(s, str) and s else None


ELO2_COLUMNS = ['winner_elo2', 'loser_elo2', 'winner_elo2_surf', 'loser_elo2_surf', 'winner_elo2_mix',
                'loser_elo2_mix', 'winner_elo2_n', 'loser_elo2_n', 'elo2_p_winner']


def elo2_features(df, params=None):
    """
    Elo2-Stand VOR jedem Match (Index wie df): Gesamt, Untergrund, Mix, Anzahl Matches für Sieger und
    Verlierer und elo2_p_winner (Siegwahrscheinlichkeit des Siegers laut Mix, ggf. Best-of-5).
    Läuft in echter Reihenfolge (chronology.py), die Rückgabe hat trotzdem die Zeilen von df.
    """
    elo = Elo2(params)
    ordered = chronological(df)
    tours = ordered['tour'].fillna('atp') if 'tour' in ordered.columns else pd.Series('atp', index=ordered.index)
    out = np.empty((len(ordered), len(ELO2_COLUMNS)))
    for i, (row, tour) in enumerate(zip(ordered.to_dict('records'), tours)):
        surface = _surface(row)
        w = elo.rating(tour, row['winner_name'], surface)
        l = elo.rating(tour, row['loser_name'], surface)
        p = elo.prob(tour, row['winner_name'], row['loser_name'], surface, _best_of(row))
        out[i] = (w[0], l[0], w[1], l[1], w[2], l[2], w[3], l[3], p)
        elo.observe(row, tour)
    return pd.DataFrame(out, index=ordered.index, columns=ELO2_COLUMNS).loc[df.index]


# ------------------------------------------------------------------ Tuning (nur Daten vor 2020)

TUNE_GRID = {
    'c': (150.0, 200.0, 250.0, 350.0),
    's': (0.3, 0.4, 0.5, 0.6),
    'ret_weight': (0.5, 1.0),
    'slam_k': (1.0, 1.1),
    'surface_weight': (0.0, 0.25, 0.5, 0.75),
    'bo5': (False, True),
}


def tune(df, eval_from, eval_to, grid=TUNE_GRID, log=print):
    """
    Kleines Grid über Elo2Params für eine Tour: Log Loss auf den gespielten Matches eval_from..eval_to
    (Jahre, inklusive). df darf nur Matches bis eval_to enthalten. Mix-Gewicht und Best-of-5 ändern die
    Ratings nicht, dafür wird nur einmal pro (c, s, ret_weight, slam_k) gerechnet.
    """
    from itertools import product
    years = match_dates(df).dt.year
    assert years.max() <= eval_to, "Tuning darf keine Daten nach eval_to sehen"
    played = ~df.apply(is_walkover, axis=1)
    mask = ((years >= eval_from) & (years <= eval_to) & played).to_numpy()
    bo5 = (pd.to_numeric(df.get('best_of'), errors='coerce') == 5).to_numpy()
    rows = []
    for c, s, ret, slam in product(grid['c'], grid['s'], grid['ret_weight'], grid['slam_k']):
        par = Elo2Params(c=c, s=s, ret_weight=ret, slam_k=slam, surface_weight=0.0)
        f = elo2_features(df, par)
        for w in grid['surface_weight']:
            mix_w = (1 - w) * f['winner_elo2'] + w * f['winner_elo2_surf']
            mix_l = (1 - w) * f['loser_elo2'] + w * f['loser_elo2_surf']
            p3 = expected(mix_w.to_numpy(), mix_l.to_numpy())
            for use_bo5 in grid['bo5']:
                p = np.where(bo5 & use_bo5, best_of_5(p3), p3) if use_bo5 else p3
                ll = float(-np.mean(np.log(np.clip(p[mask], 1e-15, 1))))
                rows.append({'c': c, 's': s, 'ret_weight': ret, 'slam_k': slam, 'surface_weight': w,
                             'bo5': use_bo5, 'log_loss': ll, 'n': int(mask.sum())})
        log(f"  c={c:g} s={s:g} ret={ret:g} slam={slam:g}: bester Log Loss "
            f"{min(r['log_loss'] for r in rows[-len(grid['surface_weight']) * len(grid['bo5']):]):.4f}")
    return pd.DataFrame(rows).sort_values('log_loss').reset_index(drop=True)


def main(argv=None):
    import argparse

    from .data_loader import load_matches

    ap = argparse.ArgumentParser(description="Elo2-Hyperparameter wählen (nur Daten vor 2020)")
    ap.add_argument('--tour', default='atp')
    ap.add_argument('--data-start', type=int, default=2015)
    ap.add_argument('--eval-from', type=int, default=2017)
    ap.add_argument('--eval-to', type=int, default=2019)
    ap.add_argument('--no-refresh', action='store_true')
    args = ap.parse_args(argv)
    if args.eval_to >= 2020:
        raise SystemExit("Hyperparameter nur auf Daten vor 2020 wählen (--eval-to <= 2019)")
    df = load_matches(first_year=args.data_start, last_year=args.eval_to, refresh=not args.no_refresh,
                      tour=args.tour)
    result = tune(df, args.eval_from, args.eval_to)
    print(result.head(15).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("Standard (c=250, s=0.4, w=0.5, ret=1.0):")
    print(result[(result['c'] == 250) & (result['s'] == 0.4) & (result['surface_weight'] == 0.5)
                 & (result['ret_weight'] == 1.0) & (result['slam_k'] == 1.0) & ~result['bo5']].to_string(index=False))


if __name__ == "__main__":
    main()
