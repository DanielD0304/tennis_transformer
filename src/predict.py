"""
Live-Vorhersagen für anstehende Matches.

Baut für alle Spieler die Match-Historie bis heute auf (gleiche Features wie in
preprocessing.create_training_samples) und lässt das trainierte Modell für ein
beliebiges Paar (Spieler A, Spieler B, Untergrund) die Siegwahrscheinlichkeit schätzen.

Datenbasis ist data/matches.csv (siehe src/update_data.py): Sackmann mit Aufschlag-Stats,
danach tennis-data.co.uk und ESPN (jedes beendete Match, auch im laufenden Turnier, ohne Stats).
Die Daten werden beim Start automatisch aktualisiert, wenn sie älter als 24 Stunden sind.

Der Untergrund wird, wenn nicht angegeben, aus dem Turniernamen bestimmt (build._tourney_lookup),
sonst aus dem Turnier, das beide Spieler gerade spielen. Ein expliziter Untergrund hat Vorrang.

Welches Modell rechnet, ist konfigurierbar (config.live_model oder MatchPredictor(model=...)):
    transformer  trainierter Transformer (models/best_model.pt, symmetrisch, mit Temperatur)
    gbdt         LightGBM (models/gbdt_<tour>.pkl, erzeugt mit python -m src.models.gbdt --tour atp)
    elo2         Elo2 (src/elo.py), braucht kein trainiertes Modell
Gewählt wird das Modell, das im Walk-forward-Backtest am besten abschneidet (siehe PR/README).

Nutzung:
    from src.predict import MatchPredictor
    p = MatchPredictor()
    p.predict("Medvedev D.", "Struff J.L.", surface="Hard")
    p.predict("Medvedev D.", "Struff J.L.", tournament="Rolex Shanghai Masters")
    p.predict_proba({'player_a': "Medvedev D.", 'player_b': "Struff J.L.", 'surface': "Hard"})
"""

import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from difflib import get_close_matches

import pandas as pd
import torch

from . import data_loader as dl
from .chronology import chronological, match_dates
from .config import default_config
from .dataset import DataSet
from .elo import Elo2, compute_elo_ratings
from .model import load_model
from .names import candidate_keys, player_key
from .preprocessing import (context_features, elo2_logit, extract_match_features, get_h2h_pct, inject_context,
                            is_home, is_walkover, pad_sequence)
from .sources.build import GRAND_SLAMS, VALID_SURFACES, _tourney_lookup
from .sources.places import tourney_country
from .train import custom_collate_fn, model_inputs

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Unter so vielen Matches in der Historie ist die Vorhersage kaum belastbar
# (im Training wurden nur Spieler mit >= 5 Matches verwendet).
MIN_HISTORY = 5
# Ohne Turniernamen: das letzte Turnier beider Spieler zählt, wenn es höchstens so lange her ist
CURRENT_TOURNEY_DAYS = 10
DEFAULT_SURFACE = 'Hard'
LIVE_MODELS = ('transformer', 'gbdt', 'elo2')


@dataclass
class Prediction:
    player_a: str
    player_b: str
    prob_a: float
    prob_b: float
    n_matches_a: int
    n_matches_b: int
    warnings: list = field(default_factory=list)
    surface: str = DEFAULT_SURFACE
    surface_source: str = 'default'  # manual, tournament, current oder default
    model: str = 'transformer'


class MatchPredictor:
    def __init__(self, config=default_config, matches=None, model_path=None, model=None, gbdt_dir=None):
        """
        matches: DataFrame im Schema von data/matches.csv (für Tests). Standard: dl.load_matches()
        ab config.data_years_start, damit ELO dieselbe Basis hat wie im Training.
        model: 'transformer', 'gbdt' oder 'elo2' (Standard: config.live_model).
        model_path: Transformer-Gewichte, gbdt_dir: Ordner mit gbdt_<tour>.pkl (Standard: models/).
        """
        self.config = config
        self.model_name = model or config.live_model
        if self.model_name not in LIVE_MODELS:
            raise ValueError(f"Unbekanntes Live-Modell {self.model_name!r} (bekannt: {', '.join(LIVE_MODELS)})")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if matches is None:
            matches = dl.load_matches(first_year=config.data_years_start)
        self._build_history(matches)

        self.model, self.temperature = None, 1.0
        if self.model_name == 'transformer':
            model_path = model_path or os.path.join(REPO_ROOT, config.best_model_path)
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Kein trainiertes Modell unter {model_path}. Erst 'python main.py' laufen lassen."
                )
            self.model, self.temperature = load_model(model_path, config, self.device)
        elif self.model_name == 'gbdt':
            self._init_gbdt(gbdt_dir or os.path.join(REPO_ROOT, 'models'))

    def _init_gbdt(self, folder):
        """LightGBM pro Tour laden und den Feature-Stand (features.FeatureBuilder) bis heute aufbauen."""
        from .evaluation.predictors import prepare_matches
        from .features import FeatureBuilder, feature_table
        from .models.gbdt import GBDTModel

        self.gbdt = {}
        for tour in sorted(set(self.tour.values())):
            path = os.path.join(folder, f'gbdt_{tour}.pkl')
            if not os.path.exists(path):
                raise FileNotFoundError(f"Kein LightGBM-Modell unter {path} (python -m src.models.gbdt --tour {tour}).")
            self.gbdt[tour] = GBDTModel.load(path)
        self.features = FeatureBuilder()
        feature_table(prepare_matches(self._history_df), self.features)

    # ------------------------------------------------------------------ Daten

    def _build_history(self, df):
        df = df.dropna(subset=['winner_name', 'loser_name', 'tourney_date'])
        df = dl.enforce_schema(df.copy())
        if 'match_num' not in df.columns:
            df['match_num'] = range(len(df))
        df = compute_elo_ratings(df)  # pro Tour, in echter Reihenfolge
        # Echtes Matchdatum (sonst Turnierstart), wie im Training
        df['match_date_obj'] = match_dates(df)
        df = df.dropna(subset=['match_date_obj'])
        df['played_on'] = df['match_date_obj']
        # Gleicher Name auf ATP und WTA: die WTA-Spielerin bekommt einen Zusatz, Historien bleiben getrennt
        tours_of = defaultdict(set)
        for tour, w, l in zip(df['tour'], df['winner_name'], df['loser_name']):
            tours_of[w].add(tour)
            tours_of[l].add(tour)
        clash = {n for n, t in tours_of.items() if len(t) > 1}
        if clash:
            wta = df['tour'] == 'wta'
            for col in ('winner_name', 'loser_name'):
                df.loc[wta & df[col].isin(clash), col] = df.loc[wta & df[col].isin(clash), col] + " (WTA)"

        rows = df[['tourney_name', 'tourney_date', 'surface']].astype(str).to_dict('records')
        if 'location' in df.columns:
            for r, loc in zip(rows, df['location']):
                r['location'] = _text(loc)
        self._surface_lookup = _tourney_lookup(rows, 'surface', VALID_SURFACES, None)

        self.history = defaultdict(list)
        self.h2h = {}
        self.players = {}  # player_key -> Name wie in den Daten
        self._index = defaultdict(set)  # alle Kandidaten-Schlüssel -> Namen
        self.matches = defaultdict(list)  # Name -> Match-Zusammenfassungen fürs Interface
        self.ranks = {}
        self.elo = {}
        self.tour = {}  # Name -> 'atp'/'wta'
        self.meta = {}  # Name -> {'dob', 'hand', 'ioc'} (neuester Stand)

        for _, row in df.iterrows():
            if is_walkover(row):
                continue
            winner, loser = row['winner_name'], row['loser_name']

            h2h_key = tuple(sorted((winner, loser)))
            if h2h_key not in self.h2h:
                self.h2h[h2h_key] = {'p1': h2h_key[0], 'wins1': 0, 'wins2': 0}
            win_h2h = get_h2h_pct(winner, loser, self.h2h)
            los_h2h = get_h2h_pct(loser, winner, self.h2h)

            self.history[winner].append(extract_match_features(row, 'winner', win_h2h))
            self.history[loser].append(extract_match_features(row, 'loser', los_h2h))

            entry = self.h2h[h2h_key]
            if winner == entry['p1']:
                entry['wins1'] += 1
            else:
                entry['wins2'] += 1

            for name in (winner, loser):
                key = player_key(name)
                if key:
                    self.players[key] = name
                for k in candidate_keys(name, reverse=False):
                    self._index[k].add(name)

            score = _text(row.get('score'))
            for name, side in ((winner, 'winner'), (loser, 'loser')):
                self.tour[name] = _text(row.get('tour')) or 'atp'
                meta = {k: _text(row.get(f'{side}_{k}')) for k in ('dob', 'hand', 'ioc')}
                self.meta[name] = {k: v or self.meta.get(name, {}).get(k, '') for k, v in meta.items()}

            for name, opp, won, rank in ((winner, loser, True, row['winner_rank']),
                                         (loser, winner, False, row['loser_rank'])):
                self.matches[name].append({
                    'date': row['played_on'].date(), 'opponent': opp, 'won': won,
                    'surface': row['surface'], 'tournament': _text(row.get('tourney_name')),
                    'round': _text(row.get('round')), 'score': score if won else _flip_score(score),
                    'has_stats': row['w_svpt'] > 0,
                })
                if rank < 9999:
                    self.ranks[name] = int(rank)

            # ELO nach dem Match
            self.elo[winner] = row['winner_elo_after']
            self.elo[loser] = row['loser_elo_after']

        self.last_match_date = df['played_on'].max()
        # Elo2 (Stand nach dem letzten Match) für predict_proba und als Transformer-Input
        self.elo2 = Elo2()
        for row in chronological(df).to_dict('records'):
            self.elo2.observe(row)
        self._history_df = df

    def resolve(self, name):
        """Name in beliebigem Format (Bookie, Kurzform) -> Name in den Daten, oder None (z.B. WTA/Doppel)."""
        if name in self.history:
            return name
        keys = candidate_keys(name)
        for key in keys:
            names = self._index.get(key)
            if names and len(names) == 1:
                return next(iter(names))
        if keys and not keys[0][1]:
            # Ohne Initial: nur Nachname, wenn eindeutig
            same_surname = {n for k, ns in self._index.items() if k[0] == keys[0][0] for n in ns}
            if len(same_surname) == 1:
                return same_surname.pop()
        return None

    def search(self, query, n=8):
        """Ähnliche Spielernamen, z.B. für Vorschläge bei Tippfehlern."""
        names = sorted(set(self.players.values()))
        q = query.strip().lower()
        hits = [name for name in names if q and q in name.lower()]
        return (hits + [m for m in get_close_matches(query, names, n=n, cutoff=0.5) if m not in hits])[:n]

    def player_info(self, name, surface=None, n_last=10):
        """Form, Ranking, ELO und letzte Matches eines Spielers (Name wie in den Daten)."""
        matches = self.matches.get(name, [])
        last = matches[-n_last:][::-1]
        on_surface = [m for m in matches if m['surface'] == surface] if surface else []
        return {
            'name': name,
            'rank': self.ranks.get(name),
            'elo': round(self.elo.get(name, 1500)),
            'n_matches': len(matches),
            'last': last,
            'last_wins': sum(m['won'] for m in last),
            'surface_wins': sum(m['won'] for m in on_surface),
            'surface_matches': len(on_surface),
        }

    def h2h_record(self, a, b):
        """(Siege a, Siege b) in den geladenen Daten."""
        stats = self.h2h.get(tuple(sorted((a, b))))
        if stats is None:
            return 0, 0
        return (stats['wins1'], stats['wins2']) if stats['p1'] == a else (stats['wins2'], stats['wins1'])

    def surface_for(self, tournament=None, players=(), match_date=None):
        """
        Untergrund automatisch: (Untergrund, Quelle). Zuerst über den Turniernamen
        (build._tourney_lookup auf allen Matches), sonst das laufende Turnier der Spieler
        (letztes Match vor höchstens CURRENT_TOURNEY_DAYS Tagen), sonst DEFAULT_SURFACE.
        """
        if tournament:
            found = self._surface_lookup(tournament)
            if found:
                return found, 'tournament'
        day = pd.Timestamp(match_date or date.today()).date()
        recent = [self.matches[p][-1] for p in players if self.matches.get(p)]
        recent = [m for m in recent if 0 <= (day - m['date']).days <= CURRENT_TOURNEY_DAYS]
        if recent:
            return max(recent, key=lambda m: m['date'])['surface'], 'current'
        return DEFAULT_SURFACE, 'default'

    # ------------------------------------------------------------ Vorhersage

    def _elo2_prob(self, pA, pB, surface, best_of=3):
        return self.elo2.prob(self.tour.get(pA, 'atp'), pA, pB, surface, best_of)

    def _sample(self, pA, pB, surface, match_date, tourney_ioc='', best_of=3):
        cfg = self.config

        def hist(p, h2h_val):
            full = self.history.get(p, [])
            rec = full[-cfg.n_recent_matches:][::-1]
            surf = [m for m in full if m['_surface'] == surface][-cfg.n_surface_matches:][::-1]
            meta = self.meta.get(p, {})
            ctx = context_features(full, match_date, h2h_val, dob=meta.get('dob'), hand=meta.get('hand'),
                                   home=is_home(meta.get('ioc'), tourney_ioc))
            return inject_context(surf, ctx), inject_context(rec, ctx)

        a_surf, a_rec = hist(pA, get_h2h_pct(pA, pB, self.h2h))
        b_surf, b_rec = hist(pB, get_h2h_pct(pB, pA, self.h2h))

        a_surf, a_s_mask = pad_sequence(a_surf, cfg.n_surface_matches, match_date)
        a_rec, a_r_mask = pad_sequence(a_rec, cfg.n_recent_matches, match_date)
        b_surf, b_s_mask = pad_sequence(b_surf, cfg.n_surface_matches, match_date)
        b_rec, b_r_mask = pad_sequence(b_rec, cfg.n_recent_matches, match_date)

        n_s, n_r = cfg.n_surface_matches, cfg.n_recent_matches
        return {
            'player_a_surface': a_surf, 'player_a_surface_mask': a_s_mask,
            'player_a_recent': a_rec, 'player_a_recent_mask': a_r_mask,
            'player_b_surface': b_surf, 'player_b_surface_mask': b_s_mask,
            'player_b_recent': b_rec, 'player_b_recent_mask': b_r_mask,
            'segment_ids': [1] * n_s + [2] * n_r + [3] * n_s + [4] * n_r,
            'label': 0, 'year': match_date.year, 'odds_a': 0.0, 'odds_b': 0.0,
            'elo2_logit': elo2_logit(self._elo2_prob(pA, pB, surface, best_of), True),
        }

    def predict(self, player_a, player_b, surface=None, match_date=None, tournament=None, best_of=None):
        """
        Wahrscheinlichkeit, dass player_a gewinnt. Namen dürfen im Bookie-Format sein.
        surface=None bestimmt den Untergrund automatisch (siehe surface_for), ein Wert überschreibt das.
        best_of=None: 5 bei Grand Slams der Herren (laut Turniername), sonst 3.
        Gibt None zurück, wenn ein Spieler nicht in den Daten ist.
        """
        pA, pB = self.resolve(player_a), self.resolve(player_b)
        if pA is None or pB is None or pA == pB:
            return None
        match_date = pd.Timestamp(match_date or date.today())
        if surface:
            surface_source = 'manual'
        else:
            surface, surface_source = self.surface_for(tournament, (pA, pB), match_date)
        tourney_ioc = tourney_country(tournament) if tournament else ''
        if best_of is None:
            slam = tournament and any(g in tournament.lower() for g in GRAND_SLAMS)
            best_of = 5 if slam and self.tour.get(pA) == 'atp' else 3

        prob_a = float(self._proba(pA, pB, surface, match_date, tourney_ioc, best_of))

        n_a, n_b = len(self.history[pA]), len(self.history[pB])
        warnings = [f"{p}: nur {n} Matches in der Historie" for p, n in ((pA, n_a), (pB, n_b)) if n < MIN_HISTORY]
        if self.tour.get(pA) != self.tour.get(pB):
            warnings.append(f"{pA} und {pB} spielen auf verschiedenen Touren")
        return Prediction(pA, pB, prob_a, 1 - prob_a, n_a, n_b, warnings, surface, surface_source, self.model_name)

    def predict_proba(self, match):
        """
        Einheitliches Interface für betscraper und Backtest: match = {'player_a', 'player_b', optional
        'surface', 'date', 'tournament', 'best_of'} -> P(player_a gewinnt), None wenn unbekannt.
        """
        pred = self.predict(match['player_a'], match['player_b'], surface=match.get('surface'),
                            match_date=match.get('date'), tournament=match.get('tournament'),
                            best_of=match.get('best_of'))
        return None if pred is None else pred.prob_a

    def _proba(self, pA, pB, surface, match_date, tourney_ioc, best_of):
        if self.model_name == 'elo2':
            return self._elo2_prob(pA, pB, surface, best_of)
        if self.model_name == 'gbdt':
            return self._gbdt_proba(pA, pB, surface, match_date, tourney_ioc, best_of)
        return self._transformer_proba(pA, pB, surface, match_date, tourney_ioc, best_of)

    @torch.no_grad()
    def _transformer_proba(self, pA, pB, surface, match_date, tourney_ioc, best_of):
        # Beide Reihenfolgen rechnen, Logits mitteln (A/B-Position spielt keine Rolle), dann Temperatur
        batch = custom_collate_fn([DataSet([s])[0] for s in (
            self._sample(pA, pB, surface, match_date, tourney_ioc, best_of),
            self._sample(pB, pA, surface, match_date, tourney_ioc, best_of),
        )])
        x, pos, seg, mask, extra = model_inputs(batch, self.device)
        logits = self.model(x, pos, seg, mask, extra if self.model.transformer.extra_dim else None)
        z = logits[:, 1] - logits[:, 0]  # Label 1 = Spieler A gewinnt
        z_sym = (z[0] - z[1]).item() / 2
        return 1 / (1 + math.exp(-z_sym / self.temperature))

    def _gbdt_proba(self, pA, pB, surface, match_date, tourney_ioc, best_of):
        from .models.gbdt import add_elo2
        tour = self.tour.get(pA, 'atp')
        meta_a, meta_b = self.meta.get(pA, {}), self.meta.get(pB, {})
        m = {'tour': tour, 'player_a': pA, 'player_b': pB, 'label': 1, 'date': match_date, 'surface': surface,
             'tourney_ioc': tourney_ioc, 'best_of': best_of, 'court': '',
             'elo_a': self.elo.get(pA, 1500), 'elo_b': self.elo.get(pB, 1500),
             'rank_a': self.ranks.get(pA, 0), 'rank_b': self.ranks.get(pB, 0),
             **{f'winner_{k}': meta_a.get(k, '') for k in ('dob', 'hand', 'ioc')},
             **{f'loser_{k}': meta_b.get(k, '') for k in ('dob', 'hand', 'ioc')}}
        row = pd.DataFrame([self.features.row(m)])
        elo2 = {}
        for x, p in (('a', pA), ('b', pB)):
            overall, surf, mix, n = self.elo2.rating(tour, p, surface)
            elo2.update({f'elo2_{x}': overall, f'elo2_surf_{x}': surf, f'elo2_mix_{x}': mix, f'elo2_n_{x}': n})
        elo2['elo2_prob_a'] = self._elo2_prob(pA, pB, surface, best_of)
        row = add_elo2(row, pd.DataFrame([elo2]))
        return float(self.gbdt[tour].predict(row)[0])


def _flip_score(score):
    """Ergebnis aus Sicht des Verlierers: '6-4 6-7(1) RET' -> '4-6 7(1)-6 RET'."""
    def flip(token):
        m = re.fullmatch(r"(\d+)-(\d+)(\(\d+\))?", token)
        if not m:
            return token
        a, b, tb = m.group(1), m.group(2), m.group(3) or ""
        # Tiebreak-Punkte des Satzverlierers stehen immer am Satzende: 7-6(3) -> 6-7(3)
        return f"{b}-{a}{tb}"
    return " ".join(flip(t) for t in score.split())


def _text(value):
    return str(value) if pd.notna(value) else ''
