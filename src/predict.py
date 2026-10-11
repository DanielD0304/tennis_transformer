"""
Live-Vorhersagen für anstehende Matches.

Baut für alle Spieler die Match-Historie bis heute auf (gleiche Features wie in
preprocessing.create_training_samples) und lässt das trainierte Modell für ein
beliebiges Paar (Spieler A, Spieler B, Untergrund) die Siegwahrscheinlichkeit schätzen.

Datenbasis ist data/matches.csv (siehe src/update_data.py): Sackmann mit Aufschlag-Stats,
danach tennis-data.co.uk und ESPN (jedes beendete Match, auch im laufenden Turnier, ohne Stats).
Die Daten werden beim Start automatisch aktualisiert, wenn sie älter als 24 Stunden sind.

Nutzung:
    from src.predict import MatchPredictor
    p = MatchPredictor()
    p.predict("Medvedev D.", "Struff J.L.", surface="Hard")
"""

import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from difflib import get_close_matches

import pandas as pd
import torch

from . import data_loader as dl
from .config import default_config
from .dataset import DataSet
from .elo import compute_elo_ratings
from .model import Model
from .names import candidate_keys, player_key
from .preprocessing import extract_match_features, get_h2h_pct, is_walkover, pad_sequence
from .train import custom_collate_fn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Unter so vielen Matches in der Historie ist die Vorhersage kaum belastbar
# (im Training wurden nur Spieler mit >= 5 Matches verwendet).
MIN_HISTORY = 5


@dataclass
class Prediction:
    player_a: str
    player_b: str
    prob_a: float
    prob_b: float
    n_matches_a: int
    n_matches_b: int
    warnings: list = field(default_factory=list)


class MatchPredictor:
    def __init__(self, config=default_config, matches=None, model_path=None):
        """
        matches: DataFrame im Schema von data/matches.csv (für Tests). Standard: dl.load_matches()
        ab config.data_years_start, damit ELO dieselbe Basis hat wie im Training.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if matches is None:
            matches = dl.load_matches(first_year=config.data_years_start)
        self._build_history(matches)

        model_path = model_path or os.path.join(REPO_ROOT, config.best_model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Kein trainiertes Modell unter {model_path}. Erst 'python main.py' laufen lassen."
            )
        self.model = Model(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            input_dim=config.input_dim,
            max_len=config.max_len,
            output_dim=config.output_dim,
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    # ------------------------------------------------------------------ Daten

    def _build_history(self, df):
        df = df.dropna(subset=['winner_name', 'loser_name', 'tourney_date'])
        df = dl.enforce_schema(df.copy())
        if 'match_num' not in df.columns:
            df['match_num'] = range(len(df))
        df = compute_elo_ratings(df)
        df['match_date_obj'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')
        df = df.dropna(subset=['match_date_obj'])
        if 'match_date' not in df.columns:
            df['match_date'] = df['tourney_date']
        # Echtes Matchdatum (fürs Interface); die Features rechnen wie im Training mit dem Turnierdatum
        played_on = pd.to_datetime(df['match_date'].astype(str).str[:8], format='%Y%m%d', errors='coerce')
        df['played_on'] = played_on.fillna(df['match_date_obj'])

        self.history = defaultdict(list)
        self.h2h = {}
        self.players = {}  # player_key -> Name wie in den Daten
        self._index = defaultdict(set)  # alle Kandidaten-Schlüssel -> Namen
        self.matches = defaultdict(list)  # Name -> Match-Zusammenfassungen fürs Interface
        self.ranks = {}
        self.elo = {}

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

            # ELO nach dem Match (in df steht der Stand davor)
            expected_w = 1 / (1 + 10 ** ((row['loser_elo'] - row['winner_elo']) / 400))
            self.elo[winner] = row['winner_elo'] + 32 * (1 - expected_w)
            self.elo[loser] = row['loser_elo'] - 32 * (1 - expected_w)

        self.last_match_date = df['played_on'].max()

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

    # ------------------------------------------------------------ Vorhersage

    def _sample(self, pA, pB, surface, match_date):
        cfg = self.config

        def hist(p, h2h_val):
            full = self.history.get(p, [])
            rec = full[-cfg.n_recent_matches:][::-1]
            surf = [m for m in full if m['_surface'] == surface][-cfg.n_surface_matches:][::-1]
            inject = lambda ms: [{**m, 'h2h_win_pct': h2h_val} for m in ms]
            return inject(surf), inject(rec)

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
        }

    @torch.no_grad()
    def predict(self, player_a, player_b, surface="Hard", match_date=None):
        """
        Wahrscheinlichkeit, dass player_a gewinnt. Namen dürfen im Bookie-Format sein.
        Gibt None zurück, wenn ein Spieler nicht in den Daten ist.
        """
        pA, pB = self.resolve(player_a), self.resolve(player_b)
        if pA is None or pB is None or pA == pB:
            return None
        match_date = pd.Timestamp(match_date or date.today())

        # Beide Reihenfolgen rechnen und mitteln, damit die A/B-Position keine Rolle spielt
        batch = custom_collate_fn([DataSet([s])[0] for s in (
            self._sample(pA, pB, surface, match_date),
            self._sample(pB, pA, surface, match_date),
        )])
        keys = ['player_a_surface', 'player_a_recent', 'player_b_surface', 'player_b_recent']
        features = torch.cat([batch[k] for k in keys], dim=1).to(self.device)
        positions = torch.cat([batch[k + '_pos'] for k in keys], dim=1).to(self.device)
        masks = torch.cat([batch[k + '_mask'] for k in keys], dim=1).to(self.device)
        masks = torch.cat([torch.ones(masks.shape[0], 1, device=self.device), masks], dim=1)
        segment_ids = batch['segment_ids'].to(self.device)

        probs = torch.softmax(self.model(features, positions, segment_ids, masks), dim=1)
        # Label 1 = Spieler A gewinnt
        prob_a = (probs[0][1].item() + probs[1][0].item()) / 2

        n_a, n_b = len(self.history[pA]), len(self.history[pB])
        warnings = [f"{p}: nur {n} Matches in der Historie" for p, n in ((pA, n_a), (pB, n_b)) if n < MIN_HISTORY]
        return Prediction(pA, pB, prob_a, 1 - prob_a, n_a, n_b, warnings)


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
