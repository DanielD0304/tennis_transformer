"""
Live-Vorhersagen für anstehende Matches.

Baut für alle Spieler die Match-Historie bis heute auf (gleiche Features wie in
preprocessing.create_training_samples) und lässt das trainierte Modell für ein
beliebiges Paar (Spieler A, Spieler B, Untergrund) die Siegwahrscheinlichkeit schätzen.

Datenbasis sind die tennis-data.co.uk-Dateien der letzten Jahre, weil nur die bis heute
aktuell sind. Die haben keine Aufschlag-Statistiken (Aces, 1st Serve %, Breakpoints ...),
diese Features sind für die Live-Historie also 0. Das Modell wurde mit Sackmann-Stats
trainiert, die Vorhersagen sind deshalb vorsichtig zu genießen.

Nutzung:
    from src.predict import MatchPredictor
    p = MatchPredictor()
    p.predict("Medvedev D.", "Struff J.L.", surface="Hard")
"""

import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date

import pandas as pd
import torch

from . import data_loader as dl
from .config import default_config
from .dataset import DataSet
from .elo import compute_elo_ratings
from .model import Model
from .names import player_key
from .preprocessing import extract_match_features, get_h2h_pct, pad_sequence
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
    def __init__(self, config=default_config, years=None, model_path=None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # data_loader arbeitet mit relativem "data"-Pfad, hier unabhängig vom Arbeitsverzeichnis
        dl.DATA_DIR = os.path.join(REPO_ROOT, "data")

        if years is None:
            this_year = date.today().year
            years = range(this_year - 3, this_year + 1)
        self._build_history(years)

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

    def _build_history(self, years):
        frames = []
        for y in years:
            df = dl.load_tuk_year(y)
            if not df.empty:
                frames.append(df)
        if not frames:
            raise RuntimeError("Keine Matchdaten von tennis-data.co.uk geladen.")

        df = pd.concat(frames, ignore_index=True)
        df = df.dropna(subset=['winner_name', 'loser_name', 'tourney_date'])
        df = dl.enforce_schema(df)
        # Reihenfolge innerhalb eines Tages beibehalten (TUK-Dateien sind chronologisch)
        df['match_num'] = range(len(df))
        df = compute_elo_ratings(df)
        df['match_date_obj'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')
        df = df.dropna(subset=['match_date_obj'])

        self.history = defaultdict(list)
        self.h2h = {}
        self.players = {}  # player_key -> Name wie in den Daten

        for _, row in df.iterrows():
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

        self.last_match_date = df['match_date_obj'].max()

    def resolve(self, name):
        """Bookie-Name -> Name in den Daten (oder None, z.B. bei WTA/Doppel)."""
        key = player_key(name)
        if key is None:
            return None
        if key in self.players:
            return self.players[key]
        if key[1]:
            # Initial passt nicht -> anderer Spieler (oft WTA-Spielerin mit gleichem Nachnamen)
            return None
        # Ohne Initial: nur Nachname, wenn eindeutig
        same_surname = [n for k, n in self.players.items() if k[0] == key[0]]
        return same_surname[0] if len(same_surname) == 1 else None

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
