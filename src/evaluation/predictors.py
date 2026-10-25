"""
Einheitliches Interface für alle Vorhersagen: p = P(Spieler A gewinnt).

Spieler A ist pro Match pseudo-zufällig (fester Seed, Hash der match_id) Sieger oder Verlierer,
damit p nicht immer die Sieger-Wahrscheinlichkeit ist. Alle Quoten werden mit umgedreht.

Ein Predictor hat zwei Methoden:
    fit(history)     history = alle Matches mit Datum < Testfenster (vom Backtest erzwungen)
    predict(window)  -> np.ndarray mit p je Zeile (NaN = keine Vorhersage)
"""
import hashlib
import math
import os

import numpy as np
import pandas as pd

from ..chronology import chronological, match_dates
from ..elo import compute_elo_ratings, elo2_features

# Quoten-Spalten im Sieger/Verlierer-Format (tennis-data) -> Kürzel in A/B-Sicht
BOOKS = {'b365': ('B365W', 'B365L'), 'ps': ('PSW', 'PSL'), 'avg': ('AvgW', 'AvgL'), 'max': ('MaxW', 'MaxL'),
         'bfe': ('BFEW', 'BFEL')}
# Markt-Benchmark: Pinnacle, sonst Betfair Exchange (Pinnacle fehlt bei tennis-data seit Februar 2026),
# sonst Durchschnittsquote. Die gewählte Quelle steht in market_source.
MARKET_SOURCES = ('ps', 'bfe', 'avg')
ORIENTATION_SEED = 42
MISSING_RANK = 9999


# ---------------------------------------------------------------------- Daten

def match_id(row):
    key = "|".join(str(row.get(c, '')) for c in ('tourney_date', 'tourney_id', 'tourney_name', 'round',
                                                  'winner_name', 'loser_name', 'match_num'))
    return hashlib.sha1(key.encode('utf-8')).hexdigest()[:16]


def a_is_winner(mid, seed=ORIENTATION_SEED):
    """Feste, reihenfolgeunabhängige A/B-Zuordnung pro Match."""
    return hashlib.sha1(f"{seed}:{mid}".encode('utf-8')).digest()[0] % 2 == 0


def remove_margin(odds_a, odds_b):
    """Faire Wahrscheinlichkeit für A aus zwei Quoten (1/o normiert, Summe beider Seiten = 1)."""
    inv_a, inv_b = 1.0 / np.asarray(odds_a, float), 1.0 / np.asarray(odds_b, float)
    return inv_a / (inv_a + inv_b)


def _valid_odds(a, b):
    return np.isfinite(a) & np.isfinite(b) & (a > 1.0) & (b > 1.0)


def prepare_matches(df, seed=ORIENTATION_SEED, with_elo=True, elo2_params=None):
    """
    Rohdaten (Schema data/matches.csv) -> Matches in A/B-Sicht, chronologisch sortiert.
    Behält alle Originalspalten (der Transformer-Adapter braucht sie) und ergänzt:
    match_id, date, year, tour, player_a, player_b, label (1 = A gewinnt), status,
    rank_a/b, elo_a/b, elo2_* (siehe elo.elo2_features), <book>_a/<book>_b für b365, ps, avg, max, bfe,
    market_source und p_market_fair (faire Wahrscheinlichkeit aus der Quelle laut MARKET_SOURCES).
    """
    df = df.dropna(subset=['winner_name', 'loser_name', 'tourney_date']).copy()
    if 'match_num' not in df.columns:
        df['match_num'] = range(len(df))
    df['match_num'] = pd.to_numeric(df['match_num'], errors='coerce').fillna(0)
    for col in ('winner_rank', 'loser_rank'):
        df[col] = pd.to_numeric(df.get(col), errors='coerce').fillna(MISSING_RANK)
    if with_elo:
        df = compute_elo_ratings(df)  # Stand VOR dem Match, pro Tour, in echter Reihenfolge
        df = df.join(elo2_features(df, elo2_params))
    else:
        df = chronological(df).reset_index(drop=True)

    # Echtes Matchdatum, sonst Turnierstart (Sackmann kennt nur den)
    df['date'] = match_dates(df)
    df = df[df['date'].notna()].copy()
    df['year'] = df['date'].dt.year
    df['tour'] = df['tour'].fillna('atp') if 'tour' in df.columns else 'atp'
    df['level'] = df['level'].fillna('main').replace('', 'main') if 'level' in df.columns else 'main'
    status = df['status'] if 'status' in df.columns else pd.Series('completed', index=df.index)
    df['status'] = status.fillna('completed').astype(str)
    if 'score' in df.columns:
        df.loc[df['score'].astype(str).str.contains('W/O', na=False), 'status'] = 'W/O'

    df['match_id'] = [match_id(r) for r in df.to_dict('records')]
    a_win = np.array([a_is_winner(m, seed) for m in df['match_id']])
    df['label'] = a_win.astype(int)
    df['player_a'] = np.where(a_win, df['winner_name'], df['loser_name'])
    df['player_b'] = np.where(a_win, df['loser_name'], df['winner_name'])
    df['rank_a'] = np.where(a_win, df['winner_rank'], df['loser_rank'])
    df['rank_b'] = np.where(a_win, df['loser_rank'], df['winner_rank'])
    if 'winner_elo' in df.columns:
        df['elo_a'] = np.where(a_win, df['winner_elo'], df['loser_elo'])
        df['elo_b'] = np.where(a_win, df['loser_elo'], df['winner_elo'])
    if 'winner_elo2' in df.columns:
        for col in ('elo2', 'elo2_surf', 'elo2_mix', 'elo2_n'):
            df[f'{col}_a'] = np.where(a_win, df[f'winner_{col}'], df[f'loser_{col}'])
            df[f'{col}_b'] = np.where(a_win, df[f'loser_{col}'], df[f'winner_{col}'])
        df['elo2_prob_a'] = np.where(a_win, df['elo2_p_winner'], 1 - df['elo2_p_winner'])
    for book, (w, l) in BOOKS.items():
        missing = pd.Series(np.nan, index=df.index)
        ow = pd.to_numeric(df[w] if w in df.columns else missing, errors='coerce').to_numpy(float)
        ol = pd.to_numeric(df[l] if l in df.columns else missing, errors='coerce').to_numpy(float)
        ok = _valid_odds(ow, ol)
        ow, ol = np.where(ok, ow, np.nan), np.where(ok, ol, np.nan)
        df[f'{book}_a'] = np.where(a_win, ow, ol)
        df[f'{book}_b'] = np.where(a_win, ol, ow)

    df['p_market_fair'], df['market_source'] = market_probability(df)
    # Faire Pinnacle-Wahrscheinlichkeit (nur PS, ohne Fallback): Preis-Proxy für Exchange/PM-Venues
    df['fair_ps'] = remove_margin(df['ps_a'], df['ps_b'])
    return chronological(df).reset_index(drop=True)


def market_probability(df, sources=MARKET_SOURCES):
    """Faire Markt-Wahrscheinlichkeit für A aus der ersten Quelle mit gültigen Quoten -> (p, Quelle)."""
    p = np.full(len(df), np.nan)
    source = np.full(len(df), 'none', dtype=object)
    for book in sources:
        if f'{book}_a' not in df.columns:
            continue
        fair = remove_margin(df[f'{book}_a'], df[f'{book}_b'])
        take = np.isnan(p) & np.isfinite(fair)
        p[take], source[take] = fair[take], book
    return p, source


# ----------------------------------------------------------------- Hilfen

def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


def fit_logistic(X, y, l2=1e-4, iters=50):
    """Logistische Regression ohne Intercept (Newton-Verfahren). Ohne Intercept bleibt p(A,B) = 1 - p(B,A)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    if X.ndim == 1:
        X = X[:, None]
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = sigmoid(X @ w)
        grad = X.T @ (p - y) + l2 * w
        hess = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(X.shape[1])
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def playable(df):
    """Gespielte Matches (W/O hat kein Ergebnis, aus dem man lernen kann)."""
    return df['status'] != 'W/O'


class Predictor:
    name = "base"

    def __init__(self):
        self.fit_max_date = None

    def fit(self, history):
        self.fit_max_date = history['date'].max() if len(history) else None

    def predict(self, window):
        raise NotImplementedError

    def _check_no_leakage(self, window):
        if self.fit_max_date is not None and len(window):
            assert self.fit_max_date < window['date'].min(), \
                f"{self.name}: Fit-Daten bis {self.fit_max_date} überschneiden Testfenster ab {window['date'].min()}"


class MarketPredictor(Predictor):
    """Markt ohne Marge: Pinnacle -> Betfair Exchange -> Durchschnittsquote (Quelle in market_source)."""
    name = "market"

    def predict(self, window):
        return market_probability(window)[0]


class RankPredictor(Predictor):
    """Logistisch auf log(Rang B) - log(Rang A), walk-forward gefittet."""
    name = "rank"
    min_train = 100

    def __init__(self):
        super().__init__()
        self.coef = 0.5  # vernünftiger Startwert, bis genug Historie da ist

    @staticmethod
    def feature(df):
        return np.log(df['rank_b'].to_numpy(float) + 1) - np.log(df['rank_a'].to_numpy(float) + 1)

    def fit(self, history):
        super().fit(history)
        h = history[playable(history)]
        if len(h) >= self.min_train:
            self.coef = fit_logistic(self.feature(h), h['label'])[0]

    def predict(self, window):
        self._check_no_leakage(window)
        return sigmoid(self.coef * self.feature(window))


class EloPredictor(Predictor):
    """Elo aus src/elo.py (K = 32, pro Tour, Aufgaben mit halbem K), Stand vor dem Match. Braucht keinen Fit."""
    name = "elo"

    def predict(self, window):
        return 1.0 / (1.0 + 10 ** ((window['elo_b'].to_numpy(float) - window['elo_a'].to_numpy(float)) / 400))


class Elo2Predictor(Predictor):
    """
    Elo2 aus src/elo.py (K im 538-Stil, Mix aus Gesamt- und Untergrund-Elo, Parameter pro Tour aus
    elo.ELO2_PARAMS, gewählt nur auf Daten vor 2020). Stand vor dem Match, braucht keinen Fit.
    """
    name = "elo2"

    def predict(self, window):
        return window['elo2_prob_a'].to_numpy(float)


class BlendPredictor(Predictor):
    """
    Logistische Regression auf logit(p_market) + logit(p_model), walk-forward gefittet
    auf den out-of-sample-Vorhersagen früherer Fenster. Das ist der Go/No-Go-Test:
    bekommt das Modell kein Gewicht, verbessert es den Markt nicht.
    Solange zu wenig Historie da ist, gilt p = p_market (Gewichte 1, 0).
    """
    min_train = 300

    def __init__(self, base, name=None):
        super().__init__()
        self.base = base
        self.name = name or f"blend_{base}"
        self.coef = np.array([1.0, 0.0])
        self.n_train = 0

    def _features(self, df):
        return np.column_stack([logit(df['p_market']), logit(df[f'p_{self.base}'])])

    def fit(self, history):
        super().fit(history)
        cols = ['p_market', f'p_{self.base}']
        if not set(cols) <= set(history.columns):
            return
        h = history[playable(history) & history[cols].notna().all(axis=1)]
        self.n_train = len(h)
        if len(h) >= self.min_train:
            self.coef = fit_logistic(self._features(h), h['label'])

    def predict(self, window):
        self._check_no_leakage(window)
        p = sigmoid(self._features(window) @ self.coef)
        return np.where(window[['p_market', f'p_{self.base}']].notna().all(axis=1), p, np.nan)


class TransformerPredictor(Predictor):
    """
    Adapter um das trainierte Modell (models/best_model.pt). Es wird nicht neu trainiert, sondern nur
    auf Matches ab first_year angewendet: val_year dient dem Checkpointing und dem Temperature Scaling,
    streng out-of-sample sind erst die Jahre danach.

    Features wie im Training (preprocessing.create_training_samples, also auch dessen Filter:
    beide Spieler bis config.max_rank, >= 5 Matches Historie, aktiv). Matches außerhalb davon bekommen NaN.
    Beide Reihenfolgen werden gerechnet, die Logits gemittelt (Symmetrie wie in predict.py), dann
    durch die Temperatur aus dem Checkpoint geteilt.
    """
    name = "transformer"

    def __init__(self, config=None, model_path=None, batch_size=256):
        super().__init__()
        from ..config import default_config
        self.config = config or default_config
        self.model_path = model_path or self.config.best_model_path
        self.batch_size = batch_size
        self.temperature = 1.0
        self._p = None  # match_id -> p(A gewinnt), einmal für alle Matches berechnet

    @property
    def first_year(self):
        """Erstes Jahr, für das der Transformer vorhersagen darf (davor: Training und Validierung)."""
        return self.config.val_year + 1

    def fit(self, history):
        # Kein Training im Walk-forward. Der Leakage-Check prüft stattdessen die Trainingsjahre.
        self.fit_max_date = None

    def _compute(self, matches):
        import torch

        from ..data_loader import enforce_schema
        from ..model import load_model
        from ..preprocessing import create_training_samples
        from ..train import symmetric_logits

        cfg = self.config
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Kein trainiertes Modell unter {self.model_path} (erst 'python main.py').")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, self.temperature = load_model(self.model_path, cfg, device)

        raw = enforce_schema(matches.copy())  # Stats-Spalten fehlen evtl. (z.B. Tests)
        samples = create_training_samples(raw, n_matches=cfg.n_surface_matches, n_recent=cfg.n_recent_matches,
                                          max_rank=cfg.max_rank)
        samples = [s for s in samples if s['year'] >= self.first_year]
        # Symmetrisch (beide Reihenfolgen) und mit der Temperatur aus dem Validierungsjahr
        z = symmetric_logits(model, samples, device, self.batch_size) / self.temperature
        p_sample_a = sigmoid(z)
        result = {}
        for s, p in zip(samples, p_sample_a):
            # label 1: Sample-Spieler A ist der Sieger
            p_winner = p if s['label'] == 1 else 1 - p
            row = matches.loc[s['match_index']]
            result[row['match_id']] = p_winner if row['label'] == 1 else 1 - p_winner
        return result

    def prepare(self, matches):
        self._p = self._compute(matches)

    def predict(self, window):
        if len(window):
            assert window['date'].min().year >= self.first_year, \
                f"transformer: Testfenster ab {window['date'].min()} liegt in Trainings- oder Validierungsjahren"
        if self._p is None:
            raise RuntimeError("TransformerPredictor.prepare(matches) zuerst aufrufen")
        return np.array([self._p.get(m, np.nan) for m in window['match_id']], dtype=float)


BASE_PREDICTORS = {'market': MarketPredictor, 'rank': RankPredictor, 'elo': EloPredictor, 'elo2': Elo2Predictor,
                   'transformer': TransformerPredictor}


STACK_BASES = ('elo2', 'gbdt')


def _stack(n, names):
    """'stack' (elo2 + gbdt), 'stack_bands' (mit Quotenbereichen), 'stack_tf' (+ Transformer), 'stack_tf_bands'
    oder explizit 'stack_<m1>+<m2>'."""
    from ..models.stack import DEFAULT_BANDS, StackPredictor
    rest = n[len('stack'):].lstrip('_')
    flags = set(rest.split('_')) if rest else set()
    if '+' in rest:
        bases = [b for b in rest.split('+') if b]
    else:
        bases = [b for b in STACK_BASES if b in names] + (['transformer'] if 'tf' in flags else [])
        if not flags <= {'tf', 'bands'}:
            raise ValueError(f"Unbekanntes Stacking {n!r} (stack, stack_bands, stack_tf, stack_<m1>+<m2>)")
    if not bases:
        raise ValueError(f"{n} braucht mindestens ein Basismodell (z.B. elo2, gbdt) in --models")
    return StackPredictor(bases, bands=DEFAULT_BANDS if 'bands' in flags else None, name=n)


def make_predictors(names, config=None, model_path=None, gbdt_refit='quarter'):
    """
    'market,rank,elo,elo2,gbdt,transformer,blend,stack' -> Predictor-Objekte in Rechenreihenfolge.
    'blend' nimmt den Transformer, falls angefordert, sonst Elo. 'blend_<modell>' wählt explizit.
    'stack*' siehe _stack. Der Markt wird für Blend, Stack und die Vergleiche immer mitgerechnet.
    """
    names = [n.strip() for n in names if n.strip()]
    if 'market' not in names:
        names = ['market'] + names
    base, combos = [], []
    for n in names:
        if n == 'blend':
            combos.append(BlendPredictor('transformer' if 'transformer' in names else 'elo', name='blend'))
        elif n.startswith('blend_'):
            combos.append(BlendPredictor(n[len('blend_'):], name=n))
        elif n.startswith('stack'):
            combos.append(_stack(n, names))
        elif n == 'transformer':
            base.append(TransformerPredictor(config, model_path))
        elif n == 'gbdt':
            from ..models.gbdt import GBDTPredictor
            base.append(GBDTPredictor(refit=gbdt_refit))
        elif n in BASE_PREDICTORS:
            base.append(BASE_PREDICTORS[n]())
        else:
            raise ValueError(f"Unbekanntes Modell: {n} (bekannt: {', '.join(BASE_PREDICTORS)}, gbdt, blend, "
                             f"blend_<modell>, stack, stack_bands, stack_tf)")
    known = {p.name for p in base}
    for c in combos:
        needed = [c.base] if isinstance(c, BlendPredictor) else list(c.bases)
        missing = [m for m in needed if m not in known]
        if missing:
            raise ValueError(f"{c.name} braucht die Modelle {', '.join(missing)} in --models")
    return base + combos
