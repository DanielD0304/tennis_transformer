"""
Stufe 2 der Modell-Leiter: LightGBM auf der flachen Feature-Tabelle (features.feature_table) plus den
Elo2-Features (elo.elo2_features, in prepare_matches auf A/B gedreht).

Training:
  - Jede Zeile kommt zweimal vor, einmal gespiegelt (A <-> B, Label umgedreht). Vorhersage symmetrisch:
    p = (p(A, B) + 1 - p(B, A)) / 2, damit die A/B-Position keine Rolle spielt.
  - Early Stopping auf einem zeitlich späteren Validierungsfenster (die letzten val_months der Historie),
    danach Kalibrierung (Platt oder Isotonic) auf genau diesem Fenster.
  - Im Walk-forward wird nur pro refit-Periode (Quartal oder Jahr) neu gefittet, immer nur auf Matches
    vor dem Testfenster.

    python -m src.models.gbdt --tour atp    # Live-Modell auf allen Daten -> models/gbdt_atp.pkl
"""
import os
import pickle
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from ..evaluation.predictors import Predictor, playable
from ..features import FEATURE_COLUMNS, feature_table
from .calibration import make_calibrator

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ELO2_FEATURES = [f'{c}_{x}' for c in ('elo2', 'elo2_surf', 'elo2_mix', 'elo2_n') for x in ('a', 'b', 'diff')] \
    + ['elo2_prob_a']
FEATURES = FEATURE_COLUMNS + ELO2_FEATURES
REFIT = {'month': 'M', 'quarter': 'Q', 'year': 'Y'}


def model_path(tour):
    return os.path.join(REPO_ROOT, 'models', f'gbdt_{tour}.pkl')


def gbdt_table(matches, builder=None):
    """Feature-Tabelle für LightGBM: feature_table + Elo2 (Stand vor dem Match), Index wie matches."""
    table = feature_table(matches, builder)
    table.index = matches.index
    return add_elo2(table, matches)


def add_elo2(table, matches):
    for c in ('elo2', 'elo2_surf', 'elo2_mix', 'elo2_n'):
        a = matches[f'{c}_a'] if f'{c}_a' in matches else np.nan
        b = matches[f'{c}_b'] if f'{c}_b' in matches else np.nan
        table[f'{c}_a'], table[f'{c}_b'] = a, b
        table[f'{c}_diff'] = table[f'{c}_a'] - table[f'{c}_b']
    table['elo2_prob_a'] = matches['elo2_prob_a'] if 'elo2_prob_a' in matches else np.nan
    return table


def swap(X):
    """Gleiche Matches aus Sicht von B: _a <-> _b, _diff negiert, Wahrscheinlichkeiten für A -> 1 - p."""
    out = X.copy()
    cols = set(X.columns)
    for c in X.columns:
        if c.endswith('_a') and c[:-2] + '_b' in cols:
            out[c], out[c[:-2] + '_b'] = X[c[:-2] + '_b'], X[c]
        elif c.endswith('_a'):  # h2h_pct_a, elo_prob_a, elo2_prob_a
            out[c] = 1 - X[c]
        elif c.endswith('_diff'):
            out[c] = -X[c]
    return out


@dataclass
class GBDTParams:
    learning_rate: float = 0.03
    num_leaves: int = 15
    min_data_in_leaf: int = 200
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    lambda_l2: float = 1.0
    max_rounds: int = 2000
    early_stopping: int = 100
    seed: int = 0
    num_threads: int = 0
    extra: dict = field(default_factory=dict)

    def lgb(self):
        return {'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
                'learning_rate': self.learning_rate, 'num_leaves': self.num_leaves,
                'min_data_in_leaf': self.min_data_in_leaf, 'feature_fraction': self.feature_fraction,
                'bagging_fraction': self.bagging_fraction, 'bagging_freq': self.bagging_freq,
                'lambda_l2': self.lambda_l2, 'seed': self.seed, 'deterministic': True,
                'num_threads': self.num_threads, **self.extra}


class GBDTModel:
    """LightGBM + symmetrische Vorhersage + Kalibrierung auf dem Validierungsfenster."""

    def __init__(self, params=None, calibration='platt', features=FEATURES):
        self.params = params or GBDTParams()
        self.calibration = calibration
        self.features = list(features)
        self.booster = None
        self.calibrator = make_calibrator(calibration)
        self.best_iteration = None

    def _augment(self, X, y):
        X = X[self.features]
        return pd.concat([X, swap(X)], ignore_index=True), np.concatenate([y, 1 - np.asarray(y)])

    def fit(self, X_train, y_train, X_val, y_val):
        import lightgbm as lgb
        Xt, yt = self._augment(X_train, np.asarray(y_train, float))
        train_set = lgb.Dataset(Xt, yt, free_raw_data=True)
        if len(X_val):
            Xv, yv = self._augment(X_val, np.asarray(y_val, float))
            valid = lgb.Dataset(Xv, yv, reference=train_set)
            self.booster = lgb.train(self.params.lgb(), train_set, num_boost_round=self.params.max_rounds,
                                     valid_sets=[valid],
                                     callbacks=[lgb.early_stopping(self.params.early_stopping, verbose=False)])
            self.best_iteration = self.booster.best_iteration or self.params.max_rounds
            self.calibrator.fit(self.predict_raw(X_val), y_val)
        else:
            # Ohne Validierungsfenster: feste, vorsichtige Rundenzahl und keine Kalibrierung
            self.booster = lgb.train(self.params.lgb(), train_set, num_boost_round=300)
            self.best_iteration = 300
        return self

    def predict_raw(self, X):
        X = X[self.features]
        p_ab = self.booster.predict(X, num_iteration=self.best_iteration)
        p_ba = self.booster.predict(swap(X), num_iteration=self.best_iteration)
        return (p_ab + 1 - p_ba) / 2

    def predict(self, X):
        return self.calibrator.predict(self.predict_raw(X))

    def importance(self, kind='gain'):
        gain = self.booster.feature_importance(importance_type=kind)
        return pd.Series(gain / max(gain.sum(), 1e-12), index=self.booster.feature_name())

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'booster': self.booster.model_to_string(), 'calibrator': self.calibrator,
                         'calibration': self.calibration, 'features': self.features,
                         'params': asdict(self.params), 'best_iteration': self.best_iteration}, f)

    @classmethod
    def load(cls, path):
        import lightgbm as lgb
        with open(path, 'rb') as f:
            d = pickle.load(f)
        m = cls(GBDTParams(**d['params']), d['calibration'], d['features'])
        m.booster = lgb.Booster(model_str=d['booster'])
        m.calibrator, m.best_iteration = d['calibrator'], d['best_iteration']
        return m


def split_by_time(dates, val_months):
    """Grenze zwischen Training und Validierung: die letzten val_months vor dem Ende der Daten."""
    return dates.max() - pd.DateOffset(months=val_months)


def fit_on_history(table, labels, dates, params=None, calibration='platt', val_months=12):
    """Training < Grenze < Validierung (zeitlich danach), alles aus der übergebenen Historie."""
    cut = split_by_time(dates, val_months)
    train, val = dates <= cut, dates > cut
    return GBDTModel(params, calibration).fit(table[train], labels[train], table[val], labels[val])


class GBDTPredictor(Predictor):
    """
    Walk-forward-Adapter. prepare(matches) baut die Feature-Tabelle einmal für alle Matches (jede Zeile
    nur aus Matches davor), fit(history) merkt sich die Historie, gefittet wird beim ersten Fenster
    einer neuen refit-Periode (refit = 'month' | 'quarter' | 'year').
    """
    name = "gbdt"

    def __init__(self, refit='quarter', val_months=12, calibration='platt', params=None, min_train=2000):
        super().__init__()
        if refit not in REFIT:
            raise ValueError(f"refit muss einer von {', '.join(REFIT)} sein")
        self.refit, self.val_months, self.calibration = refit, val_months, calibration
        self.params = params or GBDTParams()
        self.min_train = min_train
        self.table = None
        self.model = None
        self.period = None
        self.history_ids = None
        self.fits = []  # (Periode, Trainingsende, n, beste Runde, Importance)

    def prepare(self, matches):
        table = gbdt_table(matches).set_index('match_id')
        self.table = table[~table.index.duplicated()]

    def fit(self, history):
        super().fit(history)
        h = history[playable(history)]
        self.history_ids = h['match_id'].to_numpy()

    def _train(self, period):
        rows = self.table.loc[self.history_ids]
        assert rows['date'].max() <= self.fit_max_date, "gbdt: Trainingszeilen nach dem Ende der Historie"
        dates = rows['date']
        self.model = fit_on_history(rows, rows['label'].to_numpy(float), dates, self.params, self.calibration,
                                    self.val_months)
        self.period = period
        self.fits.append({'period': str(period), 'train_until': dates.max(), 'n': len(rows),
                          'best_iteration': self.model.best_iteration, 'importance': self.model.importance()})

    def predict(self, window):
        if self.table is None:
            raise RuntimeError("GBDTPredictor.prepare(matches) zuerst aufrufen")
        self._check_no_leakage(window)
        if len(window) == 0:
            return np.array([])
        period = pd.Period(window['date'].min(), REFIT[self.refit])
        if self.model is None or period != self.period:
            if len(self.history_ids) < self.min_train:
                return np.full(len(window), np.nan)
            self._train(period)
        return self.model.predict(self.table.loc[window['match_id']])

    def importance(self):
        """Mittlerer Gain-Anteil je Feature über alle Fits, absteigend."""
        if not self.fits:
            return pd.Series(dtype=float)
        return pd.concat([f['importance'] for f in self.fits], axis=1).mean(axis=1).sort_values(ascending=False)


def train_live(matches, params=None, calibration='platt', val_months=12):
    """Live-Modell auf allen gespielten Matches (Ausgabe von prepare_matches)."""
    table = gbdt_table(matches)
    rows = table[playable(matches).to_numpy()]
    return fit_on_history(rows, rows['label'].to_numpy(float), rows['date'], params, calibration, val_months)


def main(argv=None):
    import argparse

    from ..config import default_config
    from ..data_loader import load_matches
    from ..evaluation.predictors import prepare_matches

    ap = argparse.ArgumentParser(description="LightGBM-Live-Modell auf allen Daten trainieren")
    ap.add_argument('--tour', default='atp')
    ap.add_argument('--calibration', default='platt')
    ap.add_argument('--val-months', type=int, default=12)
    ap.add_argument('--no-refresh', action='store_true')
    args = ap.parse_args(argv)
    df = load_matches(first_year=default_config.data_years_start, refresh=not args.no_refresh, tour=args.tour)
    model = train_live(prepare_matches(df), calibration=args.calibration, val_months=args.val_months)
    path = model_path(args.tour)
    model.save(path)
    print(f"Gespeichert: {path} (beste Runde {model.best_iteration})")
    print(model.importance().sort_values(ascending=False).head(20).to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
