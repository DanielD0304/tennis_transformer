"""
Stufe 4 der Modell-Leiter: Stacking mit dem Markt.

Logistische Regression ohne Intercept auf logit(p_market) und den Logits der Basismodelle, walk-forward
gefittet auf deren out-of-sample-Vorhersagen aus früheren Fenstern. Ohne Intercept und mit nur
ungeraden Features (Logits, optional mal einer Indikatorfunktion des Quotenbereichs, die nur vom
Favoriten-p abhängt) bleibt p(A, B) = 1 - p(B, A).

Der Go/No-Go-Test läuft im Backtest über evaluation.metrics (gepaarter Bootstrap, Diebold-Mariano):
nur wenn der Stack den Log Loss des Markts signifikant senkt, steckt im Modell Information, die der
Markt nicht schon hat.
"""
import numpy as np

from ..evaluation.predictors import Predictor, fit_logistic, logit, playable, sigmoid

# Quotenbereiche nach Favoriten-Wahrscheinlichkeit des Markts: [0.5, 0.6), [0.6, 0.75), [0.75, 1]
DEFAULT_BANDS = (0.6, 0.75)


def odds_band(p_market, edges=DEFAULT_BANDS):
    """Index des Quotenbereichs (0 = ausgeglichen) aus max(p, 1 - p), symmetrisch in A/B."""
    fav = np.maximum(p_market, 1 - p_market)
    return np.digitize(fav, edges)


class StackPredictor(Predictor):
    min_train = 300

    def __init__(self, bases=('elo2', 'gbdt'), bands=None, name=None, l2=1.0):
        super().__init__()
        self.bases = tuple(bases)
        self.bands = tuple(bands) if bands else None
        self.name = name or "stack"
        self.l2 = l2
        self.columns = ['market'] + list(self.bases)
        self.coef = None
        self.n_train = 0

    def feature_names(self):
        names = [f'logit_{c}' for c in self.columns]
        if self.bands:
            for k in range(1, len(self.bands) + 1):
                names += [f'logit_{c}*band{k}' for c in self.columns]
        return names

    def _features(self, df):
        X = np.column_stack([logit(df[f'p_{c}']) for c in self.columns])
        if self.bands:
            band = odds_band(df['p_market'].to_numpy(float), self.bands)
            X = np.column_stack([X] + [X * (band == k)[:, None] for k in range(1, len(self.bands) + 1)])
        return X

    def _complete(self, df):
        cols = [f'p_{c}' for c in self.columns]
        if not set(cols) <= set(df.columns):
            return np.zeros(len(df), bool)
        return df[cols].notna().all(axis=1).to_numpy()

    def fit(self, history):
        super().fit(history)
        h = history[playable(history).to_numpy() & self._complete(history)]
        self.n_train = len(h)
        if len(h) >= self.min_train:
            self.coef = fit_logistic(self._features(h), h['label'], l2=self.l2)

    def predict(self, window):
        self._check_no_leakage(window)
        ok = self._complete(window)
        if self.coef is None:
            # Zu wenig Historie: p = Markt (Gewicht 1 auf logit(p_market))
            return np.where(ok, window['p_market'].to_numpy(float), np.nan)
        p = sigmoid(self._features(window) @ self.coef)
        return np.where(ok, p, np.nan)

    def weights(self):
        if self.coef is None:
            return {}
        return dict(zip(self.feature_names(), self.coef))
