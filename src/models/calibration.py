"""
Kalibrierung von Wahrscheinlichkeiten für P(A gewinnt), immer symmetrisch: c(1 - p) = 1 - c(p).
Damit bleibt p(A, B) = 1 - p(B, A) auch nach der Kalibrierung erhalten.

    Platt (ohne Intercept):  p' = sigmoid(a * logit(p))
    Temperature Scaling:     p' = sigmoid(z / T) auf Logits z, mathematisch Platt mit a = 1 / T
    Isotonic (symmetrisch):  monotone Treppenfunktion, gefittet auf (p, y) und (1 - p, 1 - y)
"""
import numpy as np

from ..evaluation.predictors import fit_logistic, logit, sigmoid


def fit_temperature(z, y):
    """Temperatur T für Logits z (Logit von P(A gewinnt)), minimiert den Log Loss auf (z, y)."""
    a = fit_logistic(np.asarray(z, float), y, l2=1e-6)[0]
    return float(1.0 / a) if a > 1e-6 else float('inf')


def apply_temperature(z, temperature):
    return sigmoid(np.asarray(z, float) / temperature)


class Platt:
    name = 'platt'

    def __init__(self):
        self.a = 1.0

    def fit(self, p, y):
        self.a = float(fit_logistic(logit(p), y, l2=1e-6)[0])
        return self

    def predict(self, p):
        return sigmoid(self.a * logit(p))


def _pav(y, w):
    """Pool-Adjacent-Violators: monoton steigende Folge, die sum w (y - f)^2 minimiert."""
    values, weights, sizes = [], [], []
    for yi, wi in zip(y, w):
        values.append(yi)
        weights.append(wi)
        sizes.append(1)
        while len(values) > 1 and values[-2] >= values[-1]:
            wsum = weights[-2] + weights[-1]
            values[-2] = (values[-2] * weights[-2] + values[-1] * weights[-1]) / wsum
            weights[-2] = wsum
            sizes[-2] += sizes[-1]
            values.pop(), weights.pop(), sizes.pop()
    return np.repeat(values, sizes)


class Isotonic:
    name = 'isotonic'

    def __init__(self, eps=1e-3):
        self.eps = eps
        self.x, self.f = np.array([0.0, 1.0]), np.array([0.0, 1.0])

    def fit(self, p, y):
        p, y = np.asarray(p, float), np.asarray(y, float)
        x = np.concatenate([p, 1 - p])
        t = np.concatenate([y, 1 - y])
        order = np.argsort(x, kind='stable')
        x, t = x[order], t[order]
        f = _pav(t, np.ones_like(t))
        # Stützstellen: Mittelwert je gleichem x (Treppenfunktion, dazwischen linear)
        ux, idx = np.unique(x, return_inverse=True)
        uf = np.bincount(idx, weights=f) / np.bincount(idx)
        self.x, self.f = ux, np.clip(uf, self.eps, 1 - self.eps)
        return self

    def predict(self, p):
        p = np.asarray(p, float)
        # Exakt symmetrisch: Mittel aus c(p) und 1 - c(1 - p)
        return (np.interp(p, self.x, self.f) + 1 - np.interp(1 - p, self.x, self.f)) / 2


class Identity:
    name = 'none'

    def fit(self, p, y):
        return self

    def predict(self, p):
        return np.asarray(p, float)


CALIBRATORS = {'platt': Platt, 'isotonic': Isotonic, 'none': Identity}


def make_calibrator(kind='platt'):
    if kind not in CALIBRATORS:
        raise ValueError(f"Unbekannte Kalibrierung {kind!r} (bekannt: {', '.join(CALIBRATORS)})")
    return CALIBRATORS[kind]()
