"""
Metriken für Wahrscheinlichkeiten und der Vergleich gegen den Markt.

Für Wetten zählt die Kalibrierung, deshalb ist Log Loss die Hauptmetrik (Accuracy nur zur Info).
Verglichen wird immer auf derselben Stichprobe: Matches mit scharfem Marktpreis (Pinnacle, seit
Februar 2026 Betfair Exchange, siehe SHARP_SOURCES), ohne W/O und mit einer Vorhersage von jedem Modell
(common_sample).
"""
import math

import numpy as np
import pandas as pd

EPS = 1e-15
# Marktquellen, gegen die verglichen wird (Durchschnittsquoten sind zu weich für den Benchmark)
SHARP_SOURCES = ('ps', 'bfe')


def _arrays(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if y.shape != p.shape:
        raise ValueError("y und p müssen gleich lang sein")
    return y, p


def pointwise_log_loss(y, p):
    y, p = _arrays(y, p)
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def log_loss(y, p):
    return float(np.mean(pointwise_log_loss(y, p)))


def brier(y, p):
    y, p = _arrays(y, p)
    return float(np.mean((p - y) ** 2))


def accuracy(y, p):
    """Anteil richtiger Tipps (p > 0.5 -> A). p = 0.5 zählt als halb richtig."""
    y, p = _arrays(y, p)
    hit = np.where(p == 0.5, 0.5, ((p > 0.5) == (y == 1)).astype(float))
    return float(np.mean(hit))


def calibration_table(y, p, bins=10):
    """Pro Bin (gleich breit auf [0, 1] oder eigene Grenzen): Anzahl, mittleres p, echte Siegquote."""
    y, p = _arrays(y, p)
    edges = np.linspace(0, 1, bins + 1) if np.isscalar(bins) else np.asarray(bins, float)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)
    rows = []
    for i in range(len(edges) - 1):
        m = idx == i
        if not m.any():
            continue
        rows.append({'bin': f"{edges[i]:.1f}-{edges[i + 1]:.1f}", 'count': int(m.sum()),
                     'mean_p': float(p[m].mean()), 'win_rate': float(y[m].mean()),
                     'diff': float(y[m].mean() - p[m].mean())})
    return pd.DataFrame(rows, columns=['bin', 'count', 'mean_p', 'win_rate', 'diff'])


def ece(y, p, bins=10):
    """Expected Calibration Error: mit der Bin-Größe gewichtete |Siegquote - mittleres p|."""
    table = calibration_table(y, p, bins)
    if table.empty:
        return float('nan')
    return float((table['count'] * table['diff'].abs()).sum() / table['count'].sum())


def summarize(y, p):
    return {'n': len(y), 'log_loss': log_loss(y, p), 'brier': brier(y, p),
            'accuracy': accuracy(y, p), 'ece': ece(y, p)}


# --------------------------------------------------------------- Tests

def _norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


def paired_bootstrap(loss_model, loss_ref, n_boot=2000, seed=0):
    """
    Gepaarter Bootstrap auf d = loss_model - loss_ref (je Match). d < 0: Modell ist besser.
    Gibt mittlere Differenz, 95-%-KI und zweiseitigen p-Wert (H0: E[d] = 0) zurück.
    """
    d = np.asarray(loss_model, float) - np.asarray(loss_ref, float)
    n = len(d)
    if n == 0:
        return {'n': 0, 'mean_diff': float('nan'), 'ci_low': float('nan'), 'ci_high': float('nan'), 'p_value': float('nan')}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = d[rng.integers(0, n, n)].mean()
    # p-Wert: Anteil der um den Mittelwert zentrierten Bootstrap-Mittel, die mindestens so extrem sind
    centered = means - d.mean()
    p_value = float(np.mean(np.abs(centered) >= abs(d.mean())))
    low, high = np.percentile(means, [2.5, 97.5])
    return {'n': n, 'mean_diff': float(d.mean()), 'ci_low': float(low), 'ci_high': float(high),
            'p_value': max(p_value, 1.0 / n_boot)}


def diebold_mariano(loss_model, loss_ref):
    """
    Diebold-Mariano-Test (h = 1, Matches als unabhängig angenommen) auf d = loss_model - loss_ref.
    Normal-Approximation, zweiseitig.
    """
    d = np.asarray(loss_model, float) - np.asarray(loss_ref, float)
    n = len(d)
    if n < 2 or d.std(ddof=1) == 0:
        return {'n': n, 'stat': float('nan'), 'p_value': float('nan')}
    stat = d.mean() / (d.std(ddof=1) / math.sqrt(n))
    return {'n': n, 'stat': float(stat), 'p_value': float(2 * _norm_sf(abs(stat)))}


# --------------------------------------------------------- Vergleich

def common_sample(predictions, models, sources=SHARP_SOURCES):
    """
    Gleiche Stichprobe für alle: scharfer Marktpreis vorhanden (sources=None: egal), kein W/O,
    p von jedem Modell.
    """
    cols = [f'p_{m}' for m in models]
    mask = predictions['status'] != 'W/O'
    if sources is not None:
        mask &= predictions['market_source'].isin(sources)
    return predictions[mask & predictions[cols].notna().all(axis=1)]


def metric_table(predictions, models, by='year', sources=SHARP_SOURCES):
    """Log Loss, Brier, Accuracy, ECE pro Modell und Gruppe (Standard: Jahr) plus 'Gesamt'."""
    sample = common_sample(predictions, models, sources)
    groups = [(str(k), g) for k, g in sample.groupby(by)] if by else []
    groups.append(('Gesamt', sample))
    rows = []
    for key, g in groups:
        for m in models:
            if len(g) == 0:
                continue
            rows.append({by or 'group': key, 'model': m, **summarize(g['label'], g[f'p_{m}'])})
    return pd.DataFrame(rows)


def significance_vs_market(predictions, models, n_boot=2000, seed=0, ref_model='market', sources=SHARP_SOURCES):
    """
    Log-Loss-Differenz jedes Modells gegen den Markt (oder ref_model): Bootstrap-KI, Bootstrap- und
    DM-p-Wert. Negativ = besser als die Referenz.
    """
    sample = common_sample(predictions, models, sources)
    ref = pointwise_log_loss(sample['label'], sample[f'p_{ref_model}'])
    rows = []
    for m in models:
        if m == ref_model:
            continue
        loss = pointwise_log_loss(sample['label'], sample[f'p_{m}'])
        boot = paired_bootstrap(loss, ref, n_boot=n_boot, seed=seed)
        dm = diebold_mariano(loss, ref)
        rows.append({'model': m, 'n': boot['n'], 'mean_diff': boot['mean_diff'], 'ci_low': boot['ci_low'],
                     'ci_high': boot['ci_high'], 'p_boot': boot['p_value'], 'dm_stat': dm['stat'],
                     'p_dm': dm['p_value']})
    return pd.DataFrame(rows)


def go_no_go(significance, model='blend', alpha=0.05):
    """
    Go nur, wenn das Modell (Blend, Stack) den Log Loss des Markts signifikant senkt:
    obere KI-Grenze der Differenz < 0 und beide p-Werte (Bootstrap, Diebold-Mariano) < alpha.
    """
    row = significance[significance['model'] == model]
    if row.empty:
        return False
    r = row.iloc[0]
    p_dm = r['p_dm'] if 'p_dm' in r and pd.notna(r['p_dm']) else 0.0
    return bool(r['ci_high'] < 0 and r['p_boot'] < alpha and p_dm < alpha)


def segments(predictions, odds_edges=(0.6, 0.75)):
    """Segment-Spalten für den Vergleich: Quotenbereich (Favoriten-p des Markts), Turnierstufe, Untergrund."""
    p = predictions['p_market'].to_numpy(float)
    fav = np.maximum(p, 1 - p)
    edges = [0.5, *odds_edges, 1.0]
    labels = [f"Fav {lo:.2f}-{hi:.2f}" for lo, hi in zip(edges[:-1], edges[1:])]
    band = pd.Series(np.array(labels, dtype=object)[np.clip(np.digitize(fav, odds_edges), 0, len(labels) - 1)],
                     index=predictions.index)
    unknown = pd.Series('unbekannt', index=predictions.index)
    level = predictions['tourney_level'].astype(str) if 'tourney_level' in predictions else unknown
    surface = predictions['surface'].astype(str) if 'surface' in predictions else unknown
    return {'Quotenbereich': band, 'Turnierstufe': level, 'Untergrund': surface}


def segment_table(predictions, model, models=None, ref_model='market', n_boot=500, seed=0, min_n=200):
    """
    Log-Loss-Differenz Modell - Referenz je Segment (Quotenbereich, Turnierstufe, Untergrund) auf der
    gemeinsamen Stichprobe, mit Bootstrap-KI. Segmente mit weniger als min_n Matches fallen weg.
    """
    sample = common_sample(predictions, models or [ref_model, model])
    loss_m = pointwise_log_loss(sample['label'], sample[f'p_{model}'])
    loss_r = pointwise_log_loss(sample['label'], sample[f'p_{ref_model}'])
    rows = []
    for kind, values in segments(sample).items():
        for key in sorted(values.unique()):
            m = (values == key).to_numpy()
            if m.sum() < min_n:
                continue
            boot = paired_bootstrap(loss_m[m], loss_r[m], n_boot=n_boot, seed=seed)
            rows.append({'segment': kind, 'value': key, 'n': int(m.sum()), f'll_{model}': float(loss_m[m].mean()),
                         f'll_{ref_model}': float(loss_r[m].mean()), 'mean_diff': boot['mean_diff'],
                         'ci_low': boot['ci_low'], 'ci_high': boot['ci_high'], 'p_boot': boot['p_value']})
    return pd.DataFrame(rows)
