"""
Timing-Backtest (Phase 4): Lohnt es sich, bei Polymarket oder Kalshi früh einzusteigen, wenn das eigene Modell
(elo2, gbdt) vom Preis abweicht?

    python -m src.evaluation.timing                         # alle Venues, Level und Zeitpunkte
    python -m src.evaluation.timing --models elo2 --venues kalshi --levels challenger
    python -m src.evaluation.timing --pm-spread 0.03 --pm-fee 0 --min-volume-kalshi 500

Ablauf:
1. Walk-forward-Vorhersagen p (backtest.run, Elo und Fits nur mit Matches vor dem Testfenster, inkl. der
   Leakage-Asserts in backtest.walk_forward und Predictor._check_no_leakage). Elo und Features laufen über alle
   Level einer Tour gemeinsam.
2. Jeder abgerechnete Markt aus data/odds/history wird über build.MatchIndex (Namen + Datum) einem Match
   zugeordnet. Der Preis zum Zeitpunkt t ist der letzte Preis <= t (Eröffnung, -24 h, -6 h, -1 h, Closing;
   siehe odds/history.price_at), nie ein späterer.
3. Einstieg zum erreichbaren Preis: Kalshi kauft zum Ask aus den Candlesticks (B über 1 - Bid von A),
   Polymarket zum Preis + halbem Spread (--pm-spread, historisch gibt es dort kein Bid/Ask). Danach Gebühren
   laut venues.py (Kalshi-Formel, Polymarket-Taker-Fee). Bei Polymarket ist die „Eröffnung“ der erste echte
   Kurs (der erste Punkt ist meist der Mittelpreis 0,5 eines leeren Orderbuchs), Preise von genau 0,5 gelten
   als Platzhalter und werden nicht gehandelt.
4. Wette, wenn p * Netto-Quote - 1 > Schwelle. Die Schwelle wird walk-forward gewählt: pro Quartal die
   Schwelle mit dem höchsten Gewinn in allen Quartalen davor (erstes Quartal: DEFAULT_THRESHOLD).
5. Kennzahlen pro Venue x Level x Zeitpunkt x Modell: Wetten, Yield mit Bootstrap-KI, CLV gegen das Closing
   der Venue und (wo vorhanden) gegen Pinnacle/Betfair, Log Loss von p gegen den Preis zum Zeitpunkt t
   (gepaarter Bootstrap aus metrics). Go/No-Go mit Holm-Korrektur über alle Segmente und Robustheit über die
   Quartale (Gewinn in mindestens zwei Quartalen einzeln bestätigt, kein Quartal mit mehr als 60 % der Wetten).

Ausgewertet wird das Ergebnis der Markt-Auflösung (das zahlt die Venue aus). Märkte mit 50-50-Auflösung fehlen.
Ehrliche Grenzen: Polymarket hat historisch nur Mittel-/Letztkurse in 10-Minuten-Auflösung, das Volumen ist
das Gesamtvolumen des Markts (als Filter ein leichter Blick in die Zukunft, deshalb standardmäßig aus).
Kalshi: Volumen bis t aus den Stunden-Candles, Bid/Ask zum Periodenende.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from ..sources.build import MatchIndex
from ..sources.common import yyyymmdd
from . import metrics
from .betting import bootstrap_yield
from .predictors import fit_logistic, logit, sigmoid
from .venues import POLYMARKET_FEE, get_venue

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(REPO_ROOT, "data", "eval")
TABLE_PATH = os.path.join(OUT_DIR, "timing.parquet")
BLEND_PATH = os.path.join(OUT_DIR, "timing_blend.json")

# Zeitpunkt -> Spalten der Historie (ts, Preis, Bid, Ask, Volumen bis t)
POINTS = {
    'open': ('open_ts', 'open_price_a', 'open_bid_a', 'open_ask_a', None),
    '24h': ('t24h_ts', 't24h_price_a', 't24h_bid_a', 't24h_ask_a', 't24h_volume'),
    '6h': ('t6h_ts', 't6h_price_a', 't6h_bid_a', 't6h_ask_a', 't6h_volume'),
    '1h': ('t1h_ts', 't1h_price_a', 't1h_bid_a', 't1h_ask_a', 't1h_volume'),
    'close': ('close_ts', 'close_price_a', 'close_bid_a', 'close_ask_a', 'close_volume'),
}
# Polymarket: Eröffnung = erster echter Kurs (der erste Punkt ist oft der Mittelpreis 0,5 eines leeren Orderbuchs)
PM_POINTS = {**POINTS, 'open': ('move_ts', 'move_price_a', None, None, None)}
PLACEHOLDER = 0.5  # Polymarket-Preis von genau 0,5: leeres Orderbuch, nicht handelbar
THRESHOLDS = (0.0, 0.02, 0.05, 0.08, 0.12, 0.2)
DEFAULT_THRESHOLD = 0.05
EXTREME = 0.03  # Preis unter 3 % / über 97 %: Verdacht auf Live-Handel oder schon entschieden
MAX_STALE = pd.Timedelta(hours=2)  # wie odds/report.py: Closing viel älter als der Start -> Startzeit verschoben
MIN_BETS = 30  # Segmente mit weniger Wetten gehen nicht in den Mehrfachtest
# GO verlangt, dass sich der Gewinn in mindestens so vielen Quartalen einzeln bestätigt (je >= MIN_BETS Wetten und
# untere Grenze des Yield-KI > 0 im Quartal). Ein einzelner Monat reicht nicht.
MIN_PERIODS = 2
# ... und dass kein Quartal mehr als diesen Anteil der Wetten trägt. Anlass: Polymarket ITF -1 h, blend_gbdt (Lauf
# 27.10.2026) mit 97 % der Wetten im Oktober 2026, als Polymarket viele dünne ITF-Märkte ohne aussagekräftigen
# Preis listete. Zwei Quartale mit je >= 30 Wetten waren dort nur formal erfüllt.
MAX_PERIOD_SHARE = 0.6
# Polymarket-Eröffnung: nur Mittel-/Letztkurs ohne Orderbuch-Tiefe. Bei Kalshi (echtes Ask) verliert dieselbe
# Strategie zur Eröffnung, der PM-Gewinn ist deshalb nicht handelbar und geht nicht in den Mehrfachtest.
NOT_TRADABLE = {('polymarket', 'open')}
ALPHA = 0.05
GO_MIN_CLV = 0.0


# ------------------------------------------------------------------ Verknüpfung

def link_predictions(history, predictions):
    """
    Pro Markt (Zeile der Historie) das Match aus den Walk-forward-Vorhersagen. Neue Spalten: match_id, flip
    (player_a des Markts ist player_b der Vorhersage), level_pred und für jedes p_<modell> sowie p_market
    (Pinnacle/Betfair ohne Marge, nur wenn market_source scharf ist) der Wert aus Sicht von player_a des Markts.
    """
    preds = predictions.reset_index(drop=True)
    index = MatchIndex()
    for i, r in enumerate(preds[['player_a', 'player_b', 'date']].to_dict('records')):
        day = yyyymmdd(r['date'])
        index.add({'source': 'pred', 'winner_name': r['player_a'], 'loser_name': r['player_b'],
                   'match_date': day, 'tourney_date': day, '_i': i})
    found, flip = [], []
    for r in history[['player_a', 'player_b', 'start_time_utc']].to_dict('records'):
        day = yyyymmdd(r['start_time_utc'])
        other, same = index.find({'source': 'market', 'winner_name': r['player_a'], 'loser_name': r['player_b'],
                                  'match_date': day, 'tourney_date': day}) if day else (None, None)
        found.append(other['_i'] if other is not None else -1)
        flip.append(other is not None and not same)
    out = history.copy().reset_index(drop=True)
    idx, flip = np.asarray(found), np.asarray(flip)
    ok = idx >= 0
    out['match_id'] = np.where(ok, preds['match_id'].to_numpy(object)[np.where(ok, idx, 0)], None)
    out['flip'] = flip
    p_cols = [c for c in preds.columns if c.startswith('p_')]
    sharp = preds['market_source'].isin(metrics.SHARP_SOURCES).to_numpy() if 'market_source' in preds else None
    for c in p_cols:
        v = preds[c].to_numpy(float)
        if c == 'p_market' and sharp is not None:
            v = np.where(sharp, v, np.nan)
        v = np.where(ok, v[np.where(ok, idx, 0)], np.nan)
        out[c] = np.where(flip, 1 - v, v)
    if 'level' in preds:
        # Polymarket kennzeichnet nur ITF, Challenger und Haupttour nicht: dann das Level aus den Matchdaten
        out['level_pred'] = np.where(ok, preds['level'].to_numpy(object)[np.where(ok, idx, 0)], None)
        level = out['level'] if 'level' in out else pd.Series(None, index=out.index, dtype=object)
        out['level'] = level.where(level.notna(), out['level_pred'])
    return out


# ---------------------------------------------------------------- Preise und Gebühren

def columns(point, venue):
    """(ts, Preis, Bid, Ask, Volumen bis t) der Historie für Zeitpunkt und Venue."""
    return (PM_POINTS if venue == 'polymarket' else POINTS)[point]


def entry_prices(df, point, venue, pm_spread=0.02):
    """
    Erreichbare Kaufpreise (0–1) für A und B zum Zeitpunkt point. Kalshi: A zum Ask, B zu 1 - Bid von A (ein
    Nein auf A ist ein Ja auf B); fehlt eine Seite des Orderbuchs, gibt es keinen Preis. Polymarket: Preis +/-
    halber Spread. Ergebnis: (Mitte bzw. Preis von A, Kaufpreis A, Kaufpreis B), NaN = kein Preis.
    """
    _, price_col, bid_col, ask_col, _ = columns(point, venue)
    mid = pd.to_numeric(df[price_col], errors='coerce').to_numpy(float) if price_col in df else np.full(len(df), np.nan)
    if venue == 'kalshi':
        bid = pd.to_numeric(df.get(bid_col), errors='coerce').to_numpy(float) if bid_col in df else np.full(len(df), np.nan)
        ask = pd.to_numeric(df.get(ask_col), errors='coerce').to_numpy(float) if ask_col in df else np.full(len(df), np.nan)
        buy_a = np.where((ask > 0) & (ask < 1), ask, np.nan)
        buy_b = np.where((bid > 0) & (bid < 1), 1 - bid, np.nan)
    else:
        half = pm_spread / 2
        mid = np.where(mid == PLACEHOLDER, np.nan, mid)
        buy_a, buy_b = mid + half, (1 - mid) + half
    ok = np.isfinite(mid) & (mid >= EXTREME) & (mid <= 1 - EXTREME)
    buy_a = np.where(ok & (buy_a < 1), buy_a, np.nan)
    buy_b = np.where(ok & (buy_b < 1), buy_b, np.nan)
    return np.where(ok, mid, np.nan), buy_a, buy_b


def venue_profile(venue, pm_fee=POLYMARKET_FEE):
    return get_venue('polymarket', taker_fee=pm_fee) if venue == 'polymarket' else get_venue(venue)


def net_odds(buy_price, profile):
    """Kaufpreis (0–1) -> Netto-Dezimalquote nach Gebühr (Kalshi-Formel bzw. Taker-Fee)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return profile.net_odds_from_gross(1 / np.asarray(buy_price, float))


def usable(df, point, venue, min_volume=0.0):
    """Märkte mit Sieger, nicht verschobenem Start, Preis zum Zeitpunkt und (optional) Mindestvolumen."""
    ts_col, price_col, _, _, vol_col = columns(point, venue)
    if price_col not in df or ts_col not in df:
        return np.zeros(len(df), dtype=bool)
    start = pd.to_datetime(df['start_time_utc'], utc=True)
    stale = (start - pd.to_datetime(df['close_ts'], utc=True)) > MAX_STALE
    ok = df['winner'].isin(['a', 'b']) & ~stale & df[price_col].notna()
    ok &= pd.to_datetime(df[ts_col], utc=True) <= start  # nie ein Preis nach der geplanten Startzeit
    if venue == 'polymarket':
        ok &= df[price_col] != PLACEHOLDER
    if min_volume:
        if venue == 'kalshi' and vol_col:
            ok &= pd.to_numeric(df[vol_col], errors='coerce').fillna(0) >= min_volume
        else:  # Polymarket: nur Gesamtvolumen bekannt
            ok &= pd.to_numeric(df['volume'], errors='coerce').fillna(0) >= min_volume
    return ok.to_numpy().copy()


# --------------------------------------------------------------------- Wetten

def candidate_bets(df, point, venue, model, pm_spread=0.02, pm_fee=POLYMARKET_FEE, min_volume=0.0):
    """
    Alle möglichen Einstiege (eine Zeile pro Markt, Seite mit dem höheren EV) zum Zeitpunkt point mit dem p des
    Modells. Spalten: event_id, start_time_utc, period, level, side, p, price (Kaufpreis), net_odds, ev, won,
    profit (pro Einheit Einsatz), clv (gegen Closing der Venue), clv_sharp (gegen Pinnacle/Betfair ohne Marge).
    """
    profile = venue_profile(venue, pm_fee)
    ok = usable(df, point, venue, min_volume).copy()
    p = df[f'p_{model}'].to_numpy(float)
    mid, buy_a, buy_b = entry_prices(df, point, venue, pm_spread)
    net_a, net_b = net_odds(buy_a, profile), net_odds(buy_b, profile)
    with np.errstate(invalid='ignore'):
        ev_a, ev_b = p * net_a - 1, (1 - p) * net_b - 1
    ev_a, ev_b = np.where(np.isfinite(ev_a), ev_a, -np.inf), np.where(np.isfinite(ev_b), ev_b, -np.inf)
    on_a = ev_a >= ev_b
    ev = np.where(on_a, ev_a, ev_b)
    ok &= np.isfinite(p) & np.isfinite(ev)
    a_won = (df['winner'] == 'a').to_numpy()
    won = np.where(on_a, a_won, ~a_won)
    net = np.where(on_a, net_a, net_b)
    price = np.where(on_a, buy_a, buy_b)
    close_a = pd.to_numeric(df['close_price_a'], errors='coerce').to_numpy(float)
    close_side = np.where(on_a, close_a, 1 - close_a)
    sharp_a = df['p_market'].to_numpy(float) if 'p_market' in df else np.full(len(df), np.nan)
    sharp_side = np.where(on_a, sharp_a, 1 - sharp_a)
    start = pd.to_datetime(df['start_time_utc'], utc=True)
    level = df['level'].fillna('?') if 'level' in df else pd.Series('?', index=df.index)
    out = pd.DataFrame({
        'event_id': df['event_id'].to_numpy(), 'start_time_utc': start, 'period': start.dt.tz_localize(None).dt.to_period('Q'),
        'venue': venue, 'level': level.to_numpy(), 'point': point, 'model': model, 'side': np.where(on_a, 'a', 'b'),
        'p': np.where(on_a, p, 1 - p), 'mid': np.where(on_a, mid, 1 - mid), 'price': price, 'net_odds': net, 'ev': ev,
        'won': won, 'profit': np.where(won, net - 1, -1.0),
        'clv': close_side / price - 1, 'clv_sharp': sharp_side / price - 1,
    })
    return out[ok].reset_index(drop=True)


def choose_threshold(past, thresholds=THRESHOLDS, default=DEFAULT_THRESHOLD):
    """Schwelle mit dem höchsten Gewinn (flat) auf den vergangenen Kandidaten; ohne Vergangenheit: default.
    Ist keine Schwelle profitabel, die strengste (so wenig Wetten wie möglich)."""
    if past is None or len(past) == 0:
        return default
    profits = [past.loc[past['ev'] > t, 'profit'].sum() for t in thresholds]
    best = int(np.argmax(profits))
    return thresholds[best] if profits[best] > 0 else max(thresholds)


def walk_forward_bets(candidates, thresholds=THRESHOLDS, default=DEFAULT_THRESHOLD):
    """Pro Quartal die Schwelle nur aus früheren Quartalen wählen und anwenden. Spalte 'threshold' ergänzt."""
    out = []
    periods = sorted(candidates['period'].unique())
    for period in periods:
        past = candidates[candidates['period'] < period]
        assert (past['start_time_utc'] < candidates.loc[candidates['period'] == period, 'start_time_utc'].min()).all(), \
            "Leakage: Schwelle aus Matches im oder nach dem Testquartal"
        t = choose_threshold(past, thresholds, default)
        now = candidates[(candidates['period'] == period) & (candidates['ev'] > t)]
        out.append(now.assign(threshold=t))
    if not out:
        return candidates.iloc[0:0].assign(threshold=np.nan)
    return pd.concat(out, ignore_index=True)


def yield_p_value(profits):
    """
    Einseitiger p-Wert für H0: Yield <= 0 bei Flat-Einsätzen (Gewinn pro Einheit), Normal-Approximation des
    Mittelwerts. Anders als ein Bootstrap-p-Wert hat er keine Untergrenze 1/n_boot, die Holm über viele
    Segmente sonst unnötig streng macht.
    """
    import math
    x = np.asarray(profits, float)
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return float('nan')
    z = x.mean() / (x.std(ddof=1) / math.sqrt(n))
    return float(0.5 * math.erfc(z / math.sqrt(2)))


def holm(p_values):
    """Holm-Bonferroni: adjustierte p-Werte (gleiche Reihenfolge, NaN bleibt NaN)."""
    p = np.asarray(p_values, float)
    out = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    m, running = len(order), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


# ------------------------------------------------------------------- Auswertung

def _blend_data(df, point, model, venue):
    _, price_col, _, _, _ = columns(point, venue)
    ok = usable(df, point, venue) & np.isfinite(df[f'p_{model}'].to_numpy(float))
    q = pd.to_numeric(df[price_col], errors='coerce').to_numpy(float)
    X = np.column_stack([logit(q), logit(df[f'p_{model}'].to_numpy(float))])
    return ok, X, (df['winner'] == 'a').to_numpy(float)


def blend_weights(df, point, model, venue, min_train=300):
    """Gewichte [Preis, Modell] des Blends auf allen Märkten (für den Live-Einsatz im Paper-Trading) oder None."""
    ok, X, y = _blend_data(df, point, model, venue)
    if ok.sum() < min_train:
        return None
    return [float(w) for w in fit_logistic(X[ok], y[ok])] + [int(ok.sum())]


def blend_column(df, point, model, venue, min_train=300):
    """
    p_blend_<model>: logistische Regression auf [logit(Preis zum Zeitpunkt), logit(p_model)], pro Quartal nur auf
    früheren Quartalen gefittet (ohne Fit: der Preis selbst). Prüft, ob das Modell dem frühen Preis etwas hinzufügt.
    """
    ok, X, y = _blend_data(df, point, model, venue)
    start = pd.to_datetime(df['start_time_utc'], utc=True)
    period = start.dt.tz_localize(None).dt.to_period('Q').to_numpy()
    out = np.full(len(df), np.nan)
    for per in sorted(set(period[ok])):
        test = ok & (period == per)
        train = ok & (period < per)
        coef = fit_logistic(X[train], y[train]) if train.sum() >= min_train else np.array([1.0, 0.0])
        out[test] = sigmoid(X[test] @ coef)
    return out


def log_loss_vs_price(df, point, model, venue, n_boot=1000, seed=0):
    """Log Loss von p_model gegen den Preis zum Zeitpunkt (gleiche Märkte). mean_diff < 0: Modell besser."""
    _, price_col, _, _, _ = columns(point, venue)
    q = pd.to_numeric(df[price_col], errors='coerce').to_numpy(float)
    p = df[f'p_{model}'].to_numpy(float)
    ok = usable(df, point, venue) & np.isfinite(p) & np.isfinite(q)
    if ok.sum() == 0:
        return {'n_ll': 0, 'll_model': np.nan, 'll_price': np.nan, 'll_diff': np.nan, 'll_ci_low': np.nan,
                'll_ci_high': np.nan}
    y = (df['winner'] == 'a').to_numpy(float)[ok]
    loss_m, loss_q = metrics.pointwise_log_loss(y, p[ok]), metrics.pointwise_log_loss(y, q[ok])
    boot = metrics.paired_bootstrap(loss_m, loss_q, n_boot=n_boot, seed=seed)
    return {'n_ll': int(ok.sum()), 'll_model': float(loss_m.mean()), 'll_price': float(loss_q.mean()),
            'll_diff': boot['mean_diff'], 'll_ci_low': boot['ci_low'], 'll_ci_high': boot['ci_high']}


def period_stats(bets, n_boot=2000, seed=0):
    """
    Robustheit über die Zeit: n_periods (Quartale mit >= MIN_BETS Wetten), n_periods_pos (davon mit unterer Grenze
    des Yield-KI > 0 im Quartal) und max_period_share (größter Anteil eines Quartals an allen Wetten).
    """
    if 'period' not in bets or len(bets) == 0:
        return {'n_periods': 0, 'n_periods_pos': 0, 'max_period_share': np.nan}
    sizes = bets.groupby('period').size()
    n_pos = 0
    for period in sizes.index[sizes >= MIN_BETS]:
        profit = bets.loc[bets['period'] == period, 'profit']
        low, _ = bootstrap_yield(profit, np.ones(len(profit)), n_boot=n_boot, seed=seed)
        n_pos += bool(low > 0)
    return {'n_periods': int((sizes >= MIN_BETS).sum()), 'n_periods_pos': n_pos,
            'max_period_share': float(sizes.max() / len(bets))}


def summarize_bets(bets, n_boot=2000, seed=0):
    n = len(bets)
    if n == 0:
        return {'n_bets': 0, 'yield': np.nan, 'yield_ci_low': np.nan, 'yield_ci_high': np.nan, 'p_yield': np.nan,
                'clv': np.nan, 'clv_sharp': np.nan, 'n_sharp': 0, 'avg_price': np.nan, 'hit_rate': np.nan,
                **period_stats(bets)}
    stakes = np.ones(n)
    low, high = bootstrap_yield(bets['profit'], stakes, n_boot=n_boot, seed=seed)
    sharp = bets['clv_sharp'].dropna()
    return {'n_bets': n, 'yield': float(bets['profit'].mean()), 'yield_ci_low': low, 'yield_ci_high': high,
            'p_yield': yield_p_value(bets['profit']), 'clv': float(bets['clv'].mean()),
            'clv_sharp': float(sharp.mean()) if len(sharp) else np.nan, 'n_sharp': len(sharp),
            'avg_price': float(bets['price'].mean()), 'hit_rate': float(bets['won'].mean()),
            **period_stats(bets, n_boot, seed)}


def run_segments(linked, venues, levels, points, models, pm_spread=0.02, pm_fee=POLYMARKET_FEE, min_volume=None,
                 n_boot=2000, seed=0, with_blend=True):
    """
    linked: {venue: Ausgabe von link_predictions}. Gibt (Tabelle pro Segment, alle Walk-forward-Wetten, Blend-Gewichte
    {venue: {point: {model: [w_preis, w_modell, n]}}}) zurück. Segment = Venue x Level x Zeitpunkt x Modell; 'all'
    fasst die Level zusammen.
    """
    min_volume = min_volume or {}
    rows, all_bets, weights = [], [], {}
    for venue in venues:
        df = linked.get(venue)
        if df is None or df.empty:
            continue
        df = df[df['match_id'].notna()].reset_index(drop=True)
        for point in points:
            frame = df
            names = list(models)
            if with_blend:
                frame = df.copy()
                for m in models:
                    frame[f'p_blend_{m}'] = blend_column(frame, point, m, venue)
                    w = blend_weights(frame, point, m, venue)
                    if w:
                        weights.setdefault(venue, {}).setdefault(point, {})[m] = w
                names += [f'blend_{m}' for m in models]
            for model in names:
                cands = candidate_bets(frame, point, venue, model, pm_spread, pm_fee, min_volume.get(venue, 0.0))
                bets = walk_forward_bets(cands)
                all_bets.append(bets)
                level_values = frame['level'].fillna('?')
                for level in list(levels) + ['all']:
                    seg_bets = bets if level == 'all' else bets[bets['level'] == level]
                    seg = frame if level == 'all' else frame[level_values == level]
                    if level != 'all' and seg.empty:
                        continue
                    rows.append({'venue': venue, 'level': level, 'point': point, 'model': model,
                                 'n_markets': int(usable(seg, point, venue).sum()),
                                 **summarize_bets(seg_bets, n_boot, seed),
                                 **log_loss_vs_price(seg, point, model, venue, n_boot=min(n_boot, 1000), seed=seed)})
    table = pd.DataFrame(rows)
    if len(table):
        table = add_decision(table)
    bets = pd.concat(all_bets, ignore_index=True) if all_bets else pd.DataFrame()
    return table, bets, weights


def add_decision(table, alpha=ALPHA, min_bets=MIN_BETS, min_periods=MIN_PERIODS, max_share=MAX_PERIOD_SHARE):
    """
    Holm über alle Einzel-Segmente (ohne 'all') mit >= min_bets Wetten, ohne nicht handelbare Zeitpunkte
    (NOT_TRADABLE). GO: p_holm < alpha, KI-Untergrenze > 0, mittlerer CLV > 0, in >= min_periods Quartalen
    einzeln KI-Untergrenze > 0 (n_periods_pos) und kein Quartal mit mehr als max_share der Wetten. Kandidat:
    Yield > 0 und nominal p < 0,1 bzw. GO-Werte ohne diese Robustheit über die Zeit (Hinweis, keine Entscheidung).
    """
    table = table.copy()
    tradable = ~pd.Series([(v, p) in NOT_TRADABLE for v, p in zip(table['venue'], table['point'])], index=table.index)
    test = (table['level'] != 'all') & (table['n_bets'] >= min_bets) & tradable
    table['p_holm'] = np.nan
    table.loc[test, 'p_holm'] = holm(table.loc[test, 'p_yield'].to_numpy())
    strong = test & (table['p_holm'] < alpha) & (table['yield_ci_low'] > 0) & (table['clv'] > GO_MIN_CLV)
    go = strong & (table['n_periods_pos'] >= min_periods) & (table['max_period_share'] <= max_share)
    candidate = test & ~go & ((table['yield'] > 0) & (table['p_yield'] < 0.1) | strong)
    table['decision'] = np.where(go, 'GO', np.where(candidate, 'Kandidat', np.where(
        test, 'NO-GO', np.where(tradable, 'zu wenig', 'nicht handelbar'))))
    return table


# ---------------------------------------------------------------------- Ausgabe

def markdown(table, cols=None):
    """Einfache Markdown-Tabelle (ohne tabulate)."""
    cols = cols or list(table.columns)

    def fmt(c, v):
        if isinstance(v, float):
            if not np.isfinite(v):
                return '–'
            if c in ('yield', 'yield_ci_low', 'yield_ci_high', 'clv', 'clv_sharp'):
                return f"{v:+.1%}"
            if c in ('hit_rate', 'max_period_share'):
                return f"{v:.0%}"
            if c.startswith('ll'):
                return f"{v:+.4f}" if c in ('ll_diff', 'll_ci_low', 'll_ci_high') else f"{v:.4f}"
            return f"{v:.3f}"
        return str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in table[cols].to_dict('records'):
        lines.append("| " + " | ".join(fmt(c, r[c]) for c in cols) + " |")
    return "\n".join(lines)


SHOW = ['venue', 'level', 'point', 'model', 'n_markets', 'n_bets', 'n_periods', 'n_periods_pos', 'max_period_share',
        'yield', 'yield_ci_low', 'yield_ci_high', 'clv', 'clv_sharp', 'll_diff', 'll_ci_low', 'll_ci_high', 'p_holm', 'decision']


def load_linked(venues, predictions, root=None, log=print):
    from ..odds.history import HISTORY_DIR, read_history
    out = {}
    for venue in venues:
        hist = read_history(venue, root or HISTORY_DIR)
        if hist.empty:
            log(f"{venue}: keine Historie (python -m src.odds.collect --history)")
            continue
        missing = [c for c in columns('24h', venue)[:2] + columns('open', venue)[:2]
                   if c not in hist.columns or hist[c].isna().all()]
        if missing:
            log(f"{venue}: Historie ohne Timing-Spalten ({', '.join(missing)}), bitte mit --refresh neu laden")
        hist = hist[hist['tour'].isin(predictions['tour'].unique())]
        linked = []
        for tour, g in hist.groupby('tour'):
            linked.append(link_predictions(g, predictions[predictions['tour'] == tour]))
        out[venue] = pd.concat(linked, ignore_index=True) if linked else hist.iloc[0:0]
        rate = out[venue]['match_id'].notna().groupby(out[venue]['level'].fillna('?')).agg(['size', 'mean'])
        log(f"{venue}: {len(out[venue])} Märkte, zugeordnet je Level: "
            + ", ".join(f"{k} {int(r['size'])} ({r['mean']:.0%})" for k, r in rate.iterrows()))
    return out


def main(argv=None):
    from ..data_loader import load_matches
    from .backtest import run

    ap = argparse.ArgumentParser(description="Timing-Backtest gegen Polymarket/Kalshi-Preise")
    ap.add_argument('--start', type=int, default=2024)
    ap.add_argument('--end', type=int, default=2026)
    ap.add_argument('--data-start', type=int, default=2015)
    ap.add_argument('--models', default='elo2,gbdt')
    ap.add_argument('--gbdt-refit', default='quarter')
    ap.add_argument('--venues', default='polymarket,kalshi')
    ap.add_argument('--levels', default='main,challenger,itf')
    ap.add_argument('--points', default=','.join(POINTS))
    ap.add_argument('--tours', default='atp,wta')
    ap.add_argument('--pm-spread', type=float, default=0.02, help="Polymarket: angenommener Spread (Kauf = Preis + Hälfte)")
    ap.add_argument('--pm-fee', type=float, default=POLYMARKET_FEE,
                    help="Polymarket: Rate der Taker-Gebühr, auch für Märkte aus der Zeit vor der Gebühr (0 = ohne)")
    ap.add_argument('--min-volume-kalshi', type=float, default=100.0, help="Kalshi: Kontrakte gehandelt bis t")
    ap.add_argument('--min-volume-pm', type=float, default=0.0, help="Polymarket: Gesamtvolumen in $ (Look-ahead!)")
    ap.add_argument('--no-blend', action='store_true')
    ap.add_argument('--bootstrap', type=int, default=2000)
    ap.add_argument('--predictions', default=None, help="vorhandene Vorhersagen (Parquet) statt neuem Walk-forward")
    ap.add_argument('--no-refresh', action='store_true')
    args = ap.parse_args(argv)

    models = [m for m in args.models.split(',') if m]
    if args.predictions:
        predictions = pd.read_parquet(args.predictions)
    else:
        df = load_matches(first_year=args.data_start, last_year=args.end, refresh=not args.no_refresh)
        parts = []
        for tour in args.tours.split(','):
            print(f"[{tour.upper()}] Walk-forward {args.start}-{args.end} ({', '.join(models)}) ...")
            preds, _ = run(df[df['tour'] == tour], models, args.start, args.end, freq='month', warmup=0,
                           gbdt_refit=args.gbdt_refit)
            parts.append(preds)
        predictions = pd.concat(parts, ignore_index=True)
        os.makedirs(OUT_DIR, exist_ok=True)
        predictions.to_parquet(os.path.join(OUT_DIR, "timing_predictions.parquet"), index=False)

    venues = args.venues.split(',')
    linked = load_linked(venues, predictions)
    table, bets, weights = run_segments(linked, venues, args.levels.split(','), args.points.split(','), models,
                               pm_spread=args.pm_spread, pm_fee=args.pm_fee,
                               min_volume={'kalshi': args.min_volume_kalshi, 'polymarket': args.min_volume_pm},
                               n_boot=args.bootstrap, with_blend=not args.no_blend)
    os.makedirs(OUT_DIR, exist_ok=True)
    table.to_parquet(TABLE_PATH, index=False)  # die Segmente mit Kandidat/GO nimmt src/paper (Modell-Strategie)
    bets.to_parquet(os.path.join(OUT_DIR, "timing_bets.parquet"), index=False)
    with open(BLEND_PATH, "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=1)  # für src/paper (Modell-Strategie im Blend mit dem Preis)
    with open(os.path.join(OUT_DIR, "timing.md"), "w", encoding="utf-8") as f:
        f.write(markdown(table, [c for c in SHOW if c in table.columns]) + "\n")
    pd.set_option('display.width', 250)
    print(table[[c for c in SHOW if c in table.columns]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    go = table[table['decision'] == 'GO']
    print(f"\nGO-Segmente (Holm, alpha={ALPHA}): {len(go)}")
    for r in table[table['decision'].isin(['GO', 'Kandidat'])].to_dict('records'):
        print(f"  {r['decision']}: {r['venue']} {r['level']} {r['point']} {r['model']}: {r['n_bets']} Wetten, "
              f"Yield {r['yield']:+.1%} [{r['yield_ci_low']:+.1%}, {r['yield_ci_high']:+.1%}], CLV {r['clv']:+.1%}, "
              f"Quartale mit KI > 0: {r['n_periods_pos']}/{r['n_periods']}, größtes Quartal {r['max_period_share']:.0%}")
    print(f"Gespeichert: {OUT_DIR}/timing.parquet, timing_bets.parquet, timing.md, timing_blend.json")


if __name__ == '__main__':
    main()
