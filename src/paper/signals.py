"""
Neue Paper-Signale aus den aktuellen Snapshots.

- model_signals (Strategie aus dem Timing-Backtest): eigenes Modell (elo2/gbdt) gegen den aktuellen Kaufpreis bei
  Polymarket/Kalshi (Ask, sonst Preis + halber Spread), Gebühren laut venues.py. Liegen Blend-Gewichte aus dem
  Timing-Backtest vor (data/eval/timing_blend.json), ist p der Blend aus aktuellem Preis und Modell (Gewichte des
  Zeitpunkts, der der Zeit bis zum Start am nächsten liegt). Das rohe Modell weicht sonst bei sehr vielen Märkten
  weit vom Preis ab, meist weil es schlechter ist als der Markt.
- sharp_signals (Strategie A): faire Pinnacle/Betfair-Wahrscheinlichkeit gegen den besten Netto-Preis aller
  anderen Venues (evaluation/sharp.py).

Einsatz: Viertel-Kelly (Deckel 2 % der Bankroll der Strategie) mit p Richtung Markt geschrumpft, dazu flat.
"""
import json
import os

import numpy as np
import pandas as pd

from ..evaluation import sharp
from ..evaluation.betting import Kelly
from ..evaluation.predictors import logit, sigmoid
from .store import bet_id, normalize

MODEL_THRESHOLD = 0.05
MAX_AGE = pd.Timedelta(minutes=90)  # nur Snapshots, die höchstens so alt sind, gelten als „aktueller Preis“
HORIZON = pd.Timedelta(hours=48)  # nur Matches, die innerhalb dieser Zeit beginnen
FLAT_STAKE = 10.0
KELLY = Kelly(0.25, 0.02)


BLEND_HOURS = {'24h': 24.0, '6h': 6.0, '1h': 1.0, 'close': 0.0}


def load_blend(path=None):
    """Blend-Gewichte aus dem Timing-Backtest ({venue: {point: {model: [w_preis, w_modell, n]}}}) oder {}."""
    from ..evaluation.timing import BLEND_PATH
    path = path or BLEND_PATH
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def blended(p, q, hours, weights, venue, model):
    """p im Blend mit dem Preis q (A) -> (p, verwendeter Zeitpunkt oder None, wenn keine Gewichte da sind)."""
    by_point = (weights or {}).get(venue) or {}
    points = [k for k in BLEND_HOURS if model in (by_point.get(k) or {})]
    if not points or q is None or not (0 < q < 1):
        return p, None
    point = min(points, key=lambda k: abs(BLEND_HOURS[k] - hours))
    w_price, w_model = by_point[point][model][:2]
    return float(sigmoid(w_price * logit(q) + w_model * logit(p))), point


def latest(snaps, now, max_age=MAX_AGE):
    """Je (venue, event_id) der neueste Snapshot, höchstens max_age alt."""
    recent = snaps[(snaps['ts_utc'] <= now) & (snaps['ts_utc'] >= now - max_age)]
    return recent.sort_values('ts_utc').drop_duplicates(['venue', 'event_id'], keep='last').reset_index(drop=True)


def upcoming(df, now, horizon=HORIZON):
    start = pd.to_datetime(df['start_time_utc'], utc=True)
    return df[start.notna() & (start > now) & (start <= now + horizon)]


def _stakes(p, odds, net, bankroll, shrink=sharp.SHRINK):
    p_k = sharp.shrink_to_market(p, 1 / odds, shrink)
    return KELLY.amount(bankroll, p_k, net)


def model_signals(snaps, prob, now, bankroll=1000.0, threshold=MODEL_THRESHOLD, strategy='modell',
                  pm_spread=sharp.PM_SPREAD, log=print, blend=None, model='elo2'):
    """
    snaps: Snapshots (schema.load_snapshots). prob(row) -> P(player_a gewinnt) oder None (Spieler unbekannt).
    blend: Gewichte aus load_blend() (leer: rohes Modell). Liefert neue Wetten (DataFrame im Schema von store.COLUMNS).
    """
    cur = upcoming(latest(snaps[snaps['venue'].isin(sharp.PREDICTION_MARKETS)], now), now)
    if cur.empty:
        return normalize(pd.DataFrame())
    gross_a, gross_b = sharp.buy_odds(cur, pm_spread)
    rows, unknown, n_blend = [], 0, 0
    for i, r in enumerate(cur.to_dict('records')):
        p = prob(r)
        if p is None or not np.isfinite(p):
            unknown += 1
            continue
        hours = (pd.Timestamp(r['start_time_utc']) - now).total_seconds() / 3600
        q = r['price_a'] if r['price_type'] == 'probability' and r['price_a'] == r['price_a'] else None
        p, used = blended(p, q, hours, blend, r['venue'], model)
        n_blend += used is not None
        prof = sharp.profile(r['venue'])
        best = None
        for side, odds, p_side in (('a', gross_a[i], p), ('b', gross_b[i], 1 - p)):
            if not np.isfinite(odds) or odds <= 1:
                continue
            net = float(prof.net_odds_from_gross(odds))
            edge = p_side * net - 1
            if best is None or edge > best[3]:
                best = (side, odds, net, edge, p_side)
        if best is None or best[3] <= threshold:
            continue
        side, odds, net, edge, p_side = best
        implied = r['price_a'] if side == 'a' else r['price_b']
        rows.append({
            'bet_id': bet_id(strategy, r['venue'], r['event_id'], side), 'created_utc': now, 'strategy': strategy,
            'venue': r['venue'], 'event_id': r['event_id'], 'tour': r.get('tour'), 'level': r.get('level'),
            'tourney': r.get('tourney'), 'player_a': r['player_a'], 'player_b': r['player_b'], 'side': side,
            'player': r[f'player_{side}'], 'start_time_utc': r['start_time_utc'], 'odds': odds, 'net_odds': net,
            'p': p_side, 'p_market': implied if implied == implied else 1 / odds, 'edge': edge,
            'stake': _stakes(p_side, odds, net, bankroll), 'stake_flat': FLAT_STAKE, 'status': 'open',
        })
    if unknown:
        log(f"  Modell: {unknown} von {len(cur)} Märkten ohne Vorhersage (Spieler nicht in den Matchdaten)")
    log(f"  Modell: {n_blend} von {len(cur) - unknown} Märkten im Blend mit dem Preis"
        + ("" if blend else " (keine Gewichte: erst python -m src.evaluation.timing)"))
    return normalize(pd.DataFrame(rows))


def sharp_signals(snaps, now, bankroll=1000.0, threshold=sharp.DEFAULT_THRESHOLD, strategy='sharp'):
    """Strategie A auf den Snapshots der letzten MAX_AGE: erstes Signal je Match und Seite, nur vor dem Start."""
    recent = snaps[(snaps['ts_utc'] <= now) & (snaps['ts_utc'] >= now - MAX_AGE)]
    sig = sharp.signals(recent, threshold)
    if sig.empty:
        return normalize(pd.DataFrame())
    sig = upcoming(sharp.first_signals(sig), now)
    if sig.empty:
        return normalize(pd.DataFrame())
    sig = sharp.stake_sizes(sig, bankroll, KELLY, flat=FLAT_STAKE)
    rows = []
    for r in sig.to_dict('records'):
        flip = bool(r['venue_flip'])
        side = ('b' if r['side'] == 'a' else 'a') if flip else r['side']  # Seite aus Sicht des Venue-Events
        rows.append({
            'bet_id': bet_id(strategy, r['venue'], r['event_id'], side), 'created_utc': now, 'strategy': strategy,
            'venue': r['venue'], 'event_id': r['event_id'], 'tour': r.get('tour'), 'level': r.get('level'),
            'tourney': r.get('tourney'), 'player_a': r['venue_player_a'], 'player_b': r['venue_player_b'],
            'side': side, 'player': r[f'venue_player_{side}'], 'start_time_utc': r['start_time_utc'],
            'odds': r['odds'], 'net_odds': r['net_odds'], 'p': r['p_fair'], 'p_market': 1 / r['odds'],
            'edge': r['edge'], 'stake': r['stake_kelly'], 'stake_flat': r['stake_flat'], 'status': 'open',
        })
    return normalize(pd.DataFrame(rows))
