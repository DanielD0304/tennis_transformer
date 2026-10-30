"""
Neue Paper-Signale aus den aktuellen Snapshots.

- model_signals (Strategie aus dem Timing-Backtest): eigenes Modell (elo2/gbdt) gegen den aktuellen Kaufpreis bei
  Polymarket/Kalshi (Ask, sonst Preis + halber Spread), Gebühren laut venues.py. Liegen Blend-Gewichte aus dem
  Timing-Backtest vor (data/eval/timing_blend.json), ist p der Blend aus aktuellem Preis und Modell (Gewichte des
  Zeitpunkts, der der Zeit bis zum Start am nächsten liegt). Das rohe Modell weicht sonst bei sehr vielen Märkten
  weit vom Preis ab, meist weil es schlechter ist als der Markt.
  Gewettet wird nur in Segmenten (Venue x Level x Stunden bis zum Start), standardmäßig in denen, die der
  Timing-Backtest als Kandidat oder GO ausweist (load_segments). So prüft das Paper-Trading genau diese Hypothesen.
- sharp_signals (Strategie A): faire Pinnacle/Betfair-Wahrscheinlichkeit gegen den besten Netto-Preis aller
  anderen Venues (evaluation/sharp.py).

Einsatz: Viertel-Kelly (Deckel 2 % der Bankroll der Strategie) mit p Richtung Markt geschrumpft, dazu flat.
"""
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..evaluation import sharp
from ..evaluation.betting import Kelly
from ..evaluation.predictors import logit, sigmoid
from .store import bet_id, normalize

MODEL_THRESHOLD = 0.05
MAX_AGE = pd.Timedelta(minutes=90)  # nur Snapshots, die höchstens so alt sind, gelten als „aktueller Preis“
HORIZON = pd.Timedelta(hours=48)  # ohne Segmente: nur Matches, die innerhalb dieser Zeit beginnen
FLAT_STAKE = 10.0
KELLY = Kelly(0.25, 0.02)


BLEND_HOURS = {'24h': 24.0, '6h': 6.0, '1h': 1.0, 'close': 0.0}
# Zeitpunkt des Timing-Backtests -> Fenster der Live-Wette in Stunden bis zum Start (von < h <= bis): ±25 % um den
# Zeitpunkt, bei -1 h eine Stunde breit, damit der Lauf alle 30 Minuten es sicher trifft, 'close' die letzte halbe
# Stunde. Die Eröffnung hat keinen festen Zeitpunkt (und bei Polymarket keinen handelbaren Preis).
POINT_WINDOWS = {'24h': (18.0, 30.0), '6h': (4.5, 7.5), '1h': (0.5, 1.5), 'close': (0.0, 0.5)}
SEGMENT_DECISIONS = ('Kandidat', 'GO')
# Ohne data/eval/timing.parquet: die Kandidaten aus docs/ERGEBNISSE_PHASE45.md (Stand 25.10.2026) als (venue, level,
# Zeitpunkt, Modell)
FALLBACK_SEGMENTS = [('polymarket', 'challenger', '24h', m) for m in ('elo2', 'gbdt', 'blend_elo2', 'blend_gbdt')] \
    + [('polymarket', 'itf', point, 'blend_gbdt') for point in ('6h', '1h', 'close')]


@dataclass(frozen=True)
class Segment:
    """Venue x Level x Fenster der Stunden bis zum Start (min_hours < Stunden <= max_hours)."""
    venue: str
    level: str
    min_hours: float
    max_hours: float

    @property
    def name(self):
        return f"{self.venue} {self.level} {self.min_hours:g}-{self.max_hours:g}h"

    def contains(self, venue, level, hours):
        return venue == self.venue and level == self.level and self.min_hours < hours <= self.max_hours


def load_segments(path=None, model=None, log=print):
    """
    Segmente aus dem Timing-Backtest (data/eval/timing.parquet): Venue x Level x Zeitpunkt mit decision 'Kandidat'
    oder 'GO', als Fenster aus POINT_WINDOWS. model ('blend_gbdt', 'elo2', ...): nur die Segmente dieses Modells, die
    Hypothese gilt für das Modell, mit dem der Backtest sie gefunden hat. Ohne Datei: FALLBACK_SEGMENTS.
    """
    from ..evaluation import timing
    path = path or timing.TABLE_PATH
    columns = ['venue', 'level', 'point', 'model']
    if os.path.exists(path):
        table = pd.read_parquet(path, columns=columns + ['decision'])
    else:
        log(f"  Modell: {path} fehlt, Segmente aus docs/ERGEBNISSE_PHASE45.md")
        table = pd.DataFrame(FALLBACK_SEGMENTS, columns=columns).assign(decision='Kandidat')
    keep = table['decision'].isin(SEGMENT_DECISIONS) & (table['level'] != 'all') \
        & table['point'].isin(list(POINT_WINDOWS))
    if model is not None:
        keep &= table['model'] == model
    found = set(zip(table.loc[keep, 'venue'], table.loc[keep, 'level'], table.loc[keep, 'point']))
    segments = [Segment(venue, level, *POINT_WINDOWS[point]) for venue, level, point in found]
    return sorted(segments, key=lambda s: (s.venue, s.level, -s.max_hours))


def parse_segments(text):
    """'polymarket:challenger:24h,kalshi:itf:0.5-2' -> [Segment, ...]. Fenster: Zeitpunkt aus POINT_WINDOWS oder
    'von-bis' in Stunden bis zum Start."""
    out = []
    for part in (p.strip() for p in str(text).split(',')):
        if not part:
            continue
        try:
            venue, level, window = part.split(':')
            lo, hi = POINT_WINDOWS[window] if window in POINT_WINDOWS else (float(x) for x in window.split('-'))
        except ValueError:
            raise ValueError(f"Segment {part!r}: erwartet venue:level:fenster, z.B. polymarket:itf:1h oder "
                             f"polymarket:itf:0.5-2") from None
        if not 0 <= lo < hi:
            raise ValueError(f"Segment {part!r}: das Fenster braucht 0 <= von < bis")
        out.append(Segment(venue, level, lo, hi))
    return out


def segments_from_arg(text, model=None, log=print):
    """--segments: 'auto' (Kandidaten/GO des Timing-Backtests für model), 'alle' (None: keine Einschränkung) oder eine
    Liste für parse_segments."""
    if text in (None, '', 'auto'):
        return load_segments(model=model, log=log)
    if text == 'alle':
        return None
    return parse_segments(text)


def in_segments(df, now, segments):
    """Nur Märkte in einem der Segmente, dazu Spalte 'segment' (Name des ersten passenden Segments)."""
    hours = (pd.to_datetime(df['start_time_utc'], utc=True) - now).dt.total_seconds() / 3600
    names = [next((s.name for s in segments if s.contains(venue, level, h)), None)
             for venue, level, h in zip(df['venue'], df['level'], hours)]
    out = df.assign(segment=names)
    return out[out['segment'].notna()].reset_index(drop=True)


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
                  pm_spread=sharp.PM_SPREAD, log=print, blend=None, model='elo2', segments=None, levels=None):
    """
    snaps: Snapshots (schema.load_snapshots). prob(row) -> P(player_a gewinnt) oder None (Spieler unbekannt).
    blend: Gewichte aus load_blend() (leer: rohes Modell). segments: Liste von Segment, gewettet wird nur in diesen
    (None: alle Märkte, die innerhalb von HORIZON beginnen). levels(df) -> df ergänzt vorher fehlende Level
    (run.fill_levels), Polymarket kennzeichnet nur ITF. Liefert neue Wetten (DataFrame im Schema von store.COLUMNS).
    """
    horizon = HORIZON if segments is None else pd.Timedelta(hours=max((s.max_hours for s in segments), default=0))
    cur = upcoming(latest(snaps[snaps['venue'].isin(sharp.PREDICTION_MARKETS)], now), now, horizon)
    if levels is not None and not cur.empty:
        cur = levels(cur.copy())
    if segments is not None and not cur.empty:
        n_all, cur = len(cur), in_segments(cur, now, segments)
        log(f"  Modell: {len(cur)} von {n_all} Märkten in den Segmenten")
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
        segment = r.get('segment')
        rows.append({
            'bet_id': bet_id(strategy, r['venue'], r['event_id'], side, segment), 'created_utc': now,
            'strategy': strategy, 'segment': segment, 'venue': r['venue'], 'event_id': r['event_id'],
            'tour': r.get('tour'), 'level': r.get('level'), 'tourney': r.get('tourney'), 'player_a': r['player_a'],
            'player_b': r['player_b'], 'side': side, 'player': r[f'player_{side}'],
            'start_time_utc': r['start_time_utc'], 'odds': odds, 'net_odds': net, 'p': p_side,
            'p_market': implied if implied == implied else 1 / odds, 'edge': edge,
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
