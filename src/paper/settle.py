"""
Offene Paper-Wetten abrechnen und CLV berechnen.

Ergebnisquellen (in dieser Reihenfolge):
1. Polymarket/Kalshi-Auflösung aus data/odds/history (für Wetten bei diesen Venues; gleiche event_id). Eine
   50-50-Auflösung (Absage, W/O) gilt als void.
2. Matchdaten data/matches.csv (Sackmann, tennis-data, ESPN, Märkte) über Namen + Datum (odds/matching.py).
   W/O ist immer void, RET nach der Regel der Venue (venues.Venue.ret_rule: Prediction Markets zahlen den
   Spieler aus, der weiterkommt, deutsche Bücher stornieren laut Annahme in evaluation/sharp.DE_BOOK).

CLV = genommene Quote / Schlussquote - 1, Schlussquote = letzter Snapshot derselben Venue und desselben Events vor
der Startzeit (bei Prediction Markets 1 / Mittelpreis). clv_sharp analog gegen die letzte faire
Pinnacle/Betfair-Wahrscheinlichkeit vor dem Start (sharp.closing_fair).
"""
import numpy as np
import pandas as pd

from ..evaluation import sharp
from ..odds.matching import link_matches

SETTLE_AFTER = pd.Timedelta(hours=3)  # so lange nach der geplanten Startzeit frühestens abrechnen


def _side_won(side, a_won):
    return bool(a_won) if side == 'a' else not bool(a_won)


def market_results(history):
    """{(venue, event_id): 'a' | 'b' | None} aus den Historie-DataFrames ({venue: df})."""
    out = {}
    for venue, df in (history or {}).items():
        for e, w in zip(df['event_id'].astype(str), df['winner']):
            out[(venue, e)] = w if w in ('a', 'b') else None
    return out


def closing_odds(bets, snaps):
    """Schlussquote je Wette (letzter Snapshot derselben Venue/des Events vor dem Start) oder NaN."""
    if snaps is None or snaps.empty:
        return np.full(len(bets), np.nan)
    snaps = snaps.sort_values('ts_utc')
    out = []
    for r in bets.to_dict('records'):
        s = snaps[(snaps['venue'] == r['venue']) & (snaps['event_id'].astype(str) == str(r['event_id']))]
        start = r['start_time_utc']
        if start is not None and start == start:
            s = s[s['ts_utc'] < start]
        if s.empty:
            out.append(np.nan)
            continue
        last = s.iloc[-1]
        if last['price_type'] == 'probability':
            price = last['price_a'] if r['side'] == 'a' else last['price_b']
            if not (price == price) and r['side'] == 'b' and last['price_a'] == last['price_a']:
                price = 1 - last['price_a']
            out.append(1 / price if price and price == price and price > 0 else np.nan)
        else:
            out.append(last[f"odds_{r['side']}"])
    return np.asarray(out, float)


def sharp_close(bets, snaps):
    """Faire Sharp-Schlusswahrscheinlichkeit für die gewettete Seite (oder NaN)."""
    if snaps is None or snaps.empty or not snaps['venue'].isin(sharp.SHARP).any():
        return np.full(len(bets), np.nan)
    linked = sharp._oriented(sharp.link_events(snaps))
    keys = linked.drop_duplicates(['venue', 'event_id']).set_index(['venue', 'event_id'])[['match_key', 'flip']]
    close = sharp.closing_fair(snaps)
    out = []
    for r in bets.to_dict('records'):
        k = (r['venue'], r['event_id'])
        if k not in keys.index or keys.loc[k, 'match_key'] not in close.index:
            out.append(np.nan)
            continue
        p_a = close[keys.loc[k, 'match_key']]
        side_a = (r['side'] == 'a') != bool(keys.loc[k, 'flip'])  # Seite aus Sicht des Sharp-Events
        out.append(p_a if side_a else 1 - p_a)
    return np.asarray(out, float)


def due(bets, now, settle_after=SETTLE_AFTER):
    """Offene Wetten, deren geplanter Start mindestens settle_after zurückliegt (bool-Series)."""
    start = pd.to_datetime(bets['start_time_utc'], utc=True)
    return (bets['status'] == 'open') & start.notna() & (start <= now - settle_after)


def settle(bets, now, history=None, matches=None, snaps=None, settle_after=SETTLE_AFTER):
    """
    Offene Wetten abrechnen, deren Start mindestens settle_after zurückliegt. history: {venue: Historie-DataFrame},
    matches: Liste von Match-Dicts (wie data/matches.csv), snaps: Snapshots für die CLV.
    Gibt (alle Wetten, Anzahl neu abgerechnet) zurück.
    """
    bets = bets.copy()
    open_due = due(bets, now, settle_after)
    if not open_due.any():
        return bets, 0
    todo = bets[open_due]
    results = market_results(history)
    linked = link_matches(todo[['player_a', 'player_b', 'start_time_utc']].assign(ts_utc=todo['created_utc']),
                          matches or []) if matches else None
    n = 0
    for k, (i, r) in enumerate(todo.iterrows()):
        prof = sharp.profile(r['venue'])
        status, source = None, None
        key = (r['venue'], str(r['event_id']))
        if key in results:
            winner = results[key]
            status = 'void' if winner is None else ('won' if winner == r['side'] else 'lost')
            source = r['venue']
        elif linked is not None and bool(linked['match_found'].iloc[k]):
            m_status = linked['match_status'].iloc[k] or 'completed'
            if prof.is_void([m_status])[0]:
                status = 'void'
            else:
                status = 'won' if _side_won(r['side'], linked['a_won'].iloc[k]) else 'lost'
            source = f"matches ({m_status})"
        if status is None:
            continue
        stake, flat = float(r['stake'] or 0.0), float(r['stake_flat'] or 0.0)
        mult = (r['net_odds'] - 1) if status == 'won' else (-1.0 if status == 'lost' else 0.0)
        bets.loc[i, ['status', 'result_source']] = [status, source]
        bets.loc[i, 'settled_utc'] = now
        bets.loc[i, 'profit'] = stake * mult
        bets.loc[i, 'profit_flat'] = flat * mult
        n += 1
    done = open_due & (bets['status'] != 'open')
    if done.any() and snaps is not None:
        close = closing_odds(bets[done], snaps)
        bets.loc[done, 'close_odds'] = close
        bets.loc[done, 'clv'] = bets.loc[done, 'odds'].to_numpy(float) / close - 1
        p_sharp = sharp_close(bets[done], snaps)
        bets.loc[done, 'clv_sharp'] = p_sharp * bets.loc[done, 'odds'].to_numpy(float) - 1
    return bets, n
