"""
Historische Preise abgerechneter Tennis-Match-Märkte von Polymarket und Kalshi.

Pro Match eine Zeile mit Eröffnung (erster Preis) und Closing (letzter Preis vor dem geplanten
Matchbeginn) aus Sicht von Spieler A, dazu das Ergebnis aus der Markt-Auflösung.
Gespeichert unter data/odds/history/{venue}.parquet. Ein zweiter Lauf lädt nur neue Märkte.

Grenzen (ehrlich):
- „Matchbeginn“ ist die geplante Startzeit der Venue. Tennis beginnt oft später (Order of Play
  „not before“), das Closing ist dann etwas älter als der echte Schlusskurs. Liegt die Startzeit
  zu spät, kann schon Live-Handel drinstecken; report.py weist extreme Closing-Preise deshalb aus.
- Polymarket liefert historisch nur Preise (Mitte bzw. letzter Kurs) in 10-Minuten-Auflösung,
  keine Bid/Ask. Kalshi liefert stündliche Candles mit Bid/Ask.
- Bei Kalshi wird nur der Markt von Spieler A geladen (spart die Hälfte der Anfragen).
"""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

from . import kalshi, polymarket
from .http import JsonClient
from .schema import HISTORY_DIR, mid, now_utc, utc

COLUMNS = ['venue', 'event_id', 'market_id', 'tour', 'level', 'tourney', 'start_time_utc', 'player_a', 'player_b',
           'open_ts', 'open_price_a', 'close_ts', 'close_price_a', 'close_bid_a', 'close_ask_a', 'close_last_a',
           'close_1h_price_a', 'n_points', 'volume', 'winner', 'source_url', 'loaded_utc']
POLY_FIDELITY = 10  # Minuten
MAX_WINDOW = timedelta(days=14)  # so weit vor dem Start wird höchstens geladen
DEFAULT_LEVELS = ('main', 'challenger')
SETTLE_DELAY = timedelta(hours=6)  # jüngere Matches sind oft noch nicht abgerechnet
# Zusätzlich der letzte Preis mindestens 1 h vor der geplanten Startzeit: robust gegen Matches, die früher
# beginnen, und gegen Kalshis Stunden-Candles, deren letzte Periode bis genau zur Startzeit reicht
SAFE_BUFFER = timedelta(hours=1)


def history_path(venue, root=HISTORY_DIR):
    return os.path.join(root, f"{venue}.parquet")


# ------------------------------------------------------- Closing-Ableitung

def opening_closing(points, start, buffer=SAFE_BUFFER):
    """
    points: [(datetime, preis, bid, ask), ...] in beliebiger Reihenfolge, preis/bid/ask dürfen None sein.
    Nur Punkte mit ts <= start zählen. Liefert dict mit open_ts/open_price_a/close_ts/close_price_a/
    close_bid_a/close_ask_a/close_1h_price_a (letzter Preis bis start - buffer)/n_points oder None, wenn es
    vor dem Start keinen Preis gibt.
    """
    pre = sorted((p for p in points if p[0] <= start and p[1] is not None), key=lambda p: p[0])
    if not pre:
        return None
    first, last = pre[0], pre[-1]
    safe = [p for p in pre if p[0] <= start - buffer]
    return {'open_ts': first[0], 'open_price_a': first[1], 'close_ts': last[0], 'close_price_a': last[1],
            'close_bid_a': last[2], 'close_ask_a': last[3], 'close_1h_price_a': safe[-1][1] if safe else None,
            'n_points': len(pre)}


def polymarket_points(history):
    """CLOB prices-history -> Punkte (ts, preis, None, None)."""
    return [(utc(int(h['t'])), float(h['p']), None, None) for h in history.get('history') or []]


def _candle_value(part, key):
    """Candle-Feld in Dollar: live '<key>_dollars', historisch '<key>' (beides Strings oder None)."""
    if not part:
        return None
    value = part.get(f'{key}_dollars', part.get(key))
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def kalshi_points(candles):
    """Candlesticks -> Punkte (Ende der Periode, Mitte aus Bid/Ask oder letzter Kurs, Bid, Ask)."""
    out = []
    for c in candles.get('candlesticks') or []:
        bid, ask = _candle_value(c.get('yes_bid'), 'close'), _candle_value(c.get('yes_ask'), 'close')
        last = _candle_value(c.get('price'), 'close')
        price = mid(bid, ask) or last
        out.append((utc(int(c['end_period_ts'])), price, bid if bid else None, ask if ask is not None and ask < 1 else None, last))
    return out


# ---------------------------------------------------------------- Polymarket

def polymarket_events(client, log=print):
    """
    Alle geschlossenen Sieger-Märkte mit Tag 'tennis' (Keyset-Paginierung über /markets, das ist
    viel schlanker als /events mit allen Zusatzmärkten). Jeder Markt wird als Event mit genau
    diesem Markt zurückgegeben, damit polymarket.match_info ihn wie ein Event behandeln kann.
    Märkte ohne sportsMarketType (ein paar aus 2024) fehlen dabei.
    """
    params = {'tag_id': polymarket.TENNIS_TAG, 'closed': 'true', 'limit': 100, 'sports_market_types': 'moneyline'}
    events = []
    while True:
        page = client.get_json(f"{polymarket.GAMMA}/markets/keyset", params)
        for m in page.get('markets') or []:
            event = dict((m.get('events') or [{}])[0])
            events.append({**event, 'markets': [m]})
        cursor = page.get('next_cursor')
        if not cursor or not page.get('markets'):
            break
        params['after_cursor'] = cursor
        if len(events) % 5000 == 0:
            log(f"    {len(events)} Märkte ...")
    return events


def polymarket_row(info, history, loaded):
    start = info['start_time_utc']
    oc = opening_closing(polymarket_points(history), start) if start else None
    if oc is None:
        return None
    return {'venue': polymarket.VENUE, 'event_id': info['event_id'], 'market_id': info['token_a'],
            'tour': info['tour'], 'level': None, 'tourney': info['tourney'], 'start_time_utc': start,
            'player_a': info['player_a'], 'player_b': info['player_b'], **oc, 'close_last_a': oc['close_price_a'],
            'volume': info['volume'], 'winner': info['winner'], 'source_url': info['source_url'], 'loaded_utc': loaded}


def load_polymarket(client, known=(), since=None, limit=None, workers=6, log=print):
    log("Polymarket: geschlossene Tennis-Events laden ...")
    events = polymarket_events(client, log)
    infos = [i for i in map(polymarket.match_info, events) if i and i['closed'] and i['start_time_utc']]
    if since:
        infos = [i for i in infos if i['start_time_utc'] >= since]
    log(f"  {len(events)} Sieger-Märkte, davon {len(infos)} Einzel-Matches "
        f"({sum(i['winner'] is not None for i in infos)} mit Sieger)")
    todo = [i for i in infos if i['event_id'] not in known][:limit]
    log(f"  {len(todo)} neu zu laden")
    loaded = now_utc()

    def one(info):
        start = info['start_time_utc']
        lo = max(info['created_utc'] or start - MAX_WINDOW, start - MAX_WINDOW) - timedelta(hours=1)
        history = client.get_json(f"{polymarket.CLOB}/prices-history", {
            'market': info['token_a'], 'startTs': int(lo.timestamp()),
            'endTs': int((start + timedelta(hours=1)).timestamp()), 'fidelity': POLY_FIDELITY})
        return polymarket_row(info, history, loaded)

    return _run(one, todo, workers, log)


# -------------------------------------------------------------------- Kalshi

def kalshi_markets(client, series, log=print):
    """Alle abgerechneten Märkte einer Serie (aktuelle API + Archiv), je Markt mit Flag 'historical'."""
    out = {}
    sources = [("/historical/markets", {'series_ticker': series, 'limit': 1000}, True),
               ("/markets", {'series_ticker': series, 'status': 'settled', 'limit': 1000}, False)]
    for path, params, hist in sources:
        for m in kalshi.paginate(client, path, 'markets', params):
            out.setdefault(m['ticker'], {**m, 'historical': hist})
    return out.values()


def kalshi_row(event, milestone, market_a, market_b, candles, loaded):
    start = utc(milestone['start_date'])
    points = kalshi_points(candles)
    oc = opening_closing([p[:4] for p in points], start)
    if oc is None:
        return None
    last = next((p[4] for p in sorted(points, key=lambda p: p[0], reverse=True) if p[0] <= start and p[4] is not None), None)
    tour, level = kalshi.tour_level(milestone)
    volume = [v for v in (kalshi._fp(market_a, 'volume'), kalshi._fp(market_b, 'volume')) if v is not None]
    return {'venue': kalshi.VENUE, 'event_id': event, 'market_id': market_a['ticker'], 'tour': tour, 'level': level,
            'tourney': (milestone.get('details') or {}).get('tournament_name'), 'start_time_utc': start,
            'player_a': market_a['yes_sub_title'], 'player_b': market_b['yes_sub_title'], **oc, 'close_last_a': last,
            'volume': sum(volume) if volume else None, 'winner': kalshi.winner_of(market_a),
            'source_url': f"{kalshi.API}/events/{event}", 'loaded_utc': loaded}


def load_kalshi(client, known=(), since=None, limit=None, levels=DEFAULT_LEVELS, workers=6, log=print):
    since = since or datetime(2024, 1, 1, tzinfo=timezone.utc)
    log("Kalshi: Milestones laden ...")
    cutoff = now_utc() - SETTLE_DELAY
    stones = [m for m in kalshi.milestones(client, since, cutoff) if kalshi.tour_level(m)[1] in levels]
    by_event = {kalshi.event_ticker(m): m for m in stones if kalshi.event_ticker(m)}
    log(f"  {len(by_event)} Einzel-Matches ({', '.join(levels)}) seit {since:%Y-%m-%d}")
    series = sorted({e.split('-')[0] for e in by_event})
    markets = {}
    for s in series:
        for m in kalshi_markets(client, s, log):
            if m['event_ticker'] in by_event:
                markets.setdefault(m['event_ticker'], []).append(m)
    log(f"  Märkte gefunden für {len(markets)} Matches in {len(series)} Serien")
    todo = [e for e in markets if e not in known and len(markets[e]) == 2][:limit]
    log(f"  {len(todo)} neu zu laden")
    loaded = now_utc()

    def one(event):
        milestone = by_event[event]
        pair = kalshi.order_markets(markets[event], milestone)
        if len(markets[event]) != 2 or len(pair) != 2:  # Turniersieger-Events haben viele Märkte
            return None
        a, b = pair
        start = utc(milestone['start_date'])
        lo = max(utc(a.get('open_time')) or start - MAX_WINDOW, start - MAX_WINDOW)
        params = {'start_ts': int(lo.timestamp()), 'end_ts': int(start.timestamp()), 'period_interval': 60}
        if a['historical']:
            url = f"{kalshi.API}/historical/markets/{a['ticker']}/candlesticks"
        else:
            url = f"{kalshi.API}/series/{event.split('-')[0]}/markets/{a['ticker']}/candlesticks"
        return kalshi_row(event, milestone, a, b, client.get_json(url, params), loaded)

    return _run(one, todo, workers, log)


# --------------------------------------------------------------- Ablauf

def _run(fn, items, workers, log):
    rows, errors = [], 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fn, x) for x in items]
        for i, f in enumerate(as_completed(futures), 1):
            try:
                row = f.result()
            except Exception as e:
                errors += 1
                if errors <= 5:
                    log(f"    Fehler: {e}")
                continue
            if row:
                rows.append(row)
            if i % 1000 == 0:
                log(f"    {i}/{len(items)} geladen, {len(rows)} mit Preis vor Start")
    if errors:
        log(f"  {errors} Märkte mit Fehler übersprungen")
    return rows


def read_history(venue, root=HISTORY_DIR):
    import pandas as pd
    path = history_path(venue, root)
    if not os.path.exists(path):
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_parquet(path)


def save_history(venue, rows, root=HISTORY_DIR):
    """Neue Zeilen mit vorhandenen zusammenführen (gleiche event_id: neue gewinnt)."""
    import pandas as pd
    new = pd.DataFrame(rows, columns=COLUMNS)
    old = read_history(venue, root)
    df = pd.concat([old, new], ignore_index=True) if len(old) else new
    df = df.drop_duplicates('event_id', keep='last')
    for c in ('start_time_utc', 'open_ts', 'close_ts', 'loaded_utc'):
        df[c] = pd.to_datetime(df[c], utc=True)
    for c in ('open_price_a', 'close_price_a', 'close_bid_a', 'close_ask_a', 'close_last_a', 'close_1h_price_a', 'volume'):
        df[c] = pd.to_numeric(df[c], errors='coerce').astype(float)
    df['n_points'] = pd.to_numeric(df['n_points'], errors='coerce').fillna(0).astype(int)
    os.makedirs(root, exist_ok=True)
    tmp = history_path(venue, root) + ".tmp"
    df.sort_values('start_time_utc').to_parquet(tmp, index=False)
    os.replace(tmp, history_path(venue, root))
    return df


LOADERS = {'polymarket': load_polymarket, 'kalshi': load_kalshi}
RATE = {'polymarket': 8, 'kalshi': 4}  # Anfragen pro Sekunde (Kalshi antwortet sonst mit 429)


def update(venues=('polymarket', 'kalshi'), since=None, limit=None, levels=DEFAULT_LEVELS, refresh=False,
           root=HISTORY_DIR, log=print):
    """Historie je Venue nachladen. Fehler einer Venue stoppen die andere nicht."""
    result = {}
    for venue in venues:
        client = JsonClient(max_per_sec=RATE[venue])
        try:
            known = set() if refresh else set(read_history(venue, root)['event_id'])
            kwargs = {'levels': levels} if venue == 'kalshi' else {}
            rows = LOADERS[venue](client, known=known, since=since, limit=limit, log=log, **kwargs)
            df = save_history(venue, rows, root)
            log(f"  {venue}: {len(rows)} neue Matches, insgesamt {len(df)} in {history_path(venue, root)} "
                f"({sum(client.calls.values())} Anfragen)")
            result[venue] = df
        except Exception as e:
            log(f"  {venue}: Historie fehlgeschlagen: {type(e).__name__}: {e}")
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description="Historische Polymarket-/Kalshi-Preise für Tennis-Matches laden")
    p.add_argument('--venues', default='polymarket,kalshi')
    p.add_argument('--since', help="nur Matches ab Datum (YYYY-MM-DD)")
    p.add_argument('--limit', type=int, help="höchstens so viele neue Märkte pro Venue (zum Testen)")
    p.add_argument('--levels', default=','.join(DEFAULT_LEVELS), help="Kalshi: main,challenger,itf")
    p.add_argument('--refresh', action='store_true', help="auch schon geladene Märkte neu laden")
    args = p.parse_args(argv)
    since = utc(args.since + "T00:00:00Z") if args.since else None
    update(args.venues.split(','), since, args.limit, tuple(args.levels.split(',')), args.refresh)


if __name__ == '__main__':
    main()
