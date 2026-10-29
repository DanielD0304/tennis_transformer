"""
Polymarket: Tennis-Einzelmatch-Märkte (Sieger) über die Gamma-API, Preise aus dem CLOB.

Seit Okt. 2025 liegen die Matches in den Serien 'atp' (10365) und 'wta' (10366), ältere nur unter
dem Tag 'tennis' (864). Ein Match-Markt hat sportsMarketType 'moneyline' und zwei Spieler als
Outcomes. Alte Märkte haben nur Nachnamen als Outcomes, die vollen Namen stehen im Titel
('Turnier: Vorname Nachname vs Vorname Nachname').
Turnier-Futures ('US Open Winner'), Doppel ('A / B vs C / D') und Zusatzmärkte (Sätze, Games)
werden ausgelassen.
"""
import json
import re

from ..names import candidate_keys
from .http import JsonClient
from .schema import Snapshot, mid, now_utc, utc

VENUE = 'polymarket'
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
SERIES = {'atp': '10365', 'wta': '10366'}
TENNIS_TAG = '864'
TAG_TOURS = {'101232': 'atp', '102123': 'wta'}
EVENT_URL = "https://polymarket.com/event/{}"
MAX_OFFSET = 2000  # die Gamma-API erlaubt keine größeren Offsets (danach /events/keyset)


def _list(value):
    if isinstance(value, list):
        return value
    try:
        return json.loads(value or "[]")
    except (TypeError, ValueError):
        return []


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _title_players(title):
    """'Pan Pacific Open: Leylah Fernandez vs Himeno Sakatsume' -> ['Leylah Fernandez', 'Himeno Sakatsume']."""
    part = str(title or "").rsplit(": ", 1)[-1]
    names = re.split(r"\s+vs\.?\s+", part, flags=re.IGNORECASE)
    return [n.strip() for n in names] if len(names) == 2 else []


def _same_surname(outcome, full):
    keys = candidate_keys(outcome)
    return bool(keys) and keys[0][0] in {k[0] for k in candidate_keys(full)}


def _players(event, market):
    """Volle Namen in der Reihenfolge der Outcomes (Token A, Token B)."""
    outcomes = [str(o) for o in _list(market.get('outcomes'))]
    full = _title_players(market.get('question') or event.get('title')) or _title_players(event.get('title'))
    full += [t.get('name') for t in event.get('teams') or [] if t.get('name')]
    out = []
    for o in outcomes:
        hit = next((f for f in full if f == o), None) or next((f for f in full if _same_surname(o, f)), None)
        out.append(hit or o)
    return out


def _is_doubles(names):
    return any("/" in n for n in names)


def match_market(event):
    """Der Sieger-Markt eines Einzelmatch-Events oder None (Futures, Doppel, nur Zusatzmärkte)."""
    markets = event.get('markets') or []
    for m in markets:
        outcomes = _list(m.get('outcomes'))
        if len(outcomes) != 2 or {str(o).lower() for o in outcomes} & {'yes', 'no', 'over', 'under'}:
            continue
        kind = m.get('sportsMarketType')
        if kind == 'moneyline' or (kind is None and len(markets) == 1):
            if not _title_players(m.get('question') or event.get('title')):
                continue
            if _is_doubles(_players(event, m)):
                return None
            return m
    return None


def tour_of(event):
    slug = event.get('seriesSlug') or next((s.get('slug') for s in event.get('series') or []), None)
    if slug in SERIES:
        return slug
    prefix = str(event.get('slug') or '').split('-')[0]
    if prefix in SERIES:  # 'atp-mayot-cassone-2026-09-26'
        return prefix
    league = str((event.get('eventMetadata') or {}).get('league') or event.get('title') or '')
    if prefix == 'itf' and re.match(r"W\d", league):  # ITF 'W15 Monastir' (Frauen), 'M15 Baku' (Männer)
        return 'wta'
    if prefix == 'itf' and re.match(r"M\d", league):
        return 'atp'
    for tag in event.get('tags') or []:
        if str(tag.get('id')) in TAG_TOURS:
            return TAG_TOURS[str(tag['id'])]
    title = str(event.get('title', ''))
    if re.search(r"\bWTA\b|\bWomen", title):
        return 'wta'
    if re.search(r"\bATP\b|\bMen\b", title):
        return 'atp'
    return None


def winner_of(market):
    """'a' / 'b' aus der Auflösung, None bei 50-50 (Absage, W/O) oder offenem Markt."""
    if not market.get('closed'):
        return None
    prices = [_float(p) for p in _list(market.get('outcomePrices'))]
    if prices == [1.0, 0.0]:
        return 'a'
    if prices == [0.0, 1.0]:
        return 'b'
    return None


def resolution(market):
    """Ergebnis aus Sicht von Token A: 'a' / 'b', 'void' bei 50-50 (Absage, W/O), None solange nicht aufgelöst."""
    winner = winner_of(market)
    if winner:
        return winner
    prices = [_float(p) for p in _list(market.get('outcomePrices'))]
    return 'void' if market.get('closed') and prices == [0.5, 0.5] else None


def fetch_resolution(client, market_id, player_a=None):
    """Auflösung eines einzelnen Match-Markts per ID (Gamma /markets/{id}), siehe resolution."""
    market = client.get_json(f"{GAMMA}/markets/{market_id}")
    return resolution(market[0] if isinstance(market, list) else market)


def match_info(event):
    """Gemeinsame Felder eines Match-Markts (für Snapshots und Historie) oder None."""
    m = match_market(event)
    if m is None:
        return None
    tokens = _list(m.get('clobTokenIds'))
    players = _players(event, m)
    if len(tokens) != 2 or len(players) != 2 or players[0] == players[1]:
        return None
    meta = event.get('eventMetadata') or {}
    title = str(m.get('question') or event.get('title') or "")
    return {
        'event_id': str(m['id']),
        'token_a': str(tokens[0]),
        'token_b': str(tokens[1]),
        'player_a': players[0],
        'player_b': players[1],
        'tour': tour_of(event),
        'level': 'itf' if str(event.get('slug') or '').startswith('itf-') else None,
        'tourney': meta.get('league') or (title.rsplit(": ", 1)[0] if ": " in title else None),
        'start_time_utc': utc(m.get('gameStartTime') or event.get('startTime')),
        'created_utc': utc(m.get('startDate') or event.get('startDate') or m.get('createdAt')),
        'volume': _float(m.get('volumeNum') if m.get('volumeNum') is not None else m.get('volume')),
        'liquidity': _float(m.get('liquidityNum') if m.get('liquidityNum') is not None else m.get('liquidity')),
        'best_bid': _float(m.get('bestBid')),
        'best_ask': _float(m.get('bestAsk')),
        'outcome_prices': [_float(p) for p in _list(m.get('outcomePrices'))],
        'closed': bool(m.get('closed')),
        'winner': winner_of(m),
        'source_url': EVENT_URL.format(event.get('slug') or m.get('slug')),
    }


def parse_events(events, ts=None, midpoints=None):
    """Offene Match-Märkte -> Snapshots. midpoints: {token_id: Preis} aus dem CLOB (optional)."""
    ts = ts or now_utc()
    midpoints = midpoints or {}
    out = []
    for event in events:
        info = match_info(event)
        if info is None or info['closed']:
            continue
        bid, ask = info['best_bid'], info['best_ask']
        prices = info['outcome_prices'] + [None, None]
        pa = _float(midpoints.get(info['token_a'])) or mid(bid, ask) or prices[0]
        pb = _float(midpoints.get(info['token_b'])) or (1 - pa if pa is not None else prices[1])
        out.append(Snapshot(
            ts_utc=ts, venue=VENUE, event_id=info['event_id'], player_a=info['player_a'],
            player_b=info['player_b'], price_a=pa, price_b=pb, price_type='probability', tour=info['tour'],
            level=info['level'], tourney=info['tourney'], start_time_utc=info['start_time_utc'],
            best_bid_a=bid, best_ask_a=ask,
            # Binärmarkt: Bid auf B = 1 - Ask auf A und umgekehrt
            best_bid_b=1 - ask if ask is not None else None, best_ask_b=1 - bid if bid is not None else None,
            volume=info['volume'], liquidity=info['liquidity'], source_url=info['source_url'],
        ))
    return out


def open_events(client, series=SERIES):
    """Alle offenen Events der Tennis-Serien (Offset-Paginierung, 100 pro Seite)."""
    events = []
    for sid in series.values():
        for offset in range(0, MAX_OFFSET, 100):
            page = client.get_json(f"{GAMMA}/events", {
                'series_id': sid, 'active': 'true', 'closed': 'false', 'limit': 100, 'offset': offset})
            events += page
            if len(page) < 100:
                break
    return events


def midpoints(client, tokens, batch=500):
    out = {}
    tokens = list(dict.fromkeys(tokens))
    for i in range(0, len(tokens), batch):
        out.update(client.post_json(f"{CLOB}/midpoints", [{'token_id': t} for t in tokens[i:i + batch]]))
    return out


def fetch_current(client=None, log=print):
    client = client or JsonClient()
    events = open_events(client)
    infos = [i for i in map(match_info, events) if i and not i['closed']]
    try:
        mids = midpoints(client, [t for i in infos for t in (i['token_a'], i['token_b'])])
    except Exception as e:  # ohne Mitte gehen auch die Gamma-Preise
        log(f"  Polymarket: CLOB-Midpoints fehlgeschlagen ({e}), nehme Gamma-Preise")
        mids = {}
    return parse_events(events, midpoints=mids)
