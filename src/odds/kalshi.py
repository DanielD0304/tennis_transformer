"""
Kalshi: Tennis-Match-Märkte über die öffentliche Market-Data-API (ohne Login).

Ein Match ist ein Event (z.B. KXATPMATCH-26OCT25MEDBAS) mit je einem Ja/Nein-Markt pro Spieler
(…-MED, …-BAS). Die Startzeit steht nicht am Markt (occurrence_datetime ist das erwartete Ende),
sondern an den Milestones (type 'tennis_tournament_singles', start_date).
Matches laufen in den Serien KXATPMATCH, KXWTAMATCH, KXATPCHALLENGERMATCH, KXWTACHALLENGERMATCH,
KXITFMATCH (ITF Herren) und KXITFWMATCH (ITF Damen), bis Mitte 2025 zum Teil in Turnier-Serien
(KXATPMIA, KXWTAMAD, ...); die Historie geht deshalb über die Milestones.
"""
from datetime import timedelta

from .http import JsonClient
from .schema import Snapshot, mid, now_utc, utc

VENUE = 'kalshi'
API = "https://api.elections.kalshi.com/trade-api/v2"
MATCH_SERIES = {
    'KXATPMATCH': ('atp', 'main'),
    'KXWTAMATCH': ('wta', 'main'),
    'KXATPCHALLENGERMATCH': ('atp', 'challenger'),
    'KXWTACHALLENGERMATCH': ('wta', 'challenger'),
    'KXITFMATCH': ('atp', 'itf'),
    'KXITFWMATCH': ('wta', 'itf'),
}
MILESTONE_TOURS = {
    'ATP': ('atp', 'main'),
    'WTA': ('wta', 'main'),
    'ATP Challenger': ('atp', 'challenger'),
    'WTA Challenger': ('wta', 'challenger'),
}


def _dollars(market, key):
    """Preisfeld in Dollar: neue API '<key>_dollars' als String, alte '<key>' in Cent oder Dollar."""
    value = market.get(f'{key}_dollars')
    if value is None:
        value = market.get(key)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value / 100 if value > 1 else value
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fp(market, key):
    for k in (f'{key}_fp', key):
        try:
            return float(market[k])
        except (KeyError, TypeError, ValueError):
            continue
    return None


def tour_level(milestone):
    """Milestone -> (tour, level). ITF über das Geschlecht, Davis Cup u.ä. -> (None, None)."""
    details = milestone.get('details') or {}
    tour = details.get('tour')
    if tour in MILESTONE_TOURS:
        return MILESTONE_TOURS[tour]
    if tour == 'ITF':
        return {'men': 'atp', 'women': 'wta'}.get(details.get('gender')), 'itf'
    return None, None


def event_ticker(milestone):
    details = milestone.get('details') or {}
    return details.get('main_game_event_ticker') or (milestone.get('primary_event_tickers') or [None])[0]


def order_markets(markets, milestone=None):
    """Die zwei Spieler-Märkte in der Reihenfolge first/second competitor (sonst wie geliefert)."""
    markets = [m for m in markets if m.get('yes_sub_title')]
    first = ((milestone or {}).get('details') or {}).get('first_competitor_id')
    if first:
        markets.sort(key=lambda m: (m.get('custom_strike') or {}).get('tennis_competitor') != first)
    return markets[:2]


def winner_of(market_a):
    """Ergebnis aus Sicht von Spieler A: 'a', 'b' oder None (fairer Preis bei Absage/W/O)."""
    return {'yes': 'a', 'no': 'b'}.get(market_a.get('result'))


def resolution(markets, player_a):
    """
    Ergebnis eines Match-Events aus Sicht von player_a (yes_sub_title seines Markts): 'a' / 'b', 'void' (abgerechnet
    ohne Sieger, z.B. Absage), None solange offen oder wenn kein Markt zu player_a passt.
    """
    market = next((m for m in markets if m.get('yes_sub_title') == player_a), None)
    if market is None:
        return None
    if market.get('result') in ('yes', 'no'):
        return winner_of(market)
    return 'void' if market.get('status') in ('settled', 'finalized') else None


def fetch_resolution(client, event_ticker, player_a):
    """Auflösung eines einzelnen Match-Events per Ticker (/markets?event_ticker=...), siehe resolution."""
    return resolution(list(paginate(client, "/markets", 'markets', {'event_ticker': event_ticker, 'limit': 100})),
                      player_a)


def parse_event(event, ts=None, milestone=None):
    """Ein offenes Match-Event (mit verschachtelten Märkten) -> Snapshot oder None."""
    markets = order_markets(event.get('markets') or [], milestone)
    if len(markets) != 2:
        return None
    a, b = markets
    series = event.get('series_ticker') or event['event_ticker'].split('-')[0]
    tour, level = tour_level(milestone) if milestone else (None, None)
    if tour is None:
        tour, level = MATCH_SERIES.get(series, (None, None))
    bid_a, ask_a = _dollars(a, 'yes_bid'), _dollars(a, 'yes_ask')
    bid_b, ask_b = _dollars(b, 'yes_bid'), _dollars(b, 'yes_ask')
    pa = mid(bid_a, ask_a) or _dollars(a, 'last_price')
    pb = mid(bid_b, ask_b) or _dollars(b, 'last_price')
    if not pa and not pb:
        return None
    volume = [v for v in (_fp(a, 'volume'), _fp(b, 'volume')) if v is not None]
    # Turnier vom Milestone: bei ITF steht am Event nur 'ITF', am Milestone 'M15 Mazatlan'
    tourney = ((milestone or {}).get('details') or {}).get('tournament_name') \
        or (event.get('product_metadata') or {}).get('competition')
    return Snapshot(
        ts_utc=ts or now_utc(), venue=VENUE, event_id=event['event_ticker'], player_a=a['yes_sub_title'],
        player_b=b['yes_sub_title'], price_a=pa or None, price_b=pb or None, price_type='probability',
        tour=tour, level=level, tourney=tourney,
        start_time_utc=utc((milestone or {}).get('start_date')), best_bid_a=bid_a, best_ask_a=ask_a,
        best_bid_b=bid_b, best_ask_b=ask_b, volume=sum(volume) if volume else None,
        source_url=f"{API}/events/{event['event_ticker']}",
    )


def paginate(client, path, key, params, max_pages=1000):
    params = dict(params)
    for _ in range(max_pages):
        page = client.get_json(f"{API}{path}", params)
        items = page.get(key) or []
        yield from items
        cursor = page.get('cursor')
        if not cursor or not items:
            return
        params['cursor'] = cursor


def milestones(client, since, until=None):
    """Alle Einzel-Milestones mit start_date >= since (datetime)."""
    params = {'type': 'tennis_tournament_singles', 'limit': 500,
              'minimum_start_date': since.strftime('%Y-%m-%dT%H:%M:%SZ')}
    out = paginate(client, "/milestones", 'milestones', params)
    if until is None:
        return list(out)
    return [m for m in out if utc(m['start_date']) <= until]


def fetch_current(client=None, log=print):
    client = client or JsonClient()
    ts = now_utc()
    by_event = {}
    try:
        for m in milestones(client, ts - timedelta(days=2)):
            by_event[event_ticker(m)] = m
    except Exception as e:  # Startzeiten sind nett, aber nicht nötig
        log(f"  Kalshi: Milestones fehlgeschlagen ({e}), Snapshots ohne Startzeit")
    out = []
    for series in MATCH_SERIES:
        params = {'series_ticker': series, 'status': 'open', 'with_nested_markets': 'true', 'limit': 200}
        for event in paginate(client, "/events", 'events', params):
            snap = parse_event(event, ts, by_event.get(event['event_ticker']))
            if snap:
                out.append(snap)
    return out
