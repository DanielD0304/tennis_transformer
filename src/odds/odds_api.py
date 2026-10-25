"""
The Odds API (https://the-odds-api.com): Sieger-Quoten (h2h) für alle aktiven Tennis-Turniere,
Regionen eu und uk (Pinnacle, Betfair Exchange, Bet365, ...).

Braucht einen API-Key in der Umgebungsvariable ODDS_API_KEY. Fehlt er, wird die Quelle mit
einem Hinweis übersprungen. Kosten: /sports ist frei, jede /odds-Anfrage kostet
Märkte × Regionen = 2 Credits (freier Tarif: 500 pro Monat). Bei 20 Minuten Takt und ~10 aktiven
Turnieren reicht das nicht, dann --odds-api-every hochsetzen (siehe collect.py).
"""
import os

from .http import JsonClient
from .schema import Snapshot, now_utc, utc

API = "https://api.the-odds-api.com/v4"
KEY_ENV = "ODDS_API_KEY"
REGIONS = "eu,uk"
# Buchmacher-Schlüssel der Odds API -> Venue-Name wie in src/evaluation/venues.py
VENUE_ALIASES = {'betfair_ex_eu': 'betfair', 'betfair_ex_uk': 'betfair'}
# Wenn derselbe Venue-Name mehrfach vorkommt (Betfair EU und UK), gewinnt der erste Eintrag hier
PREFERRED = ['betfair_ex_eu', 'betfair_ex_uk']


def api_key():
    return os.environ.get(KEY_ENV) or None


def tour_of(sport_key):
    if sport_key.startswith('tennis_atp'):
        return 'atp'
    if sport_key.startswith('tennis_wta'):
        return 'wta'
    return None


def tennis_sports(sports):
    return [s['key'] for s in sports if s.get('group') == 'Tennis' and s.get('active') and not s.get('has_outrights')]


def parse_odds(events, ts=None):
    """Antwort von /sports/{sport}/odds -> ein Snapshot pro Event und Buchmacher."""
    ts = ts or now_utc()
    out = []
    for e in events:
        a, b = e.get('home_team'), e.get('away_team')
        if not a or not b:
            continue
        seen = set()
        books = sorted(e.get('bookmakers') or [], key=lambda bk: PREFERRED.index(bk['key']) if bk['key'] in PREFERRED else len(PREFERRED))
        for book in books:
            venue = VENUE_ALIASES.get(book['key'], book['key'])
            if venue in seen:
                continue
            market = next((m for m in book.get('markets') or [] if m.get('key') == 'h2h'), None)
            if not market:
                continue
            prices = {o.get('name'): o.get('price') for o in market.get('outcomes') or []}
            if a not in prices or b not in prices:
                continue
            seen.add(venue)
            out.append(Snapshot(
                ts_utc=ts, venue=venue, event_id=str(e['id']), player_a=a, player_b=b,
                price_a=prices[a], price_b=prices[b], price_type='decimal', tour=tour_of(e.get('sport_key', '')),
                tourney=e.get('sport_title'), start_time_utc=utc(e.get('commence_time')),
                source_url=f"{API}/sports/{e.get('sport_key')}/odds",
            ))
    return out


def fetch_current(client=None, log=print, key=None):
    key = key or api_key()
    if not key:
        log(f"  Odds API übersprungen: Umgebungsvariable {KEY_ENV} ist nicht gesetzt")
        return []
    client = client or JsonClient(max_per_sec=2)
    sports = tennis_sports(client.get_json(f"{API}/sports", {'apiKey': key}))
    ts = now_utc()
    out, remaining = [], None
    for sport in sports:
        r = client.request('GET', f"{API}/sports/{sport}/odds", params={
            'apiKey': key, 'regions': REGIONS, 'markets': 'h2h', 'oddsFormat': 'decimal', 'dateFormat': 'iso'})
        remaining = r.headers.get('x-requests-remaining', remaining)
        out += parse_odds(r.json(), ts)
    log(f"  Odds API: {len(sports)} Turniere, {len(sports) + 1} Anfragen, Kontingent übrig: {remaining}")
    return out
