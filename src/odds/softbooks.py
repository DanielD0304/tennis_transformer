"""
Deutsche Soft-Books (Tipico, bwin) über die vorhandenen Playwright-Scraper in betscraper/scraper/sites.py.

Die Scraper liefern keine Event-ID und keine Startzeit. event_id ist deshalb ein Schlüssel aus Venue
und den Spielernamen; den Abgleich mit Matches macht matching.py über Namen und Zeitfenster.
Tipico blockt Cloud-IPs teilweise. Lokal mit installiertem Browser (playwright install chromium)
oder PLAYWRIGHT_CHANNEL=msedge.
"""
import os
import re
import sys

from .schema import Snapshot, now_utc

BETSCRAPER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "betscraper")


def _event_id(source, a, b):
    return f"{source}:{re.sub(r'[^a-z]+', '-', (a + ' vs ' + b).lower()).strip('-')}"


def parse_events(events, ts=None):
    """[(teams, odds, source), ...] aus scrape_all_sites -> Snapshots (Dezimalquoten)."""
    ts = ts or now_utc()
    out = []
    for teams, odds, source in events:
        if len(teams) < 2 or len(odds) < 2 or "/" in teams[0] or "/" in teams[1]:
            continue
        a, b = teams[0].strip(), teams[1].strip()
        out.append(Snapshot(ts_utc=ts, venue=source, event_id=_event_id(source, a, b), player_a=a, player_b=b,
                            price_a=odds[0], price_b=odds[1], price_type='decimal'))
    return out


def scrape():
    if BETSCRAPER_DIR not in sys.path:
        sys.path.insert(0, BETSCRAPER_DIR)
    from scraper.sites import TENNIS_SITES, scrape_all_sites
    return scrape_all_sites(TENNIS_SITES, 2)


def fetch_current(client=None, log=print):
    ts = now_utc()
    return parse_events(scrape(), ts)
