import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from playwright.sync_api import sync_playwright

# Ohne normalen User-Agent blockt Tipico Headless-Browser
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36")

# z.B. PLAYWRIGHT_CHANNEL=msedge, um das installierte Edge statt des Playwright-Chromiums zu nutzen
BROWSER_CHANNEL = os.environ.get("PLAYWRIGHT_CHANNEL") or None


@dataclass
class Site:
    source: str
    url: str
    event_selector: str
    odds_selector: str
    # None -> Namen stehen im aria-label des Events ("A vs B event")
    team_selector: str = None


FOOTBALL_SITES = [
    Site('tipico', "https://sports.tipico.de/de/heute/fussball",
         "a[data-testid=eventRowContainer][data-listitem=list-item-soccer]",
         "button[data-fs=odds-button]"),
    Site('bwin', "https://sports.bwin.de/de/sports/fu%C3%9Fball-4",
         "ms-event.grid-event", ".option-indicator", ".participant"),
]

TENNIS_SITES = [
    Site('tipico', "https://sports.tipico.de/de/heute/tennis",
         "a[data-testid=eventRowContainer][data-listitem=list-item-tennis]",
         "button[data-fs=odds-button]"),
    Site('bwin', "https://sports.bwin.de/de/sports/tennis-5",
         "ms-event.grid-event", ".option-indicator", ".participant"),
]


def parse_odd(text):
    # Tipico: "2\n1,15" (Tipp + Quote), bwin: "1.14"
    try:
        return float(text.strip().split("\n")[-1].replace(',', '.'))
    except ValueError:
        return None


def scrape_website(site, n_outcomes):
    """
    Liefert eine Liste von (teams, odds, source): die ersten zwei Teams/Spieler eines Events
    und die ersten n_outcomes Quoten (1X2 bei Fußball, Sieger bei Tennis).
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, channel=BROWSER_CHANNEL)
        page = browser.new_page(locale="de-DE", user_agent=USER_AGENT)
        page.goto(site.url, timeout=60000)

        page.wait_for_selector(site.event_selector, timeout=30000)
        event_rows = page.query_selector_all(site.event_selector)

        events = []
        for event in event_rows:
            try:
                if site.team_selector:
                    teams = [t.inner_text().strip() for t in event.query_selector_all(site.team_selector)]
                else:
                    label = (event.get_attribute('aria-label') or '').removesuffix(' event')
                    teams = [t.strip() for t in label.split(' vs ')]

                odds = [parse_odd(o.inner_text()) for o in event.query_selector_all(site.odds_selector)[:n_outcomes]]

                if len(teams) >= 2 and len(odds) == n_outcomes and None not in odds:
                    events.append((teams[:2], odds, site.source))
            except Exception as e:
                print(f"Error processing event: {type(e).__name__}: {e}")

        browser.close()
        return events


def scrape_all_sites(sites, n_outcomes):
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(scrape_website, site, n_outcomes) for site in sites]

        all_events = []
        for future in as_completed(futures):
            try:
                all_events.extend(future.result())
            except Exception as e:
                print(f"Error in scraping: {e}")
    return all_events
