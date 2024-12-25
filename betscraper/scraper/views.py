from playwright.sync_api import sync_playwright
from betObjekts.footballgame import football_game
from dataProcessing.process import process_data
from concurrent.futures import ThreadPoolExecutor
from django.shortcuts import render
from concurrent.futures import as_completed

def scrape_website(url, event_class, team_class, odds_class, source=None):
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        
        page.wait_for_selector(f'.{event_class}')
        event_rows = page.query_selector_all(f'.{event_class}')
        
        game_data = []
        
        for event in event_rows:
            try:
                teams_elements = event.query_selector_all(f'.{team_class}')
                odds_elements = event.query_selector_all(f'.{odds_class}')
                
                teams = [team.inner_text() for team in teams_elements]
                odds = [odd.inner_text() for odd in odds_elements]
                
                if len(teams) >= 2 and len(odds) >= 3:
                    game = football_game(teams[0], odds[0], odds[1], teams[1], odds[2], source)
                    game_data.append(game)
            except Exception as e:
                print(f"Error processing event: {e} - {type(e).__name__}: {str(e)}")
        
        browser.close()
        return game_data

def scrape_tipico():
    url = "https://sports.tipico.de/de"
    event_row_class = "EventRow-styles-module-event-row"
    team_class = "EventTeams-styles-module-team-title"
    odds_class = "EventOddButton-styles-module-quotation"
    return scrape_website(url, event_row_class, team_class, odds_class, source='tipico')

def scrape_betwin():
    url = "https://sports.bwin.de/de/sports"
    event_row_class = "grid-event"
    team_class = "participant"
    odds_class = "option-value"
    return scrape_website(url, event_row_class, team_class, odds_class, source='bwin')

def scrape_all_sites():
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(scrape_tipico),
            executor.submit(scrape_betwin)
            # Add more future.submit calls for other websites here
        ]
        
        all_game_data = []
        for future in as_completed(futures):
            try:
                data = future.result()
                all_game_data.extend(data)
            except Exception as e:
                print(f"Error in scraping: {e}")

    aggregated_games = {}
    for game in all_game_data:
        key = (game.team1['name'], game.team2['name'])
        if key not in aggregated_games:
            aggregated_games[key] = game
        else:
            aggregated_games[key].update_odds(game.team1['odd'], game.draw['odd'], game.team2['odd'], game.team1['source'])
    
    return list(aggregated_games.values())

def fetch_and_process_odds(request):
    all_game_data = scrape_all_sites()
    processor = process_data(all_game_data)
    result_dict = processor.process_data()

    return render(request, 'scraper/quoten.html', {'games': all_game_data, 'results': result_dict})