import requests
from bs4 import BeautifulSoup
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
import time
from betObjekts.footballgame import football_game
from dataProcessing.process import process_data
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from concurrent.futures import ThreadPoolExecutor

def scrape_website(url, event_class, team_class, odds_class, wait_time=120):
    browser_driver = FirefoxService(r"C:\Users\danie\Downloads\geckodriver-v0.35.0-win32\geckodriver.exe")
    firefox_binary_path = r"C:\Program Files\Mozilla Firefox\firefox.exe"  # Update this path if necessary
    options = Options()
    options.binary_location = firefox_binary_path
    options.headless = True
    page_to_scrape = webdriver.Firefox(service=browser_driver, options=options)
    page_to_scrape.get(url)
    
    try:
        WebDriverWait(page_to_scrape, wait_time).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, event_class))
        )
    except Exception as e:
        print(f"Error waiting for page to load: {e}")
        page_to_scrape.quit()
        return []
    
    event_rows = page_to_scrape.find_elements(By.CLASS_NAME,event_class)
    
    gameData = []
    
    for event in event_rows:
        try:
            teams_elements = event.find_elements(By.CLASS_NAME, team_class)
            odds_elements = event.find_elements(By.CLASS_NAME, odds_class)
    
            teams = [team.text for team in teams_elements]
            odds = [odd.text for odd in odds_elements]
            
            if len(teams) >= 2 and len(odds) >= 3:
                game = football_game(teams[0], odds[0], odds[1], teams[1], odds[2])
                gameData.append(game)
        except Exception as e:
            print(f"Error processing event: {e} - {type(e).__name__}: {str(e)}")
    
    page_to_scrape.quit()
    return gameData

def scrape_tipico():
    url = "https://sports.tipico.de/de"
    event_row_class = "EventRow-styles-module-event-row"
    team_class = "EventTeams-styles-module-team-title"
    odds_class = "EventOddButton-styles-module-quotation"
    return scrape_website(url, event_row_class, team_class, odds_class)

def scrape_betwin():
    url = "https://sports.bwin.de/de/sports"
    event_row_class = "grid-option-group"
    team_class = "participant"
    odds_class = "option-value"
    return scrape_website(url, event_row_class, team_class, odds_class)

def scrape_all_sites():
    with ThreadPoolExecutor() as executor:
        future_tipico = executor.submit(scrape_tipico)
        future_betwin = executor.submit(scrape_betwin)
        # Füge hier weitere future.submit-Aufrufe für die anderen Websites hinzu
        
        tipico_data = future_tipico.result()
        betwin_data = future_betwin.result()
        print(f"Tipico data: {tipico_data}")
        print(f"Betwin data: {betwin_data}")
        # Füge hier weitere result-Aufrufe für die anderen Websites hinzu
    
    all_game_data = tipico_data + betwin_data # + weitere Daten der anderen Websites
    return all_game_data

def fetch_and_process_odds(request):
    all_game_data = scrape_all_sites()
    
    # Process the aggregated data and get the result dictionary
    processor = process_data(all_game_data)
    result_dict = processor.process_data()

    return render(request, 'scraper/quoten.html', {'games': all_game_data, 'results': result_dict})