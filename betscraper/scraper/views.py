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

def scrape_tipico():
    browser_driver = FirefoxService(r"C:\Users\danie\Downloads\geckodriver-v0.35.0-win32\geckodriver.exe")
    firefox_binary_path = r"C:\Program Files\Mozilla Firefox\firefox.exe"  # Update this path if necessary
    options = Options()
    options.binary_location = firefox_binary_path
    options.headless = True
    page_to_scrape = webdriver.Firefox(service=browser_driver, options=options)
    page_to_scrape.get("https://sports.tipico.de/de")
    
    time.sleep(8)
    event_rows = page_to_scrape.find_elements(By.CLASS_NAME, "EventRow-styles-module-event-row")
    
    gameData = []
    
    for event in event_rows:
        try:
            teams_elements = event.find_elements(By.CLASS_NAME, "EventTeams-styles-module-team-title")
            odds_elements = event.find_elements(By.CLASS_NAME, "EventOddButton-styles-module-quotation")
    
            teams = [team.text for team in teams_elements]
            odds = [odd.text for odd in odds_elements]
            print(f"Teams: {teams}")
            print(f"Odds: {odds}")
            
            if len(teams) >= 2 and len(odds) >= 3:
                game = football_game(teams[0], odds[0], odds[1], teams[1], odds[2])
                gameData.append(game)
        except Exception as e:
            print(f"Error processing event: {e}")
    
    page_to_scrape.quit()
    return gameData

def scrape_another_site():
    # Implement scraping logic for another site
    gameData = []
    # Example scraping logic
    return gameData

def scrape_odds(request):
    tipico_data = scrape_tipico()
    another_site_data = scrape_another_site()
    
    all_game_data = tipico_data + another_site_data
    
    # Send the aggregated data to the data_processing module
    process_data(all_game_data)
    return render(request, 'scraper/quoten.html', {'games': all_game_data})