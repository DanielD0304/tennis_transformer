from betObjekts.footballgame import football_game
from dataProcessing.predictor import get_predictor
from dataProcessing.process import process_data
from dataProcessing.value import add_predictions, aggregate_tennis, value_bets
from django.shortcuts import render

from .sites import FOOTBALL_SITES, TENNIS_SITES, scrape_all_sites

SURFACES = ['Hard', 'Clay', 'Grass', 'Carpet']


def scrape_football():
    aggregated_games = {}
    for (team1, team2), (odd1, draw, odd2), source in scrape_all_sites(FOOTBALL_SITES, n_outcomes=3):
        key = (team1, team2)
        if key not in aggregated_games:
            aggregated_games[key] = football_game(team1, odd1, draw, team2, odd2, source)
        else:
            aggregated_games[key].update_odds(odd1, draw, odd2, source)

    return list(aggregated_games.values())


def fetch_and_process_odds(request):
    all_game_data = scrape_football()
    processor = process_data(all_game_data)
    result_dict = processor.process_data()

    return render(request, 'scraper/quoten.html', {'games': all_game_data, 'results': result_dict})


def tennis_value_bets(request):
    surface = request.GET.get('surface', 'Hard')
    if surface not in SURFACES:
        surface = 'Hard'
    try:
        min_ev = float(request.GET.get('min_ev', 1.05))
    except ValueError:
        min_ev = 1.05

    matches = aggregate_tennis(scrape_all_sites(TENNIS_SITES, n_outcomes=2))

    error = None
    try:
        add_predictions(matches, get_predictor(), surface)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    return render(request, 'scraper/tennis.html', {
        'matches': matches,
        'bets': value_bets(matches, min_ev),
        'surface': surface,
        'surfaces': SURFACES,
        'min_ev': min_ev,
        'error': error,
    })
