from datetime import date

from dataProcessing.predictor import get_predictor
from django.shortcuts import render

SURFACES = ['Hard', 'Clay', 'Grass', 'Carpet']


def predict_match(request):
    """
    Interface zum Transformer: zwei Spieler + Untergrund + Datum -> Siegwahrscheinlichkeit,
    faire Quoten, H2H und aktuelle Form beider Spieler.
    """
    player_a = request.GET.get('a', '').strip()
    player_b = request.GET.get('b', '').strip()
    surface = request.GET.get('surface', 'Hard')
    if surface not in SURFACES:
        surface = 'Hard'
    try:
        match_date = date.fromisoformat(request.GET.get('date', ''))
    except ValueError:
        match_date = date.today()

    context = {
        'a': player_a, 'b': player_b, 'surface': surface, 'surfaces': SURFACES,
        'date': match_date.isoformat(), 'match_date': match_date,
        'players': [], 'errors': [], 'result': None,
    }

    try:
        predictor = get_predictor()
    except Exception as e:
        context['model_error'] = f"{type(e).__name__}: {e}"
        return render(request, 'predictor/predict.html', context)

    context['players'] = sorted(set(predictor.players.values()))
    context['data_until'] = predictor.last_match_date.date()

    if not (player_a and player_b):
        return render(request, 'predictor/predict.html', context)

    resolved = {}
    for field, name in (('a', player_a), ('b', player_b)):
        found = predictor.resolve(name)
        if found is None:
            context['errors'].append({'name': name, 'suggestions': predictor.search(name)})
        resolved[field] = found

    if context['errors']:
        return render(request, 'predictor/predict.html', context)
    if resolved['a'] == resolved['b']:
        context['errors'].append({'name': player_b, 'suggestions': [], 'same': True})
        return render(request, 'predictor/predict.html', context)

    pred = predictor.predict(resolved['a'], resolved['b'], surface=surface, match_date=match_date)
    wins_a, wins_b = predictor.h2h_record(resolved['a'], resolved['b'])
    context['result'] = {
        'pred': pred,
        'pct_a': round(100 * pred.prob_a, 1),
        'pct_b': round(100 * pred.prob_b, 1),
        'fair_a': 1 / pred.prob_a if pred.prob_a > 0 else None,
        'fair_b': 1 / pred.prob_b if pred.prob_b > 0 else None,
        'h2h': (wins_a, wins_b),
        'info_a': predictor.player_info(resolved['a'], surface),
        'info_b': predictor.player_info(resolved['b'], surface),
    }
    return render(request, 'predictor/predict.html', context)
