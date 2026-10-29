from betObjekts.footballgame import football_game
from dataProcessing.predictor import get_predictor
from dataProcessing.process import process_data
from dataProcessing.value import add_predictions, aggregate_tennis, value_bets
from django.shortcuts import render

from .sites import FOOTBALL_SITES, TENNIS_SITES, scrape_all_sites

SURFACES = ['Hard', 'Clay', 'Grass', 'Carpet']
# Auto: pro Match aus dem laufenden Turnier der Spieler (MatchPredictor.surface_for), sonst Override
AUTO = 'Auto'


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
    surface = request.GET.get('surface', AUTO)
    if surface not in SURFACES:
        surface = AUTO
    try:
        min_ev = float(request.GET.get('min_ev', 1.05))
    except ValueError:
        min_ev = 1.05

    matches = aggregate_tennis(scrape_all_sites(TENNIS_SITES, n_outcomes=2))

    error = None
    try:
        add_predictions(matches, get_predictor(), None if surface == AUTO else surface)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    return render(request, 'scraper/tennis.html', {
        'matches': matches,
        'bets': value_bets(matches, min_ev),
        'surface': surface,
        'surfaces': [AUTO] + SURFACES,
        'min_ev': min_ev,
        'error': error,
    })


def _pct(v, sign=True):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "–"
    if v != v:
        return "–"
    return f"{v:+.1%}" if sign else f"{v:.0%}"


PCT_FIELDS = ('yield', 'yield_ci_low', 'yield_ci_high', 'yield_flat', 'clv', 'clv_sharp', 'edge', 'hit_rate')


def _fmt_rows(rows):
    """Prozentfelder als Text (+1,2 %), NaN als –."""
    import pandas as pd
    out = []
    for r in rows:
        # NaT/NaN -> None und Timestamp -> datetime, sonst scheitern die Template-Filter (date, floatformat)
        r = {k: (None if not isinstance(v, (list, tuple, dict)) and pd.isna(v) else
                 v.to_pydatetime() if isinstance(v, pd.Timestamp) else v) for k, v in r.items()}
        for k in PCT_FIELDS:
            if k in r:
                r[f'{k}_txt'] = _pct(r[k], sign=k != 'hit_rate')
        out.append(r)
    return out


def _curve_points(curve, width=640, height=180, pad=6):
    """Bankroll-Kurve [(Zeit, Wert), ...] -> SVG-Polyline-Punkte (nach Index, nicht nach Zeit)."""
    values = [v for _, v in curve]
    if len(values) < 2:
        return "", None, None
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    step = (width - 2 * pad) / (len(values) - 1)
    pts = [f"{pad + i * step:.1f},{height - pad - (v - lo) / span * (height - 2 * pad):.1f}" for i, v in enumerate(values)]
    return " ".join(pts), lo, hi


def paper_trading(request):
    """Paper-Trading-Dashboard: Bankroll-Kurve, Yield mit KI, CLV pro Venue/Strategie/Level/Segment, Go/No-Go insgesamt
    und je Segment, offene und abgerechnete Wetten."""
    from src.paper import report
    from src.paper.store import load_bets

    error = None
    try:
        bets = load_bets()
    except Exception as e:  # z.B. kaputte Datei: Seite trotzdem anzeigen
        import pandas as pd
        from src.paper.store import COLUMNS
        bets, error = pd.DataFrame(columns=COLUMNS), f"{type(e).__name__}: {e}"
    strategy = request.GET.get('strategy') or ''
    shown = bets[bets['strategy'] == strategy] if strategy else bets
    curve = report.bankroll_curve(shown)
    points, lo, hi = _curve_points(curve)
    table = report.summary(shown, n_boot=1000) if len(shown) else None
    rows = [] if table is None else table.to_dict('records')
    decision = report.go_no_go(shown, n_boot=1000)
    segments = [{'name': name, **d} for name, d in report.go_no_go_by_segment(shown, n_boot=1000).items()]
    cols = ['created_utc', 'strategy', 'segment', 'venue', 'level', 'player', 'player_a', 'player_b', 'start_time_utc',
            'odds', 'net_odds', 'p', 'edge', 'stake', 'status', 'profit', 'clv']
    recent = shown.sort_values('created_utc', ascending=False)
    return render(request, 'scraper/paper.html', {
        'error': error,
        'strategy': strategy,
        'strategies': sorted(bets['strategy'].dropna().unique()),
        'points': points, 'curve_low': lo, 'curve_high': hi, 'bankroll': curve[-1][1],
        'start_bankroll': report.START_BANKROLL,
        'rows': _fmt_rows(rows),
        'decision': decision,
        'segments': segments,
        'open_bets': _fmt_rows(recent[recent['status'] == 'open'][cols].head(200).to_dict('records')),
        'settled_bets': _fmt_rows(recent[recent['status'] != 'open'][cols].head(200).to_dict('records')),
    })
