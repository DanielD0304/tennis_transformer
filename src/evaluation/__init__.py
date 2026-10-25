"""
Evaluations-Fundament: Walk-forward-Backtest, Metriken gegen den Markt (Pinnacle ohne Marge),
Venue-Profile und eine generische Wett-Simulation.

    python -m src.evaluation.backtest --start 2020 --end 2025 --models market,rank,elo,blend

Aufbau:
    predictors.py  einheitliches Interface: p = P(Spieler A gewinnt) pro Match
    backtest.py    Walk-forward über Zeitfenster -> DataFrame `predictions` (+ CLI)
    metrics.py     Log Loss, Brier, Accuracy, ECE, Kalibrierung, Signifikanztests
    venues.py      Quotenquelle, Steuer/Gebühren, RET-Regel, Limit pro Venue
    betting.py     simulate(predictions, strategy, venue, staking)
"""
