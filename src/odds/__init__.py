"""
Quoten-Snapshots mit Zeitstempel und historische Prediction-Market-Preise (ROI-Plan Phase 2, Punkte 7–8).

    python -m src.odds.collect             # alle Venues einmal abfragen, Snapshots anhängen
    python -m src.odds.collect --loop 20   # alle 20 Minuten
    python -m src.odds.collect --history   # historische Polymarket- und Kalshi-Preise laden
    python -m src.odds.report              # Kalibrierung der PM-Closing-Preise
"""
