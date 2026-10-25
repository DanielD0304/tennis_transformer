"""
Lädt die Matchdaten fürs Training und den Predictor.

Die Daten selbst kommen aus src/sources/ (Sackmann-Archiv, tennis-data.co.uk, ESPN) und werden
von src/update_data.py zu data/matches.csv zusammengeführt. Hier wird nur noch gelesen und das
Schema für die Features vereinheitlicht.
"""

import os

import pandas as pd

from .sources.common import MATCHES_PATH, ODDS_COLUMNS

# Das Ziel-Schema (Sackmann-Style)
REQUIRED_STATS_COLS = [
    'w_ace', 'l_ace', 'w_df', 'l_df', 'w_svpt', 'l_svpt',
    'w_1stIn', 'l_1stIn', 'w_1stWon', 'l_1stWon', 'w_2ndWon', 'l_2ndWon',
    'w_SvGms', 'l_SvGms', 'w_bpSaved', 'l_bpSaved', 'w_bpFaced', 'l_bpFaced'
]

def enforce_schema(df):
    """Stats-Spalten als Zahlen (fehlend = 0), Rankings (fehlend = 9999), Datum als 'YYYYMMDD'."""
    for col in REQUIRED_STATS_COLS:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    for col in ['winner_rank', 'loser_rank']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(9999)

    if 'tourney_date' in df.columns:
        dates = pd.to_datetime(df['tourney_date'].astype(str).str[:8], format='%Y%m%d', errors='coerce')
        df['tourney_date'] = dates.dt.strftime('%Y%m%d')

    # Ältere matches.csv kennen nur ATP und nur die Haupttour
    df['tour'] = df['tour'].fillna('atp') if 'tour' in df.columns else 'atp'
    df['level'] = df['level'].fillna('main') if 'level' in df.columns else 'main'
    return df


def load_matches(first_year=None, last_year=None, refresh=True, max_age_hours=24, tour=None):
    """
    Gemeinsamer Datensatz aus data/matches.csv (Sackmann + tennis-data + ESPN, siehe src/update_data.py).
    Ist er älter als max_age_hours oder fehlt, wird er vorher aktualisiert.
    tour: 'atp' oder 'wta' filtert, None = beide.
    """
    from . import update_data
    if refresh and update_data.is_stale(max_age_hours):
        try:
            update_data.update()
        except Exception as e:
            if not os.path.exists(MATCHES_PATH):
                raise
            print(f"Warning: Update fehlgeschlagen ({e}), nutze vorhandene Daten.")

    df = pd.read_csv(MATCHES_PATH, low_memory=False,
                     dtype={'tourney_date': str, 'match_date': str, 'winner_dob': str, 'loser_dob': str})
    year = df['tourney_date'].str[:4].astype(int)
    if first_year is not None:
        df = df[year >= first_year]
    if last_year is not None:
        df = df[year <= last_year]
    df = enforce_schema(df.reset_index(drop=True))
    if tour is not None:
        df = df[df['tour'] == tour].reset_index(drop=True)
    for col in ODDS_COLUMNS:
        # Ältere matches.csv haben noch keine BFE/Max-Spalten
        df[col] = pd.to_numeric(df[col], errors='coerce') if col in df.columns else float('nan')
    return df


def load_merged_data(train_years, test_year):
    """Trainings- und Testdaten (inkl. Quoten) aus dem gemeinsamen Datensatz."""
    print(f"--- Loading matches {min(train_years)}-{test_year} ---")
    return load_matches(first_year=min(train_years), last_year=test_year)
