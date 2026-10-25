"""
Echte Reihenfolge der Matches für Elo, Historien und Features.

Sortiert wird nach dem echten Matchdatum (match_date), fehlt es, nach dem Turnierstart. Innerhalb
eines Tages entscheidet die Runde (R32 vor R16), dann match_num. Sackmann kennt nur den Turnierstart:
alle Matches eines Turniers haben dort dasselbe Datum, die Runde hält sie in der richtigen Folge.
build.py übernimmt das echte Datum aus tennis-data/ESPN, wo dasselbe Match dort vorkommt.
"""
import pandas as pd

from .sources.common import ROUND_ORDER


def _yyyymmdd(series):
    return pd.to_datetime(series.astype(str).str[:8], format='%Y%m%d', errors='coerce')


def match_dates(df):
    """Echtes Matchdatum als Timestamp, sonst Turnierstart."""
    tourney = _yyyymmdd(df['tourney_date'])
    if 'match_date' not in df.columns:
        return tourney
    return _yyyymmdd(df['match_date']).fillna(tourney)


def chronological(df):
    """df in echter Reihenfolge (Index bleibt erhalten, stabil sortiert)."""
    key = pd.DataFrame({
        'date': match_dates(df),
        'tourney': df['tourney_date'].astype(str).str[:8],
        'round': df['round'].map(ROUND_ORDER).fillna(0) if 'round' in df.columns else 0,
        'num': pd.to_numeric(df['match_num'], errors='coerce').fillna(0) if 'match_num' in df.columns else 0,
    }, index=df.index)
    order = key.sort_values(['date', 'tourney', 'round', 'num'], kind='stable', na_position='first').index
    return df.loc[order]
