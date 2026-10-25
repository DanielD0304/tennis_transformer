from .chronology import chronological
from .score import parse_score

# Gewicht eines Matches mit Aufgabe (RET/DEF) im Elo: K wird damit skaliert
RET_WEIGHT = 0.5


def is_retired(row):
    """Aufgabe oder Disqualifikation: Match nicht regulär zu Ende gespielt."""
    status = str(row.get('status', '') or '')
    if status in ('RET', 'DEF'):
        return True
    return status in ('', 'nan', 'completed') and parse_score(row.get('score')).retired


def is_walkover(row):
    return str(row.get('status', '')) == 'W/O' or 'W/O' in str(row.get('score', ''))


def expected(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def compute_elo_ratings(df, k_factor=32, start_elo=1500, ret_weight=RET_WEIGHT):
    """
    Berechnet ELO-Ratings für alle Matches im DataFrame, getrennt pro Tour (Spalte 'tour').
    Fügt 'winner_elo'/'loser_elo' (Stand VOR dem Match) und 'winner_elo_after'/'loser_elo_after' hinzu.
    Reihenfolge: echtes Matchdatum, sonst Turnierstart + Runde (siehe chronology.py).
    Walkover zählt nicht, Aufgabe (RET/DEF) mit K * ret_weight.
    """
    print("Berechne ELO-Ratings...")
    df = chronological(df).reset_index(drop=True)

    elo_ratings = {}  # (Tour, Spieler) -> aktuelles ELO
    winner_elos, loser_elos, winner_after, loser_after = [], [], [], []
    tours = df['tour'].fillna('atp') if 'tour' in df.columns else ['atp'] * len(df)

    for row, tour in zip(df.to_dict('records'), tours):
        w_key, l_key = (tour, row['winner_name']), (tour, row['loser_name'])
        w_elo = elo_ratings.get(w_key, start_elo)
        l_elo = elo_ratings.get(l_key, start_elo)
        winner_elos.append(w_elo)
        loser_elos.append(l_elo)

        if is_walkover(row):
            # kein gespieltes Match, ELO bleibt unverändert
            winner_after.append(w_elo)
            loser_after.append(l_elo)
            continue

        k = k_factor * (ret_weight if is_retired(row) else 1.0)
        delta = k * (1 - expected(w_elo, l_elo))
        elo_ratings[w_key] = w_elo + delta
        elo_ratings[l_key] = l_elo - delta
        winner_after.append(w_elo + delta)
        loser_after.append(l_elo - delta)

    df['winner_elo'] = winner_elos
    df['loser_elo'] = loser_elos
    df['winner_elo_after'] = winner_after
    df['loser_elo_after'] = loser_after
    return df
