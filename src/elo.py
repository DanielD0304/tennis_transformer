import math

def compute_elo_ratings(df, k_factor=32, start_elo=1500):
    """
    Berechnet ELO-Ratings für alle Matches im DataFrame.
    Fügt die Spalten 'winner_elo' und 'loser_elo' hinzu.
    WICHTIG: Das sind die ELOs VOR dem Match.
    """
    print("Berechne ELO-Ratings...")
    
    # Sicherstellen, dass chronologisch sortiert ist
    if 'match_num' in df.columns:
        df = df.sort_values(['tourney_date', 'match_num']).reset_index(drop=True)
    else:
        df = df.sort_values(['tourney_date']).reset_index(drop=True)
        
    elo_ratings = {} # Speichert aktuelles ELO für jeden Spieler
    
    winner_elos = []
    loser_elos = []
    
    for index, row in df.iterrows():
        w_name = row['winner_name']
        l_name = row['loser_name']
        
        # Aktuelle ELOs holen (oder Startwert)
        w_elo = elo_ratings.get(w_name, start_elo)
        l_elo = elo_ratings.get(l_name, start_elo)
        
        # Speichern für dieses Match (Status VOR dem Ergebnis)
        winner_elos.append(w_elo)
        loser_elos.append(l_elo)
        
        # Erwartungswert berechnen (Standard ELO Formel)
        expected_w = 1 / (1 + 10 ** ((l_elo - w_elo) / 400))
        expected_l = 1 / (1 + 10 ** ((w_elo - l_elo) / 400))
        
        # Neue ELOs berechnen (Winner kriegt Punkte, Loser verliert sie)
        # Score ist 1 für Winner, 0 für Loser
        new_w_elo = w_elo + k_factor * (1 - expected_w)
        new_l_elo = l_elo + k_factor * (0 - expected_l)
        
        # Dictionary updaten
        elo_ratings[w_name] = new_w_elo
        elo_ratings[l_name] = new_l_elo
        
    df['winner_elo'] = winner_elos
    df['loser_elo'] = loser_elos
    
    return df