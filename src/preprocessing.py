"""
Preprocessing module for Tennis Transformer.

This module transforms raw ATP match data into training samples for the Transformer model.
Each sample contains the match history of two players (surface-specific and recent form)
along with attention masks for variable-length sequences.

Functions:
    - save_preprocessed_samples: Run preprocessing and save to .pt file
    - extract_match_features: Extract feature dict from a single match
    - pad_sequence: Pad sequences to fixed length with attention mask
    - create_training_samples: Generate training samples from match data
"""

import torch
import math
import random
import pandas as pd
from collections import defaultdict
from . import data_loader as dl

# Feature names used for each match in the history
# Updated to include ELO and opponent rank features
FEATURE_NAMES = [
    'won', 'rank', 'aces', 'double_faults', 'first_serve_pct', 
    'days_since_match', 'opponent_elo', 'opponent_rank', 'own_elo'
]


def save_preprocessed_samples(df, out_path, n_matches=10, n_recent=15):
    """
    Führt das Preprocessing einmal aus und speichert die Samples als .pt-Datei.
    Args:
        df (pd.DataFrame): Rohdaten
        out_path (str): Speicherpfad für .pt-Datei
        n_matches (int): Surface-Matches
        n_recent (int): Recent-Matches
    """
    from .elo import compute_elo_ratings
    df = compute_elo_ratings(df)
    
    samples = create_training_samples(df, n_matches=n_matches, n_recent=n_recent)
    torch.save(samples, out_path)
    print(f"Saved {len(samples)} samples to {out_path}")


def extract_match_features(row, player_role):
    """
    Extract relevant features from a match row for a specific player.
    
    Args:
        row (pd.Series): A single row from the matches DataFrame
        player_role (str): Either 'winner' or 'loser'
    
    Returns:
        dict: Dictionary containing match features:
            - won (int): 1 if player won, 0 if lost
            - rank (int): Player's ranking at time of match (log scaled)
            - aces (int): Number of aces served
            - double_faults (int): Number of double faults
            - first_serve_pct (float): First serve percentage
            - opponent_elo (float): Normalized ELO of the opponent
            - opponent_rank (float): Log scaled rank of the opponent
            - own_elo (float): Normalized ELO of the player
            - days_since_match (float): Will be calculated dynamically in pad_sequence
    """
    if player_role == "winner":
        prefix = "w_"
        rank_col = "winner_rank"
        won = 1
        opp_rank_col = "loser_rank"
        opp_elo_col = "loser_elo"
        own_elo_col = "winner_elo"
    else:
        prefix = "l_"
        rank_col = "loser_rank"
        won = 0
        opp_rank_col = "winner_rank"
        opp_elo_col = "winner_elo"
        own_elo_col = "loser_elo"
    
    # --- Feature Extraction ---
    
    # 1. Rank (log)
    rank_value = row[rank_col]
    if rank_value is None or (isinstance(rank_value, float) and math.isnan(rank_value)) or rank_value < 1:
        log_rank = 0.0
    else:
        log_rank = math.log(rank_value + 1)
        
    # 2. Opponent Rank (log)
    opp_rank_val = row[opp_rank_col]
    if opp_rank_val is None or (isinstance(opp_rank_val, float) and math.isnan(opp_rank_val)) or opp_rank_val < 1:
        log_opp_rank = 0.0
    else:
        log_opp_rank = math.log(opp_rank_val + 1)

    # 3. ELO Features (Normalized)
    opp_elo = row.get(opp_elo_col, 1500.0)
    norm_opp_elo = opp_elo / 2000.0
    
    own_elo = row.get(own_elo_col, 1500.0)
    norm_own_elo = own_elo / 2000.0

    # 4. Stats
    aces = row[prefix + 'ace']
    dfs = row[prefix + 'df']
    svpt = row[prefix + 'svpt']
    first_in = row[prefix + '1stIn']
    first_serve_pct = first_in / svpt if svpt != 0 else 0
    
    # 5. Days since match 
    # Placeholder: days_since_match wird jetzt in pad_sequence dynamisch berechnet
    
    return {
        'won': won,
        'rank': log_rank,
        'aces': aces,
        'double_faults': dfs,
        'first_serve_pct': first_serve_pct,
        'days_since_match': 0.0, 
        'opponent_elo': norm_opp_elo,
        'opponent_rank': log_opp_rank,
        'own_elo': norm_own_elo,
        '_date_obj': row.get('match_date_obj', None),
        '_surface': row['surface'] 
    }


def pad_sequence(sequence, max_len, current_date=None):
    """
    Pad a sequence to a fixed length and create an attention mask.
    
    Pads shorter sequences with zero-valued feature dictionaries.
    The attention mask indicates which positions contain real data (1)
    and which are padding (0).
    Also dynamically calculates 'days_since_match' relative to current_date.
    
    Args:
        sequence (list[dict]): List of feature dictionaries
        max_len (int): Target length for padding
        current_date (datetime, optional): Date of the match to predict. 
                                         Used to calculate recency of history matches.
    
    Returns:
        tuple: (padded_sequence, mask)
            - padded_sequence (list[dict]): Sequence padded to max_len
            - mask (list[int]): Attention mask (1=real, 0=padding)
    """
    processed_seq = []
    
    for item in sequence:
        new_item = item.copy()
        
        # Dynamische Berechnung der Tage (log scale)
        if current_date is not None and item.get('_date_obj') is not None:
            delta = current_date - item['_date_obj']
            days = max(0, delta.days)
            new_item['days_since_match'] = math.log(1 + days)
        else:
            new_item['days_since_match'] = 0.0
            
        new_item.pop('_date_obj', None)
        new_item.pop('_surface', None)
        processed_seq.append(new_item)

    actual_len = len(processed_seq)
    pad_len = max_len - actual_len
    
    mask = [1] * actual_len + [0] * pad_len
    
    # Zero Dict erstellen
    keys = processed_seq[0].keys() if processed_seq else FEATURE_NAMES
    zero_dict = {k: 0.0 for k in keys}
    
    padded = processed_seq + [zero_dict.copy() for _ in range(pad_len)]
    return padded, mask

def count_matches_in_period(history, current_date, days=90):
    """
    Zählt, wie viele Matches ein Spieler in den letzten X Tagen hatte.
    Nutzt die player_history (Liste von Dicts mit _date_obj).
    """
    if not history or current_date is None:
        return 0
    
    count = 0
    # Grenzwert berechnen
    threshold_date = current_date - pd.Timedelta(days=days)
    
    # Rückwärts iterieren (von neu nach alt) ist effizienter
    # history[-1] ist das neueste Match VOR dem aktuellen
    for match in reversed(history):
        match_date = match.get('_date_obj')
        if match_date is None:
            continue
            
        if match_date < threshold_date:
            break # Zu alt, wir können abbrechen (da chronologisch sortiert)
            
        count += 1
        
    return count

def create_training_samples(df, n_matches=10, n_recent=15):
    """
    Create training samples from match data for the Transformer model.
    
    OPTIMIZED VERSION: Uses a single-pass approach (O(N)) instead of repeated filtering (O(N^2)).
    
    For each match, creates a sample containing:
    - Surface-specific history for both players (last N matches on same surface)
    - Recent form for both players (last N matches on any surface)
    - Attention masks for all sequences
    - Label indicating which player wins
    
    Player assignment (A/B) is randomized to prevent the model from learning
    that player_a always wins.
    
    Args:
        df (pd.DataFrame): DataFrame containing match data
        n_matches (int): Number of surface-specific matches (default: 10)
        n_recent (int): Number of recent matches (default: 15)
    
    Returns:
        list[dict]: List of training samples, each containing:
            - player_a_surface: Surface history for player A
            - player_a_surface_mask: Attention mask for player A surface
            - player_a_recent: Recent history for player A
            - player_a_recent_mask: Attention mask for player A recent
            - player_b_surface: Surface history for player B
            - player_b_surface_mask: Attention mask for player B surface
            - player_b_recent: Recent history for player B
            - player_b_recent_mask: Attention mask for player B recent
            - label: 1 if player A wins, 0 if player B wins
    """
    print("Preprocessing data (Optimized Speed)...")
    
    # 1. Datumskorrektur & Sortierung
    df['match_date_obj'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')
    
    if 'match_num' in df.columns:
        df = df.sort_values(['tourney_date', 'match_num'])
    else:
        df = df.sort_values(['tourney_date'])
    
    # 2. History Container
    # history[player_name] = [ match_1, match_2, ... ]
    player_history = defaultdict(list)
    
    samples = []
    stats = {'total': 0, 'kept': 0, 'filtered_rank': 0, 'filtered_activity': 0}
    
    count = 0
    total = len(df)
    
    for index, row in df.iterrows():
        count += 1
        if count % 5000 == 0:
            print(f"Processing match {count}/{total}...")
            
        winner = row['winner_name']
        loser = row['loser_name']
        surface = row['surface']
        current_date = row['match_date_obj']
        match_year = int(str(row['tourney_date'])[:4])
        
        stats['total'] += 1
        
        r_winner = row['winner_rank'] if not pd.isna(row['winner_rank']) else 9999
        r_loser = row['loser_rank'] if not pd.isna(row['loser_rank']) else 9999
        
        is_relevant_match = (r_winner <= 150) and (r_loser <= 150)
        
        is_top150 = (r_winner <= 150) and (r_loser <= 150)
        
        pA_active = False
        pB_active = False
        
        if is_top150: # Nur prüfen wenn Rank okay ist (spart Zeit)
            pA_matches = count_matches_in_period(player_history[winner], current_date, days=90)
            pB_matches = count_matches_in_period(player_history[loser], current_date, days=90)
            pA_active = pA_matches >= 5
            pB_active = pB_matches >= 5
        
        keep_sample = is_top150 and pA_active and pB_active
        
        if not is_top150:
            stats['filtered_rank'] += 1
        elif not (pA_active and pB_active):
            stats['filtered_activity'] += 1
            
        if keep_sample:
            stats['kept'] += 1
            
            # Label Randomization
            if random.random() > 0.5:
                pA, pB = winner, loser
                label = 1
                odds_a = row.get('B365W', 0.0)
                odds_b = row.get('B365L', 0.0)
            else:
                pA, pB = loser, winner
                label = 0
                odds_a = row.get('B365L', 0.0)
                odds_b = row.get('B365W', 0.0)
                
            def get_hist(player, surf):
                full = player_history.get(player, [])
                recent = full[-n_recent:][::-1]
                surf_hist = [m for m in full if m['_surface'] == surf]
                surf_recent = surf_hist[-n_matches:][::-1]
                return surf_recent, recent

            pA_surf_hist, pA_rec_hist = get_hist(pA, surface)
            pB_surf_hist, pB_rec_hist = get_hist(pB, surface)
            
            pA_surf, pA_surf_mask = pad_sequence(pA_surf_hist, n_matches, current_date)
            pA_rec, pA_rec_mask = pad_sequence(pA_rec_hist, n_recent, current_date)
            pB_surf, pB_surf_mask = pad_sequence(pB_surf_hist, n_matches, current_date)
            pB_rec, pB_rec_mask = pad_sequence(pB_rec_hist, n_recent, current_date)
            
            seg_ids = [1]*n_matches + [2]*n_recent + [3]*n_matches + [4]*n_recent
            
            if pd.isna(odds_a): odds_a = 0.0
            if pd.isna(odds_b): odds_b = 0.0

            sample = {
                'player_a_surface': pA_surf,
                'player_a_surface_mask': pA_surf_mask,
                'player_a_recent': pA_rec,
                'player_a_recent_mask': pA_rec_mask,
                'player_b_surface': pB_surf,
                'player_b_surface_mask': pB_surf_mask,
                'player_b_recent': pB_rec,
                'player_b_recent_mask': pB_rec_mask,
                'segment_ids': seg_ids,
                'label': label,
                'year': match_year,
                'odds_a': odds_a,
                'odds_b': odds_b
            }
            samples.append(sample)
        
        # --- HISTORY UPDATE (IMMER! Unabhängig vom Filter) ---
        # Damit die Historie lückenlos bleibt, auch wenn wir auf ein Match nicht wetten würden.
        
        w_feats = extract_match_features(row, 'winner')
        l_feats = extract_match_features(row, 'loser')
        
        player_history[winner].append(w_feats)
        player_history[loser].append(l_feats)

    print(f"Preprocessing Done.")
    print(f"Total Matches: {stats['total']}")
    print(f"Kept Samples: {stats['kept']} ({(stats['kept']/stats['total'])*100:.1f}%)")
    print(f"Filtered by Rank: {stats['filtered_rank']}")
    print(f"Filtered by Activity: {stats['filtered_activity']}")
    
    return samples

if __name__ == "__main__":
    df = dl.load_all_matches()
    save_preprocessed_samples(df, "data/preprocessed_samples.pt", n_matches=10, n_recent=15)