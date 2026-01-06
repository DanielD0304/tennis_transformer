import torch
def save_preprocessed_samples(df, out_path, n_matches=10, n_recent=15):
    """
    Führt das Preprocessing einmal aus und speichert die Samples als .pt-Datei.
    Args:
        df (pd.DataFrame): Rohdaten
        out_path (str): Speicherpfad für .pt-Datei
        n_matches (int): Surface-Matches
        n_recent (int): Recent-Matches
    """
    samples = create_training_samples(df, n_matches=n_matches, n_recent=n_recent)
    torch.save(samples, out_path)
    print(f"Saved {len(samples)} samples to {out_path}")
"""
Preprocessing module for Tennis Transformer.

This module transforms raw ATP match data into training samples for the Transformer model.
Each sample contains the match history of two players (surface-specific and recent form)
along with attention masks for variable-length sequences.

Functions:
    - filter_by_surface: Filter matches by court surface
    - filter_by_playername: Filter matches involving a specific player
    - extract_match_features: Extract feature dict from a single match
    - get_player_surface_history: Get player's last N matches on a surface
    - get_recent_history: Get player's last N matches (any surface)
    - pad_sequence: Pad sequences to fixed length with attention mask
    - create_training_samples: Generate training samples from match data
    - days_since_match: log(1+days) als Feature
"""

from . import data_loader as dl
import random


# Feature names used for each match in the history
FEATURE_NAMES = ['won', 'rank', 'aces', 'double_faults', 'first_serve_pct', 'days_since_match']


def filter_by_surface(df, surface):
    """
    Filter DataFrame to only include matches on a specific surface.
    
    Args:
        df (pd.DataFrame): DataFrame containing match data
        surface (str): Surface type ('Hard', 'Clay', 'Grass')
    
    Returns:
        pd.DataFrame: Filtered DataFrame with only matches on specified surface
    """
    return df[df['surface'] == surface]


def filter_by_playername(df, player_name):
    """
    Filter DataFrame to only include matches where a specific player participated.
    
    Args:
        df (pd.DataFrame): DataFrame containing match data
        player_name (str): Full name of the player
    
    Returns:
        pd.DataFrame: Filtered DataFrame with only matches involving the player
    """
    return df[(df['winner_name'] == player_name) | (df['loser_name'] == player_name)]


def extract_match_features(row, player_role):
    """
    Extract relevant features from a match row for a specific player.
    
    Args:
        row (pd.Series): A single row from the matches DataFrame
        player_role (str): Either 'winner' or 'loser'
    
    Returns:
        dict: Dictionary containing match features:
            - won (int): 1 if player won, 0 if lost
            - rank (int): Player's ranking at time of match
            - aces (int): Number of aces served
            - double_faults (int): Number of double faults
            - first_serve_pct (float): First serve percentage
    """
    if player_role == "winner":
        prefix = "w_"
        rank_col = "winner_rank"
        won = 1
    else:
        prefix = "l_"
        rank_col = "loser_rank"
        won = 0
    
    import math
    # Rank kann NaN oder 0 sein, daher robust log(rank+1)
    rank_value = row[rank_col]
    if rank_value is None or (isinstance(rank_value, float) and math.isnan(rank_value)) or rank_value < 1:
        log_rank = 0.0
    else:
        log_rank = math.log(rank_value + 1)
    # days_since_match: log(1 + days)
    if 'current_match_date' in row and row['current_match_date'] is not None:
        days = max(0, (row['current_match_date'] - row['tourney_date']) // 1)
        log_days = math.log(1 + days)
    else:
        log_days = 0.0
    return {
        'won': won,
        'rank': log_rank,
        'aces': row[prefix + 'ace'],
        'double_faults': row[prefix + 'df'],
        'first_serve_pct': row[prefix + '1stIn'] / row[prefix + 'svpt'] if row[prefix + 'svpt'] != 0 else 0,
        'days_since_match': log_days
    }


def get_player_surface_history(df, player_name, surface, before_date, current_tourney_id=None, current_match_num=None, n_matches=10):
    """
    Get a player's last N matches on a specific surface before a given date.
    
    Used to capture surface-specific skills and form.
    
    Args:
        df (pd.DataFrame): DataFrame containing match data
        player_name (str): Full name of the player
        surface (str): Surface type ('Hard', 'Clay', 'Grass')
        before_date (int): Date in YYYYMMDD format (matches before this date)
        current_tourney_id (str): Current tournament ID to exclude current match (default: None)
        current_match_num (int): Current match number to exclude (default: None)
        n_matches (int): Maximum number of matches to retrieve (default: 10)
    
    Returns:
        list[dict]: List of feature dictionaries, most recent first
    """
    df = filter_by_surface(df, surface) 
    df = filter_by_playername(df, player_name)
    
    # FIX: Striktes Filtern mit match_num um Data Leakage zu vermeiden
    # Das aktuelle Match darf NICHT in der Historie sein!
    if current_tourney_id is not None and current_match_num is not None and 'tourney_id' in df.columns and 'match_num' in df.columns:
        # Bedingung:
        # 1. Datum ist strikt kleiner (Vergangenheit) ODER
        # 2. Gleiches Turnier mit match_num < current_match_num (vorherige Runde des Turniers)
        mask = (df['tourney_date'] < before_date) | \
               ((df['tourney_id'] == current_tourney_id) & (df['match_num'] < current_match_num))
        df = df[mask]
    else:
        # Fallback wenn match_num fehlt
        df = df[df['tourney_date'] < before_date]
    
    df = df.sort_values(['tourney_date', 'match_num'] if 'match_num' in df.columns else 'tourney_date', ascending=False)
    df = df.head(n_matches)
    history = []
    for index, row in df.iterrows():
        row = row.copy()
        row['current_match_date'] = before_date
        if row['winner_name'] == player_name:
            features = extract_match_features(row, 'winner')
        else:
            features = extract_match_features(row, 'loser')
        history.append(features)
    return history


def get_recent_history(df, player_name, before_date, current_tourney_id=None, current_match_num=None, n_recent=15):
    """
    Get a player's last N matches on any surface before a given date.
    
    Used to capture current form regardless of surface.
    
    Args:
        df (pd.DataFrame): DataFrame containing match data
        player_name (str): Full name of the player
        before_date (int): Date in YYYYMMDD format (matches before this date)
        current_tourney_id (str): Current tournament ID to exclude current match (default: None)
        current_match_num (int): Current match number to exclude (default: None)
        n_recent (int): Maximum number of matches to retrieve (default: 15)
    
    Returns:
        list[dict]: List of feature dictionaries, most recent first
    """
    df = filter_by_playername(df, player_name)
    
    # FIX: Striktes Filtern mit match_num um Data Leakage zu vermeiden
    # Das aktuelle Match darf NICHT in der Historie sein!
    if current_tourney_id is not None and current_match_num is not None and 'tourney_id' in df.columns and 'match_num' in df.columns:
        # Bedingung:
        # 1. Datum ist strikt kleiner (Vergangenheit) ODER
        # 2. Gleiches Turnier mit match_num < current_match_num (vorherige Runde des Turniers)
        mask = (df['tourney_date'] < before_date) | \
               ((df['tourney_id'] == current_tourney_id) & (df['match_num'] < current_match_num))
        df = df[mask]
    else:
        # Fallback wenn match_num fehlt
        df = df[df['tourney_date'] < before_date]
    
    df = df.sort_values(['tourney_date', 'match_num'] if 'match_num' in df.columns else 'tourney_date', ascending=False)
    df = df.head(n_recent)
    history = []
    for index, row in df.iterrows():
        row = row.copy()
        row['current_match_date'] = before_date
        if row['winner_name'] == player_name:
            features = extract_match_features(row, 'winner')
        else:
            features = extract_match_features(row, 'loser')
        history.append(features)
    return history


def pad_sequence(sequence, max_len):
    """
    Pad a sequence to a fixed length and create an attention mask.
    
    Pads shorter sequences with zero-valued feature dictionaries.
    The attention mask indicates which positions contain real data (1)
    and which are padding (0).
    
    Args:
        sequence (list[dict]): List of feature dictionaries
        max_len (int): Target length for padding
    
    Returns:
        tuple: (padded_sequence, mask)
            - padded_sequence (list[dict]): Sequence padded to max_len
            - mask (list[int]): Attention mask (1=real, 0=padding)
    """
    actual_len = len(sequence)
    pad_len = max_len - actual_len
    mask = [1] * actual_len + [0] * pad_len
    zero_dict = {
        'won': 0,
        'rank': 0,
        'aces': 0,
        'double_faults': 0,
        'first_serve_pct': 0
    }
    padded = sequence + [zero_dict.copy() for _ in range(pad_len)]
    return padded, mask


def create_training_samples(df, n_matches=10, n_recent=15):
    """
    Create training samples from match data for the Transformer model.
    
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
    samples = []
    for index, row in df.iterrows():
        if index % 1000 == 0:
            print(f"Processing match {index}...")
        winner_name = row['winner_name']
        loser_name = row['loser_name']
        date = row['tourney_date']
        surface = row['surface']
        tourney_id = row.get('tourney_id', None)
        match_num = row.get('match_num', None)  # Extract match_num to prevent data leakage
        match_year = int(str(date)[:4])
        
        if random.random() > 0.5:
            player_a, player_b = winner_name, loser_name
            label = 1
        else:
            player_a, player_b = loser_name, winner_name
            label = 0
        a_history_surface = get_player_surface_history(df, player_a, surface, date, tourney_id, match_num)
        a_history_recent = get_recent_history(df, player_a, date, tourney_id, match_num)
        b_history_surface = get_player_surface_history(df, player_b, surface, date, tourney_id, match_num)
        b_history_recent = get_recent_history(df, player_b, date, tourney_id, match_num)
        player_a_surface, player_a_surface_mask = pad_sequence(a_history_surface, n_matches)
        player_a_recent, player_a_recent_mask = pad_sequence(a_history_recent, n_recent)
        player_b_surface, player_b_surface_mask = pad_sequence(b_history_surface, n_matches)
        player_b_recent, player_b_recent_mask = pad_sequence(b_history_recent, n_recent)

        # Segment-IDs: 1 = A-Surface, 2 = A-Recent, 3 = B-Surface, 4 = B-Recent
        seg_a_surface = [1] * n_matches
        seg_a_recent = [2] * n_recent
        seg_b_surface = [3] * n_matches
        seg_b_recent = [4] * n_recent
        segment_ids = seg_a_surface + seg_a_recent + seg_b_surface + seg_b_recent  # Länge: 2*n_matches + 2*n_recent

        sample = {
            'player_a_surface':  player_a_surface,
            'player_a_surface_mask': player_a_surface_mask,
            'player_a_recent': player_a_recent,
            'player_a_recent_mask': player_a_recent_mask,
            'player_b_surface':  player_b_surface,
            'player_b_surface_mask': player_b_surface_mask,
            'player_b_recent': player_b_recent,
            'player_b_recent_mask': player_b_recent_mask,
            'segment_ids': segment_ids,
            'label': label,
            'year': match_year     
        }
        samples.append(sample)
    return samples
        
    
if __name__ == "__main__":
    df = dl.load_all_matches()
    # Beispiel: alle Daten preprocessen und speichern
    save_preprocessed_samples(df, "preprocessed_samples.pt", n_matches=10, n_recent=15)