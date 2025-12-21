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
"""

from . import data_loader as dl
import random


# Feature names used for each match in the history
FEATURE_NAMES = ['won', 'rank', 'aces', 'double_faults', 'first_serve_pct']


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
    
    return {
        'won': won,
        'rank': row[rank_col],
        'aces': row[prefix + 'ace'],
        'double_faults': row[prefix + 'df'],
        'first_serve_pct': row[prefix + '1stIn'] / row[prefix + 'svpt']
    }


def get_player_surface_history(df, player_name, surface, before_date, n_matches=10):
    """
    Get a player's last N matches on a specific surface before a given date.
    
    Used to capture surface-specific skills and form.
    
    Args:
        df (pd.DataFrame): DataFrame containing match data
        player_name (str): Full name of the player
        surface (str): Surface type ('Hard', 'Clay', 'Grass')
        before_date (int): Date in YYYYMMDD format (matches before this date)
        n_matches (int): Maximum number of matches to retrieve (default: 10)
    
    Returns:
        list[dict]: List of feature dictionaries, most recent first
    """
    df = filter_by_surface(df, surface) 
    df = filter_by_playername(df, player_name)
    df = df[df['tourney_date'] < before_date]
    df = df.sort_values('tourney_date', ascending=False)
    df = df.head(n_matches)
    history = []
    for index, row in df.iterrows():
        if row['winner_name'] == player_name:
            features = extract_match_features(row, 'winner')
        else:
            features = extract_match_features(row, 'loser')
        history.append(features)
    return history


def get_recent_history(df, player_name, before_date, n_recent=15):
    """
    Get a player's last N matches on any surface before a given date.
    
    Used to capture current form regardless of surface.
    
    Args:
        df (pd.DataFrame): DataFrame containing match data
        player_name (str): Full name of the player
        before_date (int): Date in YYYYMMDD format (matches before this date)
        n_recent (int): Maximum number of matches to retrieve (default: 15)
    
    Returns:
        list[dict]: List of feature dictionaries, most recent first
    """
    df = filter_by_playername(df, player_name)
    df = df[df['tourney_date'] < before_date]
    df = df.sort_values('tourney_date', ascending=False)
    df = df.head(n_recent)
    history = []
    for index, row in df.iterrows():
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
        
        if random.random() > 0.5:
            player_a, player_b = winner_name, loser_name
            label = 1
        else:
            player_a, player_b = loser_name, winner_name
            label = 0
        a_history_surface = get_player_surface_history(df, player_a, surface, date)
        a_history_recent = get_recent_history(df, player_a, date)
        b_history_surface = get_player_surface_history(df, player_b, surface, date)
        b_history_recent = get_recent_history(df, player_b, date)
        player_a_surface, player_a_surface_mask = pad_sequence(a_history_surface, n_matches)
        player_a_recent, player_a_recent_mask = pad_sequence(a_history_recent, n_recent)
        player_b_surface, player_b_surface_mask = pad_sequence(b_history_surface, n_matches)
        player_b_recent, player_b_recent_mask = pad_sequence(b_history_recent, n_recent)
        sample = {
            'player_a_surface':  player_a_surface,
            'player_a_surface_mask': player_a_surface_mask,
            'player_a_recent': player_a_recent,
            'player_a_recent_mask': player_a_recent_mask,
            'player_b_surface':  player_b_surface,
            'player_b_surface_mask': player_b_surface_mask,
            'player_b_recent': player_b_recent,
            'player_b_recent_mask': player_b_recent_mask,
            'label': label        
        }
        samples.append(sample)
    return samples
        
    
if __name__ == "__main__":
    df = dl.load_all_matches()
    test_df = df.head(100)
    samples = create_training_samples(test_df)
    print(f"Created {len(samples)} samples")