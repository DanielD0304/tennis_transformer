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

import math
import random
import pandas as pd
from collections import defaultdict
from functools import lru_cache

from .chronology import chronological, match_dates
from .score import parse_score

# Matches, bei denen ein Spieler schlechter gerankt ist, fließen nicht als Trainings-Sample ein
# (in die Historie schon). Früher Top 200, gelockert für mehr Daten (vor allem WTA).
MAX_RANK = 500
# Fehlt die Spielzeit (tennis-data, ESPN), wird sie für die Müdigkeit aus den Games geschätzt
MINUTES_PER_GAME = 4.4
FATIGUE_WINDOWS = (7, 14)

# Feature names used for each match in the history
# Updated to include ELO and opponent rank features
FEATURE_NAMES = [
    'won', 'rank', 'aces', 'double_faults', 'first_serve_pct', 
    'days_since_match', 'opponent_elo', 'opponent_rank', 'own_elo',
    'bp_convert_pct', 'bp_save_pct', 'return_pts_won_pct', 'service_pts_won_pct',
    'h2h_win_pct',
    'surf_hard', 'surf_clay', 'surf_grass', 'surf_other',
    'has_stats',
    # Aus dem Spielstand (alle Quellen, auch live)
    'game_share', 'set_share', 'tiebreaks', 'n_sets', 'minutes', 'has_minutes', 'has_score', 'retired',
    # Match-Umstände
    'best_of_5', 'indoor', 'home',
    # Kontext des vorherzusagenden Matches (wie h2h_win_pct in jedes Historien-Match geschrieben)
    'fatigue_matches_7d', 'fatigue_sets_7d', 'fatigue_minutes_7d',
    'fatigue_matches_14d', 'fatigue_sets_14d', 'fatigue_minutes_14d',
    'age', 'left_handed', 'ctx_home',
]
# Match-Features außerhalb der Sequenzen (config.extra_dim), aus Sicht von Spieler A.
# elo2_logit: logit der Elo2-Siegwahrscheinlichkeit (elo.elo2_features), 0 wenn unbekannt.
EXTRA_FEATURES = ['elo2_logit']


def elo2_logit(p_winner, a_is_winner):
    """logit(P(A gewinnt)) aus der Elo2-Wahrscheinlichkeit des Siegers."""
    try:
        p = float(p_winner)
    except (TypeError, ValueError):
        return 0.0
    if not 0.0 < p < 1.0:
        return 0.0
    z = math.log(p / (1 - p))
    return z if a_is_winner else -z


# Features, die aus der Aufschlag-Statistik kommen. Fehlen die Stats (tennis-data, ESPN),
# sind sie 0 und has_stats ist 0, damit das Modell "keine Daten" von "0 Aces" unterscheiden kann.
STAT_FEATURES = [
    'aces', 'double_faults', 'first_serve_pct',
    'bp_convert_pct', 'bp_save_pct', 'return_pts_won_pct', 'service_pts_won_pct'
]
SURFACE_MAP = {
    'Hard': [1, 0, 0, 0],
    'Clay': [0, 1, 0, 0],
    'Grass': [0, 0, 1, 0],
    'Carpet': [0, 0, 0, 1]
}
DEFAULT_SURF = [1, 0, 0, 0]

def is_walkover(row):
    """Kampflose Siege sind keine gespielten Matches und fließen nicht in Historie/ELO ein."""
    return str(row.get('status', '')) == 'W/O' or 'W/O' in str(row.get('score', ''))


@lru_cache(maxsize=200_000)
def _score(score):
    return parse_score(score)


def _num(value, default=0.0):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(v) else v


def _text(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ''
    return str(value)


def player_key(row, name):
    """Historien, H2H und Elo laufen getrennt pro Tour."""
    return (_text(row.get('tour')) or 'atp', name)


def is_home(ioc, tourney_ioc):
    return 1 if ioc and tourney_ioc and ioc == tourney_ioc else 0


def age_years(dob, current_date):
    """Alter am Matchtag in Jahren (None, wenn das Geburtsdatum fehlt)."""
    dob = _text(dob)[:8]
    if len(dob) != 8 or current_date is None or pd.isna(current_date) or not dob.isdigit():
        return None
    try:
        born = pd.Timestamp(f"{dob[:4]}-{dob[4:6]}-{dob[6:8]}")
    except ValueError:
        return None
    return (current_date - born).days / 365.25


def fatigue(history, current_date, days):
    """(Matches, Sätze, Minuten) in den letzten `days` Tagen VOR current_date (nur Historie)."""
    if not history or current_date is None or pd.isna(current_date):
        return 0, 0, 0.0
    threshold = current_date - pd.Timedelta(days=days)
    n = sets = minutes = 0
    for match in reversed(history):
        d = match.get('_date_obj')
        if d is None or pd.isna(d):
            continue
        if d < threshold:
            break
        if d > current_date:
            continue  # darf nicht vorkommen (Historie liegt vor dem Match), sicherheitshalber
        n += 1
        sets += match.get('_sets', 0)
        minutes += match.get('_minutes', 0.0)
    return n, sets, minutes


def context_features(history, current_date, h2h_pct, dob=None, hand=None, home=0):
    """Werte zum vorherzusagenden Match, die in jedes Historien-Match geschrieben werden."""
    ctx = {'h2h_win_pct': h2h_pct}
    for days in FATIGUE_WINDOWS:
        n, sets, minutes = fatigue(history, current_date, days)
        ctx[f'fatigue_matches_{days}d'] = n / 5.0
        ctx[f'fatigue_sets_{days}d'] = sets / 15.0
        ctx[f'fatigue_minutes_{days}d'] = minutes / 600.0
    age = age_years(dob, current_date)
    ctx['age'] = age / 40.0 if age is not None else 0.0
    ctx['left_handed'] = 1 if _text(hand) == 'L' else 0
    ctx['ctx_home'] = home
    return ctx


def inject_context(matches, ctx):
    return [{**m, **ctx} for m in matches]


def safe_div(a, b):
    return a / b if b != 0 else 0.0

def get_h2h_pct(p1, p2, h2h_dict):
    """Gibt die Win% von p1 gegen p2 zurück."""
    key = tuple(sorted((p1, p2)))
    stats = h2h_dict.get(key, {'p1': None, 'wins1': 0, 'wins2': 0})
    
    if stats['p1'] is None: return 0.5 # Kein H2H bekannt
    
    # Wer ist wer im gespeicherten Dict?
    if p1 == stats['p1']:
        w, l = stats['wins1'], stats['wins2']
    else:
        w, l = stats['wins2'], stats['wins1']
        
    total = w + l
    return w / total if total > 0 else 0.5



def extract_match_features(row, player_role, current_h2h_pct):
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
        opp_prefix = "l_"
        rank_col = "winner_rank"
        opp_rank_col = "loser_rank"
        won = 1
        opp_elo_col = "loser_elo"
        own_elo_col = "winner_elo"
    else:
        prefix = "l_"
        opp_prefix = "w_"
        rank_col = "loser_rank"
        opp_rank_col = "winner_rank"
        won = 0
        opp_elo_col = "winner_elo"
        own_elo_col = "loser_elo"
    side = "winner" if won else "loser"
    
    # --- Feature Extraction ---
    
    # 1. Basics
    rank = math.log(row[rank_col] + 1) if row[rank_col] > 0 else 0
    opp_rank = math.log(row[opp_rank_col] + 1) if row[opp_rank_col] > 0 else 0
    
    opp_elo = row.get(opp_elo_col, 1500.0) / 2000.0
    own_elo = row.get(own_elo_col, 1500.0) / 2000.0

    # 4. Stats
    aces = row[prefix + 'ace']
    dfs = row[prefix + 'df']
    svpt = row[prefix + 'svpt']
    first_in = row[prefix + '1stIn']
    first_serve_pct = first_in / svpt if svpt != 0 else 0

    srv_pts_won = row[prefix + '1stWon'] + row[prefix + '2ndWon']
    service_pts_won_pct = safe_div(srv_pts_won, svpt)
    
    opp_svpt = row[opp_prefix + 'svpt']
    opp_srv_pts_won = row[opp_prefix + '1stWon'] + row[opp_prefix + '2ndWon']
    return_pts_won = opp_svpt - opp_srv_pts_won
    return_pts_won_pct = safe_div(return_pts_won, opp_svpt)
    
    bp_saved = row[prefix + 'bpSaved']
    bp_faced = row[prefix + 'bpFaced']
    bp_save_pct = safe_div(bp_saved, bp_faced)
    
    opp_bp_faced = row[opp_prefix + 'bpFaced']
    opp_bp_saved = row[opp_prefix + 'bpSaved']
    bp_converted = opp_bp_faced - opp_bp_saved
    bp_convert_pct = safe_div(bp_converted, opp_bp_faced)

    # 4. Surface One-Hot
    surf_vec = SURFACE_MAP.get(row['surface'], DEFAULT_SURF)

    # 5. Spielstand (alle Zahlen im Score aus Sicht des Siegers)
    score = _score(_text(row.get('score')))
    best_of = int(_num(row.get('best_of'), 3)) or 3
    minutes = _num(row.get('minutes'))
    games = score.games_w + score.games_l
    est_minutes = minutes if minutes > 0 else games * MINUTES_PER_GAME
    
    return {
        'won': won,
        'rank': rank,
        'aces': aces,
        'double_faults': dfs,
        'first_serve_pct': first_serve_pct,
        'days_since_match': 0.0, # Wird in pad_sequence gesetzt
        'opponent_elo': opp_elo,
        'opponent_rank': opp_rank,
        'own_elo': own_elo,
        'bp_convert_pct': bp_convert_pct,
        'bp_save_pct': bp_save_pct,
        'return_pts_won_pct': return_pts_won_pct,
        'service_pts_won_pct': service_pts_won_pct,
        'h2h_win_pct': current_h2h_pct,
        'surf_hard': surf_vec[0],
        'surf_clay': surf_vec[1],
        'surf_grass': surf_vec[2],
        'surf_other': surf_vec[3],
        'has_stats': 1 if svpt > 0 else 0,
        'game_share': score.game_share(bool(won)),
        'set_share': score.set_share(bool(won)),
        'tiebreaks': score.tiebreaks / 3.0,
        'n_sets': score.n_sets / best_of,
        'minutes': minutes / 180.0,
        'has_minutes': 1 if minutes > 0 else 0,
        'has_score': 1 if score.valid else 0,
        'retired': 1 if score.retired or str(row.get('status', '')) in ('RET', 'DEF') else 0,
        'best_of_5': 1 if best_of == 5 else 0,
        'indoor': 1 if _text(row.get('court')) == 'Indoor' else 0,
        'home': is_home(_text(row.get(f'{side}_ioc')), _text(row.get('tourney_ioc'))),

        '_date_obj': row.get('match_date_obj', None),
        '_surface': row['surface'],
        '_sets': score.n_sets,
        '_minutes': est_minutes,
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
        if current_date is not None and item.get('_date_obj') is not None:
            delta = current_date - item['_date_obj']
            days = max(0, delta.days)
            new_item['days_since_match'] = math.log(1 + days)
        else:
            new_item['days_since_match'] = 0.0
            
        for k in [k for k in new_item if k.startswith('_')]:
            new_item.pop(k)
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

def create_training_samples(df, n_matches=10, n_recent=15, seed=42, max_rank=MAX_RANK):
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
        max_rank (int): Beide Spieler müssen mindestens so gut gerankt sein (Standard MAX_RANK)

    Historien und H2H laufen getrennt pro Tour, Reihenfolge nach echtem Matchdatum (chronology.py).
    
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
    
    # 1. Echtes Matchdatum (sonst Turnierstart) & Sortierung
    df['match_date_obj'] = match_dates(df)
    df = chronological(df)

    player_history = defaultdict(list)
    
    h2h_tracker = {}
    samples = []
    rng = random.Random(seed)  # reproduzierbare A/B-Zuordnung
    
    for index, row in df.iterrows():
        if is_walkover(row):
            continue
        winner = player_key(row, row['winner_name'])
        loser = player_key(row, row['loser_name'])
        surface = row['surface']
        current_date = row['match_date_obj']
        year = int(str(row['tourney_date'])[:4])
        
        # H2H Key
        h2h_key = tuple(sorted((winner, loser)))
        if h2h_key not in h2h_tracker:
            h2h_tracker[h2h_key] = {'p1': h2h_key[0], 'wins1': 0, 'wins2': 0}
            
        win_h2h_pct = get_h2h_pct(winner, loser, h2h_tracker)
        los_h2h_pct = get_h2h_pct(loser, winner, h2h_tracker)
        
        # Sample Erstellung (nur wenn relevant)
        r_w = row['winner_rank'] if not pd.isna(row['winner_rank']) else 9999
        r_l = row['loser_rank'] if not pd.isna(row['loser_rank']) else 9999
        
        if r_w <= max_rank and r_l <= max_rank:
            # Check Activity
            pA_hist = player_history.get(winner, [])
            pB_hist = player_history.get(loser, [])
            
            has_history = len(pA_hist) >= 5 and len(pB_hist) >= 5
            
            pA_active = count_matches_in_period(pA_hist, current_date, days=90) >= 3
            pB_active = count_matches_in_period(pB_hist, current_date, days=90) >= 3
            
            if has_history and pA_active and pB_active:
                tourney_ioc = _text(row.get('tourney_ioc'))

                def ctx(p, side, h2h_val):
                    return context_features(player_history.get(p, []), current_date, h2h_val,
                                            dob=row.get(f'{side}_dob'), hand=row.get(f'{side}_hand'),
                                            home=is_home(_text(row.get(f'{side}_ioc')), tourney_ioc))

                if rng.random() > 0.5:
                    pA, pB = winner, loser
                    label = 1
                    odds_a = row.get('B365W', 0.0)
                    odds_b = row.get('B365L', 0.0)
                    pA_ctx, pB_ctx = ctx(winner, 'winner', win_h2h_pct), ctx(loser, 'loser', los_h2h_pct)
                else:
                    pA, pB = loser, winner
                    label = 0
                    odds_a = row.get('B365L', 0.0)
                    odds_b = row.get('B365W', 0.0)
                    pA_ctx, pB_ctx = ctx(loser, 'loser', los_h2h_pct), ctx(winner, 'winner', win_h2h_pct)
                
                def get_hist_with_context(p, surf, context):
                    full = player_history.get(p, [])
                    rec_matches = full[-n_recent:][::-1]
                    surf_matches = [m for m in full if m['_surface'] == surf][-n_matches:][::-1]
                    # inject_context kopiert die Dicts, die Historie selbst bleibt unverändert
                    return inject_context(surf_matches, context), inject_context(rec_matches, context)
                
                pA_surf_hist, pA_rec_hist = get_hist_with_context(pA, surface, pA_ctx)
                pB_surf_hist, pB_rec_hist = get_hist_with_context(pB, surface, pB_ctx)
                
                pA_surf, pA_s_mask = pad_sequence(pA_surf_hist, n_matches, current_date)
                pA_rec, pA_r_mask = pad_sequence(pA_rec_hist, n_recent, current_date)
                pB_surf, pB_s_mask = pad_sequence(pB_surf_hist, n_matches, current_date)
                pB_rec, pB_r_mask = pad_sequence(pB_rec_hist, n_recent, current_date)
                
                seg_ids = [1]*n_matches + [2]*n_recent + [3]*n_matches + [4]*n_recent
                
                sample = {
                    'player_a_surface': pA_surf, 'player_a_surface_mask': pA_s_mask,
                    'player_a_recent': pA_rec, 'player_a_recent_mask': pA_r_mask,
                    'player_b_surface': pB_surf, 'player_b_surface_mask': pB_s_mask,
                    'player_b_recent': pB_rec, 'player_b_recent_mask': pB_r_mask,
                    'segment_ids': seg_ids, 'label': label,
                    'year': year,
                    'match_index': index,  # Zeile in df, damit die Evaluation Samples Matches zuordnen kann
                    'elo2_logit': elo2_logit(row.get('elo2_p_winner'), label == 1),
                    'odds_a': odds_a if not pd.isna(odds_a) else 0.0,
                    'odds_b': odds_b if not pd.isna(odds_b) else 0.0
                }
                samples.append(sample)

        # Update History & H2H (NACH dem Sample)
        w_feats = extract_match_features(row, 'winner', win_h2h_pct)
        l_feats = extract_match_features(row, 'loser', los_h2h_pct)
        
        player_history[winner].append(w_feats)
        player_history[loser].append(l_feats)
        
        # Update H2H Tracker
        entry = h2h_tracker[h2h_key]
        if winner == entry['p1']:
            entry['wins1'] += 1
        else:
            entry['wins2'] += 1
            
    print(f"Preprocessing Done. Created {len(samples)} samples.")
    return samples

def save_preprocessed_samples(df, out_path, n_matches=10, n_recent=15, max_rank=MAX_RANK):
    """
    Führt das Preprocessing einmal aus und speichert die Samples als .pt-Datei.
    Args:
        df (pd.DataFrame): Rohdaten
        out_path (str): Speicherpfad für .pt-Datei
        n_matches (int): Surface-Matches
        n_recent (int): Recent-Matches
    """
    import torch  # erst hier: Predictor und Paper-Lauf brauchen preprocessing ohne torch

    from .elo import compute_elo_ratings
    df = compute_elo_ratings(df)

    samples = create_training_samples(df, n_matches=n_matches, n_recent=n_recent, max_rank=max_rank)
    torch.save(samples, out_path)
    print(f"Saved {len(samples)} samples to {out_path}")

