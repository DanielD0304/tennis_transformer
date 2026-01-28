import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from src.train import train, custom_collate_fn
from src.config import default_config
from src.dataset import DataSet
from src.model import Model

# ======================================================================================
# 1. STRATEGIE: FILTERED BETTING (Höhere Wahrscheinlichkeit + Mindestquote)
# ======================================================================================
def simulate_strategy_filtered(config=default_config, min_odds=1.30):
    """
    Strategie: Wette auf den Spieler mit der höheren Wahrscheinlichkeit,
    ABER NUR wenn die Quote >= min_odds (Standard: 1.30) ist.
    Dies filtert "Müll-Quoten" (1.01-1.29) heraus, die oft unprofitabel sind.
    """
    print("\n" + "="*60)
    print(f"STRATEGIE 1: FILTERED BETTING (Odds >= {min_odds:.2f})")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Modell initialisieren
    model = Model(
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        input_dim=config.input_dim,
        max_len=config.max_len,
        output_dim=config.output_dim
    ).to(device)
    
    # Bestes Modell laden
    if os.path.exists(config.best_model_path):
        model.load_state_dict(torch.load(config.best_model_path, map_location=device))
    else:
        print("Warnung: Kein trainiertes Modell gefunden! Bitte erst trainieren.")
        return
    
    # Daten laden
    if not os.path.exists(config.preprocessed_data_path):
        print("Keine vorverarbeiteten Daten gefunden!")
        return
    
    all_samples = torch.load(config.preprocessed_data_path)
    test_samples = [s for s in all_samples if s['year'] == config.test_year]
    
    if not test_samples:
        print(f"Keine Test-Samples für Jahr {config.test_year}")
        return
    
    dataset = DataSet(test_samples)
    test_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, collate_fn=custom_collate_fn)
    
    # Bankroll Management
    balance = 1000.0
    start_balance = balance
    stake = 10.0
    
    bets = 0
    wins = 0
    skipped = 0
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            # Features auf Device schieben
            features = torch.cat([batch['player_a_surface'], batch['player_a_recent'], 
                                batch['player_b_surface'], batch['player_b_recent']], dim=1).to(device)
            positions = torch.cat([batch['player_a_surface_pos'], batch['player_a_recent_pos'], 
                                 batch['player_b_surface_pos'], batch['player_b_recent_pos']], dim=1).to(device)
            masks = torch.cat([batch['player_a_surface_mask'], batch['player_a_recent_mask'], 
                             batch['player_b_surface_mask'], batch['player_b_recent_mask']], dim=1).to(device)
            
            # CLS Mask hinzufügen
            cls_mask = torch.ones(masks.shape[0], 1, device=masks.device)
            masks = torch.cat([cls_mask, masks], dim=1)
            segment_ids = batch['segment_ids'].to(device)
            
            # Forward Pass
            outputs = model(features, positions, segment_ids, masks)
            probs = torch.softmax(outputs, dim=1)
            
            labels = batch['label'].to(device)
            odds_a = batch['odds_a'].to(device)
            odds_b = batch['odds_b'].to(device)
            
            for i in range(len(labels)):
                prob_a = probs[i][1].item()
                prob_b = probs[i][0].item()
                real_outcome = labels[i].item()
                oda = odds_a[i].item()
                odb = odds_b[i].item()
                
                # Genereller Datenfilter: Ungültige Quoten ignorieren
                if oda <= 1.01 or odb <= 1.01:
                    skipped += 1
                    continue
                
                bet_placed = False
                
                # Wir wetten auf den Spieler, den das Modell favorisiert
                # Aber NUR, wenn die Quote über dem Grenzwert (min_odds) liegt
                if prob_a > prob_b:
                    if oda >= min_odds:
                        balance -= stake
                        bets += 1
                        bet_placed = True
                        if real_outcome == 1:
                            balance += stake * oda
                            wins += 1
                else:
                    if odb >= min_odds:
                        balance -= stake
                        bets += 1
                        bet_placed = True
                        if real_outcome == 0:
                            balance += stake * odb
                            wins += 1
                            
                if not bet_placed:
                    skipped += 1

    # --- ROI / Yield Berechnung ---
    profit = balance - start_balance
    roc = (profit / start_balance) * 100
    
    total_staked = bets * stake
    yield_percent = (profit / total_staked) * 100 if total_staked > 0 else 0.0

    print(f"Endkapital:       {balance:.2f}€")
    print(f"Profit/Verlust:   {profit:.2f}€")
    print(f"ROC (Bankroll):   {roc:.2f}%")
    print(f"YIELD (Echt-ROI): {yield_percent:.2f}%")
    print(f"Wetten:           {bets} (Gefiltert: {skipped})")
    if bets > 0:
        print(f"Win-Rate:         {100*wins/bets:.2f}%")
    print("="*60)


# ======================================================================================
# 2. STRATEGIE: PURE VALUE (Mathematischer Vorteil)
# ======================================================================================
def simulate_strategy_pure_value(config=default_config):
    """
    Strategie: Wette nur, wenn der erwartete Wert (Expected Value) > 1.0 ist.
    Formel: Wahrscheinlichkeit * Quote > 1.0
    """
    print("\n" + "="*60)
    print("STRATEGIE 2: PURE VALUE (Math > Bookie)")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Model(
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        input_dim=config.input_dim,
        max_len=config.max_len,
        output_dim=config.output_dim
    ).to(device)
    
    if os.path.exists(config.best_model_path):
        model.load_state_dict(torch.load(config.best_model_path, map_location=device))
    
    if not os.path.exists(config.preprocessed_data_path):
        return
    
    all_samples = torch.load(config.preprocessed_data_path)
    test_samples = [s for s in all_samples if s['year'] == config.test_year]
    
    if not test_samples: return
    
    dataset = DataSet(test_samples)
    test_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, collate_fn=custom_collate_fn)
    
    balance = 1000.0
    start_balance = balance
    stake = 10.0
    
    bets = 0
    wins = 0
    skipped = 0
    
    min_value = 1.0 
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            features = torch.cat([batch['player_a_surface'], batch['player_a_recent'], 
                                batch['player_b_surface'], batch['player_b_recent']], dim=1).to(device)
            positions = torch.cat([batch['player_a_surface_pos'], batch['player_a_recent_pos'], 
                                 batch['player_b_surface_pos'], batch['player_b_recent_pos']], dim=1).to(device)
            masks = torch.cat([batch['player_a_surface_mask'], batch['player_a_recent_mask'], 
                             batch['player_b_surface_mask'], batch['player_b_recent_mask']], dim=1).to(device)
            cls_mask = torch.ones(masks.shape[0], 1, device=masks.device)
            masks = torch.cat([cls_mask, masks], dim=1)
            segment_ids = batch['segment_ids'].to(device)
            
            outputs = model(features, positions, segment_ids, masks)
            probs = torch.softmax(outputs, dim=1)
            
            labels = batch['label'].to(device)
            odds_a = batch['odds_a'].to(device)
            odds_b = batch['odds_b'].to(device)
            
            for i in range(len(labels)):
                prob_a = probs[i][1].item()
                prob_b = probs[i][0].item()
                real_outcome = labels[i].item()
                oda = odds_a[i].item()
                odb = odds_b[i].item()
                
                if oda <= 1.01 or odb <= 1.01: continue
                
                # Value Berechnung
                ev_a = prob_a * oda
                ev_b = prob_b * odb
                
                bet_placed = False
                
                # Wir setzen auf den höchsten Value, falls er > min_value ist
                if ev_a > min_value and ev_a > ev_b:
                    balance -= stake
                    bets += 1
                    bet_placed = True
                    if real_outcome == 1:
                        balance += stake * oda
                        wins += 1
                elif ev_b > min_value:
                    balance -= stake
                    bets += 1
                    bet_placed = True
                    if real_outcome == 0:
                        balance += stake * odb
                        wins += 1
                
                if not bet_placed:
                    skipped += 1

    profit = balance - start_balance
    roc = (profit / start_balance) * 100
    total_staked = bets * stake
    yield_percent = (profit / total_staked) * 100 if total_staked > 0 else 0.0

    print(f"Endkapital:       {balance:.2f}€")
    print(f"Profit/Verlust:   {profit:.2f}€")
    print(f"ROC (Bankroll):   {roc:.2f}%")
    print(f"YIELD (Echt-ROI): {yield_percent:.2f}%")
    print(f"Wetten:           {bets} (Gefiltert: {skipped})")
    if bets > 0:
        print(f"Win-Rate:         {100*wins/bets:.2f}%")
    print("="*60)


# ======================================================================================
# 3. STRATEGIE: BASELINE (Rangliste)
# ======================================================================================
def simulate_betting_baseline(config=default_config):
    """
    Simuliert Wetten basierend auf der einfachen Strategie:
    "Setze immer auf den Spieler mit dem besseren (niedrigeren) Rang."
    """
    print("\n" + "="*60)
    print("SIMULATION: BASELINE (BETTER RANK WINS)")
    print("="*60)
    
    if not os.path.exists(config.preprocessed_data_path):
        print("Keine Daten gefunden!")
        return

    all_samples = torch.load(config.preprocessed_data_path)
    test_samples = [s for s in all_samples if s['year'] == config.test_year]
    
    if not test_samples: return

    dataset = DataSet(test_samples)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    
    balance = 1000.0
    start_balance = balance
    stake = 10.0
    bets_placed = 0
    correct_bets = 0
    
    with torch.no_grad():
        for batch in loader:
            # Rang extrahieren (Index 1 im Feature Vektor des letzten Matches)
            rank_a = batch['player_a_recent'][:, 0, 1]
            rank_b = batch['player_b_recent'][:, 0, 1]
            
            # Padding (0.0) durch Unendlich ersetzen
            rank_a = torch.where(rank_a == 0, torch.tensor(float('inf')), rank_a)
            rank_b = torch.where(rank_b == 0, torch.tensor(float('inf')), rank_b)
            
            labels = batch['label']
            odds_a = batch['odds_a']
            odds_b = batch['odds_b']
            
            for i in range(len(labels)):
                r_a = rank_a[i].item()
                r_b = rank_b[i].item()
                real_outcome = labels[i].item()
                oda = odds_a[i].item()
                odb = odds_b[i].item()
                
                if oda <= 1.01 or odb <= 1.01: continue
                
                bet_made = False
                won = False
                
                if r_a < r_b: 
                    balance -= stake
                    bets_placed += 1
                    bet_made = True
                    if real_outcome == 1:
                        balance += stake * oda
                        won = True
                elif r_b < r_a:
                    balance -= stake
                    bets_placed += 1
                    bet_made = True
                    if real_outcome == 0:
                        balance += stake * odb
                        won = True
                
                if bet_made and won:
                    correct_bets += 1

    profit = balance - start_balance
    roc = (profit / start_balance) * 100
    total_staked = bets_placed * stake
    yield_percent = (profit / total_staked) * 100 if total_staked > 0 else 0.0
    
    print(f"Endkapital:       {balance:.2f}€")
    print(f"Profit/Verlust:   {profit:.2f}€")
    print(f"ROC (Bankroll):   {roc:.2f}%")
    print(f"YIELD (Echt-ROI): {yield_percent:.2f}%")
    print(f"Wetten:           {bets_placed}")
    if bets_placed > 0:
        print(f"Win-Rate:         {100 * correct_bets / bets_placed:.2f}%")
    print("="*60 + "\n")


# ======================================================================================
# 4. DIAGNOSE: PROBABILITY CALIBRATION
# ======================================================================================
def check_calibration(config=default_config):
    """
    Überprüft, ob die Wahrscheinlichkeiten des Modells der Realität entsprechen.
    Zeigt "Overconfidence" oder "Underconfidence" an.
    """
    print("\n" + "="*60)
    print("DIAGNOSE: PROBABILITY CALIBRATION CHECK")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Model(
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        input_dim=config.input_dim,
        max_len=config.max_len,
        output_dim=config.output_dim
    ).to(device)
    
    if os.path.exists(config.best_model_path):
        model.load_state_dict(torch.load(config.best_model_path, map_location=device))
    else:
        print("Für Diagnose wird ein trainiertes Modell benötigt.")
        return

    if not os.path.exists(config.preprocessed_data_path):
        return
        
    all_samples = torch.load(config.preprocessed_data_path)
    test_samples = [s for s in all_samples if s['year'] == config.test_year]
    dataset = DataSet(test_samples)
    test_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, collate_fn=custom_collate_fn)
    
    all_probs = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            features = torch.cat([batch['player_a_surface'], batch['player_a_recent'], 
                                batch['player_b_surface'], batch['player_b_recent']], dim=1).to(device)
            positions = torch.cat([batch['player_a_surface_pos'], batch['player_a_recent_pos'], 
                                 batch['player_b_surface_pos'], batch['player_b_recent_pos']], dim=1).to(device)
            masks = torch.cat([batch['player_a_surface_mask'], batch['player_a_recent_mask'], 
                             batch['player_b_surface_mask'], batch['player_b_recent_mask']], dim=1).to(device)
            cls_mask = torch.ones(masks.shape[0], 1, device=masks.device)
            masks = torch.cat([cls_mask, masks], dim=1)
            segment_ids = batch['segment_ids'].to(device)
            
            outputs = model(features, positions, segment_ids, masks)
            probs = torch.softmax(outputs, dim=1)[:, 1] # Wahrscheinlichkeit, dass Player A gewinnt
            labels = batch['label'].to(device)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Wahrscheinlichkeits-Bins
    bins = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    print(f"{'Confidence':<15} | {'Count':<8} | {'Real Win Rate':<15} | {'Diff':<10}")
    print("-" * 60)
    
    for i in range(len(bins)-1):
        low, high = bins[i], bins[i+1]
        
        mask = (all_probs >= low) & (all_probs < high)
        if np.sum(mask) == 0: continue
        
        subset_labels = all_labels[mask]
        real_win_rate = np.mean(subset_labels)
        predicted_mean = np.mean(all_probs[mask])
        
        diff = real_win_rate - predicted_mean
        
        print(f"{low:.1f} - {high:.1f}      | {len(subset_labels):<8} | {real_win_rate:.2%}        | {diff:+.2%}")
        
    print("="*60)
    print("Hinweis: Negative 'Diff' bedeutet, das Modell ist zu optimistisch (overconfident).")

def simulate_strategy_sniper(config=default_config):
    """
    STRATEGIE 3: SNIPER (Confidence Window)
    Wette nur, wenn die Modell-Wahrscheinlichkeit im 'Gold-Bereich' liegt.
    Basierend auf Calibration Check: 70% - 80% Konfidenz.
    """
    print("\n" + "="*60)
    print("STRATEGIE 3: SNIPER (Confidence 70% - 80%)")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(config.d_model, config.num_heads, config.num_layers, config.input_dim, config.max_len, config.output_dim).to(device)
    if os.path.exists(config.best_model_path): model.load_state_dict(torch.load(config.best_model_path, map_location=device))
    if not os.path.exists(config.preprocessed_data_path): return
    all_samples = torch.load(config.preprocessed_data_path)
    test_samples = [s for s in all_samples if s['year'] == config.test_year]
    if not test_samples: return
    dataset = DataSet(test_samples)
    test_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, collate_fn=custom_collate_fn)
    
    balance = 1000.0
    start_balance = balance
    stake = 10.0
    bets = 0
    wins = 0
    skipped = 0
    
    min_conf = 0.60
    max_conf = 0.80
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            # ... (Inputs laden wie immer) ...
            features = torch.cat([batch['player_a_surface'], batch['player_a_recent'], 
                                batch['player_b_surface'], batch['player_b_recent']], dim=1).to(device)
            positions = torch.cat([batch['player_a_surface_pos'], batch['player_a_recent_pos'], 
                                 batch['player_b_surface_pos'], batch['player_b_recent_pos']], dim=1).to(device)
            masks = torch.cat([batch['player_a_surface_mask'], batch['player_a_recent_mask'], 
                             batch['player_b_surface_mask'], batch['player_b_recent_mask']], dim=1).to(device)
            cls_mask = torch.ones(masks.shape[0], 1, device=masks.device)
            masks = torch.cat([cls_mask, masks], dim=1)
            segment_ids = batch['segment_ids'].to(device)
            
            outputs = model(features, positions, segment_ids, masks)
            probs = torch.softmax(outputs, dim=1)
            
            labels = batch['label'].to(device)
            odds_a = batch['odds_a'].to(device)
            odds_b = batch['odds_b'].to(device)
            
            for i in range(len(labels)):
                prob_a = probs[i][1].item()
                prob_b = probs[i][0].item()
                real_outcome = labels[i].item()
                oda = odds_a[i].item()
                odb = odds_b[i].item()
                
                if oda <= 1.01 or odb <= 1.01: continue
                
                bet_placed = False
                
                # Prüfe Player A
                if prob_a > prob_b:
                    # Nur wetten, wenn wir im Gold-Bereich sind
                    if min_conf <= prob_a <= max_conf:
                        balance -= stake
                        bets += 1
                        bet_placed = True
                        if real_outcome == 1:
                            balance += stake * oda
                            wins += 1
                # Prüfe Player B
                else:
                    if min_conf <= prob_b <= max_conf:
                        balance -= stake
                        bets += 1
                        bet_placed = True
                        if real_outcome == 0:
                            balance += stake * odb
                            wins += 1
                            
                if not bet_placed:
                    skipped += 1

    # ROI Berechnung (wie gehabt)
    profit = balance - start_balance
    roc = (profit / start_balance) * 100
    total_staked = bets * stake
    yield_percent = (profit / total_staked) * 100 if total_staked > 0 else 0.0

    print(f"Endkapital:       {balance:.2f}€")
    print(f"Profit/Verlust:   {profit:.2f}€")
    print(f"ROC (Bankroll):   {roc:.2f}%")
    print(f"YIELD (Echt-ROI): {yield_percent:.2f}%")
    print(f"Wetten:           {bets} (Gefiltert: {skipped})")
    if bets > 0:
        print(f"Win-Rate:         {100*wins/bets:.2f}%")
    print("="*60)
    
    
def compute_baseline_accuracy(config=default_config):
    print("\n" + "="*60)
    print("Computing Baseline Accuracy (Better Ranking Wins)")
    print("="*60)
    
    # Sicherstellen, dass wir im richtigen Verzeichnis sind
    if not torch.cuda.is_available(): # Kleiner Hack, um Pfade lokal sauber zu halten
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Preprocessing, falls nötig
    if not os.path.exists(config.preprocessed_data_path):
        print(f"Preprocessed data not found at {config.preprocessed_data_path}")
        print("Running preprocessing pipeline...\n")
        from src import data_loader as dl
        from src import preprocessing as pre
        
        years = list(range(config.data_years_start, config.data_years_end + 1))
        raw_data = dl.load_all_matches(years)
        
        # ELO berechnen
        from src.elo import compute_elo_ratings
        raw_data = compute_elo_ratings(raw_data)
        
        all_samples = pre.create_training_samples(
            raw_data, 
            n_matches=config.n_surface_matches,
            n_recent=config.n_recent_matches
        )
        
        os.makedirs(os.path.dirname(config.preprocessed_data_path), exist_ok=True)
        torch.save(all_samples, config.preprocessed_data_path)
        print(f"Preprocessing complete!\n")
    else:
        all_samples = torch.load(config.preprocessed_data_path)
    
    test_samples = [s for s in all_samples if s['year'] == config.test_year]
    print(f"Using {len(test_samples)} test samples from year {config.test_year}")
    
    if len(test_samples) == 0:
        return 0.0
    
    dataset = DataSet(test_samples)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            rank_a = batch['player_a_recent'][:, 0, 1]
            rank_b = batch['player_b_recent'][:, 0, 1]
            
            rank_a = torch.where(rank_a == 0, torch.tensor(float('inf')), rank_a)
            rank_b = torch.where(rank_b == 0, torch.tensor(float('inf')), rank_b)
            
            labels = batch['label']
            
            # Prediction: Player mit niedrigerem Rang gewinnt
            predictions = torch.where(rank_a < rank_b, 1, 0)
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    baseline_accuracy = 100 * correct / total
    print(f"Total Samples: {total}")
    print(f"Baseline Accuracy (Ranking): {baseline_accuracy:.2f}%")
    print("="*60 + "\n")
    
    return baseline_accuracy


# ======================================================================================
# MAIN EXECUTION
# ======================================================================================
def main():
    
    compute_baseline_accuracy()
    
    # 2. Modell trainieren
    print("Training Transformer Model...\n")
    train()
    
    # 3. Simulationen & Auswertung
    simulate_betting_baseline()
    
    # Strategie 1 mit Filter (z.B. alles unter Quote 1.30 ignorieren)
    simulate_strategy_sniper()
    
    # Strategie 2 (Value Betting)
    simulate_strategy_pure_value()
    
    # 4. Diagnose (Calibration Check)
    check_calibration()

if __name__ == "__main__":
    main()