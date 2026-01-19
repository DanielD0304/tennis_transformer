import torch
import os
from src.train import train
from src.config import default_config
from src.dataset import DataSet
from torch.utils.data import DataLoader
from src.train import custom_collate_fn
from src.model import Model



def simulate_strategy_always_bet(config=default_config):
    """
    Strategie 1: Wette auf JEDES Spiel auf den Spieler mit der höheren Wahrscheinlichkeit.
    Egal wie die Quote ist (solange sie existiert).
    """
    print("\n" + "="*60)
    print("STRATEGIE 1: IMMER WETTEN (Highest Probability)")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = Model(
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        input_dim=config.input_dim,
        max_len=config.max_len,
        output_dim=config.output_dim
    ).to(device)
    model.load_state_dict(torch.load(config.best_model_path, map_location=device))
    
    # Load test data
    if not os.path.exists(config.preprocessed_data_path):
        print("No data found!")
        return
    
    all_samples = torch.load(config.preprocessed_data_path)
    test_samples = [s for s in all_samples if s['year'] == config.test_year]
    
    if not test_samples:
        print(f"No test samples for year {config.test_year}")
        return
    
    dataset = DataSet(test_samples)
    test_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, collate_fn=custom_collate_fn)
    
    balance = 1000.0
    start_balance = balance
    stake = 10.0
    
    bets = 0
    wins = 0
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            # Inputs auf Device
            features = torch.cat([batch['player_a_surface'], batch['player_a_recent'], 
                                batch['player_b_surface'], batch['player_b_recent']], dim=1).to(device)
            positions = torch.cat([batch['player_a_surface_pos'], batch['player_a_recent_pos'], 
                                 batch['player_b_surface_pos'], batch['player_b_recent_pos']], dim=1).to(device)
            masks = torch.cat([batch['player_a_surface_mask'], batch['player_a_recent_mask'], 
                             batch['player_b_surface_mask'], batch['player_b_recent_mask']], dim=1).to(device)
            cls_mask = torch.ones(masks.shape[0], 1, device=masks.device)
            masks = torch.cat([cls_mask, masks], dim=1)
            segment_ids = batch['segment_ids'].to(device)
            
            # Forward Pass
            outputs = model(features, positions, segment_ids, masks)
            probs = torch.softmax(outputs, dim=1)
            
            # Daten für Auswertung
            labels = batch['label'].to(device)
            odds_a = batch['odds_a'].to(device)
            odds_b = batch['odds_b'].to(device)
            
            for i in range(len(labels)):
                prob_a = probs[i][1].item()
                prob_b = probs[i][0].item()
                real_outcome = labels[i].item()
                oda = odds_a[i].item()
                odb = odds_b[i].item()
                
                # Ohne Quote keine Wette
                if oda <= 1.01 or odb <= 1.01: continue
                
                bets += 1
                balance -= stake
                
                # Wir wetten einfach auf den, der wahrscheinlicher ist
                if prob_a > prob_b:
                    # Wette auf A
                    if real_outcome == 1:
                        balance += stake * oda
                        wins += 1
                else:
                    # Wette auf B
                    if real_outcome == 0:
                        balance += stake * odb
                        wins += 1

    roi = ((balance - start_balance) / start_balance) * 100
    print(f"Endkapital: {balance:.2f}€")
    print(f"ROI: {roi:.2f}%")
    if bets > 0:
        print(f"Wetten: {bets} (Win-Rate: {100*wins/bets:.2f}%)")
    print("="*60)


def simulate_strategy_pure_value(config=default_config):
    """
    Strategie 2: PURE VALUE.
    Wette nur, wenn (Unsere Wahrscheinlichkeit * Buchmacher Quote) > 1.0
    Das wettet auch auf Außenseiter, wenn die Quote hoch genug ist!
    """
    print("\n" + "="*60)
    print("STRATEGIE 2: PURE VALUE (Math > Bookie)")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = Model(
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        input_dim=config.input_dim,
        max_len=config.max_len,
        output_dim=config.output_dim
    ).to(device)
    model.load_state_dict(torch.load(config.best_model_path, map_location=device))
    
    # Load test data
    if not os.path.exists(config.preprocessed_data_path):
        print("No data found!")
        return
    
    all_samples = torch.load(config.preprocessed_data_path)
    test_samples = [s for s in all_samples if s['year'] == config.test_year]
    
    if not test_samples:
        print(f"No test samples for year {config.test_year}")
        return
    
    dataset = DataSet(test_samples)
    test_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, collate_fn=custom_collate_fn)
    
    balance = 1000.0
    start_balance = balance
    stake = 10.0
    
    bets = 0
    wins = 0
    skipped = 0
    
    # 1.0 = Fair Value. 1.05 = Wir wollen 5% Puffer. 
    # Du kannst hier 1.0 setzen, um ALLES mitzunehmen.
    min_value = 1.0 
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            # Inputs
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
                
                # --- HIER IST DIE MAGIE ---
                # Berechne den erwarteten Wert (Expected Value - EV)
                ev_a = prob_a * oda
                ev_b = prob_b * odb
                
                # Wir wetten auf die Seite mit dem höchsten Value, FALLS Value > 1.0
                bet_placed = False
                
                # Fall 1: A hat Value und mehr Value als B
                if ev_a > min_value and ev_a > ev_b:
                    balance -= stake
                    bets += 1
                    bet_placed = True
                    if real_outcome == 1:
                        balance += stake * oda
                        wins += 1
                        
                # Fall 2: B hat Value (und mehr als A)
                elif ev_b > min_value:
                    balance -= stake
                    bets += 1
                    bet_placed = True
                    if real_outcome == 0:
                        balance += stake * odb
                        wins += 1
                
                if not bet_placed:
                    skipped += 1

    roi = ((balance - start_balance) / start_balance) * 100
    print(f"Endkapital: {balance:.2f}€")
    print(f"ROI: {roi:.2f}%")
    print(f"Wetten: {bets} (Gefiltert: {skipped})")
    if bets > 0:
        print(f"Win-Rate: {100*wins/bets:.2f}%")
    print("="*60)
    
def simulate_betting_baseline(config=default_config):
    """
    Simuliert Wetten basierend auf der einfachen Strategie:
    "Setze immer auf den Spieler mit dem besseren (niedrigeren) Rang."
    """
    print("\n" + "="*60)
    print("SIMULATION: BASELINE (BETTER RANK WINS)")
    print("="*60)
    
    # Pfad korrigieren und Daten laden
    if not torch.cuda.is_available():
         os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if not os.path.exists(config.preprocessed_data_path):
        print("Keine Daten gefunden!")
        return

    all_samples = torch.load(config.preprocessed_data_path)
    test_samples = [s for s in all_samples if s['year'] == config.test_year]
    
    if not test_samples:
        print(f"Keine Test-Samples für Jahr {config.test_year}")
        return

    dataset = DataSet(test_samples)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    
    balance = 1000.0
    initial_balance = balance
    stake = 10.0
    
    bets_placed = 0
    correct_bets = 0
    
    print(f"Simuliere Wetten auf {len(test_samples)} Matches (Jahr {config.test_year})...")
    
    with torch.no_grad():
        for batch in loader:
            # Wir nehmen den Rang aus dem letzten Match der Historie als Proxy für den aktuellen Rang
            # Index 1 ist der Rank (siehe preprocessing FEATURE_NAMES)
            # Index 0 ist das aktuellste Match in der Historie (wegen [::-1] im Preprocessing)
            rank_a = batch['player_a_recent'][:, 0, 1]
            rank_b = batch['player_b_recent'][:, 0, 1]
            
            # 0.0 (Padding) durch Unendlich ersetzen, damit diese nicht als "bester Rang" gelten
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
                
                # Ohne Quoten keine Wette
                if oda <= 1.01 or odb <= 1.01:
                    continue
                
                bet_made = False
                won = False
                
                # Strategie: Setze auf den besseren Rang
                if r_a < r_b: 
                    # A ist besser -> Wette auf A
                    balance -= stake
                    bets_placed += 1
                    bet_made = True
                    if real_outcome == 1:
                        balance += stake * oda
                        won = True
                        
                elif r_b < r_a:
                    # B ist besser -> Wette auf B
                    balance -= stake
                    bets_placed += 1
                    bet_made = True
                    if real_outcome == 0:
                        balance += stake * odb
                        won = True
                
                if bet_made and won:
                    correct_bets += 1

    roi = ((balance - initial_balance) / initial_balance) * 100
    
    print("-" * 30)
    print(f"Startkapital: {initial_balance:.2f}€")
    print(f"Endkapital:   {balance:.2f}€")
    print(f"ROI:          {roi:.2f}%")
    print(f"Wetten:       {bets_placed}")
    if bets_placed > 0:
        print(f"Win-Rate:     {100 * correct_bets / bets_placed:.2f}%")
    print("="*60 + "\n")    
    
def compute_baseline_accuracy(config=default_config):
    """
    Compute baseline accuracy: Predict winner based on ranking only.
    
    Simple heuristic: The player with better (lower) rank wins.
    This shows how much the Transformer improves over a naive strategy.
    
    Args:
        config: TrainingConfig instance
    """
    print("\n" + "="*60)
    print("Computing Baseline Accuracy (Better Ranking Wins)")
    print("="*60)
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Check if preprocessed data exists, if not preprocess it
    if not os.path.exists(config.preprocessed_data_path):
        print(f"Preprocessed data not found at {config.preprocessed_data_path}")
        print("Running preprocessing pipeline...\n")
        from src import data_loader as dl
        from src import preprocessing as pre
        
        years = list(range(config.data_years_start, config.data_years_end + 1))
        raw_data = dl.load_all_matches(years)
        all_samples = pre.create_training_samples(
            raw_data, 
            n_matches=config.n_surface_matches,
            n_recent=config.n_recent_matches
        )
        
        print(f"Saving {len(all_samples)} samples to {config.preprocessed_data_path}...")
        torch.save(all_samples, config.preprocessed_data_path)
        print(f"Preprocessing complete!\n")
    else:
        all_samples = torch.load(config.preprocessed_data_path)
    
    # Filter samples to only include test year
    test_samples = [s for s in all_samples if s['year'] == config.test_year]
    print(f"Using {len(test_samples)} test samples from year {config.test_year}")
    print(f"(Total samples available: {len(all_samples)})\n")
    
    if len(test_samples) == 0:
        print(f"ERROR: No samples found for test year {config.test_year}")
        return 0.0
    
    # Create dataset and loader
    dataset = DataSet(test_samples)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            # Get ranking features from the last match in recent history
            # Features: [won, rank, aces, double_faults, first_serve_pct, days_since_match]
            # Index 1 is rank
            rank_a = batch['player_a_recent'][:, 0, 1]  # Last match, rank feature
            rank_b = batch['player_b_recent'][:, 0, 1]
            
            rank_a = torch.where(rank_a == 0, torch.tensor(float('inf')), rank_a)
            rank_b = torch.where(rank_b == 0, torch.tensor(float('inf')), rank_b)
            
            labels = batch['label']
            
            # Predict: player with better (lower) rank wins
            # rank_a < rank_b → player A is better → predict 1
            predictions = torch.where(rank_a < rank_b, 1, 0)
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    baseline_accuracy = 100 * correct / total
    print(f"Total Samples: {total}")
    print(f"Baseline Accuracy (Ranking): {baseline_accuracy:.2f}%")
    print(f"Correct Predictions: {correct}/{total}")
    print("="*60 + "\n")
    
    return baseline_accuracy


def main():
    """Main entry point: compute baseline, then train model."""
    baseline_acc = compute_baseline_accuracy()
    
    # Train model
    print("Training Transformer Model...\n")
    train()
    
    # ROI berechnen
    simulate_betting_baseline()
    simulate_strategy_pure_value()
    simulate_strategy_always_bet()


if __name__ == "__main__":
    main()