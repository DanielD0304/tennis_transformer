import torch
import os
from src.train import train
from src.config import default_config
from src.dataset import DataSet
from torch.utils.data import DataLoader


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
    
    # Load preprocessed data
    if not torch.cuda.is_available():
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
    
    # Filter samples to only include test year (same as model test set)
    test_samples = [s for s in all_samples if s['year'] == config.test_year]
    print(f"Using {len(test_samples)} test samples from year {config.test_year}")
    print(f"(Total samples available: {len(all_samples)})\n")
    
    if len(test_samples) == 0:
        print(f"ERROR: No samples found for test year {config.test_year}")
        return 0.0
    
    # Create dataset and loader (use test year data for baseline)
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
    #baseline_acc = compute_baseline_accuracy()
    
    # Train model
    print("Training Transformer Model...\n")
    train()


if __name__ == "__main__":
    main()