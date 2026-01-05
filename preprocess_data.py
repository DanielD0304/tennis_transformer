"""
Preprocess tennis data and save to disk.
Run this once to avoid reprocessing on every training run.
"""
import torch
from src import data_loader as dl
from src import preprocessing as pre
from src.config import default_config


def preprocess_and_save(config=default_config):
    """
    Load, preprocess, and save training samples.
    
    Args:
        config: TrainingConfig instance with data parameters
    """
    print("Loading raw data...")
    years = list(range(config.data_years_start, config.data_years_end + 1))
    raw_data = dl.load_all_matches(years)
    
    print("Processing samples...")
    all_samples = pre.create_training_samples(
        raw_data, 
        n_matches=config.n_surface_matches,
        n_recent=config.n_recent_matches
    )
    
    print(f"Saving {len(all_samples)} samples to {config.preprocessed_data_path}...")
    torch.save(all_samples, config.preprocessed_data_path)
    
    print(f"Done! Saved to {config.preprocessed_data_path}")
    print(f"File can now be loaded quickly with: torch.load('{config.preprocessed_data_path}')")


if __name__ == "__main__":
    preprocess_and_save()
