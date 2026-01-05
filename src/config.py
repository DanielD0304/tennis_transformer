"""
Configuration for Tennis Transformer Training.
"""
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Training hyperparameters and paths."""
    
    # Model architecture
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 2
    input_dim: int = 6  # Number of features per match (won, rank, aces, df, 1st_serve_pct, days_since)
    n_surface_matches: int = 10  # Number of surface-specific matches in history
    n_recent_matches: int = 15  # Number of recent matches (any surface)
    max_len: int = 50  # Maximum sequence length (n_surface + n_recent) * 2
    output_dim: int = 2  # Binary classification (player A wins / player B wins)
    
    # Training parameters
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    
    # Early stopping
    patience: int = 3
    
    # Data
    train_years_end: int = 2022  # Train on data up to and including this year
    val_year: int = 2023  # Validation year
    test_year: int = 2024  # Test year
    data_years_start: int = 2015  # Start loading data from this year
    data_years_end: int = 2024  # Load data up to this year
    
    # Paths
    preprocessed_data_path: str = "preprocessed_samples.pt"
    best_model_path: str = "best_model.pt"
    
    # Visualization
    attention_layer: int = 0  # Which layer to visualize
    attention_head: int = 0  # Which attention head to visualize
    attention_sample: int = 0  # Which sample in batch to visualize


# Default configuration instance
default_config = TrainingConfig()
