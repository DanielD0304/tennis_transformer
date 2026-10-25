"""
Configuration for Tennis Transformer Training.
"""
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Training hyperparameters and paths."""
    
    dropout: float = 0.3
    # Model architecture
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 2
    input_dim: int = 39  # Anzahl Features pro Match, siehe preprocessing.FEATURE_NAMES
    n_surface_matches: int = 10  # Number of surface-specific matches in history
    n_recent_matches: int = 15  # Number of recent matches (any surface)
    max_len: int = 50  # Maximum sequence length (n_surface + n_recent) * 2
    max_rank: int = 500  # Trainings-Samples nur, wenn beide Spieler mindestens so gut gerankt sind
    output_dim: int = 2  # Binary classification (player A wins / player B wins)
    
    # Training parameters
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.0005
    
    # Anteil der Historien-Matches, deren Aufschlag-Stats im Training zufällig entfernt werden
    stats_dropout: float = 0.15
    
    # Early stopping
    patience: int = 5
    
    # Data
    train_years_end: int = 2023  # Train on data up to and including this year
    val_year: int = 2024  # Validation year
    test_year: int = 2025  # Test year
    data_years_start: int = 2015  # Start loading data from this year
    data_years_end: int = 2025  # Load data up to this year
    
    # Paths
    preprocessed_data_path: str = "data/preprocessed_samples.pt"
    best_model_path: str = "models/best_model.pt"
    
    # Visualization
    attention_layer: int = 0  # Which layer to visualize
    attention_head: int = 0  # Which attention head to visualize
    attention_sample: int = 0  # Which sample in batch to visualize


default_config = TrainingConfig()
