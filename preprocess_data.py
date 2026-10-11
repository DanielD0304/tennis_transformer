"""
Preprocess tennis data and save to disk.
Run this once to avoid reprocessing on every training run.

    python preprocess_data.py
"""
import os

from src.config import default_config
from src.train import load_or_preprocess_data


def preprocess_and_save(config=default_config):
    """
    Lädt die Matchdaten (data/matches.csv, wird bei Bedarf aktualisiert), berechnet ELO,
    erstellt die Trainings-Samples und speichert sie unter config.preprocessed_data_path.
    Eine vorhandene Datei wird dabei neu erzeugt.
    """
    if os.path.exists(config.preprocessed_data_path):
        os.remove(config.preprocessed_data_path)
    samples = load_or_preprocess_data(config)
    print(f"Done! {len(samples)} samples in {config.preprocessed_data_path}")


if __name__ == "__main__":
    preprocess_and_save()
