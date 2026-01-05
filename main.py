from src.train import train

def main():
	train()

import torch
from src.dataset import DataSet
from torch.utils.data import DataLoader

def check_baseline():
    print("Berechne Baseline (Besseres Ranking gewinnt)...")
    dataset = DataSet()
    # Wir nehmen den ganzen Datensatz zum Checken der Baseline-Logik
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    correct = 0
    total = 0
    
    for batch in loader:
        # In deinem Preprocessing ist feature index 1 das 'rank' (log rank)
        # feature vector: [won, rank, aces, df, 1st_in, days]
        
        # Wir holen uns nur das Feature "Rank" (Index 1)
        rank_a = batch['player_a_recent'][:, -1, 1] # Nimm das letzte Match der Sequenz als proxy für aktuellen Rang
        rank_b = batch['player_b_recent'][:, -1, 1] 
        
        labels = batch['label']
        
        # Logik: Wer den kleineren Rank-Wert hat (besserer Rang), sollte gewinnen
        # (Da wir log(rank+1) nutzen, ist kleiner immer noch besser)
        predictions = torch.where(rank_a < rank_b, 1, 0)
        
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
    print(f"Baseline Accuracy (Ranking): {100 * correct / total:.2f}%")

if __name__ == "__main__":
    check_baseline()