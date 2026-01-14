# Tennis Transformer 🎾

Privates Projekt um Tennis-Ergebnisse mit einem Transformer-Modell vorherzusagen.

## Ziel

Ziel ist es, anhand der letzten N Matches auf dem selben Untergrund und der aktuellen Form (recent matches) den Sieger bei einem Tennis-Spiel zu predicten.

## Architektur

Transformer-Architektur wird benutzt, weil das Ziel ein Sequence-to-Label Modell ist. Die Match-Historie eines Spielers wird als Sequenz verarbeitet, um relevante Muster zu erkennen. Das Modell nutzt:
- **Surface-specific History**: Letzten 10 Matches auf dem gleichen Untergrund
- **Recent Form**: Letzten 15 Matches (unabhängig vom Untergrund)
- **[CLS]-Token**: Globale Repräsentation für die finale Klassifikation
- **Segment Embeddings**: Unterscheidung zwischen Spieler A/B und Surface/Recent
- **Positional Encoding**: Berücksichtigung der zeitlichen Reihenfolge

## Projektstruktur

```
tennis_transformer/
├── data/                    # Gespeicherte Daten
├── src/
│   ├── config.py            # Zentrale Konfiguration (Hyperparameter, Pfade)
│   ├── data_loader.py       # ATP-Daten von GitHub laden
│   ├── preprocessing.py     # Features & Match-Historie erstellen
│   ├── attention.py         # Self-Attention (from scratch)
│   ├── encoderlayer.py      # Transformer Encoder Layer
│   ├── transformer.py       # Transformer-Block
│   ├── model.py             # Gesamtarchitektur
│   ├── dataset.py           # PyTorch Dataset
│   └── train.py             # Training Loop mit Validation & Early Stopping
├── preprocess_data.py       # Einmaliges Preprocessing (spart Zeit)
├── main.py
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/DanielD0304/tennis_transformer.git
cd tennis_transformer
pip install -r requirements.txt
```

## Verwendung

### 1. Daten vorverarbeiten (einmalig)
```bash
python -m tennis_transformer.preprocess_data
```
Dies lädt die ATP-Daten (2015-2024), verarbeitet sie und speichert sie als `preprocessed_samples.pt`. Dadurch sparst du bei jedem Training mehrere Minuten!

### 2. Modell trainieren mit automatischer Baseline-Berechnung
```bash
python main.py
```
**Automatische Baseline-Berechnung:** `main.py` berechnet automatisch die Baseline-Accuracy (Ranking-basierte Vorhersage) vor dem Training. Dies zeigt, wie viel Verbesserung das Transformer-Modell gegenüber einer naiven Strategie erreicht.

### 3. Custom Training mit eigener Config
```python
from tennis_transformer.src.config import TrainingConfig
from tennis_transformer.src.train import train

# Eigene Konfiguration
config = TrainingConfig(
    num_epochs=20,
    learning_rate=0.0005,
    batch_size=64,
    patience=5
)

train(config)
```

## Features

### Daten-Split (Zeitreihen-korrekt)
- **Training**: 2015-2022 (älteste Daten)
- **Validation**: 2023 (für Hyperparameter-Tuning)
- **Test**: 2024 (finale Evaluation)

Dies verhindert **Data Leakage**, da das Modell nie Zukunftsdaten sieht.

### Training Features
- **Best Model Checkpointing**: Speichert nur das beste Modell basierend auf Validation Accuracy
- **Early Stopping**: Stoppt Training automatisch bei Overfitting (nach 3 Epochen ohne Verbesserung)
- **Attention Visualization**: Speichert Attention-Maps nach jeder Epoche
- **Validation & Test Evaluation**: Separate Evaluation auf ungesehenen Daten

### Model Features (pro Spieler)
- `won`: 1 wenn gewonnen, 0 wenn verloren
- `rank`: log(rank+1) zur Normalisierung
- `aces`: Anzahl Aces
- `double_faults`: Anzahl Doppelfehler
- `first_serve_pct`: Erste-Aufschlag-Quote
- `days_since_match`: log(1+days) seit dem Match
- `opponent_elo`: Gegner ELO-Rating (normalisiert: elo/2000). Das ELO-Rating wird dynamisch basierend auf Spielergebissen berechnet (Start: 1500, K-Faktor: 32)
- `opponent_rank`: log(opponent_rank+1) - Ranking des Gegners bei diesem Match
- `own_elo`: Spieler-eigenes ELO-Rating (normalisiert: elo/2000) - zeigt die historische Stärke des Spielers

## Datenquelle

ATP Match-Daten von [Jeff Sackmann's Tennis ATP Repository](https://github.com/JeffSackmann/tennis_atp) (2015-2024).

## Konfiguration

Alle Hyperparameter sind zentral in `src/config.py` definiert:

```python
@dataclass
class TrainingConfig:
    # Model
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 2
    
    # Training
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    patience: int = 3
    
    # Data
    train_years_end: int = 2022
    val_year: int = 2023
    test_year: int = 2024
```

## Lernkurve & Herausforderungen

Dieses Projekt entstand als Lern- und Portfolio-Projekt. Im Verlauf gab es zahlreiche Herausforderungen und Verbesserungen:

### Data Pipeline
- **NaN/Fehlende Werte:** Ursprünglich führten fehlende oder ungültige Werte in den Features zu NaN-Losses. Lösung: Robustes Data Cleaning und Imputation mit `-1` als Sentinel.
- **ZeroDivisionError:** Division durch Null bei Feature-Berechnung (z.B. Aufschlagquoten) wurde durch explizite Checks verhindert.
- **Data Leakage (Year-Split):** Initial wurden Daten zufällig gesplittet. Lösung: Chronologischer Split nach Jahren (Train: ≤2022, Val: 2023, Test: 2024).
- **Missing Tournament Information:** Ursprünglich wurden alle Matches vom gleichen Turniertag mit `<` ausgeschlossen, was wichtige Vorrunden-Matches ignorierte. Problem: Alle Matches eines Turniers haben das gleiche `tourney_date`. Lösung: Matches vom gleichen Turnier (`tourney_id`) werden jetzt inkludiert, da sie zeitlich vor dem aktuellen Match stattfanden (Vorrunde → Finale). Kein Data Leakage, da nur frühere Runden berücksichtigt werden.

### Model Architecture
- **Softmax & CrossEntropy:** Softmax wurde aus dem Modell entfernt, da `CrossEntropyLoss` rohe Logits erwartet.
- **Sequenzaggregation:** Statt einfachem Mittelwert wird ein [CLS]-Token als globales Repräsentativ verwendet (wie bei BERT).
- **Ranking-Feature:** Der Rang wurde ursprünglich als numerisches Feature genutzt, was zu Ausreißern führte. Lösung: `log(rank+1)` als Feature.
- **Segment-Embeddings:** Um dem Modell Kontext zu geben, wurden Segment-Embeddings für Spieler A/B und Surface/Recent eingeführt.
- **Match-Alter:** Das Alter jedes Matches wird als Feature (`log(1+days_since_match)`) übergeben, damit das Modell aktuelle Form besser erkennt.

### Training & Validation
- **Validation Set:** Ursprünglich gab es nur Train/Test. Ein separates Validation-Set (2023) wurde hinzugefügt für Hyperparameter-Tuning.
- **Best Model Checkpointing:** Statt jedes Modell zu speichern, wird nur das beste basierend auf Validation Accuracy behalten.
- **Early Stopping:** Verhindert Overfitting durch automatisches Stoppen wenn Validation Loss nicht mehr sinkt.

### Code Quality
- **Hardcoded Values:** Ursprünglich waren Werte wie `input_dim=6` und `max_len=15` fest im Code. Lösung: Zentrale Config-Datei mit allen Hyperparametern.
- **Effizientes Data Loading:** Preprocessing wird nur einmal ausgeführt und als `.pt` gespeichert, statt bei jedem Training neu zu laden (spart ~5 Minuten).
- **Attention Visualization:** Statt 1000+ Bilder (pro Batch) wird nur 1 Bild pro Epoche gespeichert.
- **Kritischer Data Leakage Fix (Tournament Level):** Nach Initial-Training mit 98%+ Accuracy wurde ein **kritischer Bug** entdeckt: Das aktuelle Match war in seiner eigenen Historie enthalten! Das Modell sah das Ergebnis (`won=1/0`) als erstes Feature im `player_recent`-Vektor und "schummelte". Lösung: Striktes Filtern mit `match_num` - nur Matches mit `match_num < current_match_num` vom gleichen Turnier werden inkludiert. Das aktuelle Match ist definitiv NICHT mehr in der Historie. Nach dem Fix fiel die Accuracy realistisch auf **64.66%** (Test Accuracy, Epoch 5).
- **Preprocessing-Optimierung (Single-Pass O(N)):** Ursprüngliche Implementation hatte O(N²) Komplexität durch wiederholte Filteroperationen für jedes Match. Neue Implementation nutzt einen Single-Pass-Algorithmus mit `defaultdict`, der die Historie inkrementell aufbaut: Nur eine Iteration über alle Matches mit O(N) Zeitkomplexität. Dies reduziert die Preprocessing-Zeit um ~95% (27.672 Matches in Sekunden statt Minuten).

## Ergebnisse

### Baseline Accuracy
Das Projekt verwendet eine **Ranking-basierte Baseline**, um die Performance des Transformer-Modells zu bewerten:
- **Methode**: Einfache Heuristik - Spieler mit besserem (niedrigerem) Ranking gewinnt
- **Evaluierung**: Baseline wird auf den gleichen **Test-Jahren wie das Modell** berechnet (2024)
- **Zweck**: Zeigt, wie viel Verbesserung das Transformer-Modell gegenüber der naiven Ranking-Strategie erreicht
- **Ergebnis**: **Baseline Accuracy: 63.39%** (1.950/3.076 korrekte Vorhersagen)

Dies ermöglicht einen fairen Vergleich: Modell-Accuracy - Baseline-Accuracy = echter Mehrwert durch Deep Learning.

### Modell-Performance

Aktuelle Performance (27.672 Samples, 21.610 Training / 2.986 Validation / 3.076 Test):

**Mit neuen Features (own_elo + opponent_elo + Preprocessing-Optimierung):**
```
Epoch [1/10]:  Val Acc: 62.76%, Test Acc: 62.42%
Epoch [2/10]:  Val Acc: 62.96%, Test Acc: 63.88%
Epoch [3/10]:  Val Acc: 63.83%, Test Acc: 63.78%
Epoch [4/10]:  Val Acc: 63.30%, Test Acc: 63.88%
Epoch [5/10]:  Val Acc: 63.53%, Test Acc: 64.24%
Epoch [6/10]:  Val Acc: 64.03%, Test Acc: 64.86% ⭐ (Best Test)
Epoch [7/10]:  Val Acc: 63.60%, Test Acc: 64.04%
Epoch [8/10]:  Val Acc: 64.30%, Test Acc: 64.86% ⭐ (Best Validation)
  
Early Stopping nach Epoch 8 (keine Verbesserung in Val Loss mehr)
```

**Vergleich zum Baseline:** 
- Baseline Accuracy: 63.39%
- Beste Model Accuracy: **64.86%** (Epoch 6)
- **Improvement über Baseline: +1.47%**
