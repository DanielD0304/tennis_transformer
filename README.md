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
│   ├── loss.py              # Focal Loss Implementation
│   ├── elo.py               # ELO-Rating Berechnung
│   ├── train.py             # Training Loop mit Validation & Early Stopping
│   ├── predict.py           # Live-Vorhersagen für anstehende Matches
│   └── names.py             # Spielernamen aus verschiedenen Quellen abgleichen
├── betscraper/              # Django-App: Quoten scrapen & Value Bets anzeigen
│   ├── scraper/sites.py     # Seiten-Konfiguration (Tipico, bwin) + Playwright-Scraper
│   ├── dataProcessing/      # Buchmacher-Marge, Value-Bet-Berechnung
│   └── manage.py
├── preprocess_data.py       # Einmaliges Preprocessing (spart Zeit)
├── main.py                  # Hauptskript mit Strategien & Simulationen
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

### 4. Web-Interface
```bash
cd betscraper
python manage.py runserver
```
Dann http://127.0.0.1:8000/ öffnen. Unter **Vorhersage** zwei Spieler wählen (Autovervollständigung,
Namen in jedem Format wie „Medvedev D.“ oder „D. Medvedev“), dazu Untergrund und Datum. Die Seite zeigt
die Siegwahrscheinlichkeit des Transformers, faire Quoten, Head-to-Head sowie Ranking, ELO, Form und
die letzten Matches beider Spieler. Bei Tippfehlern gibt es Namensvorschläge. Voraussetzung ist ein
trainiertes Modell unter `models/best_model.pt`.

### 5. Live Value Bets (betscraper)
Der Scraper holt die aktuellen Tennis-Quoten von Tipico und bwin, nimmt pro Match die beste Quote,
lässt das trainierte Modell (`models/best_model.pt`) die Siegwahrscheinlichkeit schätzen und zeigt
alle Tipps mit `Wahrscheinlichkeit × Quote > Mindest-EV` an.

```bash
playwright install chromium
cd betscraper
python manage.py runserver
```
Dann http://127.0.0.1:8000/scraper/tennis/ öffnen (Untergrund und Mindest-EV sind wählbar,
die Quellen liefern keinen Untergrund). Statt des Playwright-Chromiums kann mit
`PLAYWRIGHT_CHANNEL=msedge` bzw. `chrome` ein installierter Browser genutzt werden.

Hinweise:
- Die Spieler-Historien für Live-Matches kommen aus den aktuellen tennis-data.co.uk-Dateien. Die
  enthalten keine Aufschlag-Statistiken, diese Features sind live also 0.
- Nur ATP-Hauptturniere werden erkannt; WTA, Challenger und Doppel bekommen keine Vorhersage.
- Die alte Fußball-Ansicht (1X2-Quoten + Marge) gibt es weiter unter `/scraper/scrape-odds/`.

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
Das Repository ist inzwischen nicht mehr erreichbar. Fehlen die Dateien (auch nicht lokal in `data/`),
wird automatisch auf [tennis-data.co.uk](https://www.tennis-data.co.uk/alldata.php) ausgewichen (inkl. Quoten, ohne Aufschlag-Stats).

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
- **Noise Reduction (Top 150 & Aktivität):** Matches auf niedrigerem Niveau (Challenger/Futures) oder von inaktiven Spielern erzeugten zu viel Rauschen ("Noise"). Lösung: Strikte Filterung – Ein Match wird nur trainiert/getestet, wenn beide Spieler in den Top 150 stehen UND beide mindestens 5 Spiele in den letzten 3 Monaten absolviert haben.
- **Hybrid-Datenquelle (Stats + Odds)**: Für eine echte ROI-Berechnung fehlten kompatible Daten. Jeff Sackmann liefert Stats (aber keine Quoten), Wettanbieter liefern Quoten, deswegen wurde als neue zusätzlich auswählbare Datenquelle http://www.tennis-data.co.uk benutzt.

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

## Loss Function

Das Projekt verwendet **Focal Loss** statt der Standard CrossEntropyLoss:

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
```

**Vorteile von Focal Loss:**
- Reduziert den Einfluss von leicht klassifizierbaren Samples
- Fokussiert das Training auf schwierige Fälle
- Verbessert die Kalibrierung der Wahrscheinlichkeiten

## Wett-Strategien

Das Projekt implementiert mehrere Wett-Strategien zur ROI-Analyse:

### Strategie 1: Filtered Betting
Wette auf den Spieler mit der höheren Wahrscheinlichkeit, aber nur wenn die Quote >= min_odds (Standard: 1.30) ist. Filtert "Müll-Quoten" (1.01-1.29) heraus.

### Strategie 2: Pure Value
Wette nur, wenn der erwartete Wert (Expected Value) > 1.0 ist. Formel: `Wahrscheinlichkeit * Quote > 1.0`

### Strategie 3: Sniper
Wette nur, wenn die Modell-Wahrscheinlichkeit im "Gold-Bereich" liegt (65% - 80% Konfidenz). Basiert auf dem Calibration Check.

### Baseline: Better Rank Wins
Setzt immer auf den Spieler mit dem besseren (niedrigeren) Rang.

### Diagnose: Probability Calibration Check
Überprüft, ob die Wahrscheinlichkeiten des Modells der Realität entsprechen und zeigt "Overconfidence" oder "Underconfidence" an.

## Ergebnisse

### Baseline Accuracy
Das Projekt verwendet eine **Ranking-basierte Baseline**, um die Performance des Transformer-Modells zu bewerten:
- **Methode**: Einfache Heuristik - Spieler mit besserem (niedrigerem) Ranking gewinnt
- **Evaluierung**: Baseline wird auf den gleichen **Test-Jahren wie das Modell** berechnet (2024)
- **Zweck**: Zeigt, wie viel Verbesserung das Transformer-Modell gegenüber der naiven Ranking-Strategie erreicht

Dies ermöglicht einen fairen Vergleich: Modell-Accuracy - Baseline-Accuracy = echter Mehrwert durch Deep Learning.

### Modell-Performance

#### Ergebnisse vom 29.01.2025

**Test-Daten**: 2024 (1.717 Samples)
- **Baseline (Ranking-Heuristik)**: 63.23% Accuracy
- **Transformer-Modell**: 64.44% Accuracy (Best Validation, Epoch 9)
- **Verbesserung über Baseline**: +1.21%

**Training-Metriken (10 Epochen mit Focal Loss):**
```
Epoch [1/10]:  Val Acc: 62.09%, Loss: 0.1615 → New Best
Epoch [2/10]:  Val Acc: 63.54%, Loss: 0.1607 → New Best
Epoch [3/10]:  Val Acc: 63.72%, Loss: 0.1591 → New Best
Epoch [4/10]:  Val Acc: 63.90%, Loss: 0.1586 → New Best
Epoch [5/10]:  Val Acc: 63.78%, Loss: 0.1601 (Patience: 1/3)
Epoch [6/10]:  Val Acc: 63.96%, Loss: 0.1587 → New Best (Patience: 2/3)
Epoch [7/10]:  Val Acc: 63.78%, Loss: 0.1584, LR: 0.0005
Epoch [8/10]:  Val Acc: 64.20%, Loss: 0.1579 → New Best
Epoch [9/10]:  Val Acc: 64.44%, Loss: 0.1578 → New Best ⭐
Epoch [10/10]: Val Acc: 64.26%, Loss: 0.1582 (Patience: 1/3)
```

### Wett-Simulation (ROI-Analyse)

Simulation basierend auf Modell-Vorhersagen vs. Baseline auf 1.717 Test-Matches (2024):

**Baseline (Better Rank Wins):**
- Startkapital: 1.000€
- Endkapital: 169,30€
- **ROC: -83,07%**
- **YIELD: -4,87%**
- Wetten: 1.705
- Win-Rate: 63,23%

**Strategie 3: Sniper (Confidence 70% - 80%):**
- Startkapital: 1.000€
- Endkapital: 1.030,00€
- **ROC: +3,00%** ✅
- **YIELD: +2,73%** ✅
- Wetten: 110 (Gefiltert: 1.598)
- Win-Rate: 93,64%

**Strategie 2: Pure Value (Math > Bookie):**
- Startkapital: 1.000€
- Endkapital: -482,10€
- **ROC: -148,21%**
- **YIELD: -9,55%**
- Wetten: 1.552 (Gefiltert: 156)
- Win-Rate: 31,57%

### Probability Calibration

```
Confidence      | Count    | Real Win Rate   | Diff      
------------------------------------------------------------
0.0 - 0.4      | 228      | 17.11%          | -18.65%
0.4 - 0.5      | 723      | 41.91%          | -3.85%
0.5 - 0.6      | 619      | 61.39%          | +7.43%
0.6 - 0.7      | 146      | 81.51%          | +18.24%
0.7 - 0.8      | 1        | 100.00%         | +29.80%
```

**Hinweis:** Negative 'Diff' bedeutet, das Modell ist zu optimistisch (overconfident). Die Sniper-Strategie nutzt den kalibrierten Bereich (0.6-0.8) optimal aus
