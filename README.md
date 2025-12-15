# Tennis Transformer 🎾

Privates Projekt um Tennis-Ergebnisse mit einem Transformer-Modell vorherzusagen.

## Ziel

Ziel ist es, anhand der letzten N Matches auf dem selben Untergrund den Sieger bei einem Tennis-Spiel zu predicten.

## Architektur

Transformer-Architektur wird benutzt, weil das Ziel ein Sequence-to-Label Modell ist. Die Match-Historie eines Spielers wird als Sequenz verarbeitet, um relevante Muster zu erkennen.

## Projektstruktur

```
tennis_transformer/
├── data/                    # Gespeicherte Daten
├── src/
│   ├── data_loader.py       # ATP-Daten von GitHub laden
│   ├── preprocessing.py     # Features & Match-Historie erstellen
│   ├── attention.py         # Self-Attention (from scratch)
│   ├── transformer.py       # Transformer-Block
│   ├── model.py             # Gesamtarchitektur
│   ├── dataset.py           # PyTorch Dataset
│   └── train.py             # Training Loop
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

```bash
# Daten laden und vorbereiten
python src/data_loader.py

# Modell trainieren
python main.py
```

## Datenquelle

ATP Match-Daten von [Jeff Sackmann's Tennis ATP Repository](https://github.com/JeffSackmann/tennis_atp) (2020-2024).

## Requirements

- Python 3.10+
- PyTorch
- pandas
- numpy
- scikit-learn
- matplotlib