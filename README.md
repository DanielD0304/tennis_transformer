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
│   ├── data_loader.py       # data/matches.csv lesen, Schema für die Features
│   ├── preprocessing.py     # Features & Match-Historie erstellen
│   ├── attention.py         # Self-Attention (from scratch)
│   ├── encoderlayer.py      # Transformer Encoder Layer
│   ├── transformer.py       # Transformer-Block
│   ├── model.py             # Gesamtarchitektur
│   ├── dataset.py           # PyTorch Dataset
│   ├── loss.py              # Cross-Entropy (Standard) und Focal Loss
│   ├── elo.py               # ELO-Rating (alt) und Elo2 (538-K, Untergrund-Mix, Tuning)
│   ├── models/              # LightGBM (gbdt.py), Stacking (stack.py), Kalibrierung (calibration.py)
│   ├── train.py             # Training Loop mit Validation & Early Stopping
│   ├── predict.py           # Live-Vorhersagen für anstehende Matches
│   ├── names.py             # Spielernamen aus verschiedenen Quellen abgleichen
│   ├── update_data.py       # Daten laden & data/matches.csv bauen (täglicher Refresh)
│   ├── evaluation/          # Walk-forward-Backtest, Metriken gegen den Markt, Venues, Wett-Simulation
│   └── sources/             # Sackmann-Archiv, tennis-data.co.uk, ESPN + Zusammenführung
├── betscraper/              # Django-App: Quoten scrapen & Value Bets anzeigen
│   ├── scraper/sites.py     # Seiten-Konfiguration (Tipico, bwin) + Playwright-Scraper
│   ├── dataProcessing/      # Buchmacher-Marge, Value-Bet-Berechnung
│   └── manage.py
├── preprocess_data.py       # Einmaliges Preprocessing (spart Zeit)
├── main.py                  # Training + Auswertung über src/evaluation
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
python preprocess_data.py
```
Dies lädt die Matchdaten (2015-2025, siehe Datenquellen), verarbeitet sie und speichert sie als `preprocessed_samples.pt`. Dadurch sparst du bei jedem Training mehrere Minuten!

### 2. Modell trainieren mit automatischer Baseline-Berechnung
```bash
python main.py
```
Nach dem Training wertet `main.py` das Modell mit `src.evaluation` aus: Baseline-Accuracy (Rang, Elo),
Log Loss/Brier/ECE und Signifikanz gegen den Markt (Pinnacle ohne Marge), Wett-Strategien pro Venue
und Kalibrierung. Details unter „Evaluation“.

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
Namen in jedem Format wie „Medvedev D.“ oder „D. Medvedev“), dazu optional das Turnier, Untergrund und
Datum. Mit Untergrund „Auto“ (Standard) wird er aus dem Turniernamen bestimmt, ohne Turnier aus dem
laufenden Turnier der Spieler; eine manuelle Auswahl hat Vorrang. Die Seite zeigt
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
Dann http://127.0.0.1:8000/scraper/tennis/ öffnen (Untergrund und Mindest-EV sind wählbar;
„Auto“ nimmt pro Match den Untergrund des Turniers, das die Spieler gerade spielen). Statt des Playwright-Chromiums kann mit
`PLAYWRIGHT_CHANNEL=msedge` bzw. `chrome` ein installierter Browser genutzt werden.

Hinweise:
- Die Spieler-Historien kommen aus `data/matches.csv` (siehe „Datenquellen“), inklusive aller
  bereits gespielten Runden eines laufenden Turniers.
- ATP- und WTA-Hauptturniere werden erkannt; Challenger, ITF und Doppel bekommen keine Vorhersage.
- Die alte Fußball-Ansicht (1X2-Quoten + Marge) gibt es weiter unter `/scraper/scrape-odds/`.

## Features

### Daten-Split (Zeitreihen-korrekt)
- **Training**: 2015-2023 (älteste Daten)
- **Validation**: 2024 (für Hyperparameter-Tuning)
- **Test**: 2025 (finale Evaluation)

Dies verhindert **Data Leakage**, da das Modell nie Zukunftsdaten sieht.

### Training Features
- **Best Model Checkpointing**: Speichert nur das beste Modell basierend auf Validation Accuracy
- **Early Stopping**: Stoppt Training automatisch bei Overfitting (nach 3 Epochen ohne Verbesserung)
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
- `has_stats`: 1, wenn für das Match Aufschlag-Stats vorliegen (sonst sind die Stat-Features 0)

## Datenquellen

Alle Matchdaten landen in `data/matches.csv` (Sackmann-Schema plus `match_date`, `status`, `source`).
Die Rohdaten liegen unter `data/raw/`, nichts davon ist im Repo.

| Quelle | Inhalt | Aktualität |
|---|---|---|
| [Sackmann-Archiv](https://github.com/Aneeshers/tennis-sackmann-archive) (fester Commit) | ATP und WTA: Ergebnisse **mit Aufschlag-Stats**, Spieler-Metadaten (Nation, Hand, Geburtsdatum) | bis Turnierstart 25.05.2026 (Original-Repo ist offline) |
| [tennis-data.co.uk](https://www.tennis-data.co.uk/alldata.php) | Ergebnisse, Rankings, Untergrund, Indoor/Outdoor, Quoten (Bet365, Pinnacle, Ø); ATP in `data/raw/tuk/`, WTA in `data/raw/tuk_wta/` | wöchentlich nach Turnierende |
| ESPN-Scoreboard | jedes beendete Match (ATP und WTA), auch im laufenden Turnier | täglich |

```bash
python -m src.update_data          # täglicher Refresh, lädt nur Neues (~1 Minute)
python -m src.update_data --full   # alles neu laden (~4 Minuten)
python -m src.update_data --tours atp   # nur eine Tour
```

ATP und WTA werden getrennt zusammengeführt (Spalte `tour`), Namen und Matches nie über Touren hinweg
verknüpft; Elo, Historien und H2H laufen pro Tour. Sackmann kennt nur den Turnierstart: Wo dasselbe
Match bei tennis-data oder ESPN vorkommt, wird dessen echtes Datum übernommen, sortiert wird überall
danach (sonst Turnierstart + Runde, `src/chronology.py`).

Training, `preprocess_data.py` und der `MatchPredictor` laden die Daten automatisch neu, wenn der
letzte Refresh älter als 24 Stunden ist. Beim Zusammenführen werden Spielernamen vereinheitlicht
(„Medvedev D.“, „Wu Yibing“, „Aleksandr Shevchenko“ → ein Name pro Spieler), Duplikate über
Zeitfenster + Spielerpaar erkannt und die Quoten von tennis-data an alle Matches gehängt.

Für Matches ohne Stats (tennis-data, ESPN) sind die Stat-Features 0 und `has_stats` ist 0. Im
Training werden die Stats zufällig bei 15 % der Historien-Matches entfernt (`stats_dropout`), damit
das Modell damit umgehen kann. Die ATP-Seite selbst blockt automatisierte Abrufe (Cloudflare) und
wird deshalb nicht genutzt.

Lizenz der Sackmann-Daten: CC BY-NC-SA 4.0 (nur nicht-kommerziell), Daten von Jeff Sackmann.

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

Standard ist **Cross-Entropy** (`config.loss = 'ce'`). Für Wetten zählen kalibrierte
Wahrscheinlichkeiten, und Cross-Entropy ist genau der Log Loss, an dem das Modell gemessen wird.
Der früher verwendete **Focal Loss** (gamma 2) gewichtet sichere Matches herunter und hat das Modell
systematisch untertreiben lassen (ECE 0,07 auf 2025). Er bleibt als Option (`config.loss = 'focal'`).

Nach dem Training wird auf dem Validierungsjahr eine Temperatur gefittet (Temperature Scaling auf den
symmetrischen Logits `(z(A,B) - z(B,A)) / 2`). Sie steht mit im Checkpoint und wird in Backtest und
`predict.py` angewendet.

## Evaluation

Alle Auswertungen laufen über `src/evaluation/`. Messlatte ist der **Markt**: die Pinnacle-Quote ohne
Marge, seit Februar 2026 (tennis-data hat dort keine Pinnacle-Quoten mehr) die Betfair-Exchange-Quote.
Ein Modell, das den Markt nicht verbessert, liefert keinen Wett-Edge, egal wie hoch seine Accuracy ist.

```bash
python -m src.evaluation.backtest --start 2020 --end 2026 --tour atp --models market,rank,elo,elo2,gbdt,transformer,stack
python -m src.evaluation.backtest --start 2020 --end 2026 --tour wta --models rank,elo,elo2,gbdt,transformer
python -m src.evaluation.backtest --start 2023 --end 2025 --kelly 0.25 --kelly-cap 0.02 --venues pinnacle,bet365_de
```

**Modell-Leiter** (Phase 3, jede Stufe muss die vorige im Walk-forward schlagen):

| Predictor | Datei | Modell |
|---|---|---|
| `elo2` | `src/elo.py` | K im 538-Stil `c/(n+o)^s`, Mix aus Gesamt- und Untergrund-Elo, optional Best-of-5 über die Satz-Wahrscheinlichkeit, pro Tour. Parameter (`ELO2_PARAMS`) nur auf Daten vor 2020 gewählt: `python -m src.elo --tour atp` |
| `gbdt` | `src/models/gbdt.py` | LightGBM auf `features.feature_table` + Elo2. Walk-forward neu gefittet (`--gbdt-refit quarter`/`year`), Early Stopping und Platt-Kalibrierung auf den letzten 12 Monaten vor dem Fenster, Training mit gespiegelten Zeilen, Vorhersage symmetrisiert. Live-Modell: `python -m src.models.gbdt --tour atp` |
| `transformer` | `src/train.py` | Cross-Entropy (Focal Loss per `config.loss='focal'`), Elo2-Logit als zusätzlicher Input, symmetrische Logits, Temperature Scaling auf `val_year` (Temperatur im Checkpoint) |
| `stack`, `stack_bands`, `stack_tf` | `src/models/stack.py` | Logistische Regression ohne Intercept auf `logit(p_market)` und die Logits von `elo2` + `gbdt` (`_tf`: + Transformer, `_bands`: Interaktion mit dem Quotenbereich), walk-forward auf out-of-sample-Vorhersagen |

`--warmup 1` (Standard) lässt den Walk-forward ein Jahr vor `--start` beginnen, damit Stack und Blend
von Anfang an Trainingsdaten haben. Live nutzt `predict.MatchPredictor` das Modell aus
`config.live_model` (`transformer`, `gbdt` oder `elo2`) über `predict_proba(match) -> p`. Standard ist
`gbdt`, das beste Einzelmodell im Backtest 2020–2026. Vorher einmal `python -m src.models.gbdt --tour atp`
und `--tour wta` ausführen (schreibt `models/gbdt_<tour>.pkl`).

**Stand (Backtest 2020–2026, PR #10):** Jede Stufe schlägt die vorige (ATP-Log-Loss: rank 0,6320 →
elo 0,6283 → elo2 0,6232 → gbdt 0,6089). Kein Modell und kein Stack senkt den Log Loss des Markts
(0,5891) signifikant, das Go/No-Go lautet überall **NO-GO**.

| Datei | Inhalt |
|---|---|
| `predictors.py` | Pro Match `p` = P(Spieler A gewinnt). A ist per Hash der `match_id` (fester Seed) Sieger oder Verlierer, Quoten werden mitgedreht. Predictor: `market` (ohne Marge: Pinnacle → Betfair Exchange → Ø-Quote, Quelle in `market_source`), `rank` (logistisch auf log-Rang-Differenz), `elo` (bestehendes Elo), `elo2`, `gbdt`, `transformer` (Adapter um `models/best_model.pt`), `blend` (logistische Regression auf `logit(p_market)` + `logit(p_model)`), `stack` (siehe oben) |
| `backtest.py` | Walk-forward (Standard monatlich, `--freq week/quarter/year`). Ein Predictor sieht beim Fit nur Matches mit Datum < Testfenster (Assert im Code). Ergebnis: `data/eval/predictions.parquet` |
| `metrics.py` | Log Loss (Hauptmetrik), Brier, Accuracy, ECE, Kalibrierungstabelle, gepaarter Bootstrap und Diebold-Mariano-Test auf Log-Loss-Differenzen gegen den Markt |
| `venues.py` | Venue-Profile: Preisquelle, Steuer/Gebühr, RET-Regel, max. Einsatz |
| `betting.py` | `simulate(predictions, strategy, venue, staking)`: Strategien und Staking als Parameter-Objekte |

**Metriken** werden immer auf derselben Stichprobe verglichen: Matches mit scharfem Marktpreis
(Pinnacle oder Betfair Exchange), ohne W/O und mit einer Vorhersage von jedem angeforderten Modell.
Modelle, die nur einen Teil der Jahre vorhersagen (der Transformer erst nach `val_year`, nur Matches im
Trainings-Filter `config.max_rank`, genug Historie, aktiv), bekommen eine eigene Teilstichprobe, damit die
Hauptstichprobe nicht schrumpft. `val_year` dient dem Checkpointing und dem Temperature Scaling, streng
out-of-sample ist erst das Jahr danach. Der Transformer wird im Walk-forward nicht neu trainiert.

**Venues** (Steuer und Gebühr stecken in der Netto-Quote, mit der auch EV und Kelly rechnen):

| Profil | Preis | Kosten | RET (Annahme) |
|---|---|---|---|
| `pinnacle` | `PSW/PSL` | keine | gewertet |
| `bet365_de` | `B365W/B365L` | 5,3 % Wettsteuer auf den Einsatz | gewertet |
| `avg_de` | `AvgW/AvgL` | 5,3 % Wettsteuer auf den Einsatz | gewertet |
| `betfair` | `BFEW/BFEL` (Betfair Exchange, ab August 2025) | 5 % Kommission auf den Nettogewinn | gewertet |
| `betfair_proxy` (Proxy) | faire Pinnacle-Wahrscheinlichkeit | 5 % Kommission auf den Nettogewinn | gewertet |
| `kalshi` (Proxy) | faire Pinnacle-Wahrscheinlichkeit | Fee pro Kontrakt = ceil_to_cent(0,07 · p · (1−p)) | gewertet |
| `polymarket` (Proxy) | faire Pinnacle-Wahrscheinlichkeit + Spread (Standard 2 Punkte) | Taker-Fee (Standard 0) | gewertet |

W/O ist immer void. Alle Werte sind per `Venue.with_(...)` änderbar (z. B. `ret_rule='void'`,
`max_stake=50`). Die RET-Regeln unterscheiden sich je Anbieter und müssen vor echtem Einsatz in den
AGB geprüft werden. Für Kalshi und Polymarket (und Betfair vor August 2025) gibt es historisch keine
echten Preise. Ihr Preis ist ein Proxy aus Pinnacle, deshalb sind sie im Output als „Proxy“ markiert. Die
tennis-data-Quoten sind Schlussquoten, was den erreichbaren Preis überschätzt. CLV ist im Output
vorgesehen, bleibt aber leer, bis es Quoten-Snapshots mit Zeitstempel gibt.

**Strategien** (`betting.py`, die alten Schleifen aus `main.py` als Parameter):

| Strategie | Alt | Regel |
|---|---|---|
| `Favorite(model, min_odds=1.30)` | Filtered Betting | Auf den Favoriten des Modells ab Mindestquote |
| `Value(model, min_ev=0.0)` | Pure Value | Seite mit höherem EV = p · Netto-Quote − 1, wenn > Schwelle |
| `ConfidenceBand(model, 0.65, 0.80)` | Sniper | Favorit, wenn sein p im Band liegt |
| `BetterRank()` | Baseline | Immer auf den besser platzierten Spieler |
| `Value('market', 0.02)` | neu | „Sharp vs. Rest“: faire Pinnacle-Wahrscheinlichkeit gegen den Preis der Venue |

Staking: `Flat(10)` oder `Kelly(fraction=0.25, cap=0.02)` (fraktionales Kelly auf die aktuelle
Bankroll, gedeckelt). Kennzahlen: Anzahl Wetten (und voids), Trefferquote, Ø-Quote, Yield mit
**Bootstrap-95-%-KI**, ROI auf die Bankroll, Max Drawdown.

**Go/No-Go:** Ein Modell darf erst dann live Wetten auslösen, wenn sein Stack (oder Blend) mit dem
Markt den Log Loss des Markts allein **signifikant senkt** (obere KI-Grenze der Log-Loss-Differenz < 0,
Bootstrap-p und Diebold-Mariano-p < 0,05). Die CLI gibt das Urteil pro Modell aus und bei einem GO die
Segmente (Quotenbereich, Turnierstufe, Untergrund). Ein Yield ohne KI oder auf wenigen hundert
Wetten ist kein Beleg.

`python main.py` trainiert den Transformer und ruft danach dieselbe Auswertung auf (Baseline-Accuracy,
Metriken und Signifikanz gegen den Markt, Strategien auf `bet365_de`, `pinnacle` und `betfair`, Kalibrierung).
`python main.py --eval-only` überspringt das Training.

## Ergebnisse

> **Veraltet und statistisch nicht belastbar.** Die folgenden Zahlen stammen aus den alten
> Simulationen in `main.py` (fester Split, nur Bet365 ohne Wettsteuer, keine Konfidenzintervalle,
> Accuracy statt Log Loss, kein Vergleich mit dem Markt). Der „Sniper“ mit +2,7 % Yield beruht auf
> 110 Wetten mit Quoten um 1,1: Das 95-%-KI des Yields ist dort grob ±10 %, daraus folgt nichts.
> Maßgeblich sind ab jetzt die Tabellen von `python -m src.evaluation.backtest` (siehe „Evaluation“).

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


## Quoten-Collector (`src/odds/`)

Sammelt Quoten mit Zeitstempel und lädt historische Prediction-Market-Preise (ROI-Plan, Phase 2, Punkte 7–8).
Diese Daten lassen sich später nicht nachholen. Der Collector sollte deshalb dauerhaft laufen.

| Venue | Quelle | Voraussetzung |
|---|---|---|
| `polymarket` | Gamma-API (Märkte) + CLOB (Midpoints, `prices-history`) | keine |
| `kalshi` | öffentliche Market-Data-API (Serien `KX{ATP,WTA}MATCH`, `KX{ATP,WTA}CHALLENGERMATCH`) | keine |
| `pinnacle`, `betfair`, `bet365`, … | The Odds API (Regionen `eu,uk`, Markt `h2h`) | `ODDS_API_KEY` (ohne Key übersprungen) |
| `tipico`, `bwin` | bestehende Playwright-Scraper (`betscraper/scraper/sites.py`) | `playwright install chromium` |

```bash
python -m src.odds.collect                    # alle Venues einmal abfragen
python -m src.odds.collect --loop 20          # alle 20 Minuten (Odds API nur jede 6. Runde)
python -m src.odds.collect --venues polymarket,kalshi
python -m src.odds.collect --history          # historische Polymarket-/Kalshi-Preise (dauert ~1-2 h)
python -m src.odds.report                     # Log Loss / Kalibrierung der PM-Closing-Preise
```

Ablage (unter `data/`, nicht im Repo):
- `data/odds/snapshots/YYYY-MM-DD.parquet`: ein Snapshot pro Venue, Match und Abfrage (`src/odds/schema.py`,
  laden mit `load_snapshots(start, end)`)
- `data/odds/history/{polymarket,kalshi}.parquet`: pro Match Eröffnung, Closing (letzter Preis vor der geplanten
  Startzeit), Closing 1 h vorher, Volumen und Sieger aus der Markt-Auflösung

**Dauerhaft laufen lassen** (in einer Cloud-Session läuft nichts dauerhaft). Vorher einmal `mkdir -p data/odds`
(Windows: `mkdir data\odds`), damit die Log-Umleitung funktioniert:
- Linux/macOS mit cron (alle 20 Minuten, Historie einmal pro Nacht nachladen):
  ```
  */20 * * * * cd /pfad/zu/tennis_transformer && python -m src.odds.collect --venues polymarket,kalshi,softbooks >> data/odds/collect.log 2>&1
  5 */2 * * *  cd /pfad/zu/tennis_transformer && ODDS_API_KEY=... python -m src.odds.collect --venues odds_api >> data/odds/collect.log 2>&1
  15 4 * * *   cd /pfad/zu/tennis_transformer && python -m src.odds.collect --history >> data/odds/history.log 2>&1
  ```
  Mit cron läuft jede Runde einzeln, die Odds API deshalb in einer eigenen, selteneren Zeile (Kontingent).
- Windows (Aufgabenplanung), in der Eingabeaufforderung im Repo-Ordner:
  ```
  schtasks /Create /SC MINUTE /MO 20 /TN "TennisOdds" /TR "cmd /c cd /d %CD% && python -m src.odds.collect --venues polymarket,kalshi,softbooks >> data\odds\collect.log 2>&1"
  schtasks /Create /SC HOURLY /MO 2 /TN "TennisOddsApi" /TR "cmd /c cd /d %CD% && python -m src.odds.collect --venues odds_api >> data\odds\collect.log 2>&1"
  schtasks /Create /SC DAILY /ST 04:15 /TN "TennisOddsHistory" /TR "cmd /c cd /d %CD% && python -m src.odds.collect --history >> data\odds\history.log 2>&1"
  ```
  Den Key einmal mit `setx ODDS_API_KEY "..."` setzen. Die Aufgaben laufen nur, solange der Rechner an ist.
- Auf einem Server: `nohup python -m src.odds.collect --loop 20 >> data/odds/collect.log 2>&1 &` oder als
  systemd-Service mit `Restart=always`.
