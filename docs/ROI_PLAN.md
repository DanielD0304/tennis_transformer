# Plan: Maximaler ROI bei Tenniswetten (ATP + WTA, Papier-Trading zuerst)

## Context

Ziel: herausfinden, **wo und womit** sich bei Tenniswetten ein echter, messbarer Edge holen lässt – über
deutsche Soft-Books, Exchange/Sharp-Books und Prediction Markets (Polymarket, Kalshi) hinweg. Zuerst
nur auf Papier. Die Architektur ist offen: Es bleibt, was out-of-sample am besten abschneidet.

### Ehrliche Bestandsaufnahme (aus dem Code)

| Befund | Wo | Warum es für den ROI zählt |
|---|---|---|
| Das Modell (64,4 % Accuracy) ist kaum besser als „besserer Rang gewinnt“ (63,2 %). Der Markt (Pinnacle-Closing) liegt bei ATP erfahrungsgemäß um ~70 %. | README, `main.py` | Das Modell ist **schwächer als der Markt**. Deshalb verliert „Pure Value“ −9,5 % Yield. |
| Der „Sniper“ mit +2,7 % Yield beruht auf 110 Wetten mit Quoten um 1,1. | README | Das ist Rauschen (95-%-KI geschätzt ±10 % Yield). Daraus folgt nichts. |
| Focal Loss als Loss | `src/loss.py`, `src/train.py` | Focal Loss verzerrt die Wahrscheinlichkeiten. Die Kalibrierungstabelle zeigt es: 0,6–0,7 vorhergesagt, aber 81,5 % echte Siegquote. Für Wetten zählt **Kalibrierung (Log Loss)**, nicht Accuracy. |
| Backtest nur gegen `B365W/L` | `src/preprocessing.py:322-330` | Pinnacle (`PSW/PSL`) und Ø-Quoten sind schon im Datensatz, werden aber nicht genutzt. Es gibt keine Benchmark gegen die Pinnacle-Wahrscheinlichkeit und keine Messung des Closing Line Value. |
| Die tennis-data-Quoten sind **Schlussquoten** ohne Zeitstempel | `src/sources/tuk.py` | Ein Backtest gegen Schlussquoten überschätzt den erreichbaren Preis. Für echte Aussagen braucht es **eigene Quoten-Snapshots mit Zeitstempel**. |
| Aufschlag-Stats enden mit dem Sackmann-Archiv (25.05.2026) | `src/sources/sackmann.py`, README | Seitdem ist `has_stats` = 0, live also immer. Das Modell lernt mit Stats und läuft live ohne sie: ein Train/Serve-Mismatch. |
| Kein Feature nutzt den Spielstand (`score`), obwohl ihn **alle** Quellen liefern | `src/preprocessing.py` | Game- und Satzanteile sind ein guter, überall verfügbarer Ersatz für die Aufschlag-Stats. |
| Ein Match nach Aufgabe (RET) zählt wie ein normaler Sieg (nur W/O wird ausgefiltert) | `src/preprocessing.py:is_walkover`, `src/elo.py` | Das verfälscht Elo und Form. Außerdem regeln die Buchmacher RET verschieden (void oder gewertet), was im Backtest zählt. |
| Elo ist simpel: K = 32 fix, kein Untergrund-Elo | `src/elo.py` | Ein gutes Elo (K fällt mit der Anzahl Matches, Untergrund-Mix) ist in der Literatur der stärkste Einzel-Prädiktor. |
| Das Modell ist nicht symmetrisch: p(A,B) ≠ 1 − p(B,A) | `src/model.py` | Billiger Gewinn: beide Reihenfolgen mitteln. |
| Nur ATP, Top-200-Filter; WTA fehlt | `src/sources/*`, `preprocessing.py:306` | WTA hat mehr Quoten-Ineffizienz. tennis-data und Sackmann haben WTA-Daten. |
| Live muss der Untergrund manuell gewählt werden | `betscraper/predictor/views.py`, README | Fehlerquelle. `build._tourney_lookup` kann den Untergrund schon aus dem Turniernamen ableiten. |
| Live wird nur gegen Tipico und bwin gerechnet | `betscraper/scraper/sites.py`, `dataProcessing/value.py` | 5,3 % Wettsteuer und Limits. Kein Vergleich mit Sharp-Quoten oder Prediction Markets. |

### Zentrale Erkenntnis zur Strategie

Gegen die Pinnacle-Schlussquote auf der ATP/WTA-Haupttour gewinnt fast kein öffentliches Modell. Der
realistischste Hebel ist deshalb **nicht nur „ein besseres Modell“**, sondern:

1. **Markt als Wahrheit, Venues als Opfer:** Die faire Wahrscheinlichkeit kommt aus Sharp-Quoten
   (Pinnacle oder Betfair, ohne Marge). Gewettet wird dort, wo eine Venue (Soft-Book, Polymarket, Kalshi)
   nach Gebühren und Steuern davon abweicht. Dafür braucht es kein eigenes Modell. Erfahrungsgemäß ist
   das die zuverlässigste Edge-Quelle.
2. **Eigenes Modell als Korrektiv:** Modell und Marktwahrscheinlichkeit werden gestackt (Logit-Blend).
   Gewettet wird nur, wo beide Signale auf dieselbe Seite zeigen und das Modell den Markt
   out-of-sample *verbessert*.
3. **Erfolg wird an CLV gemessen** (Closing Line Value: schlägt unser Preis die Schlussquote?), nicht am
   kurzfristigen Gewinn. CLV ist nach ~200 Wetten aussagekräftig, ROI erst nach Tausenden.

---

## Phase 1 – Evaluations-Fundament (zuerst, alles andere baut darauf auf)

Neues Modul `src/evaluation/`:
- `backtest.py`: **Walk-forward** statt eines festen Splits. Monatlich neu trainieren bzw. Elo
  fortschreiben und nur auf den Folgemonat vorhersagen. Ergebnis ist eine Tabelle
  `predictions.parquet` (match_id, Datum, Tour, p_model, p_market, alle Quoten).
- `metrics.py`: Log Loss, Brier, Kalibrierungskurve (ECE), Accuracy. Immer im **Vergleich zu drei
  Baselines**: Rang, Elo und **Pinnacle ohne Marge** (`PSW/PSL` → `1/o` normiert). Pinnacle ist die
  eigentliche Messlatte.
- `betting.py`: Simulation pro **Venue-Profil** (`venues.py`: Marge, Steuer/Gebühr, Regel bei RET,
  Limit). Startprofile:
  - DE-Soft: 5,3 % Steuer auf den Einsatz, RET je nach Buch
  - Pinnacle und Betfair: 2–5 % Kommission auf Gewinne
  - Polymarket: Taker-Fee je Markt, Spread
  - Kalshi: Fee ≈ 0,07·p·(1−p) pro Kontrakt
- Kennzahlen: Yield, ROI, Max Drawdown, Anzahl Wetten, **Bootstrap-Konfidenzintervall** für den
  Yield, CLV (sobald Snapshots da sind).
- `main.py`: Die vier `simulate_strategy_*`-Funktionen werden durch einen Aufruf von
  `evaluation.backtest` + `betting` ersetzt (Strategien als Parameter, keine kopierten Schleifen).

**Go/No-Go:** Nur ein Modell, dessen Log Loss *im Blend mit dem Markt* den von Pinnacle allein
signifikant senkt (Diebold-Mariano-Test oder Bootstrap), darf live Wetten auslösen.

## Phase 2 – Datenpipeline

Matchdaten (`src/sources/`):
1. **WTA ergänzen:** `tuk.py` um die WTA-Dateien von tennis-data (gleiches Format), `sackmann.py` um
   `tennis_wta` (sofern es im Archiv liegt), `espn.py` um das WTA-Scoreboard. Neue Spalte `tour` in
   `COLUMNS` (`common.py`). Elo und Historien werden pro Tour getrennt geführt.
2. **Score-Features** (`preprocessing.extract_match_features`): Game-Anteil, Satz-Anteil,
   Tiebreaks und Spielzeit (`minutes`, falls vorhanden) aus `score` parsen. Diese Werte gibt es in
   allen Quellen, also auch live, und sie ersetzen die Aufschlag-Stats als Kern-Feature.
3. **RET/DEF** als eigenes Flag: In Elo und Form schwächer gewichten und im Backtest nach der Regel
   der jeweiligen Venue abrechnen.
4. **Echte Reihenfolge:** Wenn `match_date` vorhanden ist, danach statt nach `tourney_date` sortieren.
   Das betrifft `days_since_match`, Müdigkeit und Elo.
5. **Zusatz-Features (billig, oft wirksam):** Müdigkeit (Matches, Sätze und Minuten der letzten
   7/14 Tage), Indoor/Outdoor, Best-of-5, Heimvorteil (Land Turnier = Land Spieler; dafür eine
   Spieler-Metadaten-Tabelle aus Sackmann `atp_players.csv`/`wta_players.csv`), Alter und Händigkeit.
6. Den **Top-200-Filter** für Trainings-Samples lockern (der Historie-Filter bleibt). Mehr Daten, vor
   allem für WTA.

Quoten (neu, entscheidend):
7. **`src/odds/` – Snapshot-Collector:** Ein Job läuft alle 15–30 Minuten und schreibt
   `data/odds_snapshots.parquet` (Zeitstempel, Venue, Match, Seite, Preis, Liquidität):
   - DE-Soft: bestehende Scraper `betscraper/scraper/sites.py` wiederverwenden (nur aufrufen und loggen)
   - Sharp: The Odds API (Region `eu`/`uk` liefert Pinnacle, Betfair Exchange u. a.; der freie Tarif
     reicht für den Anfang) **oder** die Betfair Exchange API (Konto nötig)
   - **Polymarket:** öffentliche Gamma-API (Märkte suchen) + CLOB `prices-history` (auch historisch!)
   - **Kalshi:** öffentliche Market-Data-API (Tennis-Serien, Orderbuch)
   - Matching auf Matches über die vorhandenen `src/names.player_key`/`candidate_keys`
8. **Historische Prediction-Market-Preise** von Polymarket und Kalshi rückwirkend laden. Damit lässt
   sich sofort ein Backtest „PM-Preis vs. Pinnacle-Closing“ fahren, ohne monatelang zu sammeln.
9. Live-Untergrund automatisch über `build._tourney_lookup` bestimmen (in `predict.py` anbinden).

## Phase 3 – Modell-Leiter (jede Stufe muss die vorige im Walk-forward schlagen)

| Stufe | Modell | Datei |
|---|---|---|
| 0 | Markt: Pinnacle ohne Marge | `evaluation/metrics.py` |
| 1 | Elo neu: K = 250/(n+5)^0,4 (538-Stil), Untergrund-Elo + Mix, getrennt pro Tour | `src/elo.py` (erweitern) |
| 2 | **LightGBM** auf tabellarischen Features: Elo-Differenzen, Score- und Serve/Return-Form (exponentiell gewichtet), Müdigkeit, H2H, Metadaten | neu `src/models/gbdt.py` |
| 3 | Transformer (bestehend), aber mit **Cross-Entropy** statt Focal Loss, Elo-Differenz als zusätzlichem Input, symmetrisiert und mit Temperature Scaling auf dem Validierungsset | `src/model.py`, `src/train.py`, `src/loss.py` |
| 4 | **Stacking mit dem Markt:** logistische Regression auf `[logit(p_market), logit(p_elo), logit(p_gbdt), logit(p_tf)]` plus Interaktionen nach Tour und Quotenbereich | neu `src/models/stack.py` |

Erwartung: Stufe 1 oder 2 schlägt den Transformer bei diesen Datenmengen (~20–40k Samples pro Tour)
wahrscheinlich. Wenn der Transformer im Stack kein Gewicht bekommt, fliegt er aus dem Live-Pfad. Im
Repo und in der UI bleibt er erhalten.

`predict.MatchPredictor` bekommt ein Interface `predict_proba(match) -> p`, damit `betscraper` und
der Backtest jedes Modell gleich benutzen.

## Phase 4 – Wettstrategie und Staking

In `betscraper/dataProcessing/value.py` + `src/evaluation/betting.py`:
1. **Line Shopping über alle Venues:** pro Seite den besten *Netto*-Preis nach Steuer und Gebühr.
2. **Strategie A („Sharp vs. Rest“):** Edge = p_fair(Pinnacle/Betfair) × Netto-Quote − 1 > Schwelle.
3. **Strategie B („Modell im Blend“):** Edge mit p_stack, nur wenn auch das Vorzeichen von A passt.
4. **Staking:** Viertel-Kelly mit Deckel (z. B. max. 2 % der Bankroll), p vorher Richtung Markt
   geschrumpft. Flat Stakes laufen zum Vergleich mit.
5. Schwellen und Segmente (Tour, Quotenbereich, Untergrund, Rundenart) nur im Walk-forward wählen,
   nie auf dem Testjahr.

## Phase 5 – Papier-Trading-Loop

- `src/paper/`: Täglich Vorhersagen + Snapshots → simulierte Wetten zum *damals verfügbaren* Preis
  speichern (`data/paper_bets.parquet`). Nach Matchende über die vorhandene ESPN-Quelle abrechnen und
  den CLV gegen den letzten Snapshot vor Matchbeginn berechnen.
- Django-Ansicht `/scraper/paper/`: Bankroll-Kurve, Yield mit KI, CLV pro Venue und Strategie.
- **Kriterium für echtes Geld:** ≥ 300 Papierwetten, CLV im Mittel > +2 %, untere Grenze des
  Yield-KI > 0 nach Steuer und Gebühren. Vorher rechtlich prüfen, ob Polymarket und Kalshi aus
  Deutschland nutzbar sind. Kalshi ist im Wesentlichen nur für die USA.

## Reihenfolge / Meilensteine

1. Phase 1 (Eval-Framework + Pinnacle-Benchmark): zeigt ehrlich, wo das aktuelle Modell steht.
2. Phase 2.7–2.8 (Snapshot-Collector + historische Polymarket/Kalshi-Preise): beginnt sofort
   Daten zu sammeln, die man nicht nachholen kann.
3. Phase 3 Stufe 1–2 (Elo neu, LightGBM) + Phase 2.1–2.3 (WTA, Score-Features, RET).
4. Phase 3 Stufe 3–4 (Transformer-Fixes, Stacking).
5. Phase 4 + 5 (Strategien, Paper-Loop, Dashboard).

## Kritische Dateien

- Ändern: `src/elo.py`, `src/preprocessing.py`, `src/loss.py`/`src/train.py`, `src/model.py`,
  `src/predict.py`, `src/sources/{common,tuk,sackmann,espn,build}.py`, `main.py`,
  `betscraper/dataProcessing/value.py`, `requirements.txt` (+ lightgbm, pyarrow)
- Neu: `src/evaluation/`, `src/odds/`, `src/models/{gbdt,stack}.py`, `src/paper/`
- Wiederverwenden: `src/names.py` (Namens-Matching für Quellen und Venues),
  `build._tourney_lookup` (Untergrund/Level), `betscraper/scraper/sites.py` (Soft-Book-Scraper),
  `update_data.is_stale` (Refresh-Logik)

## Verifikation

- `pytest` (bestehende Tests + neue für: Score-Parser, Venue-Gebühren, Kelly, Walk-forward ohne
  Leakage, d. h. ein Feature-Datum < Match-Datum wird per Test erzwungen)
- `python -m src.evaluation.backtest --tour atp --models market,elo,gbdt,transformer,stack` gibt eine
  Tabelle mit Log Loss, Brier und ECE pro Modell und Jahr sowie Yield mit KI pro Strategie und Venue aus
- Sanity-Check: Pinnacle ohne Marge muss beim Log Loss *besser* sein als die Rang-Baseline. Sonst ist
  das Quoten-Alignment (W/L-Tausch in `build._copy_odds`) kaputt.
- Collector 48 h laufen lassen: Snapshots aller Venues vorhanden, Match-Abgleich-Quote > 90 %
- Paper-Loop eine Woche laufen lassen: Wetten werden korrekt abgerechnet, CLV erscheint im Dashboard
