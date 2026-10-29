# Ergebnisse Phase 4/5: Timing-Backtest, Challenger/ITF, Go/No-Go (Stand 25.10.2026)

Kurzfassung: **Kein Segment trägt.** Nach Gebühren, Spread, Holm-Korrektur und der Robustheitsregel (Wetten in
mindestens zwei Quartalen) gibt es kein GO. Der einzige große Effekt (Einstieg zur Polymarket-Eröffnung,
+23 bis +33 % Yield) ist ein Preis-Artefakt und nicht handelbar (siehe unten). Übrig bleiben wenige
„Kandidaten“, die beim doppelten Spread verschwinden oder nur auf einem Monat beruhen. Sie sind Material für das
Paper-Trading, kein Grund für echtes Geld.

Reproduzieren:
```bash
python -m src.odds.collect --history --levels main,challenger,itf
python -m src.update_data
python -m src.evaluation.timing --models elo2,gbdt          # Standard: --pm-spread 0.02, Kalshi ab 100 Kontrakten
python -m src.evaluation.timing --models elo2,gbdt --pm-spread 0.04 --venues polymarket   # Sensitivität
```

## Daten-Abdeckung pro Level

Matchdaten `data/matches.csv` (2015 bis 29.10.2026, 669.205 Matches):

| Tour | Level | Sackmann (bis 01.06.2026) | tennis-data | ESPN | Polymarket | Kalshi |
|---|---|---:|---:|---:|---:|---:|
| ATP | main (inkl. Quali) | 46.814 | 1.142 | 48 | 171 | 372 |
| ATP | challenger | 90.052 | – | – | 49 | 5.317 |
| ATP | itf | 186.802 | – | 1 | 2.750 | 13.718 |
| WTA | main (inkl. Quali) | 46.040 | 1.149 | 436 | 143 | 421 |
| WTA | challenger (WTA 125) | 8.264 | – | 419 | – | 351 |
| WTA | itf | 250.292 | – | 249 | 1.367 | 12.838 |

ESPN zeigt nur vereinzelt Challenger, WTA 125 und ITF (ohne Kennzeichnung, Level über den Turniernamen) und keine
Qualifikation. Nach dem Sackmann-Ende tragen deshalb die Markt-Auflösungen Challenger und ITF (ohne Score, RET nicht
erkennbar). Seit Kalshi vollständig geladen ist, stammen die meisten davon aus Kalshi.

Prediction-Market-Historie (`data/odds/history`):
- **Polymarket:** 42.817 abgerechnete Match-Märkte (09/2025 bis 10/2026). 10.433 davon haben keine erkennbare
  Tour (ITF ohne Geschlecht), 1.042 eine 50-50-Auflösung. Zuordnung zu den Matchdaten (Walk-forward-Vorhersagen):
  Haupttour 8.012 (100 %), Challenger 7.015 (100 %), ITF 16.060 (96 %), ohne Level 1.297 (0 %).
- **Kalshi: unvollständig.** Kalshi hat die geteilte Cloud-IP dauerhaft mit 429 gedrosselt. Geladen sind 2.000 von
  23.112 Märkten (Haupttour + Challenger, 02/2025 bis 08/2026), zugeordnet: Challenger 87 %, Haupttour 21 Märkte.
  Kalshi-ITF (~39.000 Märkte) ist nicht geladen. Lokal ohne geteilte IP: `python -m src.odds.collect --history --levels main,challenger,itf`.

Walk-forward-Vorhersagen 2024 bis 2026 (elo2, gbdt; Elo und Fits über alle Level der Tour, quartalsweise Refits):
ATP main 12.374, Challenger 29.278, ITF 51.938; WTA main 11.933, WTA 125 4.686, ITF 67.681.

## Datenfehler im ersten Lauf: Gewinne zur Eröffnung

Der erste Lauf zeigte bei der Polymarket-Eröffnung +25 bis +32 % Yield auf Tausenden Wetten. Ursache:
1. **Platzhalter 0,5:** 42 % der ersten PM-Preise sind exakt 0,5. Das ist der Mittelpreis eines leeren
   Orderbuchs, kein Kurs. Korrektur: Die Historie speichert jetzt den ersten echten Kurs (`move_ts`,
   `move_price_a`). Preise von genau 0,5 gelten bei Polymarket als nicht handelbar.
2. **Auch der erste echte Kurs ist nicht handelbar:** Danach blieb der Effekt (+23 bis +33 %, auch vor Juni 2026,
   wo keine Markt-Daten in den Features stecken, also keine Leakage). Polymarket liefert historisch nur den
   Mittel- bzw. Letztkurs, keine Orderbuch-Tiefe. Kalshi hat zur Eröffnung echte Bid/Ask, und dort verliert dieselbe
   Strategie (Challenger, gbdt: −8,6 % [−15,4 %, −1,9 %]; elo2: −22,8 %). Die PM-Eröffnung ist deshalb als
   „nicht handelbar“ markiert und zählt nicht zum Mehrfachtest (`timing.NOT_TRADABLE`).
3. Daneben wurde ein Fehler im Code von #11 behoben: Polymarket-Märkte ohne Level (`NaN` im Parquet) fielen als
   Ergebnisquelle durch den Level-Filter (17.047 Märkte).
4. Zusätzliche Regel für GO: Wetten in mindestens zwei Quartalen mit je ≥ 30 Wetten. Anlass war ITF −1 h mit
   `blend_gbdt` (+21,9 % auf 1.394 Wetten, aber fast alle im Oktober 2026 und CLV nur +1,4 %). Die Regel reichte
   nicht, siehe nächster Abschnitt.

## Das GO vom 27.10.: Polymarket ITF −1 h ist ein Artefakt

Der Lauf vom 27.10. mit vollständigen Kalshi-Daten ergab ein einziges GO: Polymarket ITF −1 h, `blend_gbdt`,
+23,8 % [+17,1 %, +31,0 %] auf 1.669 Wetten, CLV aber nur +2,0 %. Es ist kein handelbarer Vorteil:

1. **Ein Monat statt zwei Quartale:** 1.618 Wetten liegen im Oktober 2026, das Q3 hat nur 47 (August 15,
   September 32). Die Regel „zwei Quartale mit je ≥ 30 Wetten“ war nur formal erfüllt. Hinzu kommt: PM-ITF-Märkte
   tragen erst ab August 2026 eine Tour. Von Mai bis Juli fehlt bei 8.932 von 9.050 Märkten das Geschlecht, und der
   Backtest ordnet sie nicht zu. PM ITF umfasst also nur August bis Oktober.
2. **Zur selben Zeit bricht die Preisqualität ein** (alle abgerechneten PM-ITF-Märkte, Preis −1 h):

   | Monat | Märkte | ohne Tour | genau 0,5 | 0,46–0,54 | ohne Volumenangabe | Log Loss Preis |
   |---|---:|---:|---:|---:|---:|---:|
   | 05/2026 | 1.660 | 100 % | 2 % | 12 % | 1 % | 0,551 |
   | 06/2026 | 3.217 | 100 % | 1 % | 12 % | 0 % | 0,533 |
   | 07/2026 | 3.818 | 97 % | 1 % | 12 % | 0 % | 0,535 |
   | 08/2026 | 3.924 | 37 % | 11 % | 22 % | 11 % | 0,560 |
   | 09/2026 | 6.181 | 0 % | 24 % | 34 % | 25 % | 0,575 |
   | 10/2026 | 6.876 | 0 % | 12 % | 25 % | 25 % | 0,566 |

3. **Der Gewinn stammt aus Märkten ohne echten Handel.** Dieselben 1.669 Wetten, aufgeteilt. Kalshi-Gegenstück
   heißt: Kalshi-Markt zum selben Match, gezählt werden die bis −1 h gehandelten Kontrakte. Das PM-Gesamtvolumen in $
   gilt für die ganze Laufzeit des Markts:

   | Merkmal | Gruppe | Wetten | Yield | Trefferquote | Ø Kaufpreis | CLV |
   |---|---|---:|---:|---:|---:|---:|
   | Kalshi-Gegenstück | ≥ 100 Kontrakte bis −1 h | 406 | −1,8 % | 50 % | 0,50 | −1,9 % |
   | | < 100 Kontrakte | 639 | +9,6 % | 54 % | 0,49 | −0,4 % |
   | | keines | 624 | +55,0 % | 72 % | 0,48 | +6,9 % |
   | PM-Gesamtvolumen | ohne Angabe | 511 | +49,1 % | 72 % | 0,48 | +6,4 % |
   | | < 100 $ | 171 | +21,8 % | 52 % | 0,45 | +2,6 % |
   | | ≥ 100 $ | 987 | +11,0 % | 54 % | 0,50 | −0,4 % |
   | Abstand des Preises zu 0,5 | ≤ 0,02 | 225 | +64,8 % | 84 % | 0,51 | +0,3 % |
   | | 0,02–0,04 | 208 | +35,3 % | 67 % | 0,50 | +1,8 % |
   | | 0,04–0,10 | 419 | +32,4 % | 64 % | 0,49 | +5,2 % |
   | | > 0,10 | 817 | +5,1 % | 49 % | 0,48 | +0,8 % |

   Auch die 47 Wetten im Q3 (+80 %) liegen fast alle (43) in Märkten ohne liquides Kalshi-Gegenstück. Doppelte
   Märkte am selben Tag sind als Ursache ausgeschlossen (Review vom 29.10.).
4. **PM-ITF-Märkte ohne Kalshi-Gegenstück sind −1 h praktisch Münzwürfe.** Ab August: 1.312 solche Märkte,
   39 % der Preise zwischen 0,46 und 0,54, Log Loss des Preises 0,684 (ln 2 = 0,693). gbdt allein ist dort besser
   (0,600). Mit Kalshi-Gegenstück ist es umgekehrt: Preis 0,516 bis 0,541, gbdt 0,560 bis 0,580. Wo Kalshi bis −1 h
   mindestens 100 Kontrakte gehandelt hat, folgt der PM-Preis dem Kalshi-Preis (Korrelation 0,99, im Mittel 0,02
   Abstand). Der Anteil der PM-ITF-Märkte ohne Kalshi-Gegenstück stieg von 5 % (August, September) auf 19 % im
   Oktober.
5. **Dieselbe Strategie bei Kalshi** (echtes Ask, ≥ 100 Kontrakte): ITF −1 h, `blend_gbdt` −9,8 %
   [−27,8 %, +10,4 %] auf 464 Wetten.

Warum gerade der Blend: Seine Gewichte für das Q4 stammen aus den Quartalen davor. Dort war der PM-Preis −1 h gut,
der Blend folgt ihm also stark (Gewicht Preis 0,90, Modell 0,19). In den dünnen Oktober-Märkten steht der Preis ohne
Information bei etwa 0,5. Dann entscheidet das Modell, und der Blend wettet auf dessen Favoriten zu einer Quote
von etwa 2,0. Bei Preisen nahe 0,5 gibt gbdt der gewetteten Seite im Mittel 0,73, getroffen werden 76 %: Das
Modell liegt richtig, der Preis ist keiner. gbdt allein bringt im selben Segment nichts (+0,6 % [−4,1 %, +5,6 %]), weil es auch in
liquiden Märkten gegen gute Preise wettet. Der Gewinn misst also Modell gegen Münzwurf, nicht Modell gegen Markt.

**Sollen Preise nahe 0,5 wie der Platzhalter 0,5 behandelt werden? Nein.** In liquiden Märkten (Kalshi-Gegenstück
mit ≥ 100 Kontrakten, PM-Volumen > 2.000 $) sind Preise um 0,5 echte enge Matches. Dort ist der Preis besser als
gbdt (298 Märkte ab August, Log Loss 0,693 gegen 0,731), der gbdt-Favorit gewinnt nur 50 %. Ein Band um 0,5 würde solche echten Kurse
streichen und dünne Märkte mit Preisen abseits von 0,5 behalten (GO-Wetten mit 0,04 bis 0,10 Abstand: +32 %).
Das Problem ist der fehlende Handel, nicht der Preis. Die PM-Historie hat aber keine Orderbuch-Tiefe, nur das
Gesamtvolumen (inklusive Live-Handel, also ein Blick in die Zukunft). Deshalb fängt die Robustheitsregel solche
Effekte ab, und im Paper-Trading zählt das echte Ask.

**Neue Regel** (`timing.add_decision`): Zusätzlich zu Holm-p < 0,05, KI-Untergrenze > 0 und CLV > 0 verlangt GO:
- in mindestens zwei Quartalen mit je ≥ 30 Wetten einzeln eine KI-Untergrenze des Yields > 0 (`n_periods_pos`),
- kein Quartal mit mehr als 60 % der Wetten (`max_period_share`).

Auf den Lauf vom 27.10. angewendet bleibt kein GO, das frühere GO wird zum Kandidaten. Die erste Bedingung allein
hätte nicht gereicht: Die 47 Q3-Wetten bestätigen den Gewinn mit +80 %, weil sie aus denselben dünnen Märkten
stammen. Erst die Konzentration (97 % der Wetten im Q4) stoppt es.

## Timing-Tabelle (gbdt und Blend aus gbdt + Preis)

Flat-Einsatz, Polymarket-Spread 2 %, Kalshi zum Ask mit Gebühr. Schwellen walk-forward pro Quartal. Yield mit 95-%-
Bootstrap-KI, CLV gegen das Closing der Venue, `ll_diff` = Log Loss Modell − Log Loss Preis zum Zeitpunkt t
(negativ = Modell besser), `n_periods` = Quartale mit ≥ 30 Wetten. Alle Modelle (inkl. elo2) siehe unten.

| venue | level | point | model | n_bets | n_periods | yield | yield_ci_low | yield_ci_high | clv | ll_diff | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kalshi | challenger | open | blend_gbdt | 0 | 0 | – | – | – | – | +0.0000 | zu wenig |
| kalshi | challenger | open | gbdt | 1089 | 1 | -8.6% | -15.4% | -1.9% | -2.9% | +0.0143 | NO-GO |
| kalshi | challenger | 24h | blend_gbdt | 0 | 0 | – | – | – | – | -0.0000 | zu wenig |
| kalshi | challenger | 24h | gbdt | 22 | 0 | +23.1% | -29.6% | +82.8% | +2.0% | +0.0352 | zu wenig |
| kalshi | challenger | 6h | blend_gbdt | 0 | 0 | – | – | – | – | -0.0000 | zu wenig |
| kalshi | challenger | 6h | gbdt | 738 | 1 | -14.0% | -23.0% | -4.3% | -1.4% | +0.0325 | NO-GO |
| kalshi | challenger | 1h | blend_gbdt | 0 | 0 | – | – | – | – | -0.0000 | zu wenig |
| kalshi | challenger | 1h | gbdt | 492 | 1 | -11.0% | -25.7% | +3.6% | -2.6% | +0.0334 | NO-GO |
| kalshi | challenger | close | blend_gbdt | 0 | 0 | – | – | – | – | -0.0000 | zu wenig |
| kalshi | challenger | close | gbdt | 505 | 1 | -10.0% | -23.7% | +3.8% | -2.5% | +0.0337 | NO-GO |
| kalshi | main | open | blend_gbdt | 0 | 0 | – | – | – | – | +0.0000 | zu wenig |
| kalshi | main | open | gbdt | 0 | 0 | – | – | – | – | +0.1195 | zu wenig |
| kalshi | main | 24h | blend_gbdt | 0 | 0 | – | – | – | – | – | zu wenig |
| kalshi | main | 24h | gbdt | 0 | 0 | – | – | – | – | – | zu wenig |
| kalshi | main | 6h | blend_gbdt | 0 | 0 | – | – | – | – | +0.0000 | zu wenig |
| kalshi | main | 6h | gbdt | 0 | 0 | – | – | – | – | +0.0144 | zu wenig |
| kalshi | main | 1h | blend_gbdt | 0 | 0 | – | – | – | – | -0.0000 | zu wenig |
| kalshi | main | 1h | gbdt | 0 | 0 | – | – | – | – | +0.0428 | zu wenig |
| kalshi | main | close | blend_gbdt | 0 | 0 | – | – | – | – | -0.0000 | zu wenig |
| kalshi | main | close | gbdt | 1 | 0 | -100.0% | -100.0% | -100.0% | -48.3% | +0.1042 | zu wenig |
| polymarket | challenger | open | blend_gbdt | 5248 | 4 | +32.5% | +27.9% | +37.3% | +32.9% | -0.0499 | nicht handelbar |
| polymarket | challenger | open | gbdt | 5806 | 4 | +29.4% | +24.9% | +33.8% | +29.8% | -0.0483 | nicht handelbar |
| polymarket | challenger | 24h | blend_gbdt | 230 | 3 | +14.4% | -2.1% | +30.0% | +16.9% | -0.0023 | Kandidat |
| polymarket | challenger | 24h | gbdt | 1452 | 4 | +6.9% | +0.1% | +14.1% | +6.4% | +0.0085 | Kandidat |
| polymarket | challenger | 6h | blend_gbdt | 547 | 2 | -3.3% | -11.0% | +4.5% | -2.1% | +0.0010 | NO-GO |
| polymarket | challenger | 6h | gbdt | 2277 | 4 | -7.5% | -13.8% | -0.8% | -3.3% | +0.0208 | NO-GO |
| polymarket | challenger | 1h | blend_gbdt | 129 | 1 | -4.1% | -21.4% | +14.5% | -1.7% | +0.0009 | NO-GO |
| polymarket | challenger | 1h | gbdt | 2254 | 4 | -5.3% | -11.8% | +1.5% | -3.8% | +0.0224 | NO-GO |
| polymarket | challenger | close | blend_gbdt | 136 | 1 | -2.9% | -20.1% | +14.2% | -2.7% | +0.0007 | NO-GO |
| polymarket | challenger | close | gbdt | 2366 | 4 | -6.0% | -12.2% | +0.8% | -4.1% | +0.0238 | NO-GO |
| polymarket | itf | open | blend_gbdt | 11965 | 3 | +25.4% | +23.2% | +27.8% | +17.7% | -0.0581 | nicht handelbar |
| polymarket | itf | open | gbdt | 12994 | 3 | +23.3% | +21.1% | +25.5% | +16.0% | -0.0540 | nicht handelbar |
| polymarket | itf | 24h | blend_gbdt | 73 | 1 | -3.6% | -28.4% | +21.1% | +6.2% | +0.0005 | NO-GO |
| polymarket | itf | 24h | gbdt | 194 | 3 | +2.8% | -16.8% | +24.3% | +5.3% | +0.0214 | NO-GO |
| polymarket | itf | 6h | blend_gbdt | 3372 | 1 | +5.3% | +1.4% | +8.9% | +1.0% | -0.0041 | Kandidat |
| polymarket | itf | 6h | gbdt | 6226 | 3 | +1.9% | -2.3% | +6.3% | +1.8% | +0.0296 | NO-GO |
| polymarket | itf | 1h | blend_gbdt | 1394 | 1 | +21.9% | +15.1% | +29.7% | +1.4% | -0.0021 | Kandidat |
| polymarket | itf | 1h | gbdt | 6673 | 3 | -0.3% | -4.7% | +4.2% | -1.3% | +0.0381 | NO-GO |
| polymarket | itf | close | blend_gbdt | 1005 | 1 | +10.8% | +1.4% | +20.1% | -2.7% | -0.0012 | Kandidat |
| polymarket | itf | close | gbdt | 6325 | 3 | -5.0% | -9.7% | -0.3% | -5.4% | +0.0467 | NO-GO |
| polymarket | main | open | blend_gbdt | 4475 | 4 | +32.9% | +28.0% | +38.2% | +28.4% | -0.0457 | nicht handelbar |
| polymarket | main | open | gbdt | 5543 | 6 | +29.8% | +25.4% | +34.4% | +26.0% | -0.0513 | nicht handelbar |
| polymarket | main | 24h | blend_gbdt | 256 | 2 | -11.8% | -30.6% | +9.7% | +0.9% | +0.0009 | NO-GO |
| polymarket | main | 24h | gbdt | 1999 | 6 | -1.4% | -8.1% | +5.3% | -0.8% | +0.0086 | NO-GO |
| polymarket | main | 6h | blend_gbdt | 423 | 2 | +1.9% | -9.9% | +14.3% | -1.4% | -0.0000 | NO-GO |
| polymarket | main | 6h | gbdt | 3011 | 6 | -0.7% | -6.4% | +5.6% | -2.4% | +0.0094 | NO-GO |
| polymarket | main | 1h | blend_gbdt | 118 | 2 | -8.3% | -33.4% | +19.4% | +6.2% | +0.0001 | NO-GO |
| polymarket | main | 1h | gbdt | 2965 | 6 | -2.0% | -7.9% | +3.8% | -3.0% | +0.0121 | NO-GO |
| polymarket | main | close | blend_gbdt | 101 | 2 | -25.9% | -48.1% | +0.7% | -3.9% | +0.0003 | NO-GO |
| polymarket | main | close | gbdt | 2728 | 6 | -1.9% | -7.9% | +4.3% | -4.3% | +0.0157 | NO-GO |

Log Loss: Ab −6 h ist der Preis bei jedem Level besser als jedes Modell allein (`ll_diff` > 0). Der Blend gewinnt
nur wenige Tausendstel. Zu −24 h ist der Preis bei Challenger ähnlich gut wie gbdt.

## Go/No-Go pro Segment

Holm-Korrektur über alle Einzel-Segmente mit ≥ 30 Wetten (ohne PM-Eröffnung). GO verlangt p_holm < 0,05, untere
KI-Grenze > 0, CLV > 0 und ≥ 2 Quartale.

- **GO: keines.** 45 Segmente NO-GO, 36 mit zu wenig Wetten, 12 nicht handelbar (PM-Eröffnung).
- **Kandidaten** (nur Hinweis, Fallzahl in Klammern):
  - Polymarket Challenger −24 h, gbdt: +6,9 % [+0,1 %, +14,1 %], CLV +6,4 % (1.452). elo2: +6,4 % [−2,0 %, +14,8 %]
    (1.331). Beim Spread 4 % verschwinden beide.
  - Polymarket Challenger −24 h, blend_elo2/blend_gbdt: +29 % (110) bzw. +14 % (230), sehr breite KI.
  - Polymarket ITF −6 h / −1 h / Closing, blend_gbdt: +5,3 % (3.372), +21,9 % (1.394), +10,8 % (1.005). Fast nur
    Oktober 2026 und CLV um 0. Das spricht eher gegen einen echten Vorteil.
- **Kalshi** (Teildaten, Challenger): alle Zeitpunkte mit genug Wetten negativ (−9 bis −23 %); −24 h hat nur 22 Wetten.
- **Haupttour**: überall NO-GO, wie schon gegen Pinnacle.

Was das heißt: Frühe Preise sind ungenauer (Log Loss −24 h 0,600, Closing 0,556). Handelbar ist der Unterschied
mit elo2/gbdt aber nicht. Der einzig plausible Ansatz ist Challenger um −24 h. Er ist zu dünn belegt und hängt am
angenommenen Spread. Er gehört ins Paper-Trading, wo echte Bid/Ask und Orderbuch-Tiefe mitgeschrieben werden.

## Sharp vs. Rest

Ohne Pinnacle/Betfair-Snapshots nicht auswertbar. Die Odds API listete in dieser Session keine aktiven
Tennis-Turniere. Die Logik ist mit Fixtures getestet und läuft im Paper-Loop mit `--odds-api`.

## Odds API

Key funktioniert (HTTP 200). Verbraucht: **0 Credits** (nur das kostenlose `/sports`), Kontingent 500/500. Aktive
Tennis-Events: keine (25.10.2026, zweimal geprüft). Historische Quoten sind im Gratis-Plan nicht enthalten.

## Alle Segmente (inkl. elo2 und blend_elo2)

| venue | level | point | model | n_markets | n_bets | n_periods | yield | yield_ci_low | yield_ci_high | clv | clv_sharp | ll_diff | ll_ci_low | ll_ci_high | p_holm | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kalshi | challenger | open | blend_elo2 | 1713 | 0 | 0 | – | – | – | – | – | +0.0000 | -0.0000 | +0.0000 | – | zu wenig |
| kalshi | challenger | open | blend_gbdt | 1713 | 0 | 0 | – | – | – | – | – | +0.0000 | -0.0000 | +0.0000 | – | zu wenig |
| kalshi | challenger | open | elo2 | 1713 | 452 | 1 | -22.8% | -35.1% | -10.6% | -5.0% | – | +0.0215 | +0.0086 | +0.0342 | 1.000 | NO-GO |
| kalshi | challenger | open | gbdt | 1713 | 1089 | 1 | -8.6% | -15.4% | -1.9% | -2.9% | – | +0.0143 | +0.0016 | +0.0282 | 1.000 | NO-GO |
| kalshi | challenger | 24h | blend_elo2 | 80 | 0 | 0 | – | – | – | – | – | -0.0000 | -0.0000 | +0.0000 | – | zu wenig |
| kalshi | challenger | 24h | blend_gbdt | 80 | 0 | 0 | – | – | – | – | – | -0.0000 | -0.0000 | +0.0000 | – | zu wenig |
| kalshi | challenger | 24h | elo2 | 80 | 24 | 0 | +34.8% | -29.6% | +104.4% | +1.9% | – | +0.0379 | -0.0255 | +0.0925 | – | zu wenig |
| kalshi | challenger | 24h | gbdt | 80 | 22 | 0 | +23.1% | -29.6% | +82.8% | +2.0% | – | +0.0352 | -0.0220 | +0.0842 | – | zu wenig |
| kalshi | challenger | 6h | blend_elo2 | 1684 | 0 | 0 | – | – | – | – | – | -0.0000 | -0.0000 | +0.0000 | – | zu wenig |
| kalshi | challenger | 6h | blend_gbdt | 1684 | 0 | 0 | – | – | – | – | – | -0.0000 | -0.0000 | +0.0000 | – | zu wenig |
| kalshi | challenger | 6h | elo2 | 1684 | 801 | 1 | -16.0% | -25.3% | -6.1% | -1.7% | – | +0.0399 | +0.0271 | +0.0521 | 1.000 | NO-GO |
| kalshi | challenger | 6h | gbdt | 1684 | 738 | 1 | -14.0% | -23.0% | -4.3% | -1.4% | – | +0.0325 | +0.0189 | +0.0449 | 1.000 | NO-GO |
| kalshi | challenger | 1h | blend_elo2 | 1708 | 0 | 0 | – | – | – | – | – | -0.0000 | -0.0000 | -0.0000 | – | zu wenig |
| kalshi | challenger | 1h | blend_gbdt | 1708 | 0 | 0 | – | – | – | – | – | -0.0000 | -0.0000 | -0.0000 | – | zu wenig |
| kalshi | challenger | 1h | elo2 | 1708 | 623 | 1 | -14.6% | -27.8% | -1.2% | -3.3% | – | +0.0407 | +0.0283 | +0.0530 | 1.000 | NO-GO |
| kalshi | challenger | 1h | gbdt | 1708 | 492 | 1 | -11.0% | -25.7% | +3.6% | -2.6% | – | +0.0334 | +0.0208 | +0.0459 | 1.000 | NO-GO |
| kalshi | challenger | close | blend_elo2 | 1713 | 0 | 0 | – | – | – | – | – | -0.0000 | -0.0000 | +0.0000 | – | zu wenig |
| kalshi | challenger | close | blend_gbdt | 1713 | 0 | 0 | – | – | – | – | – | -0.0000 | -0.0000 | +0.0000 | – | zu wenig |
| kalshi | challenger | close | elo2 | 1713 | 648 | 1 | -14.6% | -26.9% | -2.2% | -3.3% | – | +0.0409 | +0.0291 | +0.0534 | 1.000 | NO-GO |
| kalshi | challenger | close | gbdt | 1713 | 505 | 1 | -10.0% | -23.7% | +3.8% | -2.5% | – | +0.0337 | +0.0215 | +0.0465 | 1.000 | NO-GO |
| kalshi | main | open | blend_elo2 | 15 | 0 | 0 | – | – | – | – | – | +0.0000 | +0.0000 | +0.0000 | – | zu wenig |
| kalshi | main | open | blend_gbdt | 15 | 0 | 0 | – | – | – | – | – | +0.0000 | +0.0000 | +0.0000 | – | zu wenig |
| kalshi | main | open | elo2 | 15 | 1 | 0 | -100.0% | -100.0% | -100.0% | -27.5% | -1.6% | +0.1011 | -0.1265 | +0.4237 | – | zu wenig |
| kalshi | main | open | gbdt | 15 | 0 | 0 | – | – | – | – | – | +0.1195 | -0.1534 | +0.5282 | – | zu wenig |
| kalshi | main | 24h | blend_elo2 | 0 | 0 | 0 | – | – | – | – | – | – | – | – | – | zu wenig |
| kalshi | main | 24h | blend_gbdt | 0 | 0 | 0 | – | – | – | – | – | – | – | – | – | zu wenig |
| kalshi | main | 24h | elo2 | 0 | 0 | 0 | – | – | – | – | – | – | – | – | – | zu wenig |
| kalshi | main | 24h | gbdt | 0 | 0 | 0 | – | – | – | – | – | – | – | – | – | zu wenig |
| kalshi | main | 6h | blend_elo2 | 10 | 0 | 0 | – | – | – | – | – | +0.0000 | +0.0000 | +0.0000 | – | zu wenig |
| kalshi | main | 6h | blend_gbdt | 10 | 0 | 0 | – | – | – | – | – | +0.0000 | +0.0000 | +0.0000 | – | zu wenig |
| kalshi | main | 6h | elo2 | 10 | 0 | 0 | – | – | – | – | – | -0.0333 | -0.1955 | +0.1180 | – | zu wenig |
| kalshi | main | 6h | gbdt | 10 | 0 | 0 | – | – | – | – | – | +0.0144 | -0.1811 | +0.2092 | – | zu wenig |
| kalshi | main | 1h | blend_elo2 | 15 | 0 | 0 | – | – | – | – | – | -0.0000 | -0.0000 | +0.0000 | – | zu wenig |
| kalshi | main | 1h | blend_gbdt | 15 | 0 | 0 | – | – | – | – | – | -0.0000 | -0.0000 | +0.0000 | – | zu wenig |
| kalshi | main | 1h | elo2 | 15 | 0 | 0 | – | – | – | – | – | +0.0244 | -0.0832 | +0.1233 | – | zu wenig |
| kalshi | main | 1h | gbdt | 15 | 0 | 0 | – | – | – | – | – | +0.0428 | -0.0626 | +0.1473 | – | zu wenig |
| kalshi | main | close | blend_elo2 | 15 | 0 | 0 | – | – | – | – | – | -0.0000 | -0.0000 | +0.0000 | – | zu wenig |
| kalshi | main | close | blend_gbdt | 15 | 0 | 0 | – | – | – | – | – | -0.0000 | -0.0000 | +0.0000 | – | zu wenig |
| kalshi | main | close | elo2 | 15 | 1 | 0 | -100.0% | -100.0% | -100.0% | -48.3% | +135.3% | +0.0858 | -0.0267 | +0.2378 | – | zu wenig |
| kalshi | main | close | gbdt | 15 | 1 | 0 | -100.0% | -100.0% | -100.0% | -48.3% | +135.3% | +0.1042 | -0.0016 | +0.2631 | – | zu wenig |
| polymarket | challenger | open | blend_elo2 | 6700 | 5302 | 4 | +32.4% | +27.8% | +37.4% | +31.8% | +49.9% | -0.0458 | -0.0538 | -0.0377 | – | nicht handelbar |
| polymarket | challenger | open | blend_gbdt | 6700 | 5248 | 4 | +32.5% | +27.9% | +37.3% | +32.9% | +49.9% | -0.0499 | -0.0581 | -0.0414 | – | nicht handelbar |
| polymarket | challenger | open | elo2 | 6700 | 5575 | 4 | +30.0% | +25.2% | +34.6% | +29.8% | +49.9% | -0.0433 | -0.0516 | -0.0346 | – | nicht handelbar |
| polymarket | challenger | open | gbdt | 6700 | 5806 | 4 | +29.4% | +24.9% | +33.8% | +29.8% | +49.9% | -0.0483 | -0.0572 | -0.0391 | – | nicht handelbar |
| polymarket | challenger | 24h | blend_elo2 | 3039 | 110 | 1 | +29.1% | +1.7% | +57.6% | +30.7% | – | -0.0016 | -0.0045 | +0.0012 | 1.000 | Kandidat |
| polymarket | challenger | 24h | blend_gbdt | 3039 | 230 | 3 | +14.4% | -2.1% | +30.0% | +16.9% | – | -0.0023 | -0.0058 | +0.0011 | 1.000 | Kandidat |
| polymarket | challenger | 24h | elo2 | 3039 | 1331 | 4 | +6.4% | -2.0% | +14.8% | +5.2% | – | +0.0107 | +0.0032 | +0.0187 | 1.000 | Kandidat |
| polymarket | challenger | 24h | gbdt | 3039 | 1452 | 4 | +6.9% | +0.1% | +14.1% | +6.4% | – | +0.0085 | +0.0002 | +0.0174 | 1.000 | Kandidat |
| polymarket | challenger | 6h | blend_elo2 | 6875 | 86 | 1 | -12.6% | -43.0% | +22.7% | -0.2% | – | +0.0014 | +0.0002 | +0.0025 | 1.000 | NO-GO |
| polymarket | challenger | 6h | blend_gbdt | 6875 | 547 | 2 | -3.3% | -11.0% | +4.5% | -2.1% | – | +0.0010 | -0.0003 | +0.0022 | 1.000 | NO-GO |
| polymarket | challenger | 6h | elo2 | 6875 | 3208 | 4 | -6.7% | -12.3% | -1.3% | -4.1% | +5.6% | +0.0259 | +0.0212 | +0.0305 | 1.000 | NO-GO |
| polymarket | challenger | 6h | gbdt | 6875 | 2277 | 4 | -7.5% | -13.8% | -0.8% | -3.3% | – | +0.0208 | +0.0155 | +0.0255 | 1.000 | NO-GO |
| polymarket | challenger | 1h | blend_elo2 | 6944 | 64 | 1 | -11.7% | -50.2% | +33.5% | +1.9% | – | +0.0013 | +0.0004 | +0.0022 | 1.000 | NO-GO |
| polymarket | challenger | 1h | blend_gbdt | 6944 | 129 | 1 | -4.1% | -21.4% | +14.5% | -1.7% | – | +0.0009 | -0.0001 | +0.0018 | 1.000 | NO-GO |
| polymarket | challenger | 1h | elo2 | 6944 | 3293 | 4 | -7.0% | -12.7% | -1.1% | -3.8% | +2.0% | +0.0277 | +0.0227 | +0.0324 | 1.000 | NO-GO |
| polymarket | challenger | 1h | gbdt | 6944 | 2254 | 4 | -5.3% | -11.8% | +1.5% | -3.8% | – | +0.0224 | +0.0172 | +0.0274 | 1.000 | NO-GO |
| polymarket | challenger | close | blend_elo2 | 6952 | 58 | 1 | -28.2% | -60.5% | +4.6% | -4.8% | – | +0.0012 | +0.0003 | +0.0021 | 1.000 | NO-GO |
| polymarket | challenger | close | blend_gbdt | 6952 | 136 | 1 | -2.9% | -20.1% | +14.2% | -2.7% | – | +0.0007 | -0.0002 | +0.0015 | 1.000 | NO-GO |
| polymarket | challenger | close | elo2 | 6952 | 3322 | 4 | -5.9% | -11.5% | -0.2% | -4.3% | +5.6% | +0.0290 | +0.0237 | +0.0344 | 1.000 | NO-GO |
| polymarket | challenger | close | gbdt | 6952 | 2366 | 4 | -6.0% | -12.2% | +0.8% | -4.1% | – | +0.0238 | +0.0186 | +0.0293 | 1.000 | NO-GO |
| polymarket | itf | open | blend_elo2 | 13660 | 11904 | 3 | +12.8% | +10.5% | +15.3% | +13.4% | – | -0.0147 | -0.0197 | -0.0095 | – | nicht handelbar |
| polymarket | itf | open | blend_gbdt | 13660 | 11965 | 3 | +25.4% | +23.2% | +27.8% | +17.7% | – | -0.0581 | -0.0638 | -0.0522 | – | nicht handelbar |
| polymarket | itf | open | elo2 | 13660 | 12459 | 3 | +11.4% | +9.1% | +13.9% | +12.1% | – | -0.0059 | -0.0115 | -0.0001 | – | nicht handelbar |
| polymarket | itf | open | gbdt | 13660 | 12994 | 3 | +23.3% | +21.1% | +25.5% | +16.0% | – | -0.0540 | -0.0603 | -0.0476 | – | nicht handelbar |
| polymarket | itf | 24h | blend_elo2 | 474 | 5 | 0 | -20.0% | -100.0% | +60.8% | +39.8% | – | -0.0012 | -0.0070 | +0.0045 | – | zu wenig |
| polymarket | itf | 24h | blend_gbdt | 474 | 73 | 1 | -3.6% | -28.4% | +21.1% | +6.2% | – | +0.0005 | -0.0074 | +0.0082 | 1.000 | NO-GO |
| polymarket | itf | 24h | elo2 | 474 | 178 | 3 | +1.7% | -18.5% | +23.9% | +5.4% | – | +0.0155 | -0.0035 | +0.0360 | 1.000 | NO-GO |
| polymarket | itf | 24h | gbdt | 474 | 194 | 3 | +2.8% | -16.8% | +24.3% | +5.3% | – | +0.0214 | +0.0010 | +0.0433 | 1.000 | NO-GO |
| polymarket | itf | 6h | blend_elo2 | 12073 | 2 | 0 | -100.0% | -100.0% | -100.0% | -8.4% | – | -0.0003 | -0.0007 | +0.0001 | – | zu wenig |
| polymarket | itf | 6h | blend_gbdt | 12073 | 3372 | 1 | +5.3% | +1.4% | +8.9% | +1.0% | – | -0.0041 | -0.0052 | -0.0029 | 0.154 | Kandidat |
| polymarket | itf | 6h | elo2 | 12073 | 6887 | 3 | -7.7% | -12.2% | -3.3% | -1.3% | – | +0.0621 | +0.0564 | +0.0683 | 1.000 | NO-GO |
| polymarket | itf | 6h | gbdt | 12073 | 6226 | 3 | +1.9% | -2.3% | +6.3% | +1.8% | – | +0.0296 | +0.0239 | +0.0356 | 1.000 | NO-GO |
| polymarket | itf | 1h | blend_elo2 | 12738 | 2 | 0 | -100.0% | -100.0% | -100.0% | -5.4% | – | +0.0001 | -0.0002 | +0.0004 | – | zu wenig |
| polymarket | itf | 1h | blend_gbdt | 12738 | 1394 | 1 | +21.9% | +15.1% | +29.7% | +1.4% | – | -0.0021 | -0.0027 | -0.0015 | 0.000 | Kandidat |
| polymarket | itf | 1h | elo2 | 12738 | 7627 | 3 | -7.4% | -11.6% | -3.1% | -1.8% | – | +0.0739 | +0.0677 | +0.0800 | 1.000 | NO-GO |
| polymarket | itf | 1h | gbdt | 12738 | 6673 | 3 | -0.3% | -4.7% | +4.2% | -1.3% | – | +0.0381 | +0.0323 | +0.0439 | 1.000 | NO-GO |
| polymarket | itf | close | blend_elo2 | 12156 | 0 | 0 | – | – | – | – | – | -0.0000 | -0.0003 | +0.0002 | – | zu wenig |
| polymarket | itf | close | blend_gbdt | 12156 | 1005 | 1 | +10.8% | +1.4% | +20.1% | -2.7% | – | -0.0012 | -0.0018 | -0.0005 | 0.486 | Kandidat |
| polymarket | itf | close | elo2 | 12156 | 7313 | 3 | -10.7% | -15.3% | -6.0% | -5.9% | – | +0.0827 | +0.0759 | +0.0886 | 1.000 | NO-GO |
| polymarket | itf | close | gbdt | 12156 | 6325 | 3 | -5.0% | -9.7% | -0.3% | -5.4% | – | +0.0467 | +0.0407 | +0.0525 | 1.000 | NO-GO |
| polymarket | main | open | blend_elo2 | 7442 | 4522 | 4 | +30.6% | +25.5% | +35.9% | +27.7% | +22.1% | -0.0399 | -0.0468 | -0.0332 | – | nicht handelbar |
| polymarket | main | open | blend_gbdt | 7442 | 4475 | 4 | +32.9% | +28.0% | +38.2% | +28.4% | +23.4% | -0.0457 | -0.0526 | -0.0386 | – | nicht handelbar |
| polymarket | main | open | elo2 | 7442 | 5483 | 6 | +28.2% | +23.9% | +33.2% | +25.2% | +20.9% | -0.0446 | -0.0523 | -0.0377 | – | nicht handelbar |
| polymarket | main | open | gbdt | 7442 | 5543 | 6 | +29.8% | +25.4% | +34.4% | +26.0% | +21.9% | -0.0513 | -0.0592 | -0.0436 | – | nicht handelbar |
| polymarket | main | 24h | blend_elo2 | 3971 | 154 | 1 | -6.3% | -38.2% | +28.6% | +2.1% | +1.1% | +0.0029 | +0.0005 | +0.0052 | 1.000 | NO-GO |
| polymarket | main | 24h | blend_gbdt | 3971 | 256 | 2 | -11.8% | -30.6% | +9.7% | +0.9% | -1.1% | +0.0009 | -0.0013 | +0.0033 | 1.000 | NO-GO |
| polymarket | main | 24h | elo2 | 3971 | 1848 | 6 | -10.7% | -17.7% | -3.1% | -2.0% | -2.2% | +0.0161 | +0.0104 | +0.0220 | 1.000 | NO-GO |
| polymarket | main | 24h | gbdt | 3971 | 1999 | 6 | -1.4% | -8.1% | +5.3% | -0.8% | -0.8% | +0.0086 | +0.0028 | +0.0149 | 1.000 | NO-GO |
| polymarket | main | 6h | blend_elo2 | 7462 | 145 | 1 | +3.0% | -29.0% | +39.4% | -1.2% | +2.4% | +0.0013 | +0.0001 | +0.0025 | 1.000 | NO-GO |
| polymarket | main | 6h | blend_gbdt | 7462 | 423 | 2 | +1.9% | -9.9% | +14.3% | -1.4% | +0.4% | -0.0000 | -0.0012 | +0.0012 | 1.000 | NO-GO |
| polymarket | main | 6h | elo2 | 7462 | 3845 | 6 | -4.8% | -10.0% | +0.7% | -3.0% | -1.0% | +0.0163 | +0.0120 | +0.0206 | 1.000 | NO-GO |
| polymarket | main | 6h | gbdt | 7462 | 3011 | 6 | -0.7% | -6.4% | +5.6% | -2.4% | +0.3% | +0.0094 | +0.0054 | +0.0137 | 1.000 | NO-GO |
| polymarket | main | 1h | blend_elo2 | 7618 | 93 | 1 | +20.7% | -27.4% | +83.3% | +6.2% | -2.8% | +0.0011 | +0.0001 | +0.0022 | 1.000 | NO-GO |
| polymarket | main | 1h | blend_gbdt | 7618 | 118 | 2 | -8.3% | -33.4% | +19.4% | +6.2% | +1.2% | +0.0001 | -0.0008 | +0.0011 | 1.000 | NO-GO |
| polymarket | main | 1h | elo2 | 7618 | 4321 | 6 | -5.0% | -9.9% | -0.1% | -3.0% | -0.7% | +0.0189 | +0.0144 | +0.0234 | 1.000 | NO-GO |
| polymarket | main | 1h | gbdt | 7618 | 2965 | 6 | -2.0% | -7.9% | +3.8% | -3.0% | +0.5% | +0.0121 | +0.0076 | +0.0166 | 1.000 | NO-GO |
| polymarket | main | close | blend_elo2 | 7593 | 87 | 1 | -4.3% | -43.9% | +41.6% | -5.0% | +15.4% | +0.0013 | +0.0002 | +0.0023 | 1.000 | NO-GO |
| polymarket | main | close | blend_gbdt | 7593 | 101 | 2 | -25.9% | -48.1% | +0.7% | -3.9% | +18.4% | +0.0003 | -0.0006 | +0.0011 | 1.000 | NO-GO |
| polymarket | main | close | elo2 | 7593 | 3953 | 6 | -7.0% | -12.0% | -1.7% | -4.2% | +2.2% | +0.0225 | +0.0180 | +0.0273 | 1.000 | NO-GO |
| polymarket | main | close | gbdt | 7593 | 2728 | 6 | -1.9% | -7.9% | +4.3% | -4.3% | +5.0% | +0.0157 | +0.0115 | +0.0199 | 1.000 | NO-GO |
