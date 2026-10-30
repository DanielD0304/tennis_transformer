# Ergebnisse Phase 4/5: Timing-Backtest, Challenger/ITF, Go/No-Go (Stand 30.10.2026)

Kurzfassung: **Kein Segment trägt.** Nach Gebühren, Spread, Holm-Korrektur und der verschärften Robustheitsregel
(Gewinn in mindestens zwei Quartalen einzeln bestätigt, kein Quartal mit mehr als 60 % der Wetten) gibt es kein GO.
Das GO des Laufs vom 27.10. (Polymarket ITF −1 h, `blend_gbdt`, +23,8 %) ist ein Artefakt dünner PM-ITF-Märkte im
Oktober 2026 (siehe unten). Der Einstieg zur Polymarket-Eröffnung (+25 bis +33 % Yield) bleibt ein Preis-Artefakt
ohne Orderbuch-Tiefe. Kalshi ist jetzt vollständig geladen (inklusive ITF) und überall NO-GO. Übrig bleiben wenige
„Kandidaten“, die auf einem Monat, auf dünnen Märkten oder auf wenigen hundert Wetten beruhen. Sie sind Material
für das Paper-Trading, kein Grund für echtes Geld.

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
- **Polymarket:** 44.559 Match-Märkte (09/2025 bis 10/2026), 1.080 ohne Sieger (50-50-Auflösung). 10.433 haben keine
  erkennbare Tour (ITF ohne Geschlecht, fast alle von Mai bis August 2026) und fallen heraus. Zuordnung der übrigen
  34.126 zu den Matchdaten (Walk-forward-Vorhersagen): Haupttour 8.241 (100 %), Challenger 7.835 (100 %), ITF 16.722
  (94 %), ohne Level 1.328 (0 %).
- **Kalshi: vollständig** (lokal geladen, 63.431 Märkte, 02/2025 bis 10/2026). Zuordnung: Haupttour 10.542 (92 %),
  Challenger 12.664 (95 %), ITF 39.738 (81 %).

Walk-forward-Vorhersagen 01/2024 bis 29.10.2026 (elo2, gbdt; Elo und Fits über alle Level der Tour, quartalsweise
Refits): ATP main 12.371, Challenger 30.756, ITF 59.692; WTA main 12.214, WTA 125 4.962, ITF 75.126.

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
gbdt (298 Märkte ab August, Log Loss 0,693 gegen 0,731), der gbdt-Favorit gewinnt nur 50 %. Ein Band um 0,5 würde
solche echten Kurse streichen und dünne Märkte mit Preisen abseits von 0,5 behalten (GO-Wetten mit 0,04 bis 0,10
Abstand: +32 %). Das Problem ist der fehlende Handel, nicht der Preis. Die PM-Historie hat aber keine Orderbuch-Tiefe, nur das
Gesamtvolumen (inklusive Live-Handel, also ein Blick in die Zukunft). Deshalb fängt die Robustheitsregel solche
Effekte ab. Das Paper-Trading kann die Frage nur mit echtem Ask klären (siehe Go/No-Go).

**Neue Regel** (`timing.add_decision`): Zusätzlich zu Holm-p < 0,05, KI-Untergrenze > 0 und CLV > 0 verlangt GO:
- in mindestens zwei Quartalen mit je ≥ 30 Wetten einzeln eine KI-Untergrenze des Yields > 0 (`n_periods_pos`),
- kein Quartal mit mehr als 60 % der Wetten (`max_period_share`).

Auf den Lauf vom 27.10. angewendet bleibt kein GO, das frühere GO wird zum Kandidaten. Die erste Bedingung allein
hätte nicht gereicht: Die 47 Q3-Wetten bestätigen den Gewinn mit +80 %, weil sie aus denselben dünnen Märkten
stammen. Erst die Konzentration (97 % der Wetten im Q4) stoppt es.

## Timing-Tabelle (gbdt und Blend aus gbdt + Preis)

Lauf vom 30.10.2026 (Daten bis 29.10.). Flat-Einsatz, Polymarket-Spread 2 %, Kalshi zum Ask mit Gebühr (ab 100
bis t gehandelten Kontrakten). Schwellen walk-forward pro Quartal. Yield mit 95-%-Bootstrap-KI, CLV gegen das
Closing der Venue, `ll_diff` = Log Loss Modell − Log Loss Preis zum Zeitpunkt t (negativ = Modell besser).
`n_periods` = Quartale mit ≥ 30 Wetten, `n_periods_pos` = davon mit KI-Untergrenze > 0 im Quartal,
`max_period_share` = Anteil des größten Quartals an den Wetten. Alle Modelle (inkl. elo2) siehe unten.

| venue | level | point | model | n_bets | n_periods | n_periods_pos | max_period_share | yield | yield_ci_low | yield_ci_high | clv | ll_diff | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kalshi | challenger | open | blend_gbdt | 1270 | 4 | 0 | 41% | -12.3% | -20.4% | -4.4% | -4.2% | -0.0206 | NO-GO |
| kalshi | challenger | open | gbdt | 2016 | 4 | 0 | 39% | -15.1% | -21.4% | -8.8% | -3.7% | -0.0172 | NO-GO |
| kalshi | challenger | 24h | blend_gbdt | 22 | 0 | 0 | 82% | -65.7% | -91.5% | -31.7% | -4.9% | +0.0106 | zu wenig |
| kalshi | challenger | 24h | gbdt | 79 | 0 | 0 | 35% | -17.6% | -50.3% | +23.2% | +1.6% | +0.0291 | NO-GO |
| kalshi | challenger | 6h | blend_gbdt | 167 | 1 | 0 | 83% | -11.5% | -31.0% | +8.9% | -1.7% | +0.0019 | NO-GO |
| kalshi | challenger | 6h | gbdt | 1515 | 4 | 0 | 45% | -18.1% | -26.4% | -9.5% | -2.5% | +0.0155 | NO-GO |
| kalshi | challenger | 1h | blend_gbdt | 141 | 2 | 0 | 72% | -27.2% | -49.9% | -5.7% | -4.6% | +0.0027 | NO-GO |
| kalshi | challenger | 1h | gbdt | 2161 | 4 | 0 | 48% | -19.0% | -25.7% | -12.1% | -3.4% | +0.0189 | NO-GO |
| kalshi | challenger | close | blend_gbdt | 223 | 2 | 0 | 55% | -27.5% | -45.3% | -6.7% | -4.9% | +0.0027 | NO-GO |
| kalshi | challenger | close | gbdt | 2361 | 4 | 0 | 47% | -19.8% | -26.1% | -13.5% | -4.0% | +0.0191 | NO-GO |
| kalshi | itf | open | blend_gbdt | 1238 | 3 | 0 | 58% | -3.4% | -14.2% | +7.6% | +1.9% | -0.0880 | NO-GO |
| kalshi | itf | open | gbdt | 2143 | 3 | 0 | 58% | -11.9% | -19.5% | -4.5% | -2.1% | -0.0854 | NO-GO |
| kalshi | itf | 24h | blend_gbdt | 3 | 0 | 0 | 67% | -100.0% | -100.0% | -100.0% | +52.0% | -0.0145 | zu wenig |
| kalshi | itf | 24h | gbdt | 46 | 0 | 0 | 46% | -40.3% | -83.6% | +18.1% | +6.2% | +0.0156 | NO-GO |
| kalshi | itf | 6h | blend_gbdt | 437 | 2 | 0 | 82% | -26.7% | -41.8% | -9.6% | +2.5% | -0.0159 | NO-GO |
| kalshi | itf | 6h | gbdt | 1909 | 3 | 0 | 66% | -22.4% | -30.4% | -13.9% | +3.6% | +0.0010 | NO-GO |
| kalshi | itf | 1h | blend_gbdt | 476 | 2 | 0 | 85% | -9.9% | -27.9% | +8.7% | +1.5% | -0.0060 | NO-GO |
| kalshi | itf | 1h | gbdt | 3865 | 3 | 0 | 61% | -20.3% | -26.2% | -14.1% | -1.3% | +0.0191 | NO-GO |
| kalshi | itf | close | blend_gbdt | 407 | 2 | 0 | 85% | -9.3% | -27.9% | +10.4% | -6.0% | -0.0069 | NO-GO |
| kalshi | itf | close | gbdt | 4401 | 3 | 0 | 59% | -22.8% | -28.4% | -17.4% | -9.3% | +0.0190 | NO-GO |
| kalshi | main | open | blend_gbdt | 584 | 4 | 0 | 53% | -1.8% | -13.6% | +10.6% | +4.4% | -0.0554 | NO-GO |
| kalshi | main | open | gbdt | 712 | 6 | 0 | 29% | +3.4% | -7.1% | +14.8% | +5.7% | -0.0584 | NO-GO |
| kalshi | main | 24h | blend_gbdt | 413 | 1 | 0 | 95% | -26.1% | -40.1% | -10.1% | -5.1% | -0.0029 | NO-GO |
| kalshi | main | 24h | gbdt | 892 | 4 | 0 | 49% | -7.8% | -18.4% | +4.0% | -2.8% | +0.0003 | NO-GO |
| kalshi | main | 6h | blend_gbdt | 118 | 1 | 0 | 38% | -21.9% | -50.9% | +10.3% | -2.6% | +0.0008 | NO-GO |
| kalshi | main | 6h | gbdt | 2128 | 7 | 0 | 29% | -6.5% | -13.6% | +1.0% | -1.6% | +0.0089 | NO-GO |
| kalshi | main | 1h | blend_gbdt | 162 | 2 | 0 | 48% | -16.0% | -46.9% | +19.3% | -4.6% | +0.0007 | NO-GO |
| kalshi | main | 1h | gbdt | 1887 | 7 | 0 | 26% | -7.9% | -15.7% | -0.3% | -3.4% | +0.0088 | NO-GO |
| kalshi | main | close | blend_gbdt | 221 | 1 | 0 | 63% | -17.9% | -43.7% | +11.9% | -3.8% | +0.0002 | NO-GO |
| kalshi | main | close | gbdt | 2336 | 7 | 0 | 31% | -6.7% | -13.3% | +0.2% | -3.7% | +0.0087 | NO-GO |
| polymarket | challenger | open | blend_gbdt | 6279 | 4 | 4 | 44% | +32.1% | +27.6% | +37.0% | +30.8% | -0.0518 | nicht handelbar |
| polymarket | challenger | open | gbdt | 6233 | 4 | 4 | 44% | +32.6% | +27.7% | +37.3% | +31.2% | -0.0506 | nicht handelbar |
| polymarket | challenger | 24h | blend_gbdt | 298 | 3 | 1 | 65% | +13.8% | -0.9% | +28.4% | +16.0% | -0.0020 | Kandidat |
| polymarket | challenger | 24h | gbdt | 1482 | 4 | 0 | 42% | +2.8% | -4.1% | +10.1% | +6.3% | +0.0089 | NO-GO |
| polymarket | challenger | 6h | blend_gbdt | 223 | 1 | 0 | 81% | +2.9% | -12.1% | +17.5% | -2.0% | +0.0010 | NO-GO |
| polymarket | challenger | 6h | gbdt | 2355 | 4 | 0 | 44% | -8.2% | -14.8% | -1.9% | -3.7% | +0.0194 | NO-GO |
| polymarket | challenger | 1h | blend_gbdt | 258 | 1 | 0 | 89% | +1.0% | -11.6% | +13.9% | -2.2% | +0.0008 | NO-GO |
| polymarket | challenger | 1h | gbdt | 2606 | 4 | 0 | 42% | -7.5% | -13.4% | -1.4% | -4.4% | +0.0211 | NO-GO |
| polymarket | challenger | close | blend_gbdt | 263 | 1 | 0 | 91% | -1.6% | -13.2% | +10.9% | -2.4% | +0.0007 | NO-GO |
| polymarket | challenger | close | gbdt | 2431 | 4 | 0 | 46% | -7.8% | -14.3% | -1.1% | -4.4% | +0.0229 | NO-GO |
| polymarket | itf | open | blend_gbdt | 13162 | 3 | 2 | 55% | +25.5% | +23.5% | +27.3% | +17.4% | -0.0692 | nicht handelbar |
| polymarket | itf | open | gbdt | 13234 | 3 | 2 | 55% | +24.8% | +22.9% | +26.9% | +17.0% | -0.0658 | nicht handelbar |
| polymarket | itf | 24h | blend_gbdt | 23 | 0 | 0 | 70% | -27.2% | -69.1% | +22.9% | +3.0% | +0.0100 | zu wenig |
| polymarket | itf | 24h | gbdt | 66 | 1 | 0 | 53% | -11.3% | -43.1% | +22.3% | +3.0% | +0.0574 | NO-GO |
| polymarket | itf | 6h | blend_gbdt | 1696 | 1 | 1 | 99% | +18.9% | +12.7% | +25.4% | +5.7% | -0.0062 | Kandidat |
| polymarket | itf | 6h | gbdt | 6234 | 3 | 0 | 53% | +2.1% | -2.5% | +6.6% | +2.4% | +0.0206 | NO-GO |
| polymarket | itf | 1h | blend_gbdt | 1921 | 2 | 2 | 97% | +21.4% | +15.2% | +27.8% | +1.6% | -0.0034 | Kandidat |
| polymarket | itf | 1h | gbdt | 6780 | 3 | 0 | 52% | +0.7% | -4.0% | +5.2% | -0.9% | +0.0294 | NO-GO |
| polymarket | itf | close | blend_gbdt | 1386 | 1 | 1 | 100% | +10.0% | +2.8% | +17.9% | -2.7% | -0.0017 | Kandidat |
| polymarket | itf | close | gbdt | 6392 | 3 | 0 | 57% | -4.2% | -8.9% | +0.8% | -5.7% | +0.0397 | NO-GO |
| polymarket | main | open | blend_gbdt | 4955 | 4 | 3 | 40% | +29.6% | +25.2% | +34.2% | +25.7% | -0.0447 | nicht handelbar |
| polymarket | main | open | gbdt | 5359 | 6 | 5 | 37% | +30.7% | +26.6% | +34.9% | +26.8% | -0.0501 | nicht handelbar |
| polymarket | main | 24h | blend_gbdt | 311 | 2 | 0 | 49% | -14.1% | -30.1% | +4.5% | -0.3% | +0.0005 | NO-GO |
| polymarket | main | 24h | gbdt | 1912 | 6 | 0 | 40% | -1.7% | -8.3% | +5.4% | -0.9% | +0.0081 | NO-GO |
| polymarket | main | 6h | blend_gbdt | 162 | 2 | 0 | 43% | +7.5% | -20.2% | +39.1% | -0.6% | -0.0003 | NO-GO |
| polymarket | main | 6h | gbdt | 2811 | 6 | 0 | 27% | +1.0% | -5.2% | +7.3% | -2.3% | +0.0089 | NO-GO |
| polymarket | main | 1h | blend_gbdt | 176 | 2 | 0 | 68% | -6.6% | -25.1% | +13.0% | +4.5% | -0.0002 | NO-GO |
| polymarket | main | 1h | gbdt | 2828 | 6 | 0 | 36% | +0.4% | -5.7% | +7.1% | -3.1% | +0.0114 | NO-GO |
| polymarket | main | close | blend_gbdt | 164 | 2 | 0 | 74% | -22.5% | -37.7% | -5.0% | -3.1% | +0.0000 | NO-GO |
| polymarket | main | close | gbdt | 2401 | 6 | 0 | 29% | -1.6% | -8.7% | +5.6% | -4.5% | +0.0148 | NO-GO |

Log Loss: Ab −24 h ist der Preis bei jedem Level mindestens so gut wie jedes Modell allein (`ll_diff` ≥ 0), ab
−1 h überall klar besser. Nur zur Eröffnung ist das Modell besser, dort ist der Preis aber kein Kurs (siehe oben).
Der Blend gewinnt meist nur wenige Tausendstel, bei ITF etwas mehr (Kalshi −6 h −0,016, PM −6 h −0,006).

## Go/No-Go pro Segment

Holm-Korrektur über alle Einzel-Segmente mit ≥ 30 Wetten (ohne PM-Eröffnung). GO verlangt p_holm < 0,05, untere
KI-Grenze > 0, CLV > 0, in mindestens zwei Quartalen einzeln eine KI-Untergrenze > 0 und kein Quartal mit mehr als
60 % der Wetten.

- **GO: keines.** Von 120 Einzel-Segmenten sind 91 NO-GO, 6 Kandidaten, 11 haben zu wenig Wetten und 12 sind nicht
  handelbar (PM-Eröffnung). Mit der alten Regel wäre Polymarket ITF −1 h, `blend_gbdt` wieder GO: 1.921 Wetten,
  +21,4 % [+15,2 %, +27,8 %], CLV +1,6 %, 97 % der Wetten im Oktober. Das Bild ist dasselbe wie am 27.10.: mit
  liquidem Kalshi-Gegenstück −4,9 % (491 Wetten), ohne Kalshi-Gegenstück +54,5 % (691 Wetten).
- **Kandidaten** (nur Hinweis, Fallzahl in Klammern):
  - Polymarket ITF −6 h / −1 h / Closing, `blend_gbdt`: +18,9 % (1.696), +21,4 % (1.921), +10,0 % (1.386). 97 bis
    100 % der Wetten liegen im Oktober 2026. Ursache wie oben: dünne Märkte ohne Kalshi-Gegenstück.
  - Polymarket Challenger −24 h: `blend_gbdt` +13,8 % [−0,9 %, +28,4 %] (298, nur eines von drei Quartalen
    bestätigt), elo2 +6,1 % [−2,1 %, +14,5 %] (1.406, kein Quartal einzeln bestätigt), `blend_elo2` +31,5 % (99, ein
    Quartal). gbdt allein fällt auf +2,8 % [−4,1 %, +10,1 %] (1.482) und ist kein Kandidat mehr.
- **Kalshi** (jetzt vollständig): alle Segmente mit genug Wetten NO-GO. Der einzige positive Yield ist Haupttour
  Eröffnung, gbdt +3,4 % [−7,1 %, +14,8 %]. Ab −6 h liegen gbdt und elo2 auf allen Levels zwischen −6,5 und −28 %.
- **Haupttour**: überall NO-GO, wie schon gegen Pinnacle.

Sensitivität (nur Polymarket, deshalb kleinere Holm-Familie):
- **Spread 4 %:** kein GO. Die Challenger-Kandidaten elo2 und gbdt bei −24 h verschwinden. PM ITF −1 h, `blend_gbdt`
  hat dann nur noch 541 Wetten, 99 % davon im Oktober.
- **PM-Gesamtvolumen ≥ 100 $:** kein GO. PM ITF −1 h, `blend_gbdt` schrumpft von 1.921 auf 256 Wetten
  (p_holm 0,77).
- **PM-Gesamtvolumen ≥ 500 $:** kein GO. Aber gbdt allein wird bei PM ITF −1 h und Closing nominal signifikant
  (+13,5 % [+6,6 %, +20,8 %], p_holm 0,004, bzw. +13,4 %), mit negativem CLV (−3,3 % bzw. −5,7 %). Das ist der
  Blick in die Zukunft dieses Filters: Das Gesamtvolumen enthält den Live-Handel, und der ist bei Überraschungen
  höher. In Märkten über 2.000 $ gewinnt der gbdt-Favorit 65 %, unter 100 $ 80 %. Der Filter wählt also Matches mit
  Außenseitersiegen aus, und auf Außenseiter setzt gbdt bevorzugt. Der Filter bleibt deshalb aus. Diese Segmente
  scheitern am CLV und knapp an der 60-%-Regel (61 %).

Grenze der 60-%-Regel: PM ITF umfasst nur August bis Oktober, also Q3 mit zwei Monaten und Q4 mit einem. Die Regel
kann dort auch ohne Artefakt greifen. Ein GO für PM ITF braucht deshalb weitere Quartale.

Was das heißt: Frühe Preise sind ungenauer (Log Loss des PM-Preises −24 h 0,602, −1 h 0,571, Closing 0,563; Kalshi
0,585, 0,561, 0,560). Handelbar ist der Unterschied mit elo2/gbdt aber nicht. Der einzig plausible Ansatz bleibt
Challenger um −24 h. Er ist dünn belegt, bestätigt sich höchstens in einem Quartal einzeln und hängt am angenommenen
Spread.
Er gehört ins Paper-Trading, wo echte Bid/Ask und Orderbuch-Tiefe mitgeschrieben werden. Für PM ITF sagt auch das
Paper-Trading erst etwas aus, wenn nur Wetten mit echtem Ask zählen: Ohne brauchbares Ask kauft `sharp.buy_odds` zum
Preis + halben Spread, auch bei genau 0,5. Das ist die Annahme hinter dem Artefakt. Ein Preis um 0,5 ohne Handel ist
kein Kurs.

## Sharp vs. Rest

Ohne Pinnacle/Betfair-Snapshots nicht auswertbar. Die Odds API listete in dieser Session keine aktiven
Tennis-Turniere. Die Logik ist mit Fixtures getestet und läuft im Paper-Loop mit `--odds-api`.

## Odds API

Key funktioniert (HTTP 200). Verbraucht: **0 Credits** (nur das kostenlose `/sports`), Kontingent 500/500. Aktive
Tennis-Events: keine (25.10.2026, zweimal geprüft). Historische Quoten sind im Gratis-Plan nicht enthalten.

## Alle Segmente (inkl. elo2 und blend_elo2)

Lauf vom 30.10.2026, Einzel-Segmente (ohne 'all').

| venue | level | point | model | n_markets | n_bets | n_periods | n_periods_pos | max_period_share | yield | yield_ci_low | yield_ci_high | clv | clv_sharp | ll_diff | ll_ci_low | ll_ci_high | p_holm | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kalshi | challenger | open | blend_elo2 | 11987 | 1055 | 4 | 0 | 40% | -23.0% | -31.5% | -14.5% | -5.7% | – | -0.0153 | -0.0193 | -0.0111 | 1.000 | NO-GO |
| kalshi | challenger | open | blend_gbdt | 11987 | 1270 | 4 | 0 | 41% | -12.3% | -20.4% | -4.4% | -4.2% | – | -0.0206 | -0.0250 | -0.0161 | 1.000 | NO-GO |
| kalshi | challenger | open | elo2 | 11987 | 2165 | 4 | 0 | 44% | -20.8% | -27.3% | -14.1% | -5.8% | – | -0.0070 | -0.0116 | -0.0023 | 1.000 | NO-GO |
| kalshi | challenger | open | gbdt | 11987 | 2016 | 4 | 0 | 39% | -15.1% | -21.4% | -8.8% | -3.7% | – | -0.0172 | -0.0220 | -0.0122 | 1.000 | NO-GO |
| kalshi | challenger | 24h | blend_elo2 | 482 | 16 | 0 | 0 | 81% | -84.4% | -100.0% | -53.1% | -5.9% | – | +0.0068 | +0.0007 | +0.0127 | – | zu wenig |
| kalshi | challenger | 24h | blend_gbdt | 482 | 22 | 0 | 0 | 82% | -65.7% | -91.5% | -31.7% | -4.9% | – | +0.0106 | +0.0014 | +0.0203 | – | zu wenig |
| kalshi | challenger | 24h | elo2 | 482 | 97 | 2 | 0 | 42% | -17.9% | -50.5% | +20.1% | -0.2% | – | +0.0332 | +0.0126 | +0.0530 | 1.000 | NO-GO |
| kalshi | challenger | 24h | gbdt | 482 | 79 | 0 | 0 | 35% | -17.6% | -50.3% | +23.2% | +1.6% | – | +0.0291 | +0.0095 | +0.0506 | 1.000 | NO-GO |
| kalshi | challenger | 6h | blend_elo2 | 11568 | 62 | 1 | 0 | 84% | -29.5% | -77.0% | +37.8% | -4.5% | – | +0.0011 | +0.0001 | +0.0022 | 1.000 | NO-GO |
| kalshi | challenger | 6h | blend_gbdt | 11568 | 167 | 1 | 0 | 83% | -11.5% | -31.0% | +8.9% | -1.7% | – | +0.0019 | +0.0000 | +0.0036 | 1.000 | NO-GO |
| kalshi | challenger | 6h | elo2 | 11568 | 1891 | 4 | 0 | 46% | -26.9% | -34.0% | -19.5% | -3.8% | – | +0.0261 | +0.0223 | +0.0298 | 1.000 | NO-GO |
| kalshi | challenger | 6h | gbdt | 11568 | 1515 | 4 | 0 | 45% | -18.1% | -26.4% | -9.5% | -2.5% | – | +0.0155 | +0.0119 | +0.0191 | 1.000 | NO-GO |
| kalshi | challenger | 1h | blend_elo2 | 11937 | 210 | 1 | 0 | 100% | -27.2% | -50.8% | +0.2% | -6.3% | – | +0.0013 | +0.0005 | +0.0022 | 1.000 | NO-GO |
| kalshi | challenger | 1h | blend_gbdt | 11937 | 141 | 2 | 0 | 72% | -27.2% | -49.9% | -5.7% | -4.6% | – | +0.0027 | +0.0015 | +0.0042 | 1.000 | NO-GO |
| kalshi | challenger | 1h | elo2 | 11937 | 3134 | 4 | 0 | 40% | -21.8% | -28.0% | -15.9% | -4.2% | – | +0.0292 | +0.0256 | +0.0329 | 1.000 | NO-GO |
| kalshi | challenger | 1h | gbdt | 11937 | 2161 | 4 | 0 | 48% | -19.0% | -25.7% | -12.1% | -3.4% | – | +0.0189 | +0.0157 | +0.0224 | 1.000 | NO-GO |
| kalshi | challenger | close | blend_elo2 | 11987 | 589 | 1 | 0 | 100% | -24.7% | -37.9% | -11.3% | -5.4% | – | +0.0013 | +0.0006 | +0.0020 | 1.000 | NO-GO |
| kalshi | challenger | close | blend_gbdt | 11987 | 223 | 2 | 0 | 55% | -27.5% | -45.3% | -6.7% | -4.9% | – | +0.0027 | +0.0013 | +0.0040 | 1.000 | NO-GO |
| kalshi | challenger | close | elo2 | 11987 | 3419 | 4 | 0 | 39% | -21.9% | -27.3% | -16.0% | -4.8% | – | +0.0292 | +0.0257 | +0.0333 | 1.000 | NO-GO |
| kalshi | challenger | close | gbdt | 11987 | 2361 | 4 | 0 | 47% | -19.8% | -26.1% | -13.5% | -4.0% | – | +0.0191 | +0.0156 | +0.0224 | 1.000 | NO-GO |
| kalshi | itf | open | blend_elo2 | 31719 | 1176 | 3 | 0 | 46% | -17.3% | -28.0% | -6.3% | +1.1% | – | -0.0493 | -0.0523 | -0.0461 | 1.000 | NO-GO |
| kalshi | itf | open | blend_gbdt | 31719 | 1238 | 3 | 0 | 58% | -3.4% | -14.2% | +7.6% | +1.9% | – | -0.0880 | -0.0921 | -0.0837 | 1.000 | NO-GO |
| kalshi | itf | open | elo2 | 31719 | 2958 | 3 | 0 | 52% | -20.7% | -27.2% | -14.1% | -5.6% | – | -0.0421 | -0.0455 | -0.0385 | 1.000 | NO-GO |
| kalshi | itf | open | gbdt | 31719 | 2143 | 3 | 0 | 58% | -11.9% | -19.5% | -4.5% | -2.1% | – | -0.0854 | -0.0898 | -0.0808 | 1.000 | NO-GO |
| kalshi | itf | 24h | blend_elo2 | 456 | 0 | 0 | 0 | – | – | – | – | – | – | -0.0008 | -0.0089 | +0.0067 | – | zu wenig |
| kalshi | itf | 24h | blend_gbdt | 456 | 3 | 0 | 0 | 67% | -100.0% | -100.0% | -100.0% | +52.0% | – | -0.0145 | -0.0311 | +0.0031 | – | zu wenig |
| kalshi | itf | 24h | elo2 | 456 | 55 | 0 | 0 | 42% | -27.4% | -75.8% | +33.7% | +8.0% | – | +0.0664 | +0.0390 | +0.0962 | 1.000 | NO-GO |
| kalshi | itf | 24h | gbdt | 456 | 46 | 0 | 0 | 46% | -40.3% | -83.6% | +18.1% | +6.2% | – | +0.0156 | -0.0165 | +0.0504 | 1.000 | NO-GO |
| kalshi | itf | 6h | blend_elo2 | 30503 | 57 | 1 | 0 | 93% | -54.3% | -81.0% | -22.1% | +6.4% | – | -0.0031 | -0.0039 | -0.0023 | 1.000 | NO-GO |
| kalshi | itf | 6h | blend_gbdt | 30503 | 437 | 2 | 0 | 82% | -26.7% | -41.8% | -9.6% | +2.5% | – | -0.0159 | -0.0177 | -0.0141 | 1.000 | NO-GO |
| kalshi | itf | 6h | elo2 | 30503 | 2340 | 3 | 0 | 64% | -26.3% | -33.6% | -18.4% | +3.2% | – | +0.0441 | +0.0405 | +0.0477 | 1.000 | NO-GO |
| kalshi | itf | 6h | gbdt | 30503 | 1909 | 3 | 0 | 66% | -22.4% | -30.4% | -13.9% | +3.6% | – | +0.0010 | -0.0027 | +0.0048 | 1.000 | NO-GO |
| kalshi | itf | 1h | blend_elo2 | 31408 | 1 | 0 | 0 | 100% | -100.0% | -100.0% | -100.0% | -3.3% | – | -0.0005 | -0.0009 | -0.0001 | – | zu wenig |
| kalshi | itf | 1h | blend_gbdt | 31408 | 476 | 2 | 0 | 85% | -9.9% | -27.9% | +8.7% | +1.5% | – | -0.0060 | -0.0072 | -0.0046 | 1.000 | NO-GO |
| kalshi | itf | 1h | elo2 | 31408 | 5029 | 3 | 0 | 58% | -25.8% | -31.2% | -20.4% | -1.6% | – | +0.0620 | +0.0586 | +0.0655 | 1.000 | NO-GO |
| kalshi | itf | 1h | gbdt | 31408 | 3865 | 3 | 0 | 61% | -20.3% | -26.2% | -14.1% | -1.3% | – | +0.0191 | +0.0159 | +0.0226 | 1.000 | NO-GO |
| kalshi | itf | close | blend_elo2 | 31719 | 3 | 0 | 0 | 100% | -100.0% | -100.0% | -100.0% | +266.7% | – | -0.0007 | -0.0011 | -0.0003 | – | zu wenig |
| kalshi | itf | close | blend_gbdt | 31719 | 407 | 2 | 0 | 85% | -9.3% | -27.9% | +10.4% | -6.0% | – | -0.0069 | -0.0082 | -0.0056 | 1.000 | NO-GO |
| kalshi | itf | close | elo2 | 31719 | 5837 | 3 | 0 | 57% | -28.0% | -32.8% | -23.1% | -10.7% | – | +0.0622 | +0.0589 | +0.0657 | 1.000 | NO-GO |
| kalshi | itf | close | gbdt | 31719 | 4401 | 3 | 0 | 59% | -22.8% | -28.4% | -17.4% | -9.3% | – | +0.0190 | +0.0156 | +0.0224 | 1.000 | NO-GO |
| kalshi | main | open | blend_elo2 | 9548 | 304 | 4 | 0 | 33% | -0.5% | -18.1% | +18.4% | +7.3% | +4.1% | -0.0488 | -0.0548 | -0.0430 | 1.000 | NO-GO |
| kalshi | main | open | blend_gbdt | 9548 | 584 | 4 | 0 | 53% | -1.8% | -13.6% | +10.6% | +4.4% | +2.3% | -0.0554 | -0.0614 | -0.0493 | 1.000 | NO-GO |
| kalshi | main | open | elo2 | 9548 | 528 | 7 | 0 | 23% | -5.2% | -19.1% | +9.7% | +8.6% | +0.5% | -0.0508 | -0.0568 | -0.0445 | 1.000 | NO-GO |
| kalshi | main | open | gbdt | 9548 | 712 | 6 | 0 | 29% | +3.4% | -7.1% | +14.8% | +5.7% | +2.6% | -0.0584 | -0.0649 | -0.0514 | 1.000 | NO-GO |
| kalshi | main | 24h | blend_elo2 | 4536 | 300 | 1 | 0 | 97% | -32.8% | -51.1% | -14.0% | -6.4% | -5.5% | -0.0001 | -0.0021 | +0.0018 | 1.000 | NO-GO |
| kalshi | main | 24h | blend_gbdt | 4536 | 413 | 1 | 0 | 95% | -26.1% | -40.1% | -10.1% | -5.1% | -4.5% | -0.0029 | -0.0053 | -0.0004 | 1.000 | NO-GO |
| kalshi | main | 24h | elo2 | 4536 | 836 | 6 | 0 | 29% | -14.9% | -28.4% | -0.5% | -4.8% | -3.1% | +0.0104 | +0.0042 | +0.0167 | 1.000 | NO-GO |
| kalshi | main | 24h | gbdt | 4536 | 892 | 4 | 0 | 49% | -7.8% | -18.4% | +4.0% | -2.8% | -1.5% | +0.0003 | -0.0053 | +0.0059 | 1.000 | NO-GO |
| kalshi | main | 6h | blend_elo2 | 9238 | 89 | 1 | 0 | 63% | -8.8% | -54.0% | +46.5% | -2.5% | +0.5% | +0.0010 | -0.0001 | +0.0021 | 1.000 | NO-GO |
| kalshi | main | 6h | blend_gbdt | 9238 | 118 | 1 | 0 | 38% | -21.9% | -50.9% | +10.3% | -2.6% | +1.1% | +0.0008 | -0.0008 | +0.0023 | 1.000 | NO-GO |
| kalshi | main | 6h | elo2 | 9238 | 2073 | 7 | 0 | 31% | -11.6% | -19.5% | -3.1% | -2.5% | -1.5% | +0.0160 | +0.0121 | +0.0198 | 1.000 | NO-GO |
| kalshi | main | 6h | gbdt | 9238 | 2128 | 7 | 0 | 29% | -6.5% | -13.6% | +1.0% | -1.6% | -0.5% | +0.0089 | +0.0056 | +0.0123 | 1.000 | NO-GO |
| kalshi | main | 1h | blend_elo2 | 9502 | 238 | 2 | 0 | 74% | -26.2% | -52.2% | +1.9% | -4.3% | -0.2% | +0.0011 | +0.0000 | +0.0022 | 1.000 | NO-GO |
| kalshi | main | 1h | blend_gbdt | 9502 | 162 | 2 | 0 | 48% | -16.0% | -46.9% | +19.3% | -4.6% | +2.2% | +0.0007 | -0.0009 | +0.0021 | 1.000 | NO-GO |
| kalshi | main | 1h | elo2 | 9502 | 2500 | 7 | 0 | 28% | -9.4% | -16.9% | -2.5% | -3.9% | -1.5% | +0.0160 | +0.0120 | +0.0202 | 1.000 | NO-GO |
| kalshi | main | 1h | gbdt | 9502 | 1887 | 7 | 0 | 26% | -7.9% | -15.7% | -0.3% | -3.4% | -0.6% | +0.0088 | +0.0051 | +0.0125 | 1.000 | NO-GO |
| kalshi | main | close | blend_elo2 | 9548 | 616 | 2 | 0 | 81% | -6.4% | -21.7% | +11.3% | -3.3% | +1.3% | +0.0006 | -0.0003 | +0.0016 | 1.000 | NO-GO |
| kalshi | main | close | blend_gbdt | 9548 | 221 | 1 | 0 | 63% | -17.9% | -43.7% | +11.9% | -3.8% | +4.8% | +0.0002 | -0.0012 | +0.0016 | 1.000 | NO-GO |
| kalshi | main | close | elo2 | 9548 | 2857 | 7 | 0 | 26% | -9.7% | -16.1% | -3.4% | -4.0% | -0.8% | +0.0162 | +0.0126 | +0.0201 | 1.000 | NO-GO |
| kalshi | main | close | gbdt | 9548 | 2336 | 7 | 0 | 31% | -6.7% | -13.3% | +0.2% | -3.7% | -0.2% | +0.0087 | +0.0053 | +0.0121 | 1.000 | NO-GO |
| polymarket | challenger | open | blend_elo2 | 7492 | 5930 | 4 | 4 | 43% | +33.9% | +29.1% | +38.5% | +32.1% | +49.9% | -0.0481 | -0.0560 | -0.0405 | – | nicht handelbar |
| polymarket | challenger | open | blend_gbdt | 7492 | 6279 | 4 | 4 | 44% | +32.1% | +27.6% | +37.0% | +30.8% | +49.9% | -0.0518 | -0.0596 | -0.0431 | – | nicht handelbar |
| polymarket | challenger | open | elo2 | 7492 | 5863 | 4 | 4 | 39% | +32.8% | +27.7% | +37.7% | +31.8% | +38.7% | -0.0452 | -0.0533 | -0.0373 | – | nicht handelbar |
| polymarket | challenger | open | gbdt | 7492 | 6233 | 4 | 4 | 44% | +32.6% | +27.7% | +37.3% | +31.2% | +49.9% | -0.0506 | -0.0591 | -0.0412 | – | nicht handelbar |
| polymarket | challenger | 24h | blend_elo2 | 3319 | 99 | 1 | 0 | 78% | +31.5% | +2.2% | +63.1% | +33.9% | – | -0.0017 | -0.0043 | +0.0008 | 1.000 | Kandidat |
| polymarket | challenger | 24h | blend_gbdt | 3319 | 298 | 3 | 1 | 65% | +13.8% | -0.9% | +28.4% | +16.0% | – | -0.0020 | -0.0051 | +0.0012 | 1.000 | Kandidat |
| polymarket | challenger | 24h | elo2 | 3319 | 1406 | 4 | 0 | 31% | +6.1% | -2.1% | +14.5% | +5.2% | +5.1% | +0.0101 | +0.0026 | +0.0180 | 1.000 | Kandidat |
| polymarket | challenger | 24h | gbdt | 3319 | 1482 | 4 | 0 | 42% | +2.8% | -4.1% | +10.1% | +6.3% | +5.1% | +0.0089 | +0.0008 | +0.0169 | 1.000 | NO-GO |
| polymarket | challenger | 6h | blend_elo2 | 7687 | 92 | 1 | 0 | 95% | -12.0% | -43.4% | +24.2% | -1.2% | – | +0.0012 | +0.0002 | +0.0022 | 1.000 | NO-GO |
| polymarket | challenger | 6h | blend_gbdt | 7687 | 223 | 1 | 0 | 81% | +2.9% | -12.1% | +17.5% | -2.0% | – | +0.0010 | -0.0002 | +0.0021 | 1.000 | NO-GO |
| polymarket | challenger | 6h | elo2 | 7687 | 3498 | 4 | 0 | 34% | -7.9% | -13.3% | -2.2% | -4.4% | +36.7% | +0.0249 | +0.0206 | +0.0296 | 1.000 | NO-GO |
| polymarket | challenger | 6h | gbdt | 7687 | 2355 | 4 | 0 | 44% | -8.2% | -14.8% | -1.9% | -3.7% | +67.8% | +0.0194 | +0.0149 | +0.0238 | 1.000 | NO-GO |
| polymarket | challenger | 1h | blend_elo2 | 7759 | 64 | 1 | 0 | 100% | -10.7% | -48.5% | +30.9% | +2.0% | – | +0.0011 | +0.0002 | +0.0019 | 1.000 | NO-GO |
| polymarket | challenger | 1h | blend_gbdt | 7759 | 258 | 1 | 0 | 89% | +1.0% | -11.6% | +13.9% | -2.2% | – | +0.0008 | +0.0000 | +0.0016 | 1.000 | NO-GO |
| polymarket | challenger | 1h | elo2 | 7759 | 3599 | 4 | 0 | 34% | -8.1% | -13.4% | -2.8% | -4.3% | -1.8% | +0.0267 | +0.0223 | +0.0314 | 1.000 | NO-GO |
| polymarket | challenger | 1h | gbdt | 7759 | 2606 | 4 | 0 | 42% | -7.5% | -13.4% | -1.4% | -4.4% | – | +0.0211 | +0.0168 | +0.0257 | 1.000 | NO-GO |
| polymarket | challenger | close | blend_elo2 | 7767 | 58 | 1 | 0 | 100% | -27.6% | -60.3% | +5.7% | -4.8% | – | +0.0011 | +0.0002 | +0.0019 | 1.000 | NO-GO |
| polymarket | challenger | close | blend_gbdt | 7767 | 263 | 1 | 0 | 91% | -1.6% | -13.2% | +10.9% | -2.4% | – | +0.0007 | -0.0001 | +0.0016 | 1.000 | NO-GO |
| polymarket | challenger | close | elo2 | 7767 | 3433 | 4 | 0 | 37% | -7.1% | -13.0% | -1.4% | -4.5% | +5.6% | +0.0284 | +0.0238 | +0.0327 | 1.000 | NO-GO |
| polymarket | challenger | close | gbdt | 7767 | 2431 | 4 | 0 | 46% | -7.8% | -14.3% | -1.1% | -4.4% | – | +0.0229 | +0.0183 | +0.0275 | 1.000 | NO-GO |
| polymarket | itf | open | blend_elo2 | 13937 | 12037 | 3 | 2 | 56% | +15.5% | +13.3% | +17.6% | +15.3% | – | -0.0264 | -0.0314 | -0.0214 | – | nicht handelbar |
| polymarket | itf | open | blend_gbdt | 13937 | 13162 | 3 | 2 | 55% | +25.5% | +23.5% | +27.3% | +17.4% | – | -0.0692 | -0.0750 | -0.0632 | – | nicht handelbar |
| polymarket | itf | open | elo2 | 13937 | 11811 | 3 | 2 | 53% | +14.7% | +12.3% | +17.0% | +14.7% | – | -0.0177 | -0.0231 | -0.0124 | – | nicht handelbar |
| polymarket | itf | open | gbdt | 13937 | 13234 | 3 | 2 | 55% | +24.8% | +22.9% | +26.9% | +17.0% | – | -0.0658 | -0.0720 | -0.0592 | – | nicht handelbar |
| polymarket | itf | 24h | blend_elo2 | 159 | 3 | 0 | 0 | 67% | -32.7% | -100.0% | +102.0% | +20.7% | – | +0.0037 | -0.0073 | +0.0164 | – | zu wenig |
| polymarket | itf | 24h | blend_gbdt | 159 | 23 | 0 | 0 | 70% | -27.2% | -69.1% | +22.9% | +3.0% | – | +0.0100 | -0.0078 | +0.0275 | – | zu wenig |
| polymarket | itf | 24h | elo2 | 159 | 68 | 0 | 0 | 43% | -2.9% | -37.4% | +36.2% | +2.3% | – | +0.0400 | -0.0044 | +0.0836 | 1.000 | NO-GO |
| polymarket | itf | 24h | gbdt | 159 | 66 | 1 | 0 | 53% | -11.3% | -43.1% | +22.3% | +3.0% | – | +0.0574 | +0.0091 | +0.1053 | 1.000 | NO-GO |
| polymarket | itf | 6h | blend_elo2 | 12272 | 3 | 0 | 0 | 100% | -100.0% | -100.0% | -100.0% | -4.4% | – | -0.0007 | -0.0012 | -0.0003 | – | zu wenig |
| polymarket | itf | 6h | blend_gbdt | 12272 | 1696 | 1 | 1 | 99% | +18.9% | +12.7% | +25.4% | +5.7% | – | -0.0062 | -0.0074 | -0.0049 | 0.000 | Kandidat |
| polymarket | itf | 6h | elo2 | 12272 | 6938 | 3 | 0 | 54% | -7.3% | -11.8% | -2.8% | -0.8% | – | +0.0517 | +0.0459 | +0.0570 | 1.000 | NO-GO |
| polymarket | itf | 6h | gbdt | 12272 | 6234 | 3 | 0 | 53% | +2.1% | -2.5% | +6.6% | +2.4% | – | +0.0206 | +0.0153 | +0.0260 | 1.000 | NO-GO |
| polymarket | itf | 1h | blend_elo2 | 12966 | 2 | 0 | 0 | 100% | -100.0% | -100.0% | -100.0% | -4.7% | – | +0.0000 | -0.0002 | +0.0003 | – | zu wenig |
| polymarket | itf | 1h | blend_gbdt | 12966 | 1921 | 2 | 2 | 97% | +21.4% | +15.2% | +27.8% | +1.6% | – | -0.0034 | -0.0041 | -0.0026 | 0.000 | Kandidat |
| polymarket | itf | 1h | elo2 | 12966 | 7594 | 3 | 0 | 53% | -8.0% | -12.6% | -3.4% | -1.8% | – | +0.0640 | +0.0580 | +0.0696 | 1.000 | NO-GO |
| polymarket | itf | 1h | gbdt | 12966 | 6780 | 3 | 0 | 52% | +0.7% | -4.0% | +5.2% | -0.9% | – | +0.0294 | +0.0236 | +0.0352 | 1.000 | NO-GO |
| polymarket | itf | close | blend_elo2 | 12254 | 0 | 0 | 0 | – | – | – | – | – | – | -0.0001 | -0.0003 | +0.0002 | – | zu wenig |
| polymarket | itf | close | blend_gbdt | 12254 | 1386 | 1 | 1 | 100% | +10.0% | +2.8% | +17.9% | -2.7% | – | -0.0017 | -0.0024 | -0.0010 | 0.546 | Kandidat |
| polymarket | itf | close | elo2 | 12254 | 7242 | 3 | 0 | 56% | -12.0% | -16.6% | -7.1% | -6.2% | – | +0.0740 | +0.0678 | +0.0799 | 1.000 | NO-GO |
| polymarket | itf | close | gbdt | 12254 | 6392 | 3 | 0 | 57% | -4.2% | -8.9% | +0.8% | -5.7% | – | +0.0397 | +0.0345 | +0.0451 | 1.000 | NO-GO |
| polymarket | main | open | blend_elo2 | 7662 | 4685 | 4 | 3 | 37% | +26.9% | +22.4% | +31.7% | +26.7% | +22.9% | -0.0379 | -0.0452 | -0.0317 | – | nicht handelbar |
| polymarket | main | open | blend_gbdt | 7662 | 4955 | 4 | 3 | 40% | +29.6% | +25.2% | +34.2% | +25.7% | +21.8% | -0.0447 | -0.0517 | -0.0380 | – | nicht handelbar |
| polymarket | main | open | elo2 | 7662 | 5385 | 6 | 6 | 33% | +26.7% | +22.4% | +31.3% | +25.7% | +22.2% | -0.0424 | -0.0505 | -0.0360 | – | nicht handelbar |
| polymarket | main | open | gbdt | 7662 | 5359 | 6 | 5 | 37% | +30.7% | +26.6% | +34.9% | +26.8% | +23.4% | -0.0501 | -0.0583 | -0.0427 | – | nicht handelbar |
| polymarket | main | 24h | blend_elo2 | 4176 | 154 | 1 | 0 | 91% | -11.6% | -43.8% | +23.3% | +0.6% | -0.4% | +0.0031 | +0.0010 | +0.0052 | 1.000 | NO-GO |
| polymarket | main | 24h | blend_gbdt | 4176 | 311 | 2 | 0 | 49% | -14.1% | -30.1% | +4.5% | -0.3% | -1.2% | +0.0005 | -0.0018 | +0.0027 | 1.000 | NO-GO |
| polymarket | main | 24h | elo2 | 4176 | 1778 | 6 | 0 | 43% | -8.8% | -16.6% | -0.9% | -1.8% | -2.0% | +0.0174 | +0.0113 | +0.0233 | 1.000 | NO-GO |
| polymarket | main | 24h | gbdt | 4176 | 1912 | 6 | 0 | 40% | -1.7% | -8.3% | +5.4% | -0.9% | -0.5% | +0.0081 | +0.0023 | +0.0140 | 1.000 | NO-GO |
| polymarket | main | 6h | blend_elo2 | 7681 | 153 | 1 | 0 | 91% | +1.1% | -29.8% | +36.9% | -1.4% | +5.3% | +0.0014 | +0.0001 | +0.0026 | 1.000 | NO-GO |
| polymarket | main | 6h | blend_gbdt | 7681 | 162 | 2 | 0 | 43% | +7.5% | -20.2% | +39.1% | -0.6% | +26.4% | -0.0003 | -0.0014 | +0.0009 | 1.000 | NO-GO |
| polymarket | main | 6h | elo2 | 7681 | 3899 | 6 | 0 | 45% | -4.9% | -10.2% | +0.2% | -3.0% | -1.2% | +0.0166 | +0.0123 | +0.0212 | 1.000 | NO-GO |
| polymarket | main | 6h | gbdt | 7681 | 2811 | 6 | 0 | 27% | +1.0% | -5.2% | +7.3% | -2.3% | +0.6% | +0.0089 | +0.0049 | +0.0130 | 1.000 | NO-GO |
| polymarket | main | 1h | blend_elo2 | 7838 | 93 | 1 | 0 | 96% | +20.7% | -29.2% | +82.9% | +6.6% | +1.1% | +0.0012 | +0.0001 | +0.0022 | 1.000 | NO-GO |
| polymarket | main | 1h | blend_gbdt | 7838 | 176 | 2 | 0 | 68% | -6.6% | -25.1% | +13.0% | +4.5% | +21.3% | -0.0002 | -0.0011 | +0.0007 | 1.000 | NO-GO |
| polymarket | main | 1h | elo2 | 7838 | 4372 | 6 | 1 | 41% | -5.3% | -10.2% | +0.1% | -3.0% | -0.9% | +0.0191 | +0.0146 | +0.0237 | 1.000 | NO-GO |
| polymarket | main | 1h | gbdt | 7838 | 2828 | 6 | 0 | 36% | +0.4% | -5.7% | +7.1% | -3.1% | +0.6% | +0.0114 | +0.0074 | +0.0157 | 1.000 | NO-GO |
| polymarket | main | close | blend_elo2 | 7812 | 89 | 1 | 0 | 98% | -6.5% | -45.4% | +35.2% | -5.0% | +37.2% | +0.0013 | +0.0003 | +0.0022 | 1.000 | NO-GO |
| polymarket | main | close | blend_gbdt | 7812 | 164 | 2 | 0 | 74% | -22.5% | -37.7% | -5.0% | -3.1% | +20.9% | +0.0000 | -0.0008 | +0.0009 | 1.000 | NO-GO |
| polymarket | main | close | elo2 | 7812 | 3598 | 6 | 0 | 38% | -6.7% | -12.4% | -1.1% | -4.5% | +2.9% | +0.0225 | +0.0177 | +0.0270 | 1.000 | NO-GO |
| polymarket | main | close | gbdt | 7812 | 2401 | 6 | 0 | 29% | -1.6% | -8.7% | +5.6% | -4.5% | +6.2% | +0.0148 | +0.0102 | +0.0195 | 1.000 | NO-GO |
