"""
Datenquellen für Matchdaten. Alles hier kommt ohne pandas/torch aus, damit der
tägliche Download überall läuft (auch wo kompilierte Pakete blockiert sind).

    sackmann  Historie mit Aufschlag-Stats (Archiv-Mirror, fester Commit)
    tuk       tennis-data.co.uk: Ergebnisse, Rankings, Quoten (wöchentlich)
    espn      ESPN-Scoreboard: jedes beendete Match, auch im laufenden Turnier (täglich)
    build     führt alles zu data/matches.csv zusammen
"""
