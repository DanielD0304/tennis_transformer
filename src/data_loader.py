"""
Data Loader for Tennis Transformer.
Features: 
- Loads Sackmann (Stats) and TUK (Odds).
- SMART MERGE: Handles name discrepancies (e.g. 'Rafael Nadal' vs 'Nadal R.').
"""

import pandas as pd
import requests
import io
import os

DATA_DIR = "data"

# Das Ziel-Schema (Sackmann-Style)
REQUIRED_STATS_COLS = [
    'w_ace', 'l_ace', 'w_df', 'l_df', 'w_svpt', 'l_svpt',
    'w_1stIn', 'l_1stIn', 'w_1stWon', 'l_1stWon', 'w_2ndWon', 'l_2ndWon',
    'w_SvGms', 'l_SvGms', 'w_bpSaved', 'l_bpSaved', 'w_bpFaced', 'l_bpFaced'
]

# Mapping TUK -> Sackmann (für Fallback)
TUK_MAPPING = {
    'Winner': 'winner_name', 'Loser': 'loser_name',
    'WRank': 'winner_rank', 'LRank': 'loser_rank',
    'Surface': 'surface', 'Date': 'tourney_date', 
    'Wsets': 'w_sets', 'Lsets': 'l_sets', 'Comment': 'score',
    'B365W': 'B365W', 'B365L': 'B365L',
    'PSW': 'PSW', 'PSL': 'PSL',
    'AvgW': 'AvgW', 'AvgL': 'AvgL',
    'WAce': 'w_ace', 'LAce': 'l_ace', 'WDF': 'w_df', 'LDF': 'l_df',
    'WSvpt': 'w_svpt', 'LSvpt': 'l_svpt', 'W1stIn': 'w_1stIn', 'L1stIn': 'l_1stIn',
    'W1stWon': 'w_1stWon', 'L1stWon': 'l_1stWon', 'W2ndWon': 'w_2ndWon', 'L2ndWon': 'l_2ndWon',
    'WSvGms': 'w_SvGms', 'LSvGms': 'l_SvGms', 'WBpSaved': 'w_bpSaved', 'LBpSaved': 'l_bpSaved',
    'WBpFaced': 'w_bpFaced', 'LBpFaced': 'l_bpFaced'
}

def enforce_schema(df):
    for col in REQUIRED_STATS_COLS:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            
    for col in ['winner_rank', 'loser_rank']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(9999)
            
    if 'tourney_date' in df.columns:
        df['tourney_date'] = pd.to_datetime(df['tourney_date'], errors='coerce').dt.strftime('%Y%m%d')
        
    return df

def load_sackmann_year(year):
    filename = f"atp_matches_{year}.csv"
    local_path = os.path.join(DATA_DIR, filename)
    
    if os.path.exists(local_path):
        print(f"Loading Sackmann (Local): {year}")
        df = pd.read_csv(local_path)
    else:
        url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"
        print(f"Downloading Sackmann: {year}")
        try:
            df = pd.read_csv(url)
        except:
            return pd.DataFrame()

    if 'tourney_date' in df.columns:
        df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce').dt.strftime('%Y%m%d')
        
    df = enforce_schema(df)
    return df

def load_tuk_year(year):
    extensions = ['.xlsx', '.xls', '.csv']
    file_path = None
    for ext in extensions:
        path = os.path.join(DATA_DIR, f"{year}{ext}")
        if os.path.exists(path):
            file_path = path
            break
            
    if file_path:
        print(f"Loading TUK (Local): {year}")
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
    else:
        url = f"http://www.tennis-data.co.uk/{year}/{year}.xlsx"
        print(f"Downloading TUK: {year}")
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 200:
                df = pd.read_excel(io.BytesIO(r.content))
            else:
                return pd.DataFrame()
        except:
            return pd.DataFrame()

    # Bereinigung
    df.columns = df.columns.str.strip().str.replace(' ', '')
    df = df.rename(columns=TUK_MAPPING)
    
    if 'tourney_date' in df.columns:
        df['tourney_date'] = pd.to_datetime(df['tourney_date'], errors='coerce').dt.strftime('%Y%m%d')
    else:
        df['tourney_date'] = '20000101'

    return df

# --- SMART NAME MATCHING ---

def parse_tuk_name(name):
    """
    Zerlegt 'Nadal R.' in ('nadal', 'r').
    Ignoriert Groß/Kleinschreibung.
    """
    if pd.isna(name): return "", ""
    name = str(name).strip().lower()
    parts = name.split()
    
    # Fall: "Nadal" (kein Initial)
    if len(parts) == 1:
        return name, ""
    
    # Fall: "Nadal R."
    # Annahme: Letztes Element ist Initial, wenn es kurz ist
    last_part = parts[-1].replace('.', '')
    if len(last_part) <= 2: 
        initial = last_part[0] # Nehme nur ersten Buchstaben des Initials
        surname = " ".join(parts[:-1]) # Alles davor ist Nachname
        return surname, initial
    
    # Fallback: Kein erkennbares Initial
    return name, ""

def match_names(sack_full, tuk_raw):
    """
    Vergleicht Sackmann 'Rafael Nadal' mit TUK 'Nadal R.'
    """
    if pd.isna(sack_full) or pd.isna(tuk_raw): return False
    
    sack_full = str(sack_full).strip().lower()
    tuk_surname, tuk_init = parse_tuk_name(tuk_raw)
    
    if not tuk_surname: return False
    
    # 1. Nachnamen-Check: Der TUK-Nachname muss im Sackmann-Namen stecken
    # z.B. "nadal" in "rafael nadal" -> True
    if tuk_surname not in sack_full:
        return False
        
    # 2. Initial-Check: Sackmann-Vorname muss mit TUK-Initial beginnen
    # z.B. "rafael..." startet mit "r" -> True
    if tuk_init:
        if not sack_full.startswith(tuk_init):
            return False
            
    return True

def merge_sources(year):
    """
    Lädt Sackmann (Stats) und TUK (Odds) für ein Jahr und merged sie.
    Nutzt fuzzy name matching.
    """
    df_sack = load_sackmann_year(year)
    df_tuk = load_tuk_year(year)
    
    if df_sack.empty:
        print(f"Warning: No Sackmann data for {year}. Using TUK only.")
        return enforce_schema(df_tuk)
        
    if df_tuk.empty:
        print(f"Warning: No TUK data for {year}. Using Sackmann only (No Odds).")
        return df_sack
        
    print(f"Merging sources for {year} (Smart Name Match)...")
    print(f"Sackmann: {len(df_sack)}, TUK: {len(df_tuk)}")
    
    # Quoten-Spalten vorbereiten
    odds_cols = ['B365W', 'B365L', 'PSW', 'PSL', 'AvgW', 'AvgL']
    for col in odds_cols:
        df_sack[col] = 0.0
        
    merged_count = 0
    
    # Optimierung: Wir iterieren über TUK (meist weniger oder gleich viele Matches)
    # und suchen im Sackmann-DF.
    # Da O(N^2) zu langsam ist, filtern wir vor.
    
    # Wir machen es hier iterativ, da ~2000-3000 Matches pro Jahr noch in Sekunden gehen.
    # Für Speedup könnte man nach 'Winner Initial' gruppieren, aber wir halten es sicher.
    
    # Um nicht jedes Mal alle Strings zu parsen:
    # Wir erstellen temporäre Match-Keys im Sackmann DF (lowercase)
    df_sack['_search_name_w'] = df_sack['winner_name'].str.lower().fillna('')
    df_sack['_search_name_l'] = df_sack['loser_name'].str.lower().fillna('')
    
    for idx, row in df_tuk.iterrows():
        tuk_w = row.get('winner_name', '')
        tuk_l = row.get('loser_name', '')
        
        # Parse TUK Namen
        w_sur, w_ini = parse_tuk_name(tuk_w)
        l_sur, l_ini = parse_tuk_name(tuk_l)
        
        # Filter Sackmann: Winner muss Nachnamen enthalten und mit Initial starten
        # (Vektorisierte Operation für Speed)
        mask_w = (df_sack['_search_name_w'].str.contains(w_sur, regex=False)) & \
                 (df_sack['_search_name_w'].str.startswith(w_ini))
                 
        mask_l = (df_sack['_search_name_l'].str.contains(l_sur, regex=False)) & \
                 (df_sack['_search_name_l'].str.startswith(l_ini))
        
        # Matches finden, wo BEIDE Spieler passen
        candidates = df_sack[mask_w & mask_l]
        
        if len(candidates) == 1:
            # Eindeutiger Treffer -> Quoten übertragen
            match_idx = candidates.index[0]
            for col in odds_cols:
                if col in row:
                    df_sack.at[match_idx, col] = row[col]
            merged_count += 1
            
        elif len(candidates) > 1:
            # Mehrere Treffer (z.B. gleiches Match-up 2x im Jahr? Unwahrscheinlich in einem File)
            # Nimm das erste oder prüfe Datum
            match_idx = candidates.index[0]
            for col in odds_cols:
                if col in row:
                    df_sack.at[match_idx, col] = row[col]
            merged_count += 1

    print(f"Merged Odds for {merged_count} matches.")
    
    # Aufräumen
    df_sack = df_sack.drop(columns=['_search_name_w', '_search_name_l'])
    return df_sack

def load_merged_data(train_years, test_year):
    frames = []
    
    # Training: Sackmann pur
    print(f"--- Loading Training Data ---")
    for y in train_years:
        df = load_sackmann_year(y)
        frames.append(df)
        
    # Test: Merged
    print(f"--- Loading Test Data (Merged) for {test_year} ---")
    df_test = merge_sources(test_year)
    frames.append(df_test)
    
    return pd.concat(frames, ignore_index=True)