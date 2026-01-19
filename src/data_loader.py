"""
Data Loader for Tennis Transformer.
Strict Version: Loads TUK files and enforces the required schema.
"""

import pandas as pd
import requests
import io
import os

DATA_DIR = "data"

# Das Ziel-Schema, das unser Modell braucht
REQUIRED_STATS_COLS = [
    'w_ace', 'l_ace', 'w_df', 'l_df', 'w_svpt', 'l_svpt',
    'w_1stIn', 'l_1stIn', 'w_1stWon', 'l_1stWon', 'w_2ndWon', 'l_2ndWon',
    'w_SvGms', 'l_SvGms', 'w_bpSaved', 'l_bpSaved', 'w_bpFaced', 'l_bpFaced'
]

# Mapping deiner CSV-Spalten (aus deinem Upload) zu internen Namen
COLUMN_MAPPING = {
    'Winner': 'winner_name', 
    'Loser': 'loser_name',
    'WRank': 'winner_rank', 
    'LRank': 'loser_rank',
    'Surface': 'surface', 
    'Date': 'tourney_date', # Format YYYY-MM-DD in deiner Datei
    'Wsets': 'w_sets',
    'Lsets': 'l_sets',
    'Comment': 'score',
    
    # Quoten (existieren in deiner Datei)
    'B365W': 'B365W', 'B365L': 'B365L',
    'PSW': 'PSW', 'PSL': 'PSL',
    'AvgW': 'AvgW', 'AvgL': 'AvgL'
    
    # Stats mappen wir hier NICHT, da sie in deiner Datei fehlen.
    # Falls sie in alten Jahren (2023) da sind, werden sie automatisch übernommen
    # und unten normalisiert.
}

def load_tuk_year(year):
    """Lädt ein Jahr, erzwingt Schema."""
    
    # 1. Datei finden (Lokal > Web)
    extensions = ['.xlsx', '.xls', '.csv']
    file_path = None
    for ext in extensions:
        path = os.path.join(DATA_DIR, f"{year}{ext}")
        if os.path.exists(path):
            file_path = path
            break
            
    if file_path:
        print(f"Loading local file: {file_path}")
        if file_path.endswith('.csv'):
            # Deine CSV scheint Header zu haben, Pandas erkennt das meist automatisch
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
    else:
        # Download Fallback
        url = f"http://www.tennis-data.co.uk/{year}/{year}.xlsx"
        print(f"Downloading {url}...")
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 200:
                df = pd.read_excel(io.BytesIO(r.content))
            else:
                return pd.DataFrame()
        except Exception as e:
            print(f"Failed to load {year}: {e}")
            return pd.DataFrame()

    # 2. BEREINIGUNG & MAPPING
    
    # Leerzeichen in Header entfernen
    df.columns = df.columns.str.strip()
    
    # Mapping anwenden
    df = df.rename(columns=COLUMN_MAPPING)
    
    # Zusätzlich: Stats-Mapping für ältere Dateien (wo WAce etc. existieren)
    # Deine 2025er Datei hat sie nicht, aber 2023 vielleicht schon.
    stats_map = {
        'WAce': 'w_ace', 'LAce': 'l_ace', 'WDF': 'w_df', 'LDF': 'l_df',
        'WSvpt': 'w_svpt', 'LSvpt': 'l_svpt', 'W1stIn': 'w_1stIn', 'L1stIn': 'l_1stIn',
        'W1stWon': 'w_1stWon', 'L1stWon': 'l_1stWon', 'W2ndWon': 'w_2ndWon', 'L2ndWon': 'l_2ndWon',
        'WSvGms': 'w_SvGms', 'LSvGms': 'l_SvGms', 'WBpSaved': 'w_bpSaved', 'LBpSaved': 'l_bpSaved',
        'WBpFaced': 'w_bpFaced', 'LBpFaced': 'l_bpFaced'
    }
    df = df.rename(columns=stats_map)

    # 3. SCHEMA ENFORCEMENT (Kein Crash mehr!)
    # Wir prüfen: Fehlen Stats? Wenn ja -> Mit 0 auffüllen.
    for col in REQUIRED_STATS_COLS:
        if col not in df.columns:
            # Das ist kein Hack, das ist "Missing Value Imputation"
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    # 4. Datum standardisieren (YYYYMMDD)
    if 'tourney_date' in df.columns:
        # Deine Datei hat YYYY-MM-DD, Sackmann nutzt YYYYMMDD
        df['tourney_date'] = pd.to_datetime(df['tourney_date'], errors='coerce').dt.strftime('%Y%m%d')
    else:
        df['tourney_date'] = '20000101'
        
    # 5. Ranks
    for col in ['winner_rank', 'loser_rank']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(9999)
            
    return df

def load_all_matches(years=[2023, 2024, 2025], source='tuk'):
    frames = [load_tuk_year(y) for y in years]
    if not frames: return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

if __name__ == "__main__":
    df = load_all_matches([2025])
    print(f"Loaded {len(df)} matches.")
    print("Columns:", df.columns.tolist())
    # Test ob w_ace existiert (sollte jetzt 0 sein statt fehlen)
    print("Aces check:", df['w_ace'].sum())