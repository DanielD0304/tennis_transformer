import pandas as pd

def load_single_year(year):
    print(f"Loading: {year}")
    dataframe = pd.read_csv(f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv")
    return dataframe

def load_all_matches(years=[2020, 2021, 2022, 2023, 2024]):
    dataframes = [load_single_year(year) for year in years]
    data = pd.concat(dataframes, ignore_index=True)
    return data


if __name__ == "__main__":
    df = load_all_matches()
    print(f"Total matches: {df.shape[0]}")
    print(f"Columns: {list(df.columns)}")
    print(df.head())