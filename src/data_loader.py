"""
Loads the tennis data from the Github of JeffSackmann.

source: https://github.com/JeffSackmann/tennis_atp
"""


import pandas as pd


def load_single_year(year):
    """
    Loads one particular year

    Args:
        year (integer): year from the ATP data

    Returns:
        Dataframe: dataframe that includes the atp data
    """
    print(f"Loading: {year}")
    dataframe = pd.read_csv(f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv")
    dataframe = dataframe.fillna(-1)  # or another sentinel value
    return dataframe

def load_all_matches(years=[2020, 2021, 2022, 2023, 2024]):
    """
    combines every year loaded from load_single_year

    Args:
        years (list, optional): every year that data should be from. Defaults to [2020, 2021, 2022, 2023, 2024].

    Returns:
        Dataframe: includes data from every year that was loaded
    """
    data_frames = [load_single_year(year) for year in years]
    data = pd.concat(data_frames, ignore_index=True)
    return data


if __name__ == "__main__":
    df = load_all_matches()
    print(f"Total matches: {df.shape[0]}")
    print(f"Columns: {list(df.columns)}")
    print(df.head())