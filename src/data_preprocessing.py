import pandas as pd

def load_data(path:str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def handle_missing_values(df:pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.dropna(inplace=True)
    return df
