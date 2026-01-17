import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_categorical_features(df:pd.DataFrame, categorial_cols:list) -> pd.DataFrame:
    df = df.copy()
    le = LabelEncoder()

    for col in categorial_cols:
        df[col] = le.fit_transform(df[col])

    return df


def split_features_target(df:pd.DataFrame, target_col:str):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return X,y
