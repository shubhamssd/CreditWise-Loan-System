# src/feature_engineering.py
from sklearn.preprocessing import LabelEncoder
import joblib

def encode_and_save(df, categorical_cols, encoder_path="models/encoders.pkl"):
    df = df.copy()
    encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    joblib.dump(encoders, encoder_path)
    return df

categorical_cols = [
    "Employment_Status", "Marital_Status", "Dependents",
    "Loan_Purpose", "Property_Area", "Education_Level",
    "Gender", "Employer_Category"
]

df = encode_and_save(df, categorical_cols)


def split_features_target(df:pd.DataFrame, target_col:str):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return X,y
