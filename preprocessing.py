import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(filepath="adult.csv"):
    df = pd.read_csv(filepath)
    df.replace(" ?", np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


def preprocess_data(df):
    df = df.copy()

    # Encode target
    df['income'] = df['income'].apply(lambda x: 1 if '>50K' in x else 0)

    X = df.drop("income", axis=1)
    y = df["income"]

    # One-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


def split_data(X, y, test_size=0.2):
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=42
    )
