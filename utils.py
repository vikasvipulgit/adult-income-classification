"""Utility functions: data loading and preprocessing helpers.

These are minimal implementations to get started. Adapt to your dataset.
"""
import os
import pandas as pd


def load_data(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def preprocess(df):
    # Minimal preprocessing: drop non-numeric columns and fill NaNs
    numeric = df.select_dtypes(include=["number"]).copy()
    numeric = numeric.fillna(0)
    return numeric.values
