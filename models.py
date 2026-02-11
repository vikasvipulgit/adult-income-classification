"""Model helpers for training / loading / predicting.

This file provides small placeholders to make the project runnable.
Replace with your real training and persistence logic.
"""
import os
import pickle
import numpy as np


MODEL_PATH = "model.pkl"


def load_model(path=MODEL_PATH):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    # fallback: simple dummy model
    class DummyModel:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    return DummyModel()


def predict(model, X):
    return model.predict(X)


def save_model(model, path=MODEL_PATH):
    with open(path, "wb") as f:
        pickle.dump(model, f)
