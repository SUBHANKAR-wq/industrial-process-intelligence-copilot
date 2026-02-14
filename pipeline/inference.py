import os
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd

# ⭐ absolute path (CLOUD SAFE)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "autoencoder_fixed.h5"
)

SENSOR_COLS = [
    "temperature",
    "pressure",
    "flow_rate",
    "vibration"
]

autoencoder = None


def get_model():
    global autoencoder
    if autoencoder is None:
        autoencoder = load_model(
            MODEL_PATH,
            compile=False,
            safe_mode=False
        )
    return autoencoder


def run_inference(df, X):

    model = get_model()
    X_recon = model.predict(X, verbose=0)

    sensor_errors = (X - X_recon) ** 2
    output_df = df.copy()

    for i, sensor in enumerate(SENSOR_COLS):
        output_df[f"{sensor}_error"] = sensor_errors[:, i]

    total_error = sensor_errors.sum(axis=1, keepdims=True)

    for i, sensor in enumerate(SENSOR_COLS):
        output_df[f"{sensor}_contribution"] = (
            sensor_errors[:, i] / (total_error[:, 0] + 1e-12)
        )

    output_df["reconstruction_error"] = (
        total_error[:, 0] / X.shape[1]
    )

    return output_df