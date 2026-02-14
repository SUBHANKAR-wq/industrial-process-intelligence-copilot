from tensorflow.keras.models import load_model  # type: ignore
import numpy as np
import pandas as pd

MODEL_PATH = "models/autoencoder.keras"

SENSOR_COLS = [
    "temperature",
    "pressure",
    "flow_rate",
    "vibration"
]

# Load model ONLY ONCE
autoencoder = load_model(MODEL_PATH)


def run_inference(df, X):

    # X = scaled data
    X_recon = autoencoder.predict(X, verbose=0)

    sensor_errors = (X - X_recon) ** 2

    # KEEP ORIGINAL DF (timestamp safe)
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