import os
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
import pandas as pd

# =============================
# CLOUD SAFE PATH
# =============================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "autoencoder_new.h5"
)

# =============================
# CONFIG
# =============================

SENSOR_COLS = [
    "temperature",
    "pressure",
    "flow_rate",
    "vibration"
]

# =============================
# GLOBAL MODEL (LOAD ONCE)
# =============================

autoencoder = None


def get_model():
    """
    Lazy loading:
    Model loads only once when first request comes.
    Prevents Railway startup timeout (504).
    """
    global autoencoder

    if autoencoder is None:
        print("Loading autoencoder model...")

        autoencoder = load_model(
            MODEL_PATH,
            compile=False   # IMPORTANT (inference only)
        )

        print("Model loaded successfully!")

    return autoencoder


# =============================
# INFERENCE
# =============================

def run_inference(df, X):

    # Load model safely
    model = get_model()

    # Reconstruction
    X_recon = model.predict(X, verbose=0)

    # Reconstruction error
    sensor_errors = (X - X_recon) ** 2

    # Keep original dataframe
    output_df = df.copy()

    # Per sensor error
    for i, sensor in enumerate(SENSOR_COLS):
        output_df[f"{sensor}_error"] = sensor_errors[:, i]

    # Total error
    total_error = sensor_errors.sum(axis=1, keepdims=True)

    # Contribution percentage
    for i, sensor in enumerate(SENSOR_COLS):
        output_df[f"{sensor}_contribution"] = (
            sensor_errors[:, i] / (total_error[:, 0] + 1e-12)
        )

    # Final reconstruction error
    output_df["reconstruction_error"] = (
        total_error[:, 0] / X.shape[1]
    )

    return output_df