import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model # type: ignore

# -----------------------------
# CONFIG
# -----------------------------
ae_data = "data/processed/autoencoder_output_keras_withsensor_errors.csv"

original_data = "data/processed/anomaly_triggers.csv"

MODEL_PATH = "models/autoencoder.keras"

OUTPUT_PATH = "data/processed/output_with_sensor_errors.csv"

SENSOR_COLS = ["temperature", "pressure", "flow_rate", "vibration"]


df = pd.read_csv(ae_data, parse_dates=["timestamp"])
df_original = pd.read_csv(original_data, parse_dates=["timestamp"])

# Ensure both dataframes have the same number of rows
if len(df) != len(df_original):
    raise ValueError("Row mismatch between AE output and original data")


MODEL_PATH = "models/autoencoder.keras" 
autoencoder = load_model(MODEL_PATH)

X = df_original[SENSOR_COLS].values.astype("float32")


X_recon = autoencoder.predict(X, verbose=0)


# SENSOR-WISE RECONSTRUCTION ERROR

sensor_errors = (X - X_recon) ** 2

for i, sensor in enumerate(SENSOR_COLS):
    df[f"{sensor}_error"] = sensor_errors[:, i]


# NORMALIZED CONTRIBUTIONS

total_sensor_error = sensor_errors.sum(axis=1, keepdims=True)

for i, sensor in enumerate(SENSOR_COLS):
    df[f"{sensor}_contribution"] = (
        sensor_errors[:, i] / (total_sensor_error[:, 0] + 1e-12)
    )
df["state"] = df_original["state"]

# SAVE OUTPUT

df.to_csv(OUTPUT_PATH, index=False)

print("Sensor-wise reconstruction error computed CORRECTLY")
print(f" Saved to: {OUTPUT_PATH}")
