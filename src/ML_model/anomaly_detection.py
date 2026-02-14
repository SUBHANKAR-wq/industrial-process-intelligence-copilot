import numpy as np
import pandas as pd
from tensorflow.keras.models import Model# type: ignore
from tensorflow.keras.layers import Input, Dense# type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

# -----------------------------
# PATHS
# -----------------------------
DATA_PATH = "data/processed/preprocessed_data_to_train.csv"
OUTPUT_PATH = "data/processed/autoencoder_output_keras_withsensor_errors.csv"

SENSOR_COLS = ["temperature", "pressure", "flow_rate", "vibration"]

# -----------------------------
# LOAD DATA
# -----------------------------
def load_data(path):
    return pd.read_csv(path, parse_dates=["timestamp"])

# -----------------------------
# AUTOENCODER
# -----------------------------
def build_autoencoder(input_dim, lr):
    input_layer = Input(shape=(input_dim,))

    encoded = Dense(16, activation="relu")(input_layer)
    encoded = Dense(8, activation="relu")(encoded)

    decoded = Dense(16, activation="relu")(encoded)
    output_layer = Dense(input_dim, activation="linear")(decoded)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(
        optimizer=Adam(learning_rate=lr),
        loss="mse"
    )
    autoencoder.summary()
    return autoencoder

# -----------------------------
# TRAIN
# -----------------------------
def train_autoencoder(data, epochs, batch_size, lr):
    input_dim = data.shape[1]
    model = build_autoencoder(input_dim, lr)

    model.fit(
        data,
        data,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=1
    )
    return model

# -----------------------------
# RECONSTRUCTION ERROR
# -----------------------------
def compute_reconstruction_error(x_true, x_pred, df):
    error = np.mean(np.square(x_true - x_pred), axis=1)
    df["reconstruction_error"] = error
    return df

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    df = load_data(DATA_PATH)

    if "source" not in df.columns:
        raise ValueError("'source' column missing. Run preprocessing first.")

    train_df = df[df["source"] == "synthetic"]

    X_train = train_df[SENSOR_COLS].values.astype("float32")
    X_all = df[SENSOR_COLS].values.astype("float32")

    autoencoder = train_autoencoder(
        X_train,
        epochs=50,
        batch_size=32,
        lr=0.001
    )
    autoencoder.save("models/autoencoder.keras")
    print("Autoencoder model saved")


    X_reconstructed = autoencoder.predict(X_all)
    df_result = compute_reconstruction_error(X_all, X_reconstructed, df)

    df_result.to_csv(OUTPUT_PATH, index=False)
    print(" Autoencoder processing completed")
