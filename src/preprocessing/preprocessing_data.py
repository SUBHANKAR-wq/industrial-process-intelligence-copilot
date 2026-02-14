import joblib
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

input_path = 'data/processed/sensor_data_normalized.csv'
output_path = 'data/processed/preprocessed_data_to_train.csv'
sensor_colms = ["temperature", "pressure", "flow_rate", "vibration"]

def load_data(path):
    return pd.read_csv(path, parse_dates=["timestamp"])#convert timestamp as a string to datetime

def split_normal_data(data):
    data = data[data['source'] == 'synthetic']
    print("Filtered synthetic normal data")
    return data

def fit_scaler(data):
    scaler = StandardScaler()
    scaler.fit(data[sensor_colms])
    print("Scaler fitting completed")
    return scaler

def apply_scaler(data, scaler):
    data = data.copy()
    data[sensor_colms] = scaler.transform(data[sensor_colms])
    print("Data scaling completed")
    return data
def save_scaler(scaler, path="models/scaler.pkl"):
    joblib.dump(scaler, path)
    print(f"Scaler saved to {path}")
    
def preprocess(input_path, output_path):
    assert os.path.exists(input_path), f"Input file not found: {input_path}"

    df = load_data(input_path)

    # applying scaler for only normal synthetic data as we are training on this
    train_df = split_normal_data(df)
    scaler = fit_scaler(train_df)

    save_scaler(scaler)
    df_processed = apply_scaler(df, scaler)
    df_processed.to_csv(output_path, index=False)

    print(" Preprocessing completed")
    return df_processed


if __name__ == "__main__":
    preprocess(input_path, output_path)
