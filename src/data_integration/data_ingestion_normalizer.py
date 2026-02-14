import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

sensor_data_path = 'data/raw/sensor_data_raw.csv'
NASA_sensor_data_path = 'data/raw/train_FD001.txt'

output_data_path = 'data/processed/sensor_data_normalized.csv'

# loading synthetic sensor data
def load_synthetic_sensor_data(path):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["source"] = "synthetic"
    return df

# loading NASA sensor data
def load_NASA_sensor_data(path):
    
    colms = (   ["engine_id","cycle"]+
                [f"op_setting_{i}" for i in range(1,4)]+
                [f"s{i}" for i in range(1,22)] )
    
    df = pd.read_csv(path, sep= r"\s+", header=None, names=colms)

    # creating cycles as timestamps
    # making starting time for nasa as 1-1-24 
    
    start_time = datetime(2024, 1, 1)

    df["timestamp"] = df['cycle'].apply(lambda x: start_time + timedelta(seconds=int(x)*5))
   
    # mapping NASA features with relevant synthetic data features

    map_data = pd.DataFrame({
        "timestamp": df["timestamp"],
        "temperature": df["s2"],
        "pressure": df["s7"],
        "flow_rate": df["s4"],
        "vibration": df["s11"],
        "source": "nasa"
    })

    return map_data


# def normalize_sensor_data(df):
    
#     scaler = MinMaxScaler()
#     sensor_col = ["temperature", "pressure", "flow_rate", "vibration"]
#     df[sensor_col] = scaler.fit_transform(df[sensor_col])

#     return df



if __name__ == "__main__":
   
    synthetic_data = load_synthetic_sensor_data(sensor_data_path)
    nasa_data = load_NASA_sensor_data(NASA_sensor_data_path)

    # Drop label before ML processing becoz this may create leakage in model training of insupervised learning
    if "anomaly_label" in synthetic_data.columns:
        synthetic_df = synthetic_data.drop(columns=["anomaly_label"])

    
    
    combined_data = pd.concat([synthetic_data, nasa_data], ignore_index=True)



    combined_data.to_csv(output_data_path, index=False)

    print("Data ingestion and normalization completed successfully.")
    print(f"Normalized data saved to: {output_data_path}")
    print(combined_data.head())



    