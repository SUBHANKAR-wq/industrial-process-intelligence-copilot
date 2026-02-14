import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

# no. of samples
num_points  = 20000
# no. of samples having these characteristics 
# normal_points = 16000
drift_start = 2000
anomaly_start = 2000
# interval for each sensors
interval_seconds = 5

np.random.seed(42)

# function for generating time stamps
def generate_time_stamp(start_time, num_points,interval_seconds):
    time_stamp = []
    for i in range(num_points):
        time_stamp.append(start_time + timedelta(seconds=i*interval_seconds))
    return time_stamp

# function for generating sensor data
def generate_normal_dis_data(mean,std,size):
    return np.random.normal(mean,std,size)

def apply_gradual_drift(data , drift_rate):
    # linspace will genrate values between 0 to drift_rate*len(data) which have same rate of change
    drift = np.linspace(0,drift_rate*(len(data)),len(data)) 
    return data + drift

def apply_sudden_anomaly(data,spike_value):
    anomaly_data = data.copy()
    for i in range(len(anomaly_data)):
        if random.random() < 0.2:  # 20% chance to introduce an anomaly
            anomaly_data[i] += spike_value 

    return anomaly_data


def generate_sensor_data():
    start_time = datetime.now()

    # Generate timestamps
    time_stamp = generate_time_stamp(start_time,num_points,interval_seconds)

    # normal sensor data genration
    temperature = generate_normal_dis_data(mean = 70,std = 0.5,size =num_points)
    pressure = generate_normal_dis_data(mean = 5,std = 0.1,size =num_points)
    flow_rate = generate_normal_dis_data(mean = 100,std = 2,size =num_points)
    vibration = generate_normal_dis_data(mean = 0.02,std = 0.005,size =num_points)
    anomaly_label = np.zeros(num_points)


    # applying gradual drift to temprature sensor data and vibration sensor data

    temperature[drift_start:anomaly_start] = apply_gradual_drift(temperature[drift_start:anomaly_start], drift_rate=0.002) 
    vibration[drift_start:anomaly_start] = apply_gradual_drift(vibration[drift_start:anomaly_start], drift_rate=0.0005)

    # applying sudden anomaly to pressure sensor data and flow_rate sensor data
        # -------- Sudden Anomalies --------
    pressure[anomaly_start:] = apply_sudden_anomaly(pressure[anomaly_start:], spike_value=1.5
    )

    flow_rate[anomaly_start:] = apply_sudden_anomaly(
        flow_rate[anomaly_start:], spike_value=-20
    )

    anomaly_label[anomaly_start:] = 1
    # creating dataframe
    data = {
        "timestamp": time_stamp,
        "temperature": temperature,
        "pressure": pressure,
        "flow_rate": flow_rate,
        "vibration": vibration,
        "anomaly_label": anomaly_label
    }

    return pd.DataFrame(data)


if __name__ == "__main__":
    df = generate_sensor_data()

    output_path = r"D:\industrial-process-intelligence-copilot\data\raw\sensor_data_raw.csv"

    df.to_csv(output_path, index=False)

    print("Sensor data generated successfully")
    print(f" Saved to: {output_path}")


