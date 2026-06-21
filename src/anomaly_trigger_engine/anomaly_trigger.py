import numpy as np
import pandas as pd

input_path = "data/processed/autoencoder_output_keras_withsensor_errors.csv"
output_path = "data/processed/anomaly_triggers.csv"

df = pd.read_csv(input_path,parse_dates = ["timestamp"])

# we will set the threshold as per the normal data 
base_line_df = df[df["source"]=="synthetic"]

mean_error = base_line_df["reconstruction_error"].mean()
std_error = base_line_df["reconstruction_error"].std()

# Percentile-based thresholds (RECOMMENDED)

normal_threshold = np.percentile(
    base_line_df["reconstruction_error"], 95
)

anomaly_threshold = np.percentile(
    base_line_df["reconstruction_error"], 99
)
import json

thresholds = {
    "normal_threshold": float(normal_threshold),
    "anomaly_threshold": float(anomaly_threshold)
}

with open("models/thresholds.json", "w") as f:
    json.dump(thresholds, f, indent=4)

print(" Thresholds saved to models/thresholds.json")

print(f"Normal threshold  : {normal_threshold}")
print(f"Anomaly threshold : {anomaly_threshold}")


def classify_state(error):
    if error <= normal_threshold:
        return "normal"
    elif error <= anomaly_threshold:
        return "drift"
    else:
        return "anomaly"
    
df["state"] = df["reconstruction_error"].apply(classify_state)
print(" State labeling completed")
df.to_csv(output_path,index=False)
print(f" Saved to: {output_path}")

