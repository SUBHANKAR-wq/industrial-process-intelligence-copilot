import json

with open("models/thresholds.json", "r") as f:
    thresholds = json.load(f)

normal_threshold = thresholds["normal_threshold"]
anomaly_threshold = thresholds["anomaly_threshold"]
def classify_state(error):
    if error <= normal_threshold:
        return "normal"
    elif error <= anomaly_threshold:
        return "drift"
    else:
        return "anomaly"