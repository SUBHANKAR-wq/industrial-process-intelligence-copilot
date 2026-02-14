import pandas as pd
from datetime import datetime


def data_generator(data):
    """
    Convert raw sensor input into structured dataframe

    Supports:
    - list  -> [temp, pressure, flow, vibration]
    - dict  -> {"temperature":..., ...}
    """

    # ===== IF INPUT IS DICTIONARY =====
    if isinstance(data, dict):

        features = {
            "timestamp": datetime.now(),
            "temperature": float(data["temperature"]),
            "pressure": float(data["pressure"]),
            "flow_rate": float(data["flow_rate"]),
            "vibration": float(data["vibration"]),
        }

    # ===== IF INPUT IS LIST =====
    else:

        features = {
            "timestamp": datetime.now(),
            "temperature": float(data[0]),
            "pressure": float(data[1]),
            "flow_rate": float(data[2]),
            "vibration": float(data[3]),
        }

    df = pd.DataFrame([features])

    return df