import pandas as pd
from datetime import datetime


def data_generator(data):
    """
    Convert raw sensor list into structured format
    """

    features = {
            "timestamp": datetime.now(),
            "temperature": float(data["temperature"]),
            "pressure": float(data["pressure"]),
            "flow_rate": float(data["flow_rate"]),
            "vibration": float(data["vibration"]),
        }
    

    df = pd.DataFrame([features])

    return df