import pandas as pd
from datetime import datetime


def data_generator(data):
    """
    Convert raw sensor list into structured format
    """

    features = {
        "timestamp": datetime.now(),
        "temperature": float(data[0]),
        "pressure": float(data[1]),
        "flow_rate": float(data[2]),
        "vibration": float(data[3]),
    }

    df = pd.DataFrame([features])

    return df