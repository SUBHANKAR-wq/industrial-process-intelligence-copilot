import pandas as pd
from datetime import datetime
import json  # Add this import

def data_generator(data):
    """
    Convert raw sensor input into structured dataframe

    Supports:
    - JSON string  -> '{"temperature":..., ...}'
    - list         -> [temp, pressure, flow, vibration]
    - dict         -> {"temperature":..., ...}
    """

    # ===== IF INPUT IS A STRING (JSON) =====
    if isinstance(data, str):
        try:
            # Parse the JSON string into a dictionary
            data = json.loads(data)
        except json.JSONDecodeError:
            # If it's not valid JSON, maybe it's a string representation of a list?
            # This handles cases like "[1,2,3,4]"
            if data.startswith('[') and data.endswith(']'):
                import ast
                data = ast.literal_eval(data)
            else:
                raise ValueError(f"Could not parse input string: {data}")

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
    elif isinstance(data, (list, tuple)):
        features = {
            "timestamp": datetime.now(),
            "temperature": float(data[0]),
            "pressure": float(data[1]),
            "flow_rate": float(data[2]),
            "vibration": float(data[3]),
        }
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    df = pd.DataFrame([features])
    return df