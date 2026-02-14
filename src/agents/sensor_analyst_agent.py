"""
Sensor Analyst Agent
--------------------
Analyzes sensor-wise reconstruction contributions to:
- identify dominant sensors
- assess severity
- compute confidence
"""

SENSORS = ["temperature", "pressure", "flow_rate", "vibration"]

def dominant_sensors(row):
    # Use CONTRIBUTIONS, not errors
    contributions = {
        sensor: row[f"{sensor}_contribution"]
        for sensor in SENSORS
    }

    # Sort by contribution (descending)
    sorted_sensors = sorted(
        contributions.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Select sensors contributing >= 10%
    dominant = [
        sensor for sensor, value in sorted_sensors
        if value >= 0.10
    ]

    # Fallback: ensure at least one sensor
    if not dominant:
        dominant = [sorted_sensors[0][0]]

    return dominant


def severity(row):
    state = row["state"]
    error = row["reconstruction_error"]

    if state == "normal":
        return "none"

    elif state == "drift":
        # simple, deterministic rule
        return "low" if error < 2e-05 else "medium"

    else:  # anomaly
        return "high"


def confidence_score(row, dominant):
    # Confidence = sum of dominant sensor contributions
    return sum(
        row[f"{sensor}_contribution"]
        for sensor in dominant
    )


def sensor_analyst_agent(row):
    dom = dominant_sensors(row)

    return {
        "dominant_sensors": dom,
        "severity": severity(row),
        "confidence": round(confidence_score(row, dom), 3)
    }
