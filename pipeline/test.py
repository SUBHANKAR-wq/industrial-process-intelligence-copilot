from pipeline.main_pipeline import main_pipeline
import numpy as np
import time

step = 0


def simulate_sensor_data():

    global step

    temperature = np.random.normal(70, 0.5)
    pressure = np.random.normal(5, 0.1)
    flow_rate = np.random.normal(100, 2)
    vibration = np.random.normal(0.02, 0.005)

    cycle = step % 5

    if cycle in [0, 1]:

        mode = "NORMAL"

    elif cycle == 2:

        drift = np.random.uniform(2, 4)

        temperature += drift
        vibration += drift * 0.002

        mode = "DRIFT"

    else:

        pressure += 1.5
        flow_rate -= 20

        temperature += 0.5
        vibration += 0.01

        mode = "ANOMALY"

    step += 1

    return {
        "temperature": round(temperature, 2),
        "pressure": round(pressure, 2),
        "flow_rate": round(flow_rate, 2),
        "vibration": round(vibration, 4),
        "mode": mode
    }


while True:

    sensor_data = simulate_sensor_data()

    print("\n" + "=" * 60)
    print("Sensor Input:")
    print(sensor_data)

    try:

        result = main_pipeline(sensor_data)

        print("\nPipeline Result")

        print(
            f"State : {result['state']}"
        )

        print(
            f"Score : {result['anomaly_score']:.5f}"
        )

        print(
            f"Actual : {sensor_data['mode']}"
        )

        if result["root_cause"]:
            print(
                "Root Cause:",
                result["root_cause"]
            )

    except Exception as e:
        print("Error:", e)

    time.sleep(2)