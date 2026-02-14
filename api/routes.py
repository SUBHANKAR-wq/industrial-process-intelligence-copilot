from fastapi import APIRouter
from pipeline.main_pipeline import main_pipeline
import traceback

router = APIRouter()


@router.post("/predict")
def predict(data: dict):
    """
    Run full pipeline
    """
    try:
        sensor_values = data["sensor_values"]

        result = main_pipeline(sensor_values)

        return result

    except Exception as e:
        print("ERROR OCCURRED:", e)
        traceback.print_exc()
        return {"error": str(e)}