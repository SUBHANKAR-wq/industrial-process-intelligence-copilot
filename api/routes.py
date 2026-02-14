from fastapi import APIRouter
from pipeline.main_pipeline import main_pipeline

router = APIRouter()


@router.post("/predict")
def predict(data: dict):
    """
    Run full pipeline
    """

    sensor_values = data["sensor_values"]

    result = main_pipeline(sensor_values)

    return result