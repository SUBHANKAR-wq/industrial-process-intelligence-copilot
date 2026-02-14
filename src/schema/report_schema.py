from pydantic import BaseModel, Field
from typing import List

class ReportOutput(BaseModel):
    title: str = Field(
        description="Short headline summarizing the system condition"
    )

    timestamp: str = Field(
        description="Timestamp of the detected event"
    )

    system_state: str = Field(
        description="Overall system state: normal | drift | anomaly"
    )

    severity: str = Field(
        description="Severity level: none | low | medium | high"
    )

    dominant_sensors: List[str] = Field(
        description="Sensors contributing most to the issue"
    )

    probable_cause: str = Field(
        description="Most likely physical root cause"
    )

    explanation: str = Field(
        description="Engineering explanation of the issue"
    )

    recommended_action: str = Field(
        description="Operational action to be taken"
    )

    confidence: float = Field(
        description="Overall confidence score (0–1)"
    )

    summary: str = Field(
        description="One-paragraph executive summary"
    )
