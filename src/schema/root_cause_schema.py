from pydantic import BaseModel, Field

class RootCauseOutput(BaseModel):
    probable_cause: str = Field(
        description="Most likely physical cause of the issue"
    )
    explanation: str = Field(
        description="Engineering reasoning behind the diagnosis"
    )
    inspection_focus: str = Field(
        description="Component that should be inspected first"
    )
    confidence: float = Field(
        description="Confidence level between 0 and 1"
    )
