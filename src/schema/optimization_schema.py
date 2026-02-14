from pydantic import BaseModel, Field

class OptimizationOutput(BaseModel):
    recommended_action: str = Field(
        description="monitor | schedule_maintenance | shutdown"
    )
    justification: str = Field(
        description="Reason for the chosen action"
    )
