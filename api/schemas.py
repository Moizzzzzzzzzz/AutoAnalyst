from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class AnalysisConfig(BaseModel):
    """Configuration overrides for the analysis run."""
    run_ml: bool = Field(default=True, description="Whether to run the AutoML agent")
    max_retries: int = Field(default=2, description="Max retries per agent failure")

class AnalysisResponse(BaseModel):
    """Response returned immediately upon task submission."""
    session_id: str
    status: str
    message: str

class StatusResponse(BaseModel):
    """Response for checking the progress of an ongoing session."""
    session_id: str
    status: str
    progress_percent: int
    current_step: str
    agent_log: List[str]

class HealthResponse(BaseModel):
    """Basic API health check response."""
    status: str
    version: str
    uptime_seconds: float