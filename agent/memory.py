import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

class SessionMemory(BaseModel):
    """
    Holds the complete state of a single analysis session.
    Passed between LangGraph nodes during execution.
    """
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: str = Field(default="INIT", description="Current state of the pipeline")
    user_prompt: str = Field(..., description="The original natural language request")
    
    # Dataset Metadata
    dataset_path: Optional[str] = None
    dataset_summary: Dict[str, Any] = Field(default_factory=dict)
    
    # Agent Outputs
    eda_results: Dict[str, Any] = Field(default_factory=dict)
    hypotheses: List[Dict[str, Any]] = Field(default_factory=list)
    model_results: Dict[str, Any] = Field(default_factory=dict)
    visualizations: List[str] = Field(default_factory=list, description="List of file paths to generated charts")
    
    # Tracking & Diagnostics
    agent_reasoning: List[str] = Field(default_factory=list, description="Chain-of-thought log")
    errors: List[Dict[str, str]] = Field(default_factory=list)
    total_tokens_used: int = Field(default=0)
    
    # Timestamps
    timestamp_start: datetime = Field(default_factory=datetime.utcnow)
    timestamp_end: Optional[datetime] = None

    def add_log(self, agent_name: str, message: str) -> None:
        """Appends a timestamped reasoning log."""
        timestamp = datetime.utcnow().strftime("%H:%M:%S")
        self.agent_reasoning.append(f"[{timestamp}] {agent_name}: {message}")

    def add_error(self, agent_name: str, error_msg: str) -> None:
        """Records a non-fatal error for the final report's limitations section."""
        self.errors.append({"agent": agent_name, "error": error_msg})
