import uuid
import logging
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException, status
from typing import Optional
from api.schemas import AnalysisResponse, StatusResponse, AnalysisConfig

logger = logging.getLogger(__name__)
router = APIRouter()

# Temporary in-memory storage for active sessions (Will map to Redis later)
ACTIVE_SESSIONS = {}

async def run_analysis_pipeline(session_id: str, file_path: str, prompt: str, config: dict):
    """
    Placeholder for the main LangGraph execution.
    Will be implemented in Phase 2.
    """
    logger.info(f"Starting pipeline for {session_id}. Prompt: {prompt}")
    ACTIVE_SESSIONS[session_id]["status"] = "PROCESSING"
    # Orchestrator logic will go here
    pass

@router.post("/analyze", response_model=AnalysisResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_analysis(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    prompt: str = Form(...),
    run_ml: Optional[bool] = Form(True)
):
    """
    Accepts a dataset and a prompt, validates them, and starts a background agent job.
    """
    if not file.filename.endswith(('.csv', '.xlsx', '.xls', '.json')):
        raise HTTPException(
            status_code=400, 
            detail="Unsupported file format. Please upload CSV, Excel, or JSON."
        )

    session_id = str(uuid.uuid4())
    
    # In a real scenario, save the file to a secure temp directory or S3 bucket here.
    file_path = f"/tmp/{session_id}_{file.filename}"
    
    # Initialize mock state
    ACTIVE_SESSIONS[session_id] = {
        "status": "QUEUED",
        "current_step": "INGEST",
        "progress": 0,
        "logs": ["Job queued successfully."]
    }

    config = AnalysisConfig(run_ml=run_ml)

    # Queue the background task
    background_tasks.add_task(
        run_analysis_pipeline, 
        session_id=session_id, 
        file_path=file_path, 
        prompt=prompt,
        config=config.model_dump()
    )

    return AnalysisResponse(
        session_id=session_id,
        status="QUEUED",
        message="Data received. Autonomous analysis started."
    )

@router.get("/status/{session_id}", response_model=StatusResponse)
async def get_status(session_id: str):
    """Poll for the status of an ongoing analysis."""
    if session_id not in ACTIVE_SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    session_data = ACTIVE_SESSIONS[session_id]
    return StatusResponse(
        session_id=session_id,
        status=session_data["status"],
        progress_percent=session_data["progress"],
        current_step=session_data["current_step"],
        agent_log=session_data["logs"]
    )