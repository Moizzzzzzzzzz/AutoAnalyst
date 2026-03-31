import time
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router as api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Application startup timestamp for health checks
START_TIME = time.time()

def create_app() -> FastAPI:
    """Application factory to initialize FastAPI and middleware."""
    app = FastAPI(
        title="AutoAnalyst API",
        description="Autonomous Data Analysis Agent Backend",
        version="0.1.0"
    )

    # Configure CORS for local development and future HuggingFace Space UI
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Restrict this in true production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routes
    app.include_router(api_router)
    
    logger.info("AutoAnalyst API initialized successfully.")
    return app

app = create_app()

@app.get("/health", tags=["System"])
async def health_check() -> dict:
    """Returns the operational status of the API."""
    uptime = time.time() - START_TIME
    return {
        "status": "ok",
        "version": "0.1.0",
        "uptime_seconds": round(uptime, 2)
    }