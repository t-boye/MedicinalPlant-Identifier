from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import plants, recognition, auth
from app.core.config import settings
from app.core.logging_config import get_logger

# Initialize logger
logger = get_logger(__name__)

app = FastAPI(
    title="Medicinal Plant Identifier API",
    description="API for identifying medicinal plants and accessing their properties",
    version="1.0.0"
)

logger.info("Starting Medicinal Plant Identifier API...")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(plants.router, prefix="/api/plants", tags=["Plants"])
app.include_router(recognition.router, prefix="/api/recognition", tags=["Recognition"])

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {
        "message": "Medicinal Plant Identifier API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    logger.debug("Health check endpoint accessed")
    return {"status": "healthy"}

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info("Application startup complete")
    logger.info(f"CORS enabled for origins: {settings.ALLOWED_ORIGINS}")

@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("Application shutting down...")
