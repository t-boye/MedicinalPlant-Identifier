from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import plants, recognition, auth
from app.core.config import settings

app = FastAPI(
    title="Medicinal Plant Identifier API",
    description="API for identifying medicinal plants and accessing their properties",
    version="1.0.0"
)

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
    return {
        "message": "Medicinal Plant Identifier API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
