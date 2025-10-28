import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import List, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """
    Application settings with environment variable support
    """
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Medicinal Plant Identifier"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True

    # CORS - Parse comma-separated list from env
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:5173",  # Frontend dev server
        "http://localhost:3000",
        "http://localhost:8081",
        "exp://localhost:8081"
    ]

    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/medicinal_plants_db"

    # Security
    SECRET_KEY: str = "your-secret-key-here-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # ML Model Configuration
    MODEL_PATH: str = "./saved_models/plant_cnn_efficientnet.keras"
    MODEL_INPUT_SIZE: Tuple[int, int] = (224, 224)
    CONFIDENCE_THRESHOLD: float = 0.3

    # Training Configuration
    TRAINING_DATA_DIR: str = "./ml_model/data/raw"
    MODEL_SAVE_DIR: str = "./saved_models"
    BASE_MODEL: str = "efficientnet"
    BATCH_SIZE: int = 32
    EPOCHS: int = 50
    LEARNING_RATE: float = 0.001

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/app.log"

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Parse ALLOWED_ORIGINS from comma-separated string if provided as string
        if isinstance(self.ALLOWED_ORIGINS, str):
            self.ALLOWED_ORIGINS = [origin.strip() for origin in self.ALLOWED_ORIGINS.split(',')]

        # Parse MODEL_INPUT_SIZE from comma-separated string if provided as string
        if isinstance(self.MODEL_INPUT_SIZE, str):
            size = tuple(map(int, self.MODEL_INPUT_SIZE.split(',')))
            self.MODEL_INPUT_SIZE = size

        # Create necessary directories
        Path(self.MODEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.LOG_FILE).parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
