from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Medicinal Plant Identifier"

    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:8081", "exp://localhost:8081"]

    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost/medicinal_plants_db"

    # Security
    SECRET_KEY: str = "your-secret-key-here-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # ML Model
    MODEL_PATH: str = "models/plant_classifier.h5"
    MODEL_INPUT_SIZE: tuple = (224, 224)

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
