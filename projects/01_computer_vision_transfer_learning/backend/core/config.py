"""
VisionAI Platform Configuration.
Environment-based configuration for image classification API.
"""

from pydantic_settings import BaseSettings
from typing import List
import secrets


class Settings(BaseSettings):
    """Application settings."""

    # App
    APP_NAME: str = "VisionAI - Image Classification Platform"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "Production-ready AI-powered image classification API"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"

    # API
    API_V1_PREFIX: str = "/api/v1"
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    ALGORITHM: str = "HS256"

    # Database
    DATABASE_URL: str = "postgresql://visionai:visionai123@db:5432/visionai"
    DATABASE_ECHO: bool = False

    # Redis
    REDIS_URL: str = "redis://redis:6379/0"
    CACHE_EXPIRE_SECONDS: int = 3600

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 30
    RATE_LIMIT_PER_HOUR: int = 500

    # Model
    MODEL_NAME: str = "resnet50"
    MODEL_PATH: str = "./models/best_model_resnet50.pth"
    NUM_CLASSES: int = 1000  # ImageNet classes
    IMAGE_SIZE: int = 224
    DEVICE: str = "cuda"

    # Upload
    UPLOAD_DIR: str = "./uploads"
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

    # CORS
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://frontend:3000"
    ]

    # Monitoring
    ENABLE_METRICS: bool = True
    PROMETHEUS_PORT: int = 9090

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"

    # Pagination
    DEFAULT_PAGE_SIZE: int = 20
    MAX_PAGE_SIZE: int = 100

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
