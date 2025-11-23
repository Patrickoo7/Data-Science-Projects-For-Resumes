"""
StockPredict Platform Configuration.
Environment-based settings for time series forecasting API.
"""

from pydantic_settings import BaseSettings
from typing import List
import secrets


class Settings(BaseSettings):
    """Application settings."""

    # App
    APP_NAME: str = "StockPredict - AI Stock Forecasting Platform"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "Production-ready stock price prediction with deep learning"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"

    # API
    API_V1_PREFIX: str = "/api/v1"
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    ALGORITHM: str = "HS256"

    # Database
    DATABASE_URL: str = "postgresql://stockpredict:stockpredict123@db:5432/stockpredict"
    DATABASE_ECHO: bool = False

    # Redis
    REDIS_URL: str = "redis://redis:6379/0"
    CACHE_EXPIRE_SECONDS: int = 3600

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 20
    RATE_LIMIT_PER_HOUR: int = 300

    # Model
    MODEL_TYPE: str = "lstm"  # lstm, gru, transformer, ensemble
    MODEL_PATH: str = "./models/lstm_model.pth"
    SEQUENCE_LENGTH: int = 60
    PREDICTION_DAYS: int = 30
    DEVICE: str = "cuda"

    # Stock Data
    DEFAULT_TICKER: str = "AAPL"
    DATA_START_DATE: str = "2020-01-01"
    ALLOWED_TICKERS: List[str] = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA"]

    # Technical Indicators
    USE_INDICATORS: bool = True
    INDICATORS: List[str] = ["SMA_20", "SMA_50", "RSI", "MACD", "BBANDS"]

    # CORS
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8501",  # Streamlit default
        "http://frontend:8501"
    ]

    # Monitoring
    ENABLE_METRICS: bool = True
    PROMETHEUS_PORT: int = 9090

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"

    # Backtesting
    INITIAL_CAPITAL: float = 10000.0
    COMMISSION: float = 0.001  # 0.1%

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
