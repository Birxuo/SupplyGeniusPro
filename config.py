from pydantic import BaseSettings, validator
from typing import Dict, List, Optional
import os

class Settings(BaseSettings):
    # Application Settings
    APP_NAME: str = "SupplyGenius Pro"
    DEBUG: bool = False
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    
    # IBM Granite Model Settings
    MODEL_CONFIG: dict = {
        "temperature": 0.7,
        "max_tokens": 1000,
        "top_p": 0.95,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.1
    }
    
    # Cache Settings
    CACHE_TTL: int = 3600
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Security Settings
    SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALLOWED_HOSTS: List[str] = ["*"]
    
    # Rate Limiting
    RATE_LIMIT_DEFAULT: str = "100/minute"
    RATE_LIMIT_PREMIUM: str = "1000/minute"
    
    # Supply Chain Settings
    INVENTORY_ALERT_THRESHOLD: float = 0.2
    LEAD_TIME_BUFFER: int = 7
    SAFETY_STOCK_FACTOR: float = 1.5
    
    # Monitoring
    METRICS_ENABLED: bool = True
    PROMETHEUS_PORT: int = 9090
    LOG_LEVEL: str = "INFO"
    
    # Background Tasks
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"
    TASK_QUEUE_MAX_SIZE: int = 1000

    # Database Settings
    DATABASE_URL: str = "sqlite:///./supplygenius.db"
    
    @validator("SECRET_KEY", pre=True)
    def validate_secret_key(cls, v: Optional[str]) -> str:
        if not v:
            return os.urandom(32).hex()
        return v

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()