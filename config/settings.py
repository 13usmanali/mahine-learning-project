import os
from typing import Dict, Any

class Config:
    """Application configuration"""
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'cyber-sentinel-secret-key'
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    # Model settings
    MODEL_PATH = os.environ.get('MODEL_PATH', 'models/cyber_sentinel_model.pkl')
    
    # API settings
    API_HOST = os.environ.get('API_HOST', '0.0.0.0')
    API_PORT = int(os.environ.get('API_PORT', 5000))
    
    # Logging settings
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    
    # CORS settings (if needed)
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*')
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            key: getattr(cls, key)
            for key in dir(cls)
            if not key.startswith('_') and not callable(getattr(cls, key))
        }