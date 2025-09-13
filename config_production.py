import os
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

class ProductionConfig:
    """
    Production configuration for Agriverse with Oracle Cloud integration.
    """
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'production-secret-key-change-this'
    
    # Oracle Cloud Database Configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'oracle+oracledb://username:password@adb.us-ashburn-1.oraclecloud.com:1522/agriverse_prod'
    
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 20,
        'max_overflow': 30,
        'pool_pre_ping': True,
        'pool_recycle': 3600,
        'connect_args': {
            'encoding': 'utf-8',
            'nencoding': 'utf-8'
        }
    }
    
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Mail settings
    MAIL_SERVER = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
    MAIL_PORT = int(os.environ.get('MAIL_PORT', 587))
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'true').lower() in ['true', 'on', '1']
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    
    # Stripe settings
    STRIPE_SECRET_KEY = os.environ.get('STRIPE_SECRET_KEY')
    STRIPE_PUBLISHABLE_KEY = os.environ.get('STRIPE_PUBLISHABLE_KEY')
    
    # Upload settings
    UPLOAD_FOLDER = os.path.join(basedir, 'app', 'static', 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Redis Configuration for Caching
    REDIS_URL = os.environ.get('REDIS_URL') or 'redis://localhost:6379/0'
    CACHE_TYPE = 'redis'
    CACHE_REDIS_URL = REDIS_URL
    CACHE_DEFAULT_TIMEOUT = 300  # 5 minutes
    
    # Oracle Cloud AI Services
    OCI_CONFIG_PATH = os.environ.get('OCI_CONFIG_PATH', '~/.oci/config')
    OCI_PROFILE = os.environ.get('OCI_PROFILE', 'DEFAULT')
    
    # ML Model Configuration
    ML_MODEL_PATH = os.path.join(basedir, 'app', 'ml_models', 'saved_models')
    ML_CACHE_TIMEOUT = 3600  # 1 hour
    
    # Security Settings
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = 3600  # 1 hour
    
    # Logging Configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE', 'agriverse.log')
    
    # Performance Settings
    JSON_SORT_KEYS = False
    JSONIFY_PRETTYPRINT_REGULAR = False
    
    # API Rate Limiting
    RATELIMIT_STORAGE_URL = REDIS_URL
    RATELIMIT_DEFAULT = "1000 per hour"
    
    # CORS Settings
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*').split(',')
    
    # Monitoring and Analytics
    ENABLE_METRICS = os.environ.get('ENABLE_METRICS', 'true').lower() == 'true'
    METRICS_ENDPOINT = os.environ.get('METRICS_ENDPOINT', '/metrics')

class DevelopmentConfig:
    """
    Development configuration for local development.
    """
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-change-this-in-production'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'instance', 'agriverse.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Mail settings
    MAIL_SERVER = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
    MAIL_PORT = int(os.environ.get('MAIL_PORT', 587))
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'true').lower() in ['true', 'on', '1']
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    
    # Stripe settings
    STRIPE_SECRET_KEY = os.environ.get('STRIPE_SECRET_KEY')
    STRIPE_PUBLISHABLE_KEY = os.environ.get('STRIPE_PUBLISHABLE_KEY')
    
    # Upload settings
    UPLOAD_FOLDER = os.path.join(basedir, 'app', 'static', 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Development settings
    DEBUG = True
    TESTING = False
    
    # ML Model Configuration
    ML_MODEL_PATH = os.path.join(basedir, 'app', 'ml_models', 'saved_models')
    ML_CACHE_TIMEOUT = 300  # 5 minutes for development

class TestingConfig:
    """
    Testing configuration for unit tests.
    """
    SECRET_KEY = 'testing-secret-key'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    TESTING = True
    WTF_CSRF_ENABLED = False
    
    # Disable mail sending during tests
    MAIL_SUPPRESS_SEND = True
    
    # ML Model Configuration
    ML_MODEL_PATH = os.path.join(basedir, 'app', 'ml_models', 'test_models')
    ML_CACHE_TIMEOUT = 60  # 1 minute for testing

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
