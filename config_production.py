# Production Configuration for Agriverse
# Email: harshag1772004@gmail.com

import os

class ProductionConfig:
    SECRET_KEY = '@q0x1vCkjz4XgQuzXUrV7E1FBNw5uoxd'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'postgresql://agriverse:7%G9NvnQ9mdWBAl^O9q4@localhost/agriverse'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Email
    MAIL_SERVER = 'smtp.gmail.com'
    MAIL_PORT = 587
    MAIL_USE_TLS = True
    MAIL_USERNAME = 'harshag1772004@gmail.com'
    MAIL_PASSWORD = 'k3oKuqQB0jG^6gFluLGYUIf3'
    
    # APIs
    WEATHER_API_KEY = 'OWM_WEATHER_HoqIUfM51aU1MFgy51MKSlBdKLBFHynhuOhY2s30'
    OPENAI_API_KEY = 'sk-OPENAI_31riaUcpDdRM7NxjTPAFDzrFnC5F8CtlatmPdjnBqB8aIcwv'
    DEEPSEEK_API_KEY = 'sk-DEEPSEEK_jHhOBICYiLDq27NZseMT5iaspUTGdmiA6hHOCptAxqZ0DYIi'
    GOOGLE_MAPS_API_KEY = 'AIzaMAPS_4M7wKhNZaUWIbqoWGNggkEW3nKXMKubFZrk'
    
    # Stripe
    STRIPE_PUBLISHABLE_KEY = 'pk_test_STRIPE_qCQU2cXD2zp1vUftK0QfOZl0'
    STRIPE_SECRET_KEY = 'AZFB1TSRGQ8stwrNkgEFAySjNJ*NC7vF'
    
    # Redis
    REDIS_URL = 'redis://localhost:6379/0'
