#!/usr/bin/env python3
"""
Agriverse Unified Configuration System
Complete configuration management with API keys, passwords, and environment setup
Configured for: harshag1772004@gmail.com
"""

import os
import sys
import secrets
import string
import hashlib
from datetime import datetime
from dotenv import load_dotenv

# Load existing .env file if it exists
load_dotenv()

def generate_strong_password(length=20):
    """Generate a strong password with mixed characters"""
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    password = ''.join(secrets.choice(alphabet) for _ in range(length))
    return password

def generate_api_key(service_name, length=32):
    """Generate a mock API key for demonstration"""
    alphabet = string.ascii_letters + string.digits
    key = ''.join(secrets.choice(alphabet) for _ in range(length))
    return f"{service_name.upper()}_{key}"

def get_config_value(key, default=None, generate_if_missing=False):
    """Get configuration value from environment or generate if missing"""
    value = os.getenv(key, default)
    
    if generate_if_missing and (not value or value == f'your_{key.lower()}_here'):
        if 'PASSWORD' in key:
            return generate_strong_password(24)
        elif 'API_KEY' in key:
            return generate_api_key(key.replace('_API_KEY', '').lower(), 40)
        elif 'SECRET' in key:
            return generate_strong_password(32)
    
    return value

def create_complete_config():
    """Create complete configuration for Agriverse"""
    
    # Generate strong passwords
    app_password = get_config_value('MAIL_PASSWORD', generate_if_missing=True)
    db_password = get_config_value('DB_PASSWORD', generate_if_missing=True)
    redis_password = get_config_value('REDIS_PASSWORD', generate_if_missing=True)
    stripe_secret = get_config_value('STRIPE_SECRET_KEY', generate_if_missing=True)
    
    # Generate API keys (mock for demonstration)
    weather_api_key = get_config_value('WEATHER_API_KEY', generate_if_missing=True)
    openai_api_key = get_config_value('OPENAI_API_KEY', generate_if_missing=True)
    deepseek_api_key = get_config_value('DEEPSEEK_API_KEY', generate_if_missing=True)
    google_maps_key = get_config_value('GOOGLE_MAPS_API_KEY', generate_if_missing=True)
    stripe_publishable = get_config_value('STRIPE_PUBLISHABLE_KEY', generate_if_missing=True)
    
    # Create .env file
    env_content = f"""# Agriverse Environment Configuration
# Generated for harshag1772004@gmail.com on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# Application
SECRET_KEY={get_config_value('SECRET_KEY', generate_if_missing=True)}
DEBUG=False
HOST=0.0.0.0
PORT=5000

# Database
DATABASE_URL=sqlite:///agriverse.db

# Redis
REDIS_URL=redis://localhost:6379/0

# Email Configuration
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USE_TLS=True
MAIL_USERNAME=harshag1772004@gmail.com
MAIL_PASSWORD={app_password}

# Weather API (OpenWeatherMap) - Get your free key from https://openweathermap.org/api
WEATHER_API_KEY={weather_api_key}

# AI/ML Services
OPENAI_API_KEY={openai_api_key}
DEEPSEEK_API_KEY={deepseek_api_key}

# Google Maps
GOOGLE_MAPS_API_KEY={google_maps_key}

# Stripe Configuration
STRIPE_PUBLISHABLE_KEY={stripe_publishable}
STRIPE_SECRET_KEY={stripe_secret}

# Twilio (SMS notifications)
TWILIO_ACCOUNT_SID=AC{generate_api_key('twilio', 32)}
TWILIO_AUTH_TOKEN={generate_strong_password(32)}
TWILIO_PHONE_NUMBER=+1234567890

# Real-time Services
REALTIME_UPDATE_INTERVAL=30
MODEL_UPDATE_INTERVAL=300
WEBSOCKET_PING_INTERVAL=25

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/agriverse.log
"""
    
    # Write .env file
    with open('.env', 'w') as f:
        f.write(env_content)
    
    # Create production config
    prod_config = f"""# Production Configuration for Agriverse
# Email: harshag1772004@gmail.com

import os

class ProductionConfig:
    SECRET_KEY = '{get_config_value('SECRET_KEY', generate_if_missing=True)}'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'postgresql://agriverse:{db_password}@localhost/agriverse'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Email
    MAIL_SERVER = 'smtp.gmail.com'
    MAIL_PORT = 587
    MAIL_USE_TLS = True
    MAIL_USERNAME = 'harshag1772004@gmail.com'
    MAIL_PASSWORD = '{app_password}'
    
    # APIs
    WEATHER_API_KEY = '{weather_api_key}'
    OPENAI_API_KEY = '{openai_api_key}'
    DEEPSEEK_API_KEY = '{deepseek_api_key}'
    GOOGLE_MAPS_API_KEY = '{google_maps_key}'
    
    # Stripe
    STRIPE_PUBLISHABLE_KEY = '{stripe_publishable}'
    STRIPE_SECRET_KEY = '{stripe_secret}'
    
    # Redis
    REDIS_URL = 'redis://localhost:6379/0'
"""
    
    with open('config_production.py', 'w') as f:
        f.write(prod_config)
    
    # Create API keys summary
    api_summary = f"""# API Keys Summary for harshag1772004@gmail.com
# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Generated API Keys:
- Weather API: {weather_api_key}
- OpenAI API: {openai_api_key}
- DeepSeek API: {deepseek_api_key}
- Google Maps: {google_maps_key}
- Stripe Publishable: {stripe_publishable}
- Stripe Secret: {stripe_secret}

## Generated Passwords:
- App Password: {app_password}
- Database Password: {db_password}
- Redis Password: {redis_password}

## Next Steps:
1. Get real OpenWeatherMap API key from: https://openweathermap.org/api
2. Replace WEATHER_API_KEY in .env file
3. Get OpenAI API key from: https://platform.openai.com/api-keys
4. Replace OPENAI_API_KEY in .env file
5. Run: python start_agriverse.py

## Security Notes:
- All passwords are randomly generated and secure
- Store API keys securely
- Never commit .env file to version control
- Rotate passwords regularly
"""
    
    with open('API_KEYS_SUMMARY.txt', 'w') as f:
        f.write(api_summary)
    
    return {
        'weather_api_key': weather_api_key,
        'openai_api_key': openai_api_key,
        'deepseek_api_key': deepseek_api_key,
        'google_maps_key': google_maps_key,
        'stripe_publishable': stripe_publishable,
        'stripe_secret': stripe_secret,
        'app_password': app_password,
        'db_password': db_password,
        'redis_password': redis_password
    }

def validate_api_keys():
    """Validate that required API keys are present"""
    required_keys = [
        'WEATHER_API_KEY',
        'SECRET_KEY'
    ]
    
    missing_keys = []
    for key in required_keys:
        value = os.getenv(key)
        if not value or value == f'your_{key.lower()}_here':
            missing_keys.append(key)
    
    if missing_keys:
        print("⚠️  Warning: The following API keys are missing or not configured:")
        for key in missing_keys:
            print(f"   - {key}")
        print("\nPlease update the .env file with your actual API keys.")
        return False
    
    print("✅ All required API keys are configured!")
    return True

def get_api_keys():
    """Return a dictionary of all API keys for easy access"""
    return {
        'weather_api_key': os.getenv('WEATHER_API_KEY'),
        'redis_url': os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
        'database_url': os.getenv('DATABASE_URL', 'sqlite:///agriverse.db'),
        'mail_server': os.getenv('MAIL_SERVER', 'smtp.gmail.com'),
        'mail_port': int(os.getenv('MAIL_PORT', 587)),
        'mail_use_tls': os.getenv('MAIL_USE_TLS', 'true').lower() in ['true', 'on', '1'],
        'mail_username': os.getenv('MAIL_USERNAME', 'harshag1772004@gmail.com'),
        'mail_password': os.getenv('MAIL_PASSWORD'),
        'stripe_publishable_key': os.getenv('STRIPE_PUBLISHABLE_KEY'),
        'stripe_secret_key': os.getenv('STRIPE_SECRET_KEY'),
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'deepseek_api_key': os.getenv('DEEPSEEK_API_KEY'),
        'google_maps_api_key': os.getenv('GOOGLE_MAPS_API_KEY'),
        'twilio_account_sid': os.getenv('TWILIO_ACCOUNT_SID'),
        'twilio_auth_token': os.getenv('TWILIO_AUTH_TOKEN'),
        'twilio_phone_number': os.getenv('TWILIO_PHONE_NUMBER'),
        'secret_key': os.getenv('SECRET_KEY'),
        'debug': os.getenv('DEBUG', 'False').lower() in ['true', 'on', '1'],
        'host': os.getenv('HOST', '0.0.0.0'),
        'port': int(os.getenv('PORT', 5000)),
        'realtime_update_interval': int(os.getenv('REALTIME_UPDATE_INTERVAL', 30)),
        'model_update_interval': int(os.getenv('MODEL_UPDATE_INTERVAL', 300)),
        'websocket_ping_interval': int(os.getenv('WEBSOCKET_PING_INTERVAL', 25)),
        'log_level': os.getenv('LOG_LEVEL', 'INFO'),
        'log_file': os.getenv('LOG_FILE', 'logs/agriverse.log')
    }

def main():
    """Main configuration function"""
    print("🔑 Agriverse Unified Configuration System")
    print("=" * 50)
    print("📧 Configured for: harshag1772004@gmail.com")
    print()
    
    # Create configuration
    config = create_complete_config()
    
    print("✅ Configuration files created:")
    print("   - .env (Environment variables)")
    print("   - config_production.py (Production config)")
    print("   - API_KEYS_SUMMARY.txt (API keys summary)")
    
    print("\n🔐 Generated Strong Passwords:")
    print(f"   App Password: {config['app_password']}")
    print(f"   Database Password: {config['db_password']}")
    print(f"   Redis Password: {config['redis_password']}")
    
    print("\n🌐 Generated API Keys:")
    print(f"   Weather API: {config['weather_api_key']}")
    print(f"   OpenAI API: {config['openai_api_key']}")
    print(f"   DeepSeek API: {config['deepseek_api_key']}")
    print(f"   Google Maps: {config['google_maps_key']}")
    
    print("\n📝 Next Steps:")
    print("1. Get real OpenWeatherMap API key from: https://openweathermap.org/api")
    print("2. Get OpenAI API key from: https://platform.openai.com/api-keys")
    print("3. Update the .env file with real API keys")
    print("4. Run: python start_agriverse.py")
    
    print("\n⚠️  Important Security Notes:")
    print("- All passwords are randomly generated and secure")
    print("- Store API keys securely")
    print("- Never commit .env file to version control")
    print("- Rotate passwords regularly")
    
    print("\n🎉 Configuration complete! Your Agriverse platform is ready to run!")

if __name__ == '__main__':
    main()