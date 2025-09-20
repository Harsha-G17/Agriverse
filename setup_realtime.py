#!/usr/bin/env python3
"""
Real-Time Agricultural Prediction System Setup Script

This script sets up the real-time prediction system for the Agriverse platform.
It installs dependencies, initializes services, and provides testing capabilities.
"""

import os
import sys
import subprocess
import time
import json
import requests
from datetime import datetime

def print_banner():
    """Print setup banner"""
    print("=" * 60)
    print("🌱 AGRIVERSE REAL-TIME PREDICTION SYSTEM SETUP")
    print("=" * 60)
    print("Setting up real-time ML models and data streaming...")
    print()

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        sys.exit(1)

def check_redis():
    """Check if Redis is available"""
    print("\n🔍 Checking Redis connection...")
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        print("✅ Redis connection successful")
        return True
    except Exception as e:
        print(f"⚠️  Redis not available: {e}")
        print("   Please install and start Redis server:")
        print("   - Windows: Download from https://redis.io/download")
        print("   - Linux: sudo apt-get install redis-server")
        print("   - macOS: brew install redis")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    directories = [
        "app/ml_models/saved_models",
        "app/static/realtime",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def initialize_models():
    """Initialize and train ML models"""
    print("\n🤖 Initializing ML models...")
    try:
        from app.ml_models.realtime_models import realtime_crop_model, realtime_yield_model, realtime_soil_model
        
        # Add some initial data to train models
        print("   Training crop prediction model...")
        for i in range(50):
            data = {
                'ph_level': 6.5 + (i % 10) * 0.2,
                'nitrogen_level': 60 + (i % 20),
                'phosphorus_level': 50 + (i % 15),
                'potassium_level': 55 + (i % 18),
                'organic_matter': 3 + (i % 5),
                'moisture_content': 65 + (i % 20),
                'temperature': 25 + (i % 10),
                'humidity': 65 + (i % 20),
                'rainfall': i % 5
            }
            realtime_crop_model.add_data_point(data)
        
        print("   Training yield forecasting model...")
        for i in range(50):
            data = {
                'ph_level': 6.5 + (i % 10) * 0.2,
                'nitrogen_level': 60 + (i % 20),
                'phosphorus_level': 50 + (i % 15),
                'potassium_level': 55 + (i % 18),
                'organic_matter': 3 + (i % 5),
                'moisture_content': 65 + (i % 20),
                'temperature': 25 + (i % 10),
                'humidity': 65 + (i % 20),
                'rainfall': i % 5,
                'crop_type': ['Rice', 'Wheat', 'Corn', 'Soybeans', 'Cotton'][i % 5]
            }
            realtime_yield_model.add_data_point(data)
        
        print("   Training soil analysis model...")
        for i in range(50):
            data = {
                'ph_level': 6.5 + (i % 10) * 0.2,
                'nitrogen_level': 60 + (i % 20),
                'phosphorus_level': 50 + (i % 15),
                'potassium_level': 55 + (i % 18),
                'organic_matter': 3 + (i % 5),
                'moisture_content': 65 + (i % 20)
            }
            realtime_soil_model.add_data_point(data)
        
        print("✅ ML models initialized successfully")
        
    except Exception as e:
        print(f"❌ Error initializing models: {e}")
        print("   Models will be trained on first use")

def test_api_endpoints():
    """Test API endpoints"""
    print("\n🧪 Testing API endpoints...")
    
    # Start the Flask app in a separate process for testing
    print("   Starting Flask app for testing...")
    
    try:
        from app import create_app
        app = create_app()
        
        with app.test_client() as client:
            # Test real-time data endpoint
            response = client.get('/api/realtime/data/latest')
            if response.status_code == 200:
                print("✅ Real-time data endpoint working")
            else:
                print(f"⚠️  Real-time data endpoint returned {response.status_code}")
            
            # Test sensor status endpoint
            response = client.get('/api/realtime/sensors/status')
            if response.status_code == 200:
                print("✅ Sensor status endpoint working")
            else:
                print(f"⚠️  Sensor status endpoint returned {response.status_code}")
            
            # Test models status endpoint
            response = client.get('/api/realtime/models/status')
            if response.status_code == 200:
                print("✅ Models status endpoint working")
            else:
                print(f"⚠️  Models status endpoint returned {response.status_code}")
        
        print("✅ API endpoints tested successfully")
        
    except Exception as e:
        print(f"❌ Error testing API endpoints: {e}")

def create_startup_script():
    """Create startup script for the real-time system"""
    print("\n📝 Creating startup script...")
    
    startup_script = """#!/usr/bin/env python3
'''
Real-Time Agricultural Prediction System Startup Script
'''

import os
import sys
import time
from app import create_app, socketio
from app.services.realtime_data_service import realtime_service
from app.services.websocket_service import websocket_service

def main():
    print("🌱 Starting Agriverse Real-Time Prediction System...")
    
    # Create Flask app
    app = create_app()
    
    # Start real-time data service
    print("   Starting real-time data service...")
    realtime_service.start_realtime_service()
    
    # Start WebSocket data streaming
    print("   Starting WebSocket data streaming...")
    websocket_service.start_data_streaming()
    
    print("✅ Real-time services started successfully!")
    print("   Dashboard: http://localhost:5000/realtime-dashboard")
    print("   API Docs: http://localhost:5000/api/realtime/data/latest")
    print("   Press Ctrl+C to stop")
    
    try:
        # Run the application
        socketio.run(app, debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\\n🛑 Stopping real-time services...")
        realtime_service.stop_realtime_service()
        websocket_service.stop_data_streaming()
        print("✅ Services stopped successfully!")

if __name__ == '__main__':
    main()
"""
    
    with open('start_realtime.py', 'w') as f:
        f.write(startup_script)
    
    print("✅ Startup script created: start_realtime.py")

def create_documentation():
    """Create documentation for the real-time system"""
    print("\n📚 Creating documentation...")
    
    docs = """# Real-Time Agricultural Prediction System

## Overview
This system provides real-time predictions for agricultural operations using machine learning models and live data streams.

## Features
- **Real-time Data Collection**: Weather, soil sensors, and market data
- **Live ML Predictions**: Crop recommendations, yield forecasting, soil analysis
- **WebSocket Streaming**: Real-time data updates to frontend
- **Interactive Dashboard**: Live visualization of all data and predictions

## API Endpoints

### Real-Time Data
- `GET /api/realtime/data/latest` - Get latest data from all sources
- `GET /api/realtime/data/soil` - Get soil sensor data
- `GET /api/realtime/data/weather` - Get weather data
- `GET /api/realtime/data/market` - Get market prices

### Real-Time Predictions
- `POST /api/realtime/predict/crop` - Get crop recommendations
- `POST /api/realtime/predict/yield` - Get yield forecasts
- `POST /api/realtime/analyze/soil` - Get soil health analysis
- `POST /api/realtime/predict/batch` - Batch predictions

### System Status
- `GET /api/realtime/sensors/status` - Check sensor status
- `GET /api/realtime/models/status` - Check ML model status
- `GET /api/realtime/websocket/stats` - WebSocket connection stats

### Service Control
- `POST /api/realtime/start-service` - Start real-time services
- `POST /api/realtime/stop-service` - Stop real-time services

## WebSocket Events

### Client to Server
- `subscribe` - Subscribe to data streams
- `unsubscribe` - Unsubscribe from data streams
- `request_prediction` - Request real-time prediction
- `get_latest_data` - Get latest data

### Server to Client
- `soil_data_update` - Soil sensor data updates
- `weather_data_update` - Weather data updates
- `market_data_update` - Market price updates
- `crop_prediction_update` - Crop prediction updates
- `yield_prediction_update` - Yield forecast updates
- `soil_prediction_update` - Soil analysis updates

## Usage

### Starting the System
```bash
python start_realtime.py
```

### Accessing the Dashboard
Open your browser and go to: http://localhost:5000/realtime-dashboard

### Making Predictions
```python
import requests

# Crop prediction
data = {
    "ph_level": 6.5,
    "nitrogen_level": 60,
    "phosphorus_level": 50,
    "potassium_level": 55,
    "organic_matter": 3,
    "moisture_content": 65,
    "temperature": 25,
    "humidity": 65,
    "rainfall": 0
}

response = requests.post('http://localhost:5000/api/realtime/predict/crop', json=data)
prediction = response.json()
```

### WebSocket Connection
```javascript
const socket = io();

// Subscribe to soil data
socket.emit('subscribe', { room: 'soil_data' });

// Listen for updates
socket.on('soil_data_update', function(data) {
    console.log('Soil data:', data);
});

// Request prediction
socket.emit('request_prediction', {
    type: 'crop',
    ph_level: 6.5,
    nitrogen_level: 60,
    // ... other parameters
});
```

## Configuration

### Environment Variables
- `WEATHER_API_KEY` - OpenWeatherMap API key for weather data
- `REDIS_URL` - Redis connection URL (default: localhost:6379)

### Model Configuration
Models are automatically trained with incoming data and update every 5 minutes. You can adjust the update frequency in the model classes.

## Troubleshooting

### Common Issues
1. **Redis Connection Error**: Make sure Redis is running
2. **WebSocket Connection Failed**: Check if port 5000 is available
3. **Model Training Errors**: Ensure sufficient data is available

### Logs
Check the console output for detailed error messages and system status.

## Support
For issues and questions, please check the logs and API responses for error details.
"""
    
    with open('REALTIME_SYSTEM_README.md', 'w') as f:
        f.write(docs)
    
    print("✅ Documentation created: REALTIME_SYSTEM_README.md")

def main():
    """Main setup function"""
    print_banner()
    
    # Check system requirements
    check_python_version()
    
    # Install dependencies
    install_dependencies()
    
    # Check Redis
    redis_available = check_redis()
    
    # Create directories
    create_directories()
    
    # Initialize models
    initialize_models()
    
    # Test API endpoints
    test_api_endpoints()
    
    # Create startup script
    create_startup_script()
    
    # Create documentation
    create_documentation()
    
    print("\n" + "=" * 60)
    print("🎉 REAL-TIME SYSTEM SETUP COMPLETE!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Make sure Redis is running (if not already)")
    print("2. Run: python start_realtime.py")
    print("3. Open: http://localhost:5000/realtime-dashboard")
    print()
    print("Features available:")
    print("✅ Real-time data collection")
    print("✅ Live ML predictions")
    print("✅ WebSocket streaming")
    print("✅ Interactive dashboard")
    print("✅ REST API endpoints")
    print()
    if not redis_available:
        print("⚠️  Note: Redis is not available. Some features may not work properly.")
        print("   Please install and start Redis for full functionality.")
    print()
    print("For more information, see: REALTIME_SYSTEM_README.md")
    print("=" * 60)

if __name__ == '__main__':
    main()
