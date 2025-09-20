#!/usr/bin/env python3
"""
Agriverse - AI-Powered Agricultural Intelligence Platform
Unified Startup Script with Modern Frontend and Real-Time Capabilities
Configured for:
"""

import os
import sys
import time
import signal
from datetime import datetime

def print_banner():
    """Print startup banner"""
    print("🌱" + "=" * 68 + "🌱")
    print("🌱 AGRIVERSE - AI-Powered Agricultural Intelligence Platform 🌱")
    print("🌱" + "=" * 68 + "🌱")
    print("✨ Modern Frontend + Real-Time ML + Premium Design")
    print("📧 Configured for: harshag1772004@gmail.com")
    print("🕐 Started at:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print()

def setup_environment():
    """Setup environment variables"""
    print("🔧 Setting up environment...")
    
    # Set environment variables
    os.environ['SECRET_KEY'] = 'G7!bPz@3Lm#9Kx8Qw2E5Rt6Yu1Io9Pz4As7Df8Gh3Jk6Lm9Nq2Wx5Zc8Vb1Mn4'
    os.environ['WEATHER_API_KEY'] = 'demo_weather_key_for_testing'
    os.environ['REDIS_URL'] = 'redis://localhost:6379/0'
    os.environ['DEBUG'] = 'True'
    os.environ['MAIL_USERNAME'] = 'harshag1772004@gmail.com'
    os.environ['MAIL_PASSWORD'] = 'Oupoc9$Xb6W9YroVdvwbrVwG'
    
    print("✅ Environment configured")

def check_dependencies():
    """Check if all dependencies are available"""
    print("🔍 Checking dependencies...")
    
    try:
        import flask
        import flask_socketio
        import redis
        import numpy
        import pandas
        
        print("✅ All core dependencies available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Run: pip install -r requirements.txt")
        return False

def check_redis():
    """Check Redis connection"""
    print("🔍 Checking Redis connection...")
    
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        r.ping()
        print("✅ Redis connection successful")
        return True
    except Exception as e:
        print(f"⚠️  Redis not available: {e}")
        print("   The app will work with limited real-time features")
        return False

def start_services():
    """Start all Agriverse services"""
    print("🚀 Starting Agriverse services...")
    
    try:
        from app import create_app, socketio
        from app.services.realtime_data_service import realtime_service
        from app.services.websocket_service import websocket_service
        
        # Create Flask app
        print("   🔄 Initializing Flask application...")
        app = create_app()

        with app.app_context():
            # Start real-time data service
            print("   📡 Starting real-time data service...")
            realtime_service.start_realtime_service()

            # Start WebSocket data streaming
            print("   🌐 Starting WebSocket data streaming...")
            websocket_service.start_data_streaming()

            print("✅ All services started successfully!")
            print()
            # Display access information
            print("🌐 Access your platform:")
            print("   🏠 Homepage: http://localhost:5000")
            print("   📊 Dashboard: http://localhost:5000/realtime-dashboard")
            print("   🔌 API: http://localhost:5000/api/realtime/data/latest")
            print()
            print("🎯 Features available:")
            print("   ✨ Modern, responsive frontend design")
            print("   🤖 AI-powered crop predictions")
            print("   📈 Real-time analytics and monitoring")
            print("   🌤️ Live weather data integration")
            print("   📱 Mobile-optimized interface")
            print("   🔄 WebSocket real-time updates")
            print()
            print("Press Ctrl+C to stop all services")
            print("=" * 70)
            # Run the application
            socketio.run(app, debug=True, host='0.0.0.0', port=5000)
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping all services...")
        try:
            realtime_service.stop_realtime_service()
            websocket_service.stop_data_streaming()
        except:
            pass
        print("✅ All services stopped successfully!")
        print("👋 Thank you for using Agriverse!")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        print("💡 Make sure all dependencies are installed: pip install -r requirements.txt")
        print("💡 If Redis is not available, the app will work with limited real-time features")

def main():
    """Main startup function"""
    print_banner()
    
    # Setup environment
    setup_environment()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check Redis (optional)
    redis_available = check_redis()
    
    # Start services
    start_services()

if __name__ == '__main__':
    main()
