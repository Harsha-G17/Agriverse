import asyncio
import aiohttp
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
import redis
from flask import current_app
import threading
import queue
import schedule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WeatherData:
    """Weather data structure for real-time updates"""
    temperature: float
    humidity: float
    rainfall: float
    wind_speed: float
    pressure: float
    uv_index: float
    timestamp: datetime
    location: str

@dataclass
class SoilSensorData:
    """Soil sensor data structure for real-time updates"""
    ph_level: float
    nitrogen_level: float
    phosphorus_level: float
    potassium_level: float
    organic_matter: float
    moisture_content: float
    temperature: float
    timestamp: datetime
    sensor_id: str
    location: str

@dataclass
class MarketData:
    """Market data structure for real-time price updates"""
    crop_name: str
    current_price: float
    price_change: float
    volume: float
    timestamp: datetime
    market: str

class RealTimeDataService:
    """
    Real-time data service for fetching and processing agricultural data
    from various sources including weather APIs, IoT sensors, and market data
    """
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client or redis.Redis(host='localhost', port=6379, db=0)
        self.data_queue = queue.Queue()
        self.running = False
        self.threads = []
        
        # API configurations
        self.weather_api_key = None  # Set later from current_app
        self.weather_base_url = "http://api.openweathermap.org/data/2.5"
        
        # Mock sensor data for demonstration
        self.mock_sensors = {
            'sensor_001': {'location': 'Field A', 'lat': 28.6139, 'lon': 77.2090},
            'sensor_002': {'location': 'Field B', 'lat': 28.6140, 'lon': 77.2091},
            'sensor_003': {'location': 'Field C', 'lat': 28.6141, 'lon': 77.2092}
        }
        
        # Market data sources
        self.market_sources = [
            'rice', 'wheat', 'corn', 'soybeans', 'cotton', 'sugarcane'
        ]
    
    async def fetch_weather_data(self, lat: float, lon: float, location: str) -> WeatherData:
        """Fetch real-time weather data from OpenWeatherMap API"""
        try:
            if self.weather_api_key is None:
                from flask import current_app
                self.weather_api_key = current_app.config.get('WEATHER_API_KEY', 'your_openweather_api_key')
            url = f"{self.weather_base_url}/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.weather_api_key,
                'units': 'metric'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        return WeatherData(
                            temperature=data['main']['temp'],
                            humidity=data['main']['humidity'],
                            rainfall=data.get('rain', {}).get('1h', 0),
                            wind_speed=data['wind']['speed'],
                            pressure=data['main']['pressure'],
                            uv_index=data.get('uvi', 0),
                            timestamp=datetime.now(),
                            location=location
                        )
                    else:
                        logger.warning(f"Weather API error: {response.status}")
                        return self._generate_mock_weather_data(location)
        except Exception as e:
            logger.error(f"Error fetching weather data: {str(e)}")
            return self._generate_mock_weather_data(location)
    
    def _generate_mock_weather_data(self, location: str) -> WeatherData:
        """Generate mock weather data for demonstration"""
        # Simulate realistic weather patterns
        base_temp = 25 + np.random.normal(0, 5)
        base_humidity = 65 + np.random.normal(0, 15)
        
        return WeatherData(
            temperature=max(-10, min(45, base_temp)),
            humidity=max(0, min(100, base_humidity)),
            rainfall=max(0, np.random.exponential(2)),
            wind_speed=max(0, np.random.exponential(3)),
            pressure=1013 + np.random.normal(0, 20),
            uv_index=max(0, min(11, np.random.exponential(3))),
            timestamp=datetime.now(),
            location=location
        )
    
    async def fetch_soil_sensor_data(self, sensor_id: str) -> SoilSensorData:
        """Fetch real-time soil sensor data (mock implementation)"""
        try:
            # In production, this would connect to actual IoT sensors
            # For now, we'll generate realistic mock data
            
            sensor_info = self.mock_sensors.get(sensor_id, {
                'location': 'Unknown Field',
                'lat': 28.6139,
                'lon': 77.2090
            })
            
            # Generate realistic soil data with some variation
            base_ph = 6.5 + np.random.normal(0, 0.5)
            base_nitrogen = 60 + np.random.normal(0, 15)
            base_phosphorus = 50 + np.random.normal(0, 12)
            base_potassium = 55 + np.random.normal(0, 10)
            base_organic = 3 + np.random.normal(0, 1)
            base_moisture = 65 + np.random.normal(0, 15)
            base_temp = 22 + np.random.normal(0, 3)
            
            return SoilSensorData(
                ph_level=max(4.0, min(9.0, base_ph)),
                nitrogen_level=max(0, min(100, base_nitrogen)),
                phosphorus_level=max(0, min(100, base_phosphorus)),
                potassium_level=max(0, min(100, base_potassium)),
                organic_matter=max(0, min(10, base_organic)),
                moisture_content=max(0, min(100, base_moisture)),
                temperature=max(5, min(40, base_temp)),
                timestamp=datetime.now(),
                sensor_id=sensor_id,
                location=sensor_info['location']
            )
        except Exception as e:
            logger.error(f"Error fetching soil sensor data: {str(e)}")
            raise
    
    async def fetch_market_data(self, crop_name: str) -> MarketData:
        """Fetch real-time market data for crops"""
        try:
            # In production, this would connect to actual market data APIs
            # For now, we'll generate realistic mock data
            
            # Base prices for different crops (per kg)
            base_prices = {
                'rice': 45.0,
                'wheat': 25.0,
                'corn': 20.0,
                'soybeans': 60.0,
                'cotton': 120.0,
                'sugarcane': 3.5
            }
            
            base_price = base_prices.get(crop_name.lower(), 30.0)
            
            # Simulate price fluctuations
            price_change = np.random.normal(0, 0.05)  # 5% standard deviation
            current_price = base_price * (1 + price_change)
            
            return MarketData(
                crop_name=crop_name,
                current_price=max(0.1, current_price),
                price_change=price_change * 100,  # Percentage change
                volume=np.random.exponential(1000),
                timestamp=datetime.now(),
                market='National'
            )
        except Exception as e:
            logger.error(f"Error fetching market data: {str(e)}")
            raise
    
    async def fetch_all_sensor_data(self) -> List[SoilSensorData]:
        """Fetch data from all soil sensors"""
        tasks = []
        for sensor_id in self.mock_sensors.keys():
            tasks.append(self.fetch_soil_sensor_data(sensor_id))
        
        return await asyncio.gather(*tasks)
    
    async def fetch_all_weather_data(self) -> List[WeatherData]:
        """Fetch weather data for all sensor locations"""
        tasks = []
        for sensor_id, info in self.mock_sensors.items():
            tasks.append(self.fetch_weather_data(info['lat'], info['lon'], info['location']))
        
        return await asyncio.gather(*tasks)
    
    async def fetch_all_market_data(self) -> List[MarketData]:
        """Fetch market data for all crops"""
        tasks = []
        for crop in self.market_sources:
            tasks.append(self.fetch_market_data(crop))
        
        return await asyncio.gather(*tasks)
    
    def cache_data(self, key: str, data: Any, ttl: int = 300):
        """Cache data in Redis with TTL"""
        try:
            serialized_data = json.dumps(data, default=str)
            self.redis_client.setex(key, ttl, serialized_data)
        except Exception as e:
            logger.error(f"Error caching data: {str(e)}")
    
    def get_cached_data(self, key: str) -> Optional[Any]:
        """Retrieve cached data from Redis"""
        try:
            cached_data = self.redis_client.get(key)
            if cached_data:
                return json.loads(cached_data)
            return None
        except Exception as e:
            logger.error(f"Error retrieving cached data: {str(e)}")
            return None
    
    async def process_realtime_data(self):
        """Main data processing loop for real-time updates"""
        while self.running:
            try:
                # Fetch all data sources
                soil_data = await self.fetch_all_sensor_data()
                weather_data = await self.fetch_all_weather_data()
                market_data = await self.fetch_all_market_data()
                
                # Cache the data
                self.cache_data('realtime_soil_data', [self._serialize_soil_data(d) for d in soil_data], 60)
                self.cache_data('realtime_weather_data', [self._serialize_weather_data(d) for d in weather_data], 300)
                self.cache_data('realtime_market_data', [self._serialize_market_data(d) for d in market_data], 1800)
                
                # Put data in queue for real-time processing
                self.data_queue.put({
                    'type': 'realtime_update',
                    'timestamp': datetime.now().isoformat(),
                    'soil_data': [self._serialize_soil_data(d) for d in soil_data],
                    'weather_data': [self._serialize_weather_data(d) for d in weather_data],
                    'market_data': [self._serialize_market_data(d) for d in market_data]
                })
                
                logger.info(f"Processed real-time data at {datetime.now()}")
                
                # Wait before next update
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in real-time data processing: {str(e)}")
                await asyncio.sleep(60)  # Wait longer on error
    
    def _serialize_soil_data(self, data: SoilSensorData) -> Dict:
        """Serialize soil data for JSON storage"""
        return {
            'ph_level': data.ph_level,
            'nitrogen_level': data.nitrogen_level,
            'phosphorus_level': data.phosphorus_level,
            'potassium_level': data.potassium_level,
            'organic_matter': data.organic_matter,
            'moisture_content': data.moisture_content,
            'temperature': data.temperature,
            'timestamp': data.timestamp.isoformat(),
            'sensor_id': data.sensor_id,
            'location': data.location
        }
    
    def _serialize_weather_data(self, data: WeatherData) -> Dict:
        """Serialize weather data for JSON storage"""
        return {
            'temperature': data.temperature,
            'humidity': data.humidity,
            'rainfall': data.rainfall,
            'wind_speed': data.wind_speed,
            'pressure': data.pressure,
            'uv_index': data.uv_index,
            'timestamp': data.timestamp.isoformat(),
            'location': data.location
        }
    
    def _serialize_market_data(self, data: MarketData) -> Dict:
        """Serialize market data for JSON storage"""
        return {
            'crop_name': data.crop_name,
            'current_price': data.current_price,
            'price_change': data.price_change,
            'volume': data.volume,
            'timestamp': data.timestamp.isoformat(),
            'market': data.market
        }
    
    def start_realtime_service(self):
        """Start the real-time data service"""
        if not self.running:
            self.running = True
            
            # Start the main data processing loop in a separate thread
            def run_async_loop():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.process_realtime_data())
            
            thread = threading.Thread(target=run_async_loop, daemon=True)
            thread.start()
            self.threads.append(thread)
            
            logger.info("Real-time data service started")
    
    def stop_realtime_service(self):
        """Stop the real-time data service"""
        self.running = False
        for thread in self.threads:
            thread.join(timeout=5)
        self.threads.clear()
        logger.info("Real-time data service stopped")
    
    def get_latest_data(self) -> Dict:
        """Get the latest cached data"""
        return {
            'soil_data': self.get_cached_data('realtime_soil_data') or [],
            'weather_data': self.get_cached_data('realtime_weather_data') or [],
            'market_data': self.get_cached_data('realtime_market_data') or [],
            'last_updated': datetime.now().isoformat()
        }
    
    def get_data_queue(self) -> queue.Queue:
        """Get the data queue for real-time streaming"""
        return self.data_queue

# Global instance
realtime_service = RealTimeDataService()
