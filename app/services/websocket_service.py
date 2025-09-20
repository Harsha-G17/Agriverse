import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Set
import threading
import queue
from flask import Flask
from flask_socketio import SocketIO, emit, join_room, leave_room
import redis
from .realtime_data_service import realtime_service
from ..ml_models.realtime_models import realtime_crop_model, realtime_yield_model, realtime_soil_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSocketService:
    """
    WebSocket service for real-time data streaming and ML predictions
    """
    
    def __init__(self, app: Flask = None, socketio: SocketIO = None):
        self.app = app
        self.socketio = socketio
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.connected_clients: Set[str] = set()
        self.room_subscriptions: Dict[str, Set[str]] = {
            'soil_data': set(),
            'weather_data': set(),
            'market_data': set(),
            'crop_predictions': set(),
            'yield_forecasts': set(),
            'soil_analysis': set()
        }
        self.running = False
        self.data_thread = None
        
        if app and socketio:
            self.init_app(app, socketio)
    
    def init_app(self, app: Flask, socketio: SocketIO):
        """Initialize the WebSocket service with Flask app and SocketIO"""
        self.app = app
        self.socketio = socketio
        self._register_handlers()
    
    def _register_handlers(self):
        """Register WebSocket event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            client_id = request.sid
            self.connected_clients.add(client_id)
            logger.info(f"Client connected: {client_id}")
            
            # Send initial data
            self._send_initial_data(client_id)
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            client_id = request.sid
            self.connected_clients.discard(client_id)
            
            # Remove from all rooms
            for room in self.room_subscriptions.values():
                room.discard(client_id)
            
            logger.info(f"Client disconnected: {client_id}")
        
        @self.socketio.on('subscribe')
        def handle_subscribe(data):
            """Handle subscription to data streams"""
            client_id = request.sid
            room = data.get('room')
            
            if room in self.room_subscriptions:
                join_room(room)
                self.room_subscriptions[room].add(client_id)
                logger.info(f"Client {client_id} subscribed to {room}")
                
                # Send current data for the room
                self._send_room_data(room, client_id)
            else:
                emit('error', {'message': f'Invalid room: {room}'})
        
        @self.socketio.on('unsubscribe')
        def handle_unsubscribe(data):
            """Handle unsubscription from data streams"""
            client_id = request.sid
            room = data.get('room')
            
            if room in self.room_subscriptions:
                leave_room(room)
                self.room_subscriptions[room].discard(client_id)
                logger.info(f"Client {client_id} unsubscribed from {room}")
        
        @self.socketio.on('request_prediction')
        def handle_prediction_request(data):
            """Handle real-time prediction requests"""
            client_id = request.sid
            prediction_type = data.get('type')
            
            try:
                if prediction_type == 'crop':
                    result = self._handle_crop_prediction(data)
                elif prediction_type == 'yield':
                    result = self._handle_yield_prediction(data)
                elif prediction_type == 'soil':
                    result = self._handle_soil_analysis(data)
                else:
                    result = {'error': 'Invalid prediction type'}
                
                emit('prediction_result', result)
                
            except Exception as e:
                logger.error(f"Error handling prediction request: {str(e)}")
                emit('prediction_error', {'error': str(e)})
        
        @self.socketio.on('get_latest_data')
        def handle_latest_data_request():
            """Handle request for latest data"""
            client_id = request.sid
            latest_data = realtime_service.get_latest_data()
            emit('latest_data', latest_data)
    
    def _send_initial_data(self, client_id: str):
        """Send initial data to newly connected client"""
        try:
            # Get latest data
            latest_data = realtime_service.get_latest_data()
            
            # Send each data type
            if latest_data['soil_data']:
                emit('soil_data_update', {
                    'data': latest_data['soil_data'],
                    'timestamp': latest_data['last_updated']
                }, room=client_id)
            
            if latest_data['weather_data']:
                emit('weather_data_update', {
                    'data': latest_data['weather_data'],
                    'timestamp': latest_data['last_updated']
                }, room=client_id)
            
            if latest_data['market_data']:
                emit('market_data_update', {
                    'data': latest_data['market_data'],
                    'timestamp': latest_data['last_updated']
                }, room=client_id)
                
        except Exception as e:
            logger.error(f"Error sending initial data: {str(e)}")
    
    def _send_room_data(self, room: str, client_id: str):
        """Send current data for a specific room"""
        try:
            latest_data = realtime_service.get_latest_data()
            
            if room == 'soil_data' and latest_data['soil_data']:
                emit('soil_data_update', {
                    'data': latest_data['soil_data'],
                    'timestamp': latest_data['last_updated']
                }, room=client_id)
            
            elif room == 'weather_data' and latest_data['weather_data']:
                emit('weather_data_update', {
                    'data': latest_data['weather_data'],
                    'timestamp': latest_data['last_updated']
                }, room=client_id)
            
            elif room == 'market_data' and latest_data['market_data']:
                emit('market_data_update', {
                    'data': latest_data['market_data'],
                    'timestamp': latest_data['last_updated']
                }, room=client_id)
                
        except Exception as e:
            logger.error(f"Error sending room data: {str(e)}")
    
    def _handle_crop_prediction(self, data: Dict) -> Dict:
        """Handle crop prediction request"""
        try:
            # Prepare soil data
            soil_data = {
                'ph_level': data.get('ph_level', 6.5),
                'nitrogen_level': data.get('nitrogen_level', 60),
                'phosphorus_level': data.get('phosphorus_level', 50),
                'potassium_level': data.get('potassium_level', 55),
                'organic_matter': data.get('organic_matter', 3),
                'moisture_content': data.get('moisture_content', 65),
                'temperature': data.get('temperature', 25),
                'humidity': data.get('humidity', 65),
                'rainfall': data.get('rainfall', 0)
            }
            
            # Get prediction
            prediction = realtime_crop_model.predict_realtime(soil_data)
            
            return {
                'type': 'crop_prediction',
                'prediction': prediction,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in crop prediction: {str(e)}")
            return {'error': str(e)}
    
    def _handle_yield_prediction(self, data: Dict) -> Dict:
        """Handle yield prediction request"""
        try:
            # Prepare data
            prediction_data = {
                'ph_level': data.get('ph_level', 6.5),
                'nitrogen_level': data.get('nitrogen_level', 60),
                'phosphorus_level': data.get('phosphorus_level', 50),
                'potassium_level': data.get('potassium_level', 55),
                'organic_matter': data.get('organic_matter', 3),
                'moisture_content': data.get('moisture_content', 65),
                'temperature': data.get('temperature', 25),
                'humidity': data.get('humidity', 65),
                'rainfall': data.get('rainfall', 0),
                'crop_type': data.get('crop_type', 'Rice')
            }
            
            # Get prediction
            prediction = realtime_yield_model.predict_realtime(prediction_data)
            
            return {
                'type': 'yield_prediction',
                'prediction': prediction,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in yield prediction: {str(e)}")
            return {'error': str(e)}
    
    def _handle_soil_analysis(self, data: Dict) -> Dict:
        """Handle soil analysis request"""
        try:
            # Prepare soil data
            soil_data = {
                'ph_level': data.get('ph_level', 6.5),
                'nitrogen_level': data.get('nitrogen_level', 60),
                'phosphorus_level': data.get('phosphorus_level', 50),
                'potassium_level': data.get('potassium_level', 55),
                'organic_matter': data.get('organic_matter', 3),
                'moisture_content': data.get('moisture_content', 65)
            }
            
            # Get analysis
            analysis = realtime_soil_model.predict_realtime(soil_data)
            
            return {
                'type': 'soil_analysis',
                'analysis': analysis,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in soil analysis: {str(e)}")
            return {'error': str(e)}
    
    def start_data_streaming(self):
        """Start the data streaming service"""
        if not self.running:
            self.running = True
            self.data_thread = threading.Thread(target=self._data_streaming_loop, daemon=True)
            self.data_thread.start()
            logger.info("WebSocket data streaming started")
    
    def stop_data_streaming(self):
        """Stop the data streaming service"""
        self.running = False
        if self.data_thread:
            self.data_thread.join(timeout=5)
        logger.info("WebSocket data streaming stopped")
    
    def _data_streaming_loop(self):
        """Main data streaming loop"""
        while self.running:
            try:
                # Get data from the real-time service queue
                data_queue = realtime_service.get_data_queue()
                
                if not data_queue.empty():
                    data = data_queue.get_nowait()
                    self._broadcast_data(data)
                
                # Sleep briefly to prevent excessive CPU usage
                threading.Event().wait(1)
                
            except Exception as e:
                logger.error(f"Error in data streaming loop: {str(e)}")
                threading.Event().wait(5)
    
    def _broadcast_data(self, data: Dict):
        """Broadcast data to subscribed clients"""
        try:
            data_type = data.get('type')
            timestamp = data.get('timestamp')
            
            if data_type == 'realtime_update':
                # Broadcast soil data
                if data.get('soil_data') and self.room_subscriptions['soil_data']:
                    self.socketio.emit('soil_data_update', {
                        'data': data['soil_data'],
                        'timestamp': timestamp
                    }, room='soil_data')
                
                # Broadcast weather data
                if data.get('weather_data') and self.room_subscriptions['weather_data']:
                    self.socketio.emit('weather_data_update', {
                        'data': data['weather_data'],
                        'timestamp': timestamp
                    }, room='weather_data')
                
                # Broadcast market data
                if data.get('market_data') and self.room_subscriptions['market_data']:
                    self.socketio.emit('market_data_update', {
                        'data': data['market_data'],
                        'timestamp': timestamp
                    }, room='market_data')
                
                logger.info(f"Broadcasted real-time data to {len(self.connected_clients)} clients")
                
        except Exception as e:
            logger.error(f"Error broadcasting data: {str(e)}")
    
    def broadcast_prediction(self, prediction_type: str, prediction_data: Dict):
        """Broadcast prediction results to subscribed clients"""
        try:
            room = f"{prediction_type}_predictions"
            if room in self.room_subscriptions and self.room_subscriptions[room]:
                self.socketio.emit(f'{prediction_type}_prediction_update', {
                    'data': prediction_data,
                    'timestamp': datetime.now().isoformat()
                }, room=room)
                
        except Exception as e:
            logger.error(f"Error broadcasting prediction: {str(e)}")
    
    def get_connection_stats(self) -> Dict:
        """Get connection statistics"""
        return {
            'connected_clients': len(self.connected_clients),
            'room_subscriptions': {room: len(clients) for room, clients in self.room_subscriptions.items()},
            'is_running': self.running
        }

# Global WebSocket service instance
websocket_service = WebSocketService()
