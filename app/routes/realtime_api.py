from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
from functools import wraps
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from ..services.realtime_data_service import realtime_service
from ..ml_models.realtime_models import realtime_crop_model, realtime_yield_model, realtime_soil_model
from ..services.websocket_service import websocket_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

realtime_api_bp = Blueprint('realtime_api', __name__, url_prefix='/api/realtime')

def farmer_or_admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return jsonify({'error': 'Authentication required'}), 401
        if current_user.role not in ['farmer', 'admin']:
            return jsonify({'error': 'Farmer or admin privileges required'}), 403
        return f(*args, **kwargs)
    return decorated_function

@realtime_api_bp.route('/data/latest', methods=['GET'])
@login_required
def get_latest_data():
    """
    Get the latest real-time data from all sources
    """
    try:
        latest_data = realtime_service.get_latest_data()
        
        return jsonify({
            'success': True,
            'data': latest_data,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting latest data: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@realtime_api_bp.route('/data/soil', methods=['GET'])
@login_required
def get_soil_data():
    """
    Get latest soil sensor data
    """
    try:
        latest_data = realtime_service.get_latest_data()
        soil_data = latest_data.get('soil_data', [])
        
        return jsonify({
            'success': True,
            'soil_data': soil_data,
            'count': len(soil_data),
            'timestamp': latest_data.get('last_updated')
        })
        
    except Exception as e:
        logger.error(f"Error getting soil data: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@realtime_api_bp.route('/data/weather', methods=['GET'])
@login_required
def get_weather_data():
    """
    Get latest weather data
    """
    try:
        latest_data = realtime_service.get_latest_data()
        weather_data = latest_data.get('weather_data', [])
        
        return jsonify({
            'success': True,
            'weather_data': weather_data,
            'count': len(weather_data),
            'timestamp': latest_data.get('last_updated')
        })
        
    except Exception as e:
        logger.error(f"Error getting weather data: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@realtime_api_bp.route('/data/market', methods=['GET'])
@login_required
def get_market_data():
    """
    Get latest market data
    """
    try:
        latest_data = realtime_service.get_latest_data()
        market_data = latest_data.get('market_data', [])
        
        return jsonify({
            'success': True,
            'market_data': market_data,
            'count': len(market_data),
            'timestamp': latest_data.get('last_updated')
        })
        
    except Exception as e:
        logger.error(f"Error getting market data: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@realtime_api_bp.route('/predict/crop', methods=['POST'])
@farmer_or_admin_required
def predict_crop_realtime():
    """
    Get real-time crop prediction based on current conditions
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['ph_level', 'nitrogen_level', 'phosphorus_level', 
                          'potassium_level', 'organic_matter', 'moisture_content']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Prepare data for prediction
        prediction_data = {
            'ph_level': float(data['ph_level']),
            'nitrogen_level': float(data['nitrogen_level']),
            'phosphorus_level': float(data['phosphorus_level']),
            'potassium_level': float(data['potassium_level']),
            'organic_matter': float(data['organic_matter']),
            'moisture_content': float(data['moisture_content']),
            'temperature': float(data.get('temperature', 25)),
            'humidity': float(data.get('humidity', 65)),
            'rainfall': float(data.get('rainfall', 0))
        }
        
        # Get real-time prediction
        prediction = realtime_crop_model.predict_realtime(prediction_data)
        
        # Broadcast prediction if WebSocket is available
        if websocket_service.socketio:
            websocket_service.broadcast_prediction('crop', prediction)
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in real-time crop prediction: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@realtime_api_bp.route('/predict/yield', methods=['POST'])
@farmer_or_admin_required
def predict_yield_realtime():
    """
    Get real-time yield prediction based on current conditions
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['ph_level', 'nitrogen_level', 'phosphorus_level', 
                          'potassium_level', 'organic_matter', 'moisture_content', 'crop_type']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Prepare data for prediction
        prediction_data = {
            'ph_level': float(data['ph_level']),
            'nitrogen_level': float(data['nitrogen_level']),
            'phosphorus_level': float(data['phosphorus_level']),
            'potassium_level': float(data['potassium_level']),
            'organic_matter': float(data['organic_matter']),
            'moisture_content': float(data['moisture_content']),
            'temperature': float(data.get('temperature', 25)),
            'humidity': float(data.get('humidity', 65)),
            'rainfall': float(data.get('rainfall', 0)),
            'crop_type': data['crop_type']
        }
        
        # Get real-time prediction
        prediction = realtime_yield_model.predict_realtime(prediction_data)
        
        # Broadcast prediction if WebSocket is available
        if websocket_service.socketio:
            websocket_service.broadcast_prediction('yield', prediction)
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in real-time yield prediction: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@realtime_api_bp.route('/analyze/soil', methods=['POST'])
@farmer_or_admin_required
def analyze_soil_realtime():
    """
    Get real-time soil health analysis
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['ph_level', 'nitrogen_level', 'phosphorus_level', 
                          'potassium_level', 'organic_matter', 'moisture_content']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Prepare data for analysis
        analysis_data = {
            'ph_level': float(data['ph_level']),
            'nitrogen_level': float(data['nitrogen_level']),
            'phosphorus_level': float(data['phosphorus_level']),
            'potassium_level': float(data['potassium_level']),
            'organic_matter': float(data['organic_matter']),
            'moisture_content': float(data['moisture_content'])
        }
        
        # Get real-time analysis
        analysis = realtime_soil_model.predict_realtime(analysis_data)
        
        # Broadcast analysis if WebSocket is available
        if websocket_service.socketio:
            websocket_service.broadcast_prediction('soil', analysis)
        
        return jsonify({
            'success': True,
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in real-time soil analysis: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@realtime_api_bp.route('/predict/batch', methods=['POST'])
@farmer_or_admin_required
def batch_predict_realtime():
    """
    Perform batch real-time predictions for multiple samples
    """
    try:
        data = request.get_json()
        
        if not data or 'samples' not in data:
            return jsonify({'error': 'No samples provided'}), 400
        
        samples = data['samples']
        if not isinstance(samples, list) or len(samples) == 0:
            return jsonify({'error': 'Samples must be a non-empty list'}), 400
        
        results = []
        
        for i, sample in enumerate(samples):
            try:
                # Validate sample
                required_fields = ['ph_level', 'nitrogen_level', 'phosphorus_level', 
                                  'potassium_level', 'organic_matter', 'moisture_content']
                
                missing_fields = [field for field in required_fields if field not in sample]
                if missing_fields:
                    results.append({
                        'index': i,
                        'error': f'Missing required fields: {missing_fields}'
                    })
                    continue
                
                # Prepare data
                prediction_data = {
                    'ph_level': float(sample['ph_level']),
                    'nitrogen_level': float(sample['nitrogen_level']),
                    'phosphorus_level': float(sample['phosphorus_level']),
                    'potassium_level': float(sample['potassium_level']),
                    'organic_matter': float(sample['organic_matter']),
                    'moisture_content': float(sample['moisture_content']),
                    'temperature': float(sample.get('temperature', 25)),
                    'humidity': float(sample.get('humidity', 65)),
                    'rainfall': float(sample.get('rainfall', 0))
                }
                
                # Get predictions
                crop_prediction = realtime_crop_model.predict_realtime(prediction_data)
                soil_analysis = realtime_soil_model.predict_realtime(prediction_data)
                
                # Get yield prediction if crop type is provided
                yield_prediction = None
                if 'crop_type' in sample:
                    yield_data = {**prediction_data, 'crop_type': sample['crop_type']}
                    yield_prediction = realtime_yield_model.predict_realtime(yield_data)
                
                results.append({
                    'index': i,
                    'crop_prediction': crop_prediction,
                    'soil_analysis': soil_analysis,
                    'yield_prediction': yield_prediction
                })
                
            except Exception as e:
                results.append({
                    'index': i,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'results': results,
            'total_samples': len(samples),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in batch real-time prediction: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@realtime_api_bp.route('/sensors/status', methods=['GET'])
@login_required
def get_sensor_status():
    """
    Get status of all sensors and data sources
    """
    try:
        latest_data = realtime_service.get_latest_data()
        
        # Check data freshness
        current_time = datetime.now()
        last_updated = datetime.fromisoformat(latest_data.get('last_updated', current_time.isoformat()))
        time_diff = (current_time - last_updated).total_seconds()
        
        # Determine status based on data freshness
        if time_diff < 60:  # Less than 1 minute
            status = 'online'
        elif time_diff < 300:  # Less than 5 minutes
            status = 'warning'
        else:
            status = 'offline'
        
        sensor_status = {
            'overall_status': status,
            'last_update': latest_data.get('last_updated'),
            'time_since_update': time_diff,
            'sensors': {
                'soil_sensors': {
                    'count': len(latest_data.get('soil_data', [])),
                    'status': 'online' if len(latest_data.get('soil_data', [])) > 0 else 'offline'
                },
                'weather_stations': {
                    'count': len(latest_data.get('weather_data', [])),
                    'status': 'online' if len(latest_data.get('weather_data', [])) > 0 else 'offline'
                },
                'market_feeds': {
                    'count': len(latest_data.get('market_data', [])),
                    'status': 'online' if len(latest_data.get('market_data', [])) > 0 else 'offline'
                }
            }
        }
        
        return jsonify({
            'success': True,
            'sensor_status': sensor_status
        })
        
    except Exception as e:
        logger.error(f"Error getting sensor status: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@realtime_api_bp.route('/models/status', methods=['GET'])
@login_required
def get_models_status():
    """
    Get status of real-time ML models
    """
    try:
        models_status = {
            'crop_prediction': {
                'trained': realtime_crop_model.is_trained,
                'last_update': realtime_crop_model.last_update.isoformat() if realtime_crop_model.last_update else None,
                'data_points': len(realtime_crop_model.data_buffer)
            },
            'yield_forecasting': {
                'trained': realtime_yield_model.is_trained,
                'last_update': realtime_yield_model.last_update.isoformat() if realtime_yield_model.last_update else None,
                'data_points': len(realtime_yield_model.data_buffer)
            },
            'soil_analysis': {
                'trained': realtime_soil_model.is_trained,
                'last_update': realtime_soil_model.last_update.isoformat() if realtime_soil_model.last_update else None,
                'data_points': len(realtime_soil_model.data_buffer)
            }
        }
        
        return jsonify({
            'success': True,
            'models_status': models_status
        })
        
    except Exception as e:
        logger.error(f"Error getting models status: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@realtime_api_bp.route('/websocket/stats', methods=['GET'])
@login_required
def get_websocket_stats():
    """
    Get WebSocket connection statistics
    """
    try:
        stats = websocket_service.get_connection_stats()
        
        return jsonify({
            'success': True,
            'websocket_stats': stats
        })
        
    except Exception as e:
        logger.error(f"Error getting WebSocket stats: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@realtime_api_bp.route('/start-service', methods=['POST'])
@login_required
def start_realtime_service():
    """
    Start the real-time data service
    """
    try:
        if current_user.role != 'admin':
            return jsonify({'error': 'Admin privileges required'}), 403
        
        # Start real-time data service
        realtime_service.start_realtime_service()
        
        # Start WebSocket data streaming
        websocket_service.start_data_streaming()
        
        return jsonify({
            'success': True,
            'message': 'Real-time services started successfully'
        })
        
    except Exception as e:
        logger.error(f"Error starting real-time service: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@realtime_api_bp.route('/stop-service', methods=['POST'])
@login_required
def stop_realtime_service():
    """
    Stop the real-time data service
    """
    try:
        if current_user.role != 'admin':
            return jsonify({'error': 'Admin privileges required'}), 403
        
        # Stop real-time data service
        realtime_service.stop_realtime_service()
        
        # Stop WebSocket data streaming
        websocket_service.stop_data_streaming()
        
        return jsonify({
            'success': True,
            'message': 'Real-time services stopped successfully'
        })
        
    except Exception as e:
        logger.error(f"Error stopping real-time service: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
