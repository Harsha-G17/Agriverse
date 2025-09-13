from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
from functools import wraps
import pandas as pd
import numpy as np
from ..ml_models import CropPredictionModel, YieldForecastingModel, SoilAnalysisModel
from ..models import db, SoilReport, User
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ml_api_bp = Blueprint('ml_api', __name__, url_prefix='/api/ml')

def farmer_or_admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return jsonify({'error': 'Authentication required'}), 401
        if current_user.role not in ['farmer', 'admin']:
            return jsonify({'error': 'Farmer or admin privileges required'}), 403
        return f(*args, **kwargs)
    return decorated_function

@ml_api_bp.route('/train-crop-model', methods=['POST'])
@login_required
def train_crop_model():
    """
    Train the crop prediction model.
    """
    try:
        crop_model = CropPredictionModel()
        
        # Train model with synthetic data
        result = crop_model.train_model(use_synthetic=True)
        
        return jsonify({
            'success': True,
            'message': 'Crop prediction model trained successfully',
            'accuracy': result['accuracy'],
            'best_params': result['best_params'],
            'feature_importance': result['feature_importance']
        })
        
    except Exception as e:
        logger.error(f"Error training crop model: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@ml_api_bp.route('/train-yield-model', methods=['POST'])
@login_required
def train_yield_model():
    """
    Train the yield forecasting model.
    """
    try:
        yield_model = YieldForecastingModel()
        
        # Train model with synthetic data
        result = yield_model.train_model(use_synthetic=True)
        
        return jsonify({
            'success': True,
            'message': 'Yield forecasting model trained successfully',
            'metrics': result
        })
        
    except Exception as e:
        logger.error(f"Error training yield model: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@ml_api_bp.route('/predict-crop', methods=['POST'])
@farmer_or_admin_required
def predict_crop():
    """
    Predict crop recommendations based on soil and weather data.
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
        
        # Create soil data DataFrame
        soil_data = pd.DataFrame([{
            'ph_level': float(data['ph_level']),
            'nitrogen_level': float(data['nitrogen_level']),
            'phosphorus_level': float(data['phosphorus_level']),
            'potassium_level': float(data['potassium_level']),
            'organic_matter': float(data['organic_matter']),
            'moisture_content': float(data['moisture_content'])
        }])
        
        # Create weather data DataFrame
        weather_data = pd.DataFrame([{
            'temperature': float(data.get('temperature', 25)),
            'humidity': float(data.get('humidity', 65)),
            'rainfall': float(data.get('rainfall', 0))
        }])
        
        # Initialize and load model
        crop_model = CropPredictionModel()
        if not crop_model.load_model():
            # Train model if not available
            crop_model.train_model(use_synthetic=True)
        
        # Get predictions
        recommendations = crop_model.predict_crop(soil_data, weather_data)
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'model_accuracy': crop_model.accuracy
        })
        
    except Exception as e:
        logger.error(f"Error in crop prediction: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@ml_api_bp.route('/predict-yield', methods=['POST'])
@farmer_or_admin_required
def predict_yield():
    """
    Predict crop yield based on soil, weather, and crop type.
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
        
        # Create soil data DataFrame
        soil_data = pd.DataFrame([{
            'ph_level': float(data['ph_level']),
            'nitrogen_level': float(data['nitrogen_level']),
            'phosphorus_level': float(data['phosphorus_level']),
            'potassium_level': float(data['potassium_level']),
            'organic_matter': float(data['organic_matter']),
            'moisture_content': float(data['moisture_content'])
        }])
        
        # Create weather data DataFrame
        weather_data = pd.DataFrame([{
            'temperature': float(data.get('temperature', 25)),
            'humidity': float(data.get('humidity', 65)),
            'rainfall': float(data.get('rainfall', 0))
        }])
        
        # Initialize and load model
        yield_model = YieldForecastingModel()
        if not yield_model.load_model():
            # Train model if not available
            yield_model.train_model(use_synthetic=True)
        
        # Get yield prediction
        yield_prediction = yield_model.predict_yield(
            soil_data, weather_data, crop_type=data['crop_type']
        )
        
        return jsonify({
            'success': True,
            'yield_prediction': yield_prediction,
            'model_metrics': yield_model.metrics
        })
        
    except Exception as e:
        logger.error(f"Error in yield prediction: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@ml_api_bp.route('/analyze-soil', methods=['POST'])
@farmer_or_admin_required
def analyze_soil():
    """
    Analyze soil health and provide recommendations.
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
        
        # Create soil data DataFrame
        soil_data = pd.DataFrame([{
            'ph_level': float(data['ph_level']),
            'nitrogen_level': float(data['nitrogen_level']),
            'phosphorus_level': float(data['phosphorus_level']),
            'potassium_level': float(data['potassium_level']),
            'organic_matter': float(data['organic_matter']),
            'moisture_content': float(data['moisture_content'])
        }])
        
        # Initialize and load model
        soil_model = SoilAnalysisModel()
        if not soil_model.load_model():
            # Train model if not available
            soil_model.train_model([])
        
        # Get soil analysis
        analysis = soil_model.analyze_soil_health(soil_data)
        
        return jsonify({
            'success': True,
            'soil_analysis': analysis
        })
        
    except Exception as e:
        logger.error(f"Error in soil analysis: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@ml_api_bp.route('/batch-predict', methods=['POST'])
@farmer_or_admin_required
def batch_predict():
    """
    Perform batch predictions for multiple soil samples.
    """
    try:
        data = request.get_json()
        
        if not data or 'samples' not in data:
            return jsonify({'error': 'No samples provided'}), 400
        
        samples = data['samples']
        if not isinstance(samples, list) or len(samples) == 0:
            return jsonify({'error': 'Samples must be a non-empty list'}), 400
        
        # Initialize models
        crop_model = CropPredictionModel()
        yield_model = YieldForecastingModel()
        soil_model = SoilAnalysisModel()
        
        # Load or train models
        if not crop_model.load_model():
            crop_model.train_model(use_synthetic=True)
        if not yield_model.load_model():
            yield_model.train_model(use_synthetic=True)
        if not soil_model.load_model():
            soil_model.train_model([])
        
        results = []
        
        for i, sample in enumerate(samples):
            try:
                # Validate sample
                required_fields = ['ph_level', 'nitrogen_level', 'phosphorus_level', 
                                  'potassium_level', 'organic_matter', 'moisture_content']
                
                for field in required_fields:
                    if field not in sample:
                        results.append({
                            'index': i,
                            'error': f'Missing required field: {field}'
                        })
                        continue
                
                # Create DataFrames
                soil_data = pd.DataFrame([{
                    'ph_level': float(sample['ph_level']),
                    'nitrogen_level': float(sample['nitrogen_level']),
                    'phosphorus_level': float(sample['phosphorus_level']),
                    'potassium_level': float(sample['potassium_level']),
                    'organic_matter': float(sample['organic_matter']),
                    'moisture_content': float(sample['moisture_content'])
                }])
                
                weather_data = pd.DataFrame([{
                    'temperature': float(sample.get('temperature', 25)),
                    'humidity': float(sample.get('humidity', 65)),
                    'rainfall': float(sample.get('rainfall', 0))
                }])
                
                # Get predictions
                crop_recommendations = crop_model.predict_crop(soil_data, weather_data)
                soil_analysis = soil_model.analyze_soil_health(soil_data)
                
                # Get yield predictions for top recommendation
                yield_predictions = {}
                if crop_recommendations:
                    top_crop = crop_recommendations[0]['name']
                    yield_pred = yield_model.predict_yield(
                        soil_data, weather_data, crop_type=top_crop
                    )
                    yield_predictions[top_crop] = yield_pred
                
                results.append({
                    'index': i,
                    'crop_recommendations': crop_recommendations,
                    'soil_analysis': soil_analysis,
                    'yield_predictions': yield_predictions
                })
                
            except Exception as e:
                results.append({
                    'index': i,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'results': results,
            'total_samples': len(samples)
        })
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@ml_api_bp.route('/model-performance', methods=['GET'])
@login_required
def get_model_performance():
    """
    Get performance metrics for all ML models.
    """
    try:
        crop_model = CropPredictionModel()
        yield_model = YieldForecastingModel()
        soil_model = SoilAnalysisModel()
        
        performance = {}
        
        # Load models and get performance
        if crop_model.load_model():
            performance['crop_prediction'] = crop_model.get_model_performance()
        else:
            performance['crop_prediction'] = {'error': 'Model not trained'}
        
        if yield_model.load_model():
            performance['yield_forecasting'] = yield_model.get_model_performance()
        else:
            performance['yield_forecasting'] = {'error': 'Model not trained'}
        
        if soil_model.load_model():
            performance['soil_analysis'] = soil_model.get_model_performance()
        else:
            performance['soil_analysis'] = {'error': 'Model not trained'}
        
        return jsonify({
            'success': True,
            'performance': performance
        })
        
    except Exception as e:
        logger.error(f"Error getting model performance: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@ml_api_bp.route('/train-all-models', methods=['POST'])
@login_required
def train_all_models():
    """
    Train all ML models at once.
    """
    try:
        results = {}
        
        # Train crop prediction model
        try:
            crop_model = CropPredictionModel()
            crop_result = crop_model.train_model(use_synthetic=True)
            results['crop_prediction'] = {
                'success': True,
                'accuracy': crop_result['accuracy'],
                'message': 'Crop prediction model trained successfully'
            }
        except Exception as e:
            results['crop_prediction'] = {
                'success': False,
                'error': str(e)
            }
        
        # Train yield forecasting model
        try:
            yield_model = YieldForecastingModel()
            yield_result = yield_model.train_model(use_synthetic=True)
            results['yield_forecasting'] = {
                'success': True,
                'metrics': yield_result,
                'message': 'Yield forecasting model trained successfully'
            }
        except Exception as e:
            results['yield_forecasting'] = {
                'success': False,
                'error': str(e)
            }
        
        # Train soil analysis model
        try:
            soil_model = SoilAnalysisModel()
            soil_result = soil_model.train_model([])
            results['soil_analysis'] = {
                'success': True,
                'message': 'Soil analysis model trained successfully'
            }
        except Exception as e:
            results['soil_analysis'] = {
                'success': False,
                'error': str(e)
            }
        
        return jsonify({
            'success': True,
            'message': 'All models training completed',
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error training all models: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
