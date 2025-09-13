from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
from flask_login import login_required, current_user
from functools import wraps
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from ..ml_models import CropPredictionModel, YieldForecastingModel, SoilAnalysisModel
from ..models import db, SoilReport, User
from ..cache_config import CacheManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ml_admin_bp = Blueprint('ml_admin', __name__, url_prefix='/admin/ml')

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'admin':
            flash('Access denied. Admin privileges required.', 'error')
            return redirect(url_for('main.index'))
        return f(*args, **kwargs)
    return decorated_function

@ml_admin_bp.route('/dashboard')
@login_required
@admin_required
def ml_dashboard():
    """
    ML models management dashboard.
    """
    try:
        # Initialize models
        crop_model = CropPredictionModel()
        yield_model = YieldForecastingModel()
        soil_model = SoilAnalysisModel()
        
        # Check model status
        models_status = {
            'crop_prediction': {
                'trained': crop_model.load_model(),
                'accuracy': crop_model.accuracy if crop_model.is_trained else None,
                'last_trained': None
            },
            'yield_forecasting': {
                'trained': yield_model.load_model(),
                'metrics': yield_model.metrics if yield_model.is_trained else None,
                'last_trained': None
            },
            'soil_analysis': {
                'trained': soil_model.load_model(),
                'last_trained': None
            }
        }
        
        # Get model files info
        model_path = 'app/ml_models/saved_models'
        model_files = []
        if os.path.exists(model_path):
            for file in os.listdir(model_path):
                if file.endswith('.joblib'):
                    file_path = os.path.join(model_path, file)
                    stat = os.stat(file_path)
                    model_files.append({
                        'name': file,
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime),
                        'type': file.split('_')[0]
                    })
        
        # Get cache statistics
        cache_stats = {}
        try:
            from flask import current_app
            cache_manager = getattr(current_app, 'cache_manager', None)
            if cache_manager:
                cache_stats = cache_manager.get_cache_stats()
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
        
        # Get recent soil reports for training data
        recent_reports = SoilReport.query.order_by(SoilReport.report_date.desc()).limit(10).all()
        
        return render_template('admin/ml_dashboard.html',
                             models_status=models_status,
                             model_files=model_files,
                             cache_stats=cache_stats,
                             recent_reports=recent_reports)
        
    except Exception as e:
        logger.error(f"Error in ML dashboard: {str(e)}")
        flash('An error occurred while loading the ML dashboard.', 'error')
        return redirect(url_for('admin.dashboard'))

@ml_admin_bp.route('/train-crop-model', methods=['POST'])
@login_required
@admin_required
def train_crop_model():
    """
    Train the crop prediction model.
    """
    try:
        crop_model = CropPredictionModel()
        
        # Get training parameters from request
        use_real_data = request.form.get('use_real_data', 'false').lower() == 'true'
        num_samples = int(request.form.get('num_samples', 10000))
        
        if use_real_data:
            # Use real soil reports for training
            soil_reports = SoilReport.query.all()
            if not soil_reports:
                return jsonify({
                    'success': False,
                    'error': 'No soil reports available for training'
                }), 400
            
            # Convert to training data
            X, y = convert_soil_reports_to_training_data(soil_reports)
            result = crop_model.train_model(X, y, use_synthetic=False)
        else:
            # Use synthetic data
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

@ml_admin_bp.route('/train-yield-model', methods=['POST'])
@login_required
@admin_required
def train_yield_model():
    """
    Train the yield forecasting model.
    """
    try:
        yield_model = YieldForecastingModel()
        
        # Get training parameters
        use_real_data = request.form.get('use_real_data', 'false').lower() == 'true'
        num_samples = int(request.form.get('num_samples', 5000))
        
        if use_real_data:
            # Use real data if available
            soil_reports = SoilReport.query.all()
            if not soil_reports:
                return jsonify({
                    'success': False,
                    'error': 'No soil reports available for training'
                }), 400
            
            # Convert to training data
            X, y = convert_soil_reports_to_yield_training_data(soil_reports)
            result = yield_model.train_model(X, y, use_synthetic=False)
        else:
            # Use synthetic data
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

@ml_admin_bp.route('/train-soil-model', methods=['POST'])
@login_required
@admin_required
def train_soil_model():
    """
    Train the soil analysis model.
    """
    try:
        soil_model = SoilAnalysisModel()
        
        # Get soil reports for training
        soil_reports = SoilReport.query.all()
        
        if not soil_reports:
            return jsonify({
                'success': False,
                'error': 'No soil reports available for training'
            }), 400
        
        # Convert to training data
        soil_data_list = []
        for report in soil_reports:
            soil_data_list.append({
                'ph_level': report.ph_level,
                'nitrogen_level': report.nitrogen_level,
                'phosphorus_level': report.phosphorus_level,
                'potassium_level': report.potassium_level,
                'organic_matter': report.organic_matter,
                'moisture_content': report.moisture_content
            })
        
        result = soil_model.train_model(soil_data_list)
        
        return jsonify({
            'success': True,
            'message': 'Soil analysis model trained successfully',
            'clustering_result': result
        })
        
    except Exception as e:
        logger.error(f"Error training soil model: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@ml_admin_bp.route('/train-all-models', methods=['POST'])
@login_required
@admin_required
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
            soil_reports = SoilReport.query.all()
            soil_data_list = []
            for report in soil_reports:
                soil_data_list.append({
                    'ph_level': report.ph_level,
                    'nitrogen_level': report.nitrogen_level,
                    'phosphorus_level': report.phosphorus_level,
                    'potassium_level': report.potassium_level,
                    'organic_matter': report.organic_matter,
                    'moisture_content': report.moisture_content
                })
            
            soil_result = soil_model.train_model(soil_data_list)
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

@ml_admin_bp.route('/model-performance')
@login_required
@admin_required
def model_performance():
    """
    Get detailed model performance metrics.
    """
    try:
        crop_model = CropPredictionModel()
        yield_model = YieldForecastingModel()
        soil_model = SoilAnalysisModel()
        
        performance = {}
        
        # Get crop model performance
        if crop_model.load_model():
            performance['crop_prediction'] = crop_model.get_model_performance()
        else:
            performance['crop_prediction'] = {'error': 'Model not trained'}
        
        # Get yield model performance
        if yield_model.load_model():
            performance['yield_forecasting'] = yield_model.get_model_performance()
        else:
            performance['yield_forecasting'] = {'error': 'Model not trained'}
        
        # Get soil model performance
        if soil_model.load_model():
            performance['soil_analysis'] = soil_model.get_model_performance()
        else:
            performance['soil_analysis'] = {'error': 'Model not trained'}
        
        return render_template('admin/ml_performance.html', performance=performance)
        
    except Exception as e:
        logger.error(f"Error getting model performance: {str(e)}")
        flash('An error occurred while loading model performance.', 'error')
        return redirect(url_for('ml_admin.ml_dashboard'))

@ml_admin_bp.route('/clear-cache', methods=['POST'])
@login_required
@admin_required
def clear_cache():
    """
    Clear ML model cache.
    """
    try:
        from flask import current_app
        cache_manager = getattr(current_app, 'cache_manager', None)
        
        if cache_manager:
            # Clear different cache patterns
            patterns = [
                'ml_prediction:*',
                'soil_analysis:*',
                'crop_recommendations:*',
                'yield_prediction:*'
            ]
            
            cleared_count = 0
            for pattern in patterns:
                if cache_manager.invalidate_cache_pattern(pattern):
                    cleared_count += 1
            
            return jsonify({
                'success': True,
                'message': f'Cache cleared successfully. {cleared_count} patterns cleared.'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Cache manager not available'
            })
        
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@ml_admin_bp.route('/export-model/<model_type>')
@login_required
@admin_required
def export_model(model_type):
    """
    Export trained model for deployment.
    """
    try:
        if model_type not in ['crop_prediction', 'yield_forecasting', 'soil_analysis']:
            return jsonify({'error': 'Invalid model type'}), 400
        
        model_path = 'app/ml_models/saved_models'
        
        # Find the latest model files
        model_files = []
        for file in os.listdir(model_path):
            if file.startswith(model_type) and file.endswith('.joblib'):
                model_files.append(file)
        
        if not model_files:
            return jsonify({'error': 'No trained model found'}), 404
        
        # Get the latest model
        latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(model_path, x)))
        
        # Create export package
        import zipfile
        import tempfile
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            with zipfile.ZipFile(tmp_file.name, 'w') as zip_file:
                # Add model files
                for file in model_files:
                    if file.startswith(model_type):
                        zip_file.write(os.path.join(model_path, file), file)
                
                # Add metadata
                metadata = {
                    'model_type': model_type,
                    'export_date': datetime.now().isoformat(),
                    'version': '1.0'
                }
                zip_file.writestr('metadata.json', json.dumps(metadata, indent=2))
            
            return jsonify({
                'success': True,
                'download_url': f'/admin/ml/download-export/{os.path.basename(tmp_file.name)}'
            })
        
    except Exception as e:
        logger.error(f"Error exporting model: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def convert_soil_reports_to_training_data(soil_reports):
    """
    Convert soil reports to training data for crop prediction.
    """
    X = []
    y = []
    
    for report in soil_reports:
        # Create feature vector
        features = [
            report.ph_level,
            report.nitrogen_level,
            report.phosphorus_level,
            report.potassium_level,
            report.organic_matter,
            report.moisture_content,
            # Add default weather data
            25,  # temperature
            65,  # humidity
            0    # rainfall
        ]
        
        X.append(features)
        
        # For now, use a simple rule to determine crop type
        # In production, this would come from historical data
        if report.ph_level >= 6.0 and report.ph_level <= 7.5:
            y.append('Rice')
        elif report.nitrogen_level >= 50:
            y.append('Wheat')
        elif report.phosphorus_level >= 40:
            y.append('Corn')
        else:
            y.append('Soybeans')
    
    return np.array(X), np.array(y)

def convert_soil_reports_to_yield_training_data(soil_reports):
    """
    Convert soil reports to training data for yield forecasting.
    """
    X = []
    y = []
    
    for report in soil_reports:
        # Create feature vector
        features = [
            report.ph_level,
            report.nitrogen_level,
            report.phosphorus_level,
            report.potassium_level,
            report.organic_matter,
            report.moisture_content,
            # Add default weather data
            25,  # temperature
            65,  # humidity
            0,   # rainfall
            # Add temporal features
            0.5, 0, 1, 0, 0,  # month, sin(day), cos(day), temp_trend, rain_trend
            # Add crop features (default)
            0.5, 120,  # yield_potential, growth_period
            # Add historical features (default)
            3000, 0, 100,  # avg_yield, trend, std
            # Add engineered features
            0.5, 0.5, 0.5, 0.5, 0.5  # soil_health, temp_suitability, moisture_stress, nutrient_balance, ph_suitability
        ]
        
        X.append(features)
        
        # Calculate expected yield based on soil conditions
        base_yield = 3000
        soil_factor = (report.nitrogen_level + report.phosphorus_level + report.potassium_level + report.organic_matter) / 400
        ph_factor = 1 - abs(report.ph_level - 6.5) / 6.5
        yield_estimate = base_yield * (0.5 + soil_factor * 0.3 + ph_factor * 0.2)
        
        y.append(yield_estimate)
    
    return np.array(X), np.array(y)
