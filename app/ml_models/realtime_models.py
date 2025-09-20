import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import joblib
import os
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import threading
import time
from collections import deque
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeMLModel:
    """
    Base class for real-time ML models with streaming data support
    """
    
    def __init__(self, model_path='app/ml_models/saved_models/', window_size=100):
        self.model_path = model_path
        self.window_size = window_size
        self.data_buffer = deque(maxlen=window_size)
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.last_update = None
        self.update_frequency = 300  # Update model every 5 minutes
        self.retrain_threshold = 0.05  # Retrain if performance drops by 5%
        
        # Create model directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)
    
    def add_data_point(self, data_point: Dict):
        """Add a new data point to the buffer"""
        self.data_buffer.append({
            **data_point,
            'timestamp': datetime.now().isoformat()
        })
        
        # Check if we need to update the model
        if self._should_update_model():
            self._update_model()
    
    def _should_update_model(self) -> bool:
        """Check if the model should be updated"""
        if not self.is_trained:
            return len(self.data_buffer) >= self.window_size // 2
        
        if self.last_update is None:
            return True
        
        time_since_update = (datetime.now() - self.last_update).total_seconds()
        return time_since_update >= self.update_frequency
    
    def _update_model(self):
        """Update the model with new data"""
        if len(self.data_buffer) < 10:  # Need minimum data points
            return
        
        try:
            # Convert buffer to DataFrame
            df = pd.DataFrame(list(self.data_buffer))
            
            # Prepare features and targets
            X, y = self._prepare_training_data(df)
            
            if X is None or y is None:
                return
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train/update model
            if self.model is None:
                self.model = self._create_model()
            
            self.model.fit(X_scaled, y)
            self.is_trained = True
            self.last_update = datetime.now()
            
            logger.info(f"Model updated with {len(self.data_buffer)} data points")
            
        except Exception as e:
            logger.error(f"Error updating model: {str(e)}")
    
    def predict_realtime(self, data_point: Dict) -> Dict:
        """Make real-time prediction"""
        if not self.is_trained:
            return {'error': 'Model not trained'}
        
        try:
            # Add data point to buffer
            self.add_data_point(data_point)
            
            # Prepare features
            features = self._prepare_features(data_point)
            if features is None:
                return {'error': 'Invalid features'}
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Make prediction
            prediction = self._make_prediction(features_scaled)
            
            # Calculate confidence
            confidence = self._calculate_confidence(features_scaled)
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat(),
                'model_updated': self.last_update.isoformat() if self.last_update else None
            }
            
        except Exception as e:
            logger.error(f"Error in real-time prediction: {str(e)}")
            return {'error': str(e)}
    
    def _prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data - to be implemented by subclasses"""
        raise NotImplementedError
    
    def _prepare_features(self, data_point: Dict) -> np.ndarray:
        """Prepare features for prediction - to be implemented by subclasses"""
        raise NotImplementedError
    
    def _create_model(self):
        """Create model instance - to be implemented by subclasses"""
        raise NotImplementedError
    
    def _make_prediction(self, features_scaled: np.ndarray):
        """Make prediction - to be implemented by subclasses"""
        raise NotImplementedError
    
    def _calculate_confidence(self, features_scaled: np.ndarray) -> float:
        """Calculate prediction confidence - to be implemented by subclasses"""
        return 0.8  # Default confidence


class RealTimeCropPredictionModel(RealTimeMLModel):
    """
    Real-time crop prediction model with streaming data support
    """
    
    def __init__(self, model_path='app/ml_models/saved_models/', window_size=100):
        super().__init__(model_path, window_size)
        self.label_encoder = None
        self.crop_conditions = {
            'Rice': {'ph_min': 6.0, 'ph_max': 7.5, 'temp_min': 20, 'temp_max': 30},
            'Wheat': {'ph_min': 6.0, 'ph_max': 7.0, 'temp_min': 15, 'temp_max': 25},
            'Corn': {'ph_min': 5.5, 'ph_max': 7.0, 'temp_min': 25, 'temp_max': 35},
            'Soybeans': {'ph_min': 6.0, 'ph_max': 7.0, 'temp_min': 20, 'temp_max': 30},
            'Cotton': {'ph_min': 5.5, 'ph_max': 8.0, 'temp_min': 25, 'temp_max': 35}
        }
    
    def _prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for crop prediction"""
        try:
            # Required features
            feature_cols = ['ph_level', 'nitrogen_level', 'phosphorus_level', 
                           'potassium_level', 'organic_matter', 'moisture_content',
                           'temperature', 'humidity', 'rainfall']
            
            # Check if all required columns exist
            missing_cols = [col for col in feature_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing columns: {missing_cols}")
                return None, None
            
            # Prepare features
            X = df[feature_cols].values
            
            # Generate labels based on optimal conditions
            y = []
            for _, row in df.iterrows():
                best_crop = self._find_best_crop(row)
                y.append(best_crop)
            
            # Encode labels
            if self.label_encoder is None:
                from sklearn.preprocessing import LabelEncoder
                self.label_encoder = LabelEncoder()
                y_encoded = self.label_encoder.fit_transform(y)
            else:
                y_encoded = self.label_encoder.transform(y)
            
            return X, y_encoded
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            return None, None
    
    def _prepare_features(self, data_point: Dict) -> np.ndarray:
        """Prepare features for crop prediction"""
        feature_cols = ['ph_level', 'nitrogen_level', 'phosphorus_level', 
                       'potassium_level', 'organic_matter', 'moisture_content',
                       'temperature', 'humidity', 'rainfall']
        
        features = []
        for col in feature_cols:
            if col in data_point:
                features.append(float(data_point[col]))
            else:
                # Use default values
                defaults = {
                    'ph_level': 6.5, 'nitrogen_level': 60, 'phosphorus_level': 50,
                    'potassium_level': 55, 'organic_matter': 3, 'moisture_content': 65,
                    'temperature': 25, 'humidity': 65, 'rainfall': 0
                }
                features.append(defaults[col])
        
        return np.array(features)
    
    def _create_model(self):
        """Create Random Forest classifier for crop prediction"""
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    
    def _make_prediction(self, features_scaled: np.ndarray):
        """Make crop prediction"""
        if self.label_encoder is None:
            return "Unknown"
        
        prediction = self.model.predict(features_scaled)[0]
        crop_name = self.label_encoder.inverse_transform([prediction])[0]
        
        # Get probabilities for all crops
        probabilities = self.model.predict_proba(features_scaled)[0]
        crop_probs = dict(zip(self.label_encoder.classes_, probabilities))
        
        # Get top 3 recommendations
        top_crops = sorted(crop_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            'recommended_crop': crop_name,
            'confidence': float(max(probabilities)),
            'all_recommendations': [
                {'crop': crop, 'probability': float(prob)} 
                for crop, prob in top_crops
            ]
        }
    
    def _find_best_crop(self, row: pd.Series) -> str:
        """Find the best crop based on soil and weather conditions"""
        best_crop = None
        best_score = -1
        
        for crop, conditions in self.crop_conditions.items():
            score = 0
            
            # pH score
            ph = row['ph_level']
            if conditions['ph_min'] <= ph <= conditions['ph_max']:
                score += 1
            else:
                score += max(0, 1 - abs(ph - (conditions['ph_min'] + conditions['ph_max']) / 2) / 2)
            
            # Temperature score
            temp = row['temperature']
            if conditions['temp_min'] <= temp <= conditions['temp_max']:
                score += 1
            else:
                score += max(0, 1 - abs(temp - (conditions['temp_min'] + conditions['temp_max']) / 2) / 10)
            
            # Nutrient scores
            nutrients = ['nitrogen_level', 'phosphorus_level', 'potassium_level']
            for nutrient in nutrients:
                if nutrient in row:
                    level = row[nutrient]
                    if level >= 50:  # Good nutrient level
                        score += 0.5
            
            if score > best_score:
                best_score = score
                best_crop = crop
        
        return best_crop or 'Rice'  # Default fallback


class RealTimeYieldForecastingModel(RealTimeMLModel):
    """
    Real-time yield forecasting model with streaming data support
    """
    
    def __init__(self, model_path='app/ml_models/saved_models/', window_size=100):
        super().__init__(model_path, window_size)
        self.yield_scaler = MinMaxScaler()
        self.crop_yield_factors = {
            'Rice': {'base_yield': 4000, 'max_yield': 8000},
            'Wheat': {'base_yield': 3000, 'max_yield': 6000},
            'Corn': {'base_yield': 5000, 'max_yield': 10000},
            'Soybeans': {'base_yield': 2000, 'max_yield': 4000},
            'Cotton': {'base_yield': 1000, 'max_yield': 2000}
        }
    
    def _prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for yield forecasting"""
        try:
            # Required features
            feature_cols = ['ph_level', 'nitrogen_level', 'phosphorus_level', 
                           'potassium_level', 'organic_matter', 'moisture_content',
                           'temperature', 'humidity', 'rainfall', 'crop_type']
            
            # Check if all required columns exist
            missing_cols = [col for col in feature_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing columns: {missing_cols}")
                return None, None
            
            # Prepare features
            X = df[feature_cols].copy()
            
            # Encode crop type
            if 'crop_type' in X.columns:
                X['crop_type_encoded'] = pd.Categorical(X['crop_type']).codes
                X = X.drop('crop_type', axis=1)
            
            X = X.values
            
            # Generate yield targets based on conditions
            y = []
            for _, row in df.iterrows():
                crop_type = row.get('crop_type', 'Rice')
                yield_val = self._calculate_expected_yield(row, crop_type)
                y.append(yield_val)
            
            return X, np.array(y)
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            return None, None
    
    def _prepare_features(self, data_point: Dict) -> np.ndarray:
        """Prepare features for yield prediction"""
        feature_cols = ['ph_level', 'nitrogen_level', 'phosphorus_level', 
                       'potassium_level', 'organic_matter', 'moisture_content',
                       'temperature', 'humidity', 'rainfall', 'crop_type']
        
        features = []
        for col in feature_cols:
            if col in data_point:
                if col == 'crop_type':
                    # Encode crop type
                    crop_types = list(self.crop_yield_factors.keys())
                    if data_point[col] in crop_types:
                        features.append(crop_types.index(data_point[col]))
                    else:
                        features.append(0)  # Default to first crop
                else:
                    features.append(float(data_point[col]))
            else:
                # Use default values
                defaults = {
                    'ph_level': 6.5, 'nitrogen_level': 60, 'phosphorus_level': 50,
                    'potassium_level': 55, 'organic_matter': 3, 'moisture_content': 65,
                    'temperature': 25, 'humidity': 65, 'rainfall': 0
                }
                if col == 'crop_type':
                    features.append(0)  # Default crop type
                else:
                    features.append(defaults[col])
        
        return np.array(features)
    
    def _create_model(self):
        """Create Random Forest regressor for yield forecasting"""
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    
    def _make_prediction(self, features_scaled: np.ndarray):
        """Make yield prediction"""
        prediction = self.model.predict(features_scaled)[0]
        
        # Calculate confidence based on prediction variance
        # Use multiple trees to estimate uncertainty
        tree_predictions = []
        for tree in self.model.estimators_[:10]:  # Use first 10 trees
            tree_pred = tree.predict(features_scaled)[0]
            tree_predictions.append(tree_pred)
        
        std_dev = np.std(tree_predictions)
        confidence = max(0, 1 - (std_dev / prediction)) if prediction > 0 else 0.5
        
        return {
            'predicted_yield': float(prediction),
            'confidence': float(confidence),
            'unit': 'kg/hectare'
        }
    
    def _calculate_expected_yield(self, row: pd.Series, crop_type: str) -> float:
        """Calculate expected yield based on conditions"""
        if crop_type not in self.crop_yield_factors:
            crop_type = 'Rice'  # Default
        
        factors = self.crop_yield_factors[crop_type]
        base_yield = factors['base_yield']
        max_yield = factors['max_yield']
        
        # Calculate yield factors
        soil_factor = (row['nitrogen_level'] + row['phosphorus_level'] + 
                      row['potassium_level'] + row['organic_matter']) / 400
        
        weather_factor = 1 - abs(row['temperature'] - 25) / 25
        moisture_factor = 1 - abs(row['moisture_content'] - 70) / 70
        ph_factor = 1 - abs(row['ph_level'] - 6.5) / 6.5
        
        # Combine factors
        yield_multiplier = (soil_factor * 0.4 + weather_factor * 0.3 + 
                          moisture_factor * 0.2 + ph_factor * 0.1)
        yield_multiplier = np.clip(yield_multiplier, 0.3, 1.5)
        
        predicted_yield = base_yield + (max_yield - base_yield) * yield_multiplier
        return np.clip(predicted_yield, base_yield * 0.3, max_yield * 1.2)


class RealTimeSoilHealthModel(RealTimeMLModel):
    """
    Real-time soil health analysis model with streaming data support
    """
    
    def __init__(self, model_path='app/ml_models/saved_models/', window_size=100):
        super().__init__(model_path, window_size)
        self.health_thresholds = {
            'excellent': 80,
            'good': 60,
            'fair': 40,
            'poor': 20
        }
    
    def _prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for soil health analysis"""
        try:
            # Required features
            feature_cols = ['ph_level', 'nitrogen_level', 'phosphorus_level', 
                           'potassium_level', 'organic_matter', 'moisture_content']
            
            # Check if all required columns exist
            missing_cols = [col for col in feature_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing columns: {missing_cols}")
                return None, None
            
            # Prepare features
            X = df[feature_cols].values
            
            # Calculate health scores as targets
            y = []
            for _, row in df.iterrows():
                health_score = self._calculate_health_score(row)
                y.append(health_score)
            
            return X, np.array(y)
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            return None, None
    
    def _prepare_features(self, data_point: Dict) -> np.ndarray:
        """Prepare features for soil health analysis"""
        feature_cols = ['ph_level', 'nitrogen_level', 'phosphorus_level', 
                       'potassium_level', 'organic_matter', 'moisture_content']
        
        features = []
        for col in feature_cols:
            if col in data_point:
                features.append(float(data_point[col]))
            else:
                # Use default values
                defaults = {
                    'ph_level': 6.5, 'nitrogen_level': 60, 'phosphorus_level': 50,
                    'potassium_level': 55, 'organic_matter': 3, 'moisture_content': 65
                }
                features.append(defaults[col])
        
        return np.array(features)
    
    def _create_model(self):
        """Create Random Forest regressor for soil health analysis"""
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    
    def _make_prediction(self, features_scaled: np.ndarray):
        """Make soil health prediction"""
        health_score = self.model.predict(features_scaled)[0]
        health_score = np.clip(health_score, 0, 100)
        
        # Determine health category
        if health_score >= self.health_thresholds['excellent']:
            category = 'excellent'
        elif health_score >= self.health_thresholds['good']:
            category = 'good'
        elif health_score >= self.health_thresholds['fair']:
            category = 'fair'
        else:
            category = 'poor'
        
        # Calculate confidence based on feature importance
        confidence = 0.8  # Base confidence
        
        return {
            'health_score': float(health_score),
            'health_category': category,
            'confidence': float(confidence),
            'recommendations': self._generate_recommendations(health_score, features_scaled[0])
        }
    
    def _calculate_health_score(self, row: pd.Series) -> float:
        """Calculate soil health score"""
        weights = {
            'ph_level': 0.20,
            'nitrogen_level': 0.25,
            'phosphorus_level': 0.20,
            'potassium_level': 0.20,
            'organic_matter': 0.10,
            'moisture_content': 0.05
        }
        
        score = 0
        
        # pH score
        ph = row['ph_level']
        if 6.0 <= ph <= 7.0:
            ph_score = 100
        elif 5.5 <= ph < 6.0 or 7.0 < ph <= 7.5:
            ph_score = 80
        else:
            ph_score = 60
        
        score += ph_score * weights['ph_level']
        
        # Nutrient scores
        for nutrient in ['nitrogen_level', 'phosphorus_level', 'potassium_level']:
            level = row[nutrient]
            if level >= 60:
                nutrient_score = 100
            elif level >= 40:
                nutrient_score = 80
            elif level >= 20:
                nutrient_score = 60
            else:
                nutrient_score = 40
            
            score += nutrient_score * weights[nutrient]
        
        # Organic matter score
        organic_matter = row['organic_matter']
        if organic_matter >= 5:
            om_score = 100
        elif organic_matter >= 3:
            om_score = 80
        else:
            om_score = 60
        
        score += om_score * weights['organic_matter']
        
        # Moisture score
        moisture = row['moisture_content']
        if 50 <= moisture <= 80:
            moisture_score = 100
        else:
            moisture_score = 80
        
        score += moisture_score * weights['moisture_content']
        
        return min(100, max(0, score))
    
    def _generate_recommendations(self, health_score: float, features: np.ndarray) -> List[Dict]:
        """Generate soil improvement recommendations"""
        recommendations = []
        
        if health_score < 40:
            recommendations.append({
                'type': 'General',
                'priority': 'high',
                'action': 'Conduct comprehensive soil testing',
                'description': 'Soil health is poor. Immediate testing and intervention required.'
            })
        
        if health_score < 60:
            recommendations.append({
                'type': 'General',
                'priority': 'medium',
                'action': 'Implement crop rotation',
                'description': 'Rotate crops to improve soil structure and nutrient balance.'
            })
        
        return recommendations


# Global instances for real-time models
realtime_crop_model = RealTimeCropPredictionModel()
realtime_yield_model = RealTimeYieldForecastingModel()
realtime_soil_model = RealTimeSoilHealthModel()
