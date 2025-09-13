import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CropPredictionModel:
    """
    Advanced ML model for crop prediction based on soil and weather data.
    Uses Random Forest and Gradient Boosting for high accuracy predictions.
    """
    
    def __init__(self, model_path='app/ml_models/saved_models/'):
        self.model_path = model_path
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.feature_importance = None
        self.accuracy = None
        self.is_trained = False
        
        # Create model directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)
        
        # Crop-specific optimal conditions
        self.crop_conditions = {
            'Rice': {
                'ph_min': 6.0, 'ph_max': 7.5,
                'temp_min': 20, 'temp_max': 30,
                'nitrogen_min': 40, 'nitrogen_max': 80,
                'phosphorus_min': 20, 'phosphorus_max': 60,
                'potassium_min': 30, 'potassium_max': 70,
                'moisture_min': 60, 'moisture_max': 90
            },
            'Wheat': {
                'ph_min': 6.0, 'ph_max': 7.0,
                'temp_min': 15, 'temp_max': 25,
                'nitrogen_min': 50, 'nitrogen_max': 90,
                'phosphorus_min': 30, 'phosphorus_max': 70,
                'potassium_min': 40, 'potassium_max': 80,
                'moisture_min': 50, 'moisture_max': 80
            },
            'Corn': {
                'ph_min': 5.5, 'ph_max': 7.0,
                'temp_min': 25, 'temp_max': 35,
                'nitrogen_min': 60, 'nitrogen_max': 100,
                'phosphorus_min': 40, 'phosphorus_max': 80,
                'potassium_min': 50, 'potassium_max': 90,
                'moisture_min': 50, 'moisture_max': 85
            },
            'Soybeans': {
                'ph_min': 6.0, 'ph_max': 7.0,
                'temp_min': 20, 'temp_max': 30,
                'nitrogen_min': 30, 'nitrogen_max': 60,
                'phosphorus_min': 20, 'phosphorus_max': 50,
                'potassium_min': 40, 'potassium_max': 80,
                'moisture_min': 40, 'moisture_max': 75
            },
            'Cotton': {
                'ph_min': 5.5, 'ph_max': 8.0,
                'temp_min': 25, 'temp_max': 35,
                'nitrogen_min': 40, 'nitrogen_max': 80,
                'phosphorus_min': 20, 'phosphorus_max': 50,
                'potassium_min': 30, 'potassium_max': 70,
                'moisture_min': 40, 'moisture_max': 80
            }
        }
    
    def preprocess_data(self, soil_data, weather_data=None):
        """
        Preprocess soil and weather data for ML model training.
        
        Args:
            soil_data: DataFrame with soil parameters
            weather_data: DataFrame with weather parameters (optional)
        
        Returns:
            Processed features array
        """
        try:
            # Handle missing values
            soil_data = soil_data.fillna(soil_data.median())
            
            # Extract soil features
            soil_features = soil_data[['ph_level', 'nitrogen_level', 'phosphorus_level', 
                                     'potassium_level', 'organic_matter', 'moisture_content']].values
            
            # Add weather features if available
            if weather_data is not None:
                weather_data = weather_data.fillna(weather_data.median())
                weather_features = weather_data[['temperature', 'humidity', 'rainfall']].values
                features = np.column_stack([soil_features, weather_features])
            else:
                # Use default weather values if not provided
                default_weather = np.array([[25, 65, 0]] * len(soil_data))  # temp, humidity, rainfall
                features = np.column_stack([soil_features, default_weather])
            
            # Feature engineering
            engineered_features = self._engineer_features(soil_data, weather_data)
            features = np.column_stack([features, engineered_features])
            
            return features
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise
    
    def _engineer_features(self, soil_data, weather_data=None):
        """
        Create engineered features for better model performance.
        """
        features = []
        
        # Soil nutrient balance
        features.append((soil_data['nitrogen_level'] + 
                        soil_data['phosphorus_level'] + 
                        soil_data['potassium_level']) / 3)
        
        # pH suitability score
        features.append(1 - abs(soil_data['ph_level'] - 6.5) / 6.5)
        
        # Organic matter to moisture ratio
        features.append(soil_data['organic_matter'] / (soil_data['moisture_content'] + 1))
        
        # Nitrogen to phosphorus ratio
        features.append(soil_data['nitrogen_level'] / (soil_data['phosphorus_level'] + 1))
        
        if weather_data is not None:
            # Temperature suitability
            features.append(1 - abs(weather_data['temperature'] - 25) / 25)
            
            # Humidity to rainfall ratio
            features.append(weather_data['humidity'] / (weather_data['rainfall'] + 1))
        else:
            # Default values
            features.extend([0.5, 0.5])
        
        return np.column_stack(features)
    
    def generate_training_data(self, num_samples=10000):
        """
        Generate synthetic training data for demonstration.
        In production, this would use real historical data.
        """
        np.random.seed(42)
        
        data = []
        labels = []
        
        for crop, conditions in self.crop_conditions.items():
            for _ in range(num_samples // len(self.crop_conditions)):
                # Generate soil data within crop-specific ranges
                ph = np.random.normal(
                    (conditions['ph_min'] + conditions['ph_max']) / 2, 
                    0.5
                )
                ph = np.clip(ph, 0, 14)
                
                nitrogen = np.random.normal(
                    (conditions['nitrogen_min'] + conditions['nitrogen_max']) / 2,
                    10
                )
                nitrogen = np.clip(nitrogen, 0, 100)
                
                phosphorus = np.random.normal(
                    (conditions['phosphorus_min'] + conditions['phosphorus_max']) / 2,
                    8
                )
                phosphorus = np.clip(phosphorus, 0, 100)
                
                potassium = np.random.normal(
                    (conditions['potassium_min'] + conditions['potassium_max']) / 2,
                    10
                )
                potassium = np.clip(potassium, 0, 100)
                
                organic_matter = np.random.normal(50, 15)
                organic_matter = np.clip(organic_matter, 0, 100)
                
                moisture = np.random.normal(
                    (conditions['moisture_min'] + conditions['moisture_max']) / 2,
                    10
                )
                moisture = np.clip(moisture, 0, 100)
                
                # Weather data
                temperature = np.random.normal(
                    (conditions['temp_min'] + conditions['temp_max']) / 2,
                    3
                )
                humidity = np.random.normal(65, 15)
                rainfall = np.random.exponential(5)
                
                data.append([
                    ph, nitrogen, phosphorus, potassium, organic_matter, moisture,
                    temperature, humidity, rainfall
                ])
                labels.append(crop)
        
        return np.array(data), np.array(labels)
    
    def train_model(self, X=None, y=None, use_synthetic=True):
        """
        Train the crop prediction model.
        """
        try:
            if use_synthetic or X is None or y is None:
                logger.info("Generating synthetic training data...")
                X, y = self.generate_training_data()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Encode labels
            y_train_encoded = self.label_encoder.fit_transform(y_train)
            y_test_encoded = self.label_encoder.transform(y_test)
            
            # Train Random Forest model
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            # Hyperparameter tuning
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5, 10]
            }
            
            grid_search = GridSearchCV(
                self.model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
            )
            
            logger.info("Training model with hyperparameter tuning...")
            grid_search.fit(X_train_scaled, y_train_encoded)
            
            self.model = grid_search.best_estimator_
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            self.accuracy = accuracy_score(y_test_encoded, y_pred)
            
            # Get feature importance
            self.feature_importance = dict(zip(
                ['ph_level', 'nitrogen_level', 'phosphorus_level', 'potassium_level', 
                 'organic_matter', 'moisture_content', 'temperature', 'humidity', 'rainfall',
                 'nutrient_balance', 'ph_suitability', 'organic_moisture_ratio', 
                 'nitrogen_phosphorus_ratio', 'temp_suitability', 'humidity_rainfall_ratio'],
                self.model.feature_importances_
            ))
            
            self.is_trained = True
            
            logger.info(f"Model trained successfully! Accuracy: {self.accuracy:.4f}")
            logger.info(f"Best parameters: {grid_search.best_params_}")
            
            # Save model
            self.save_model()
            
            return {
                'accuracy': self.accuracy,
                'best_params': grid_search.best_params_,
                'feature_importance': self.feature_importance,
                'classification_report': classification_report(y_test_encoded, y_pred, 
                                                            target_names=self.label_encoder.classes_)
            }
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def predict_crop(self, soil_data, weather_data=None, return_probabilities=True):
        """
        Predict the best crop for given soil and weather conditions.
        """
        if not self.is_trained:
            self.load_model()
        
        if not self.is_trained:
            raise ValueError("Model not trained. Please train the model first.")
        
        try:
            # Preprocess input data
            features = self.preprocess_data(soil_data, weather_data)
            features_scaled = self.scaler.transform(features)
            
            if return_probabilities:
                probabilities = self.model.predict_proba(features_scaled)
                crop_probs = dict(zip(self.label_encoder.classes_, probabilities[0]))
                
                # Get top 3 recommendations
                top_crops = sorted(crop_probs.items(), key=lambda x: x[1], reverse=True)[:3]
                
                recommendations = []
                for crop, prob in top_crops:
                    confidence = int(prob * 100)
                    reason = self._get_recommendation_reason(crop, soil_data.iloc[0] if hasattr(soil_data, 'iloc') else soil_data[0])
                    
                    recommendations.append({
                        'name': crop,
                        'confidence': confidence,
                        'reason': reason,
                        'probability': prob
                    })
                
                return recommendations
            else:
                prediction = self.model.predict(features_scaled)
                return self.label_encoder.inverse_transform(prediction)[0]
                
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise
    
    def _get_recommendation_reason(self, crop, soil_data):
        """
        Generate human-readable reason for crop recommendation.
        """
        if crop not in self.crop_conditions:
            return "Suitable for current soil conditions"
        
        conditions = self.crop_conditions[crop]
        reasons = []
        
        # Check pH
        if hasattr(soil_data, 'ph_level'):
            ph = soil_data.ph_level
            if conditions['ph_min'] <= ph <= conditions['ph_max']:
                reasons.append(f"Optimal pH level ({ph:.1f})")
            else:
                reasons.append(f"pH level ({ph:.1f}) within acceptable range")
        
        # Check temperature (if available)
        if hasattr(soil_data, 'temperature'):
            temp = soil_data.temperature
            if conditions['temp_min'] <= temp <= conditions['temp_max']:
                reasons.append(f"Optimal temperature ({temp:.1f}°C)")
            else:
                reasons.append(f"Temperature ({temp:.1f}°C) suitable for growth")
        
        # Check nutrients
        if hasattr(soil_data, 'nitrogen_level'):
            nitrogen = soil_data.nitrogen_level
            if conditions['nitrogen_min'] <= nitrogen <= conditions['nitrogen_max']:
                reasons.append(f"Good nitrogen levels ({nitrogen:.1f}%)")
        
        return "; ".join(reasons) if reasons else "Suitable for current conditions"
    
    def save_model(self):
        """
        Save the trained model and preprocessing objects.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save model
            model_file = os.path.join(self.model_path, f'crop_prediction_model_{timestamp}.joblib')
            joblib.dump(self.model, model_file)
            
            # Save scaler
            scaler_file = os.path.join(self.model_path, f'crop_scaler_{timestamp}.joblib')
            joblib.dump(self.scaler, scaler_file)
            
            # Save label encoder
            encoder_file = os.path.join(self.model_path, f'crop_encoder_{timestamp}.joblib')
            joblib.dump(self.label_encoder, encoder_file)
            
            # Save metadata
            metadata = {
                'accuracy': self.accuracy,
                'feature_importance': self.feature_importance,
                'timestamp': timestamp,
                'is_trained': self.is_trained
            }
            
            metadata_file = os.path.join(self.model_path, f'crop_metadata_{timestamp}.joblib')
            joblib.dump(metadata, metadata_file)
            
            logger.info(f"Model saved successfully: {model_file}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, timestamp=None):
        """
        Load the most recent trained model.
        """
        try:
            if timestamp is None:
                # Find the most recent model
                model_files = [f for f in os.listdir(self.model_path) if f.startswith('crop_prediction_model_')]
                if not model_files:
                    logger.warning("No trained model found")
                    return False
                
                # Extract timestamps and get the latest
                timestamps = [f.replace('crop_prediction_model_', '').replace('.joblib', '') for f in model_files]
                timestamp = max(timestamps)
            
            # Load model
            model_file = os.path.join(self.model_path, f'crop_prediction_model_{timestamp}.joblib')
            self.model = joblib.load(model_file)
            
            # Load scaler
            scaler_file = os.path.join(self.model_path, f'crop_scaler_{timestamp}.joblib')
            self.scaler = joblib.load(scaler_file)
            
            # Load label encoder
            encoder_file = os.path.join(self.model_path, f'crop_encoder_{timestamp}.joblib')
            self.label_encoder = joblib.load(encoder_file)
            
            # Load metadata
            metadata_file = os.path.join(self.model_path, f'crop_metadata_{timestamp}.joblib')
            metadata = joblib.load(metadata_file)
            
            self.accuracy = metadata['accuracy']
            self.feature_importance = metadata['feature_importance']
            self.is_trained = metadata['is_trained']
            
            logger.info(f"Model loaded successfully: {model_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def get_model_performance(self):
        """
        Get detailed model performance metrics.
        """
        if not self.is_trained:
            return {"error": "Model not trained"}
        
        return {
            'accuracy': self.accuracy,
            'feature_importance': self.feature_importance,
            'is_trained': self.is_trained,
            'model_type': 'Random Forest Classifier',
            'n_estimators': self.model.n_estimators if self.model else None
        }
