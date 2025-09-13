import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import joblib
import os
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YieldForecastingModel:
    """
    Advanced ML model for crop yield forecasting based on soil, weather, and historical data.
    Uses Random Forest and Gradient Boosting for accurate yield predictions.
    """
    
    def __init__(self, model_path='app/ml_models/saved_models/'):
        self.model_path = model_path
        self.scaler = StandardScaler()
        self.yield_scaler = MinMaxScaler()
        self.model = None
        self.feature_importance = None
        self.metrics = {}
        self.is_trained = False
        
        # Create model directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)
        
        # Crop-specific yield factors
        self.crop_yield_factors = {
            'Rice': {'base_yield': 4000, 'max_yield': 8000, 'unit': 'kg/hectare'},
            'Wheat': {'base_yield': 3000, 'max_yield': 6000, 'unit': 'kg/hectare'},
            'Corn': {'base_yield': 5000, 'max_yield': 10000, 'unit': 'kg/hectare'},
            'Soybeans': {'base_yield': 2000, 'max_yield': 4000, 'unit': 'kg/hectare'},
            'Cotton': {'base_yield': 1000, 'max_yield': 2000, 'unit': 'kg/hectare'}
        }
    
    def preprocess_data(self, soil_data, weather_data, historical_yields=None, crop_type=None):
        """
        Preprocess data for yield forecasting model.
        
        Args:
            soil_data: DataFrame with soil parameters
            weather_data: DataFrame with weather parameters
            historical_yields: DataFrame with historical yield data
            crop_type: String indicating crop type
        
        Returns:
            Processed features array and target values
        """
        try:
            # Handle missing values
            soil_data = soil_data.fillna(soil_data.median())
            weather_data = weather_data.fillna(weather_data.median())
            
            # Extract soil features
            soil_features = soil_data[['ph_level', 'nitrogen_level', 'phosphorus_level', 
                                     'potassium_level', 'organic_matter', 'moisture_content']].values
            
            # Extract weather features
            weather_features = weather_data[['temperature', 'humidity', 'rainfall']].values
            
            # Combine features
            features = np.column_stack([soil_features, weather_features])
            
            # Add temporal features
            temporal_features = self._create_temporal_features(weather_data)
            features = np.column_stack([features, temporal_features])
            
            # Add crop-specific features
            if crop_type:
                crop_features = self._create_crop_features(crop_type, len(soil_data))
                features = np.column_stack([features, crop_features])
            
            # Add historical yield features
            if historical_yields is not None:
                hist_features = self._create_historical_features(historical_yields, len(soil_data))
                features = np.column_stack([features, hist_features])
            
            # Feature engineering
            engineered_features = self._engineer_yield_features(soil_data, weather_data)
            features = np.column_stack([features, engineered_features])
            
            return features
            
        except Exception as e:
            logger.error(f"Error in yield data preprocessing: {str(e)}")
            raise
    
    def _create_temporal_features(self, weather_data):
        """
        Create temporal features for yield prediction.
        """
        features = []
        
        # Season indicators
        if 'date' in weather_data.columns:
            dates = pd.to_datetime(weather_data['date'])
            features.append((dates.dt.month % 12 + 1) / 12)  # Month normalized
            features.append(np.sin(2 * np.pi * dates.dt.dayofyear / 365))  # Seasonal cycle
            features.append(np.cos(2 * np.pi * dates.dt.dayofyear / 365))
        else:
            # Default seasonal values
            features.extend([0.5, 0, 1])
        
        # Weather trends
        if len(weather_data) > 1:
            temp_trend = np.gradient(weather_data['temperature'].values)
            rainfall_trend = np.gradient(weather_data['rainfall'].values)
            features.extend([temp_trend[-1], rainfall_trend[-1]])
        else:
            features.extend([0, 0])
        
        return np.column_stack(features) if len(features) > 1 else np.array(features).reshape(-1, 1)
    
    def _create_crop_features(self, crop_type, num_samples):
        """
        Create crop-specific features.
        """
        if crop_type in self.crop_yield_factors:
            factors = self.crop_yield_factors[crop_type]
            base_yield = factors['base_yield']
            max_yield = factors['max_yield']
        else:
            base_yield = 3000
            max_yield = 6000
        
        # Crop yield potential
        yield_potential = np.full(num_samples, base_yield / max_yield)
        
        # Crop growth period (simplified)
        growth_period = np.full(num_samples, 120)  # days
        
        return np.column_stack([yield_potential, growth_period])
    
    def _create_historical_features(self, historical_yields, num_samples):
        """
        Create features from historical yield data.
        """
        features = []
        
        if len(historical_yields) > 0:
            # Average historical yield
            avg_yield = historical_yields['yield'].mean()
            features.append(np.full(num_samples, avg_yield))
            
            # Yield trend
            if len(historical_yields) > 1:
                trend = np.polyfit(range(len(historical_yields)), historical_yields['yield'], 1)[0]
                features.append(np.full(num_samples, trend))
            else:
                features.append(np.zeros(num_samples))
            
            # Yield variability
            yield_std = historical_yields['yield'].std()
            features.append(np.full(num_samples, yield_std))
        else:
            features.extend([np.zeros(num_samples)] * 3)
        
        return np.column_stack(features)
    
    def _engineer_yield_features(self, soil_data, weather_data):
        """
        Create engineered features for yield prediction.
        """
        features = []
        
        # Soil health score
        soil_health = (
            soil_data['nitrogen_level'] * 0.3 +
            soil_data['phosphorus_level'] * 0.3 +
            soil_data['potassium_level'] * 0.2 +
            soil_data['organic_matter'] * 0.2
        ) / 100
        features.append(soil_health)
        
        # Weather suitability
        temp_suitability = 1 - abs(weather_data['temperature'] - 25) / 25
        features.append(temp_suitability)
        
        # Moisture stress index
        moisture_stress = abs(weather_data['humidity'] - 70) / 70
        features.append(moisture_stress)
        
        # Nutrient balance
        nutrient_balance = (
            soil_data['nitrogen_level'] / (soil_data['phosphorus_level'] + 1) +
            soil_data['phosphorus_level'] / (soil_data['potassium_level'] + 1)
        ) / 2
        features.append(nutrient_balance)
        
        # pH suitability
        ph_suitability = 1 - abs(soil_data['ph_level'] - 6.5) / 6.5
        features.append(ph_suitability)
        
        return np.column_stack(features)
    
    def generate_training_data(self, num_samples=5000):
        """
        Generate synthetic training data for yield forecasting.
        """
        np.random.seed(42)
        
        data = []
        yields = []
        
        for crop, factors in self.crop_yield_factors.items():
            for _ in range(num_samples // len(self.crop_yield_factors)):
                # Generate soil data
                ph = np.random.normal(6.5, 1.0)
                ph = np.clip(ph, 4.0, 9.0)
                
                nitrogen = np.random.normal(60, 20)
                nitrogen = np.clip(nitrogen, 0, 100)
                
                phosphorus = np.random.normal(50, 15)
                phosphorus = np.clip(phosphorus, 0, 100)
                
                potassium = np.random.normal(55, 18)
                potassium = np.clip(potassium, 0, 100)
                
                organic_matter = np.random.normal(50, 15)
                organic_matter = np.clip(organic_matter, 0, 100)
                
                moisture = np.random.normal(65, 15)
                moisture = np.clip(moisture, 0, 100)
                
                # Generate weather data
                temperature = np.random.normal(25, 5)
                humidity = np.random.normal(65, 15)
                rainfall = np.random.exponential(10)
                
                # Calculate expected yield based on conditions
                base_yield = factors['base_yield']
                max_yield = factors['max_yield']
                
                # Yield factors
                soil_factor = (nitrogen + phosphorus + potassium + organic_matter) / 400
                weather_factor = 1 - abs(temperature - 25) / 25
                moisture_factor = 1 - abs(moisture - 70) / 70
                ph_factor = 1 - abs(ph - 6.5) / 6.5
                
                # Add some randomness
                random_factor = np.random.normal(1, 0.1)
                
                # Calculate final yield
                yield_multiplier = (soil_factor * 0.4 + weather_factor * 0.3 + 
                                  moisture_factor * 0.2 + ph_factor * 0.1) * random_factor
                yield_multiplier = np.clip(yield_multiplier, 0.3, 1.5)
                
                predicted_yield = base_yield + (max_yield - base_yield) * yield_multiplier
                predicted_yield = np.clip(predicted_yield, base_yield * 0.3, max_yield * 1.2)
                
                data.append([
                    ph, nitrogen, phosphorus, potassium, organic_matter, moisture,
                    temperature, humidity, rainfall,
                    # Temporal features
                    0.5, 0, 1, 0, 0,  # month, sin(day), cos(day), temp_trend, rain_trend
                    # Crop features
                    base_yield / max_yield, 120,  # yield_potential, growth_period
                    # Historical features
                    base_yield, 0, 100,  # avg_yield, trend, std
                    # Engineered features
                    soil_factor, weather_factor, moisture_factor, nutrient_factor, ph_factor
                ])
                yields.append(predicted_yield)
        
        return np.array(data), np.array(yields)
    
    def train_model(self, X=None, y=None, use_synthetic=True):
        """
        Train the yield forecasting model.
        """
        try:
            if use_synthetic or X is None or y is None:
                logger.info("Generating synthetic training data for yield forecasting...")
                X, y = self.generate_training_data()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Scale target values
            y_train_scaled = self.yield_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
            y_test_scaled = self.yield_scaler.transform(y_test.reshape(-1, 1)).ravel()
            
            # Train Random Forest model
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            # Hyperparameter tuning
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [15, 20, 25],
                'min_samples_split': [2, 5, 10]
            }
            
            from sklearn.model_selection import GridSearchCV
            grid_search = GridSearchCV(
                self.model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
            )
            
            logger.info("Training yield forecasting model...")
            grid_search.fit(X_train_scaled, y_train_scaled)
            
            self.model = grid_search.best_estimator_
            
            # Evaluate model
            y_pred_scaled = self.model.predict(X_test_scaled)
            y_pred = self.yield_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            
            # Calculate metrics
            self.metrics = {
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred),
                'mape': mean_absolute_percentage_error(y_test, y_pred) * 100
            }
            
            # Get feature importance
            feature_names = [
                'ph_level', 'nitrogen_level', 'phosphorus_level', 'potassium_level',
                'organic_matter', 'moisture_content', 'temperature', 'humidity', 'rainfall',
                'month', 'sin_day', 'cos_day', 'temp_trend', 'rain_trend',
                'yield_potential', 'growth_period', 'avg_yield', 'yield_trend', 'yield_std',
                'soil_health', 'temp_suitability', 'moisture_stress', 'nutrient_balance', 'ph_suitability'
            ]
            
            self.feature_importance = dict(zip(feature_names, self.model.feature_importances_))
            
            self.is_trained = True
            
            logger.info(f"Yield forecasting model trained successfully!")
            logger.info(f"MAE: {self.metrics['mae']:.2f}")
            logger.info(f"RMSE: {self.metrics['rmse']:.2f}")
            logger.info(f"R²: {self.metrics['r2']:.4f}")
            logger.info(f"MAPE: {self.metrics['mape']:.2f}%")
            
            # Save model
            self.save_model()
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"Error training yield model: {str(e)}")
            raise
    
    def predict_yield(self, soil_data, weather_data, historical_yields=None, crop_type=None):
        """
        Predict crop yield for given conditions.
        """
        if not self.is_trained:
            self.load_model()
        
        if not self.is_trained:
            raise ValueError("Model not trained. Please train the model first.")
        
        try:
            # Preprocess input data
            features = self.preprocess_data(soil_data, weather_data, historical_yields, crop_type)
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            y_pred_scaled = self.model.predict(features_scaled)
            y_pred = self.yield_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            
            # Calculate confidence interval (simplified)
            confidence_interval = self._calculate_confidence_interval(features_scaled)
            
            return {
                'predicted_yield': float(y_pred[0]),
                'confidence_interval': confidence_interval,
                'crop_type': crop_type,
                'unit': self.crop_yield_factors.get(crop_type, {}).get('unit', 'kg/hectare'),
                'model_metrics': self.metrics
            }
            
        except Exception as e:
            logger.error(f"Error in yield prediction: {str(e)}")
            raise
    
    def _calculate_confidence_interval(self, features_scaled, confidence=0.95):
        """
        Calculate confidence interval for yield prediction.
        """
        # Simplified confidence interval calculation
        # In production, use proper uncertainty quantification methods
        predictions = []
        
        # Use multiple trees for uncertainty estimation
        for tree in self.model.estimators_[:10]:  # Use first 10 trees
            pred = tree.predict(features_scaled)
            pred_unscaled = self.yield_scaler.inverse_transform(pred.reshape(-1, 1)).ravel()
            predictions.append(pred_unscaled[0])
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        # Calculate confidence interval
        alpha = 1 - confidence
        z_score = 1.96  # For 95% confidence
        margin_error = z_score * std_pred
        
        return {
            'lower': float(mean_pred - margin_error),
            'upper': float(mean_pred + margin_error),
            'confidence': confidence
        }
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model performance with detailed metrics.
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X_test_scaled = self.scaler.transform(X_test)
        y_pred_scaled = self.model.predict(X_test_scaled)
        y_pred = self.yield_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'mape': mean_absolute_percentage_error(y_test, y_pred) * 100
        }
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_test_scaled, y_test_scaled, cv=5, scoring='neg_mean_squared_error')
        metrics['cv_rmse'] = np.sqrt(-cv_scores.mean())
        metrics['cv_std'] = np.sqrt(cv_scores.std())
        
        return metrics
    
    def save_model(self):
        """
        Save the trained model and preprocessing objects.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save model
            model_file = os.path.join(self.model_path, f'yield_forecasting_model_{timestamp}.joblib')
            joblib.dump(self.model, model_file)
            
            # Save scalers
            scaler_file = os.path.join(self.model_path, f'yield_scaler_{timestamp}.joblib')
            joblib.dump(self.scaler, scaler_file)
            
            yield_scaler_file = os.path.join(self.model_path, f'yield_target_scaler_{timestamp}.joblib')
            joblib.dump(self.yield_scaler, yield_scaler_file)
            
            # Save metadata
            metadata = {
                'metrics': self.metrics,
                'feature_importance': self.feature_importance,
                'timestamp': timestamp,
                'is_trained': self.is_trained
            }
            
            metadata_file = os.path.join(self.model_path, f'yield_metadata_{timestamp}.joblib')
            joblib.dump(metadata, metadata_file)
            
            logger.info(f"Yield forecasting model saved successfully: {model_file}")
            
        except Exception as e:
            logger.error(f"Error saving yield model: {str(e)}")
            raise
    
    def load_model(self, timestamp=None):
        """
        Load the most recent trained model.
        """
        try:
            if timestamp is None:
                # Find the most recent model
                model_files = [f for f in os.listdir(self.model_path) if f.startswith('yield_forecasting_model_')]
                if not model_files:
                    logger.warning("No trained yield model found")
                    return False
                
                # Extract timestamps and get the latest
                timestamps = [f.replace('yield_forecasting_model_', '').replace('.joblib', '') for f in model_files]
                timestamp = max(timestamps)
            
            # Load model
            model_file = os.path.join(self.model_path, f'yield_forecasting_model_{timestamp}.joblib')
            self.model = joblib.load(model_file)
            
            # Load scalers
            scaler_file = os.path.join(self.model_path, f'yield_scaler_{timestamp}.joblib')
            self.scaler = joblib.load(scaler_file)
            
            yield_scaler_file = os.path.join(self.model_path, f'yield_target_scaler_{timestamp}.joblib')
            self.yield_scaler = joblib.load(yield_scaler_file)
            
            # Load metadata
            metadata_file = os.path.join(self.model_path, f'yield_metadata_{timestamp}.joblib')
            metadata = joblib.load(metadata_file)
            
            self.metrics = metadata['metrics']
            self.feature_importance = metadata['feature_importance']
            self.is_trained = metadata['is_trained']
            
            logger.info(f"Yield forecasting model loaded successfully: {model_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading yield model: {str(e)}")
            return False
    
    def get_model_performance(self):
        """
        Get detailed model performance metrics.
        """
        if not self.is_trained:
            return {"error": "Model not trained"}
        
        return {
            'metrics': self.metrics,
            'feature_importance': self.feature_importance,
            'is_trained': self.is_trained,
            'model_type': 'Random Forest Regressor'
        }
