import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SoilAnalysisModel:
    """
    Advanced soil analysis model for soil health assessment and recommendations.
    Uses clustering and classification techniques for soil type identification.
    """
    
    def __init__(self, model_path='app/ml_models/saved_models/'):
        self.model_path = model_path
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=3)
        self.cluster_model = None
        self.soil_health_thresholds = {
            'excellent': 80,
            'good': 60,
            'fair': 40,
            'poor': 20
        }
        self.is_trained = False
        
        # Create model directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)
    
    def analyze_soil_health(self, soil_data):
        """
        Analyze soil health based on multiple parameters.
        
        Args:
            soil_data: DataFrame or dict with soil parameters
        
        Returns:
            Dictionary with soil health analysis
        """
        try:
            if isinstance(soil_data, dict):
                soil_data = pd.DataFrame([soil_data])
            
            # Calculate soil health score
            health_score = self._calculate_soil_health_score(soil_data)
            
            # Determine soil health category
            health_category = self._categorize_soil_health(health_score)
            
            # Get nutrient analysis
            nutrient_analysis = self._analyze_nutrients(soil_data)
            
            # Get pH analysis
            ph_analysis = self._analyze_ph(soil_data)
            
            # Get recommendations
            recommendations = self._generate_recommendations(soil_data, health_score)
            
            return {
                'health_score': health_score,
                'health_category': health_category,
                'nutrient_analysis': nutrient_analysis,
                'ph_analysis': ph_analysis,
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in soil health analysis: {str(e)}")
            raise
    
    def _calculate_soil_health_score(self, soil_data):
        """
        Calculate overall soil health score (0-100).
        """
        # Weighted scoring system
        weights = {
            'ph_level': 0.20,
            'nitrogen_level': 0.25,
            'phosphorus_level': 0.20,
            'potassium_level': 0.20,
            'organic_matter': 0.10,
            'moisture_content': 0.05
        }
        
        score = 0
        
        # pH score (optimal range: 6.0-7.0)
        ph = soil_data['ph_level'].iloc[0] if hasattr(soil_data, 'iloc') else soil_data['ph_level']
        if 6.0 <= ph <= 7.0:
            ph_score = 100
        elif 5.5 <= ph < 6.0 or 7.0 < ph <= 7.5:
            ph_score = 80
        elif 5.0 <= ph < 5.5 or 7.5 < ph <= 8.0:
            ph_score = 60
        else:
            ph_score = 40
        
        score += ph_score * weights['ph_level']
        
        # Nutrient scores
        for nutrient in ['nitrogen_level', 'phosphorus_level', 'potassium_level']:
            level = soil_data[nutrient].iloc[0] if hasattr(soil_data, 'iloc') else soil_data[nutrient]
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
        organic_matter = soil_data['organic_matter'].iloc[0] if hasattr(soil_data, 'iloc') else soil_data['organic_matter']
        if organic_matter >= 5:
            om_score = 100
        elif organic_matter >= 3:
            om_score = 80
        elif organic_matter >= 1:
            om_score = 60
        else:
            om_score = 40
        
        score += om_score * weights['organic_matter']
        
        # Moisture score
        moisture = soil_data['moisture_content'].iloc[0] if hasattr(soil_data, 'iloc') else soil_data['moisture_content']
        if 50 <= moisture <= 80:
            moisture_score = 100
        elif 40 <= moisture < 50 or 80 < moisture <= 90:
            moisture_score = 80
        else:
            moisture_score = 60
        
        score += moisture_score * weights['moisture_content']
        
        return min(100, max(0, score))
    
    def _categorize_soil_health(self, health_score):
        """
        Categorize soil health based on score.
        """
        if health_score >= self.soil_health_thresholds['excellent']:
            return 'excellent'
        elif health_score >= self.soil_health_thresholds['good']:
            return 'good'
        elif health_score >= self.soil_health_thresholds['fair']:
            return 'fair'
        else:
            return 'poor'
    
    def _analyze_nutrients(self, soil_data):
        """
        Analyze nutrient levels and provide recommendations.
        """
        analysis = {}
        
        nutrients = ['nitrogen_level', 'phosphorus_level', 'potassium_level']
        nutrient_names = ['Nitrogen', 'Phosphorus', 'Potassium']
        
        for nutrient, name in zip(nutrients, nutrient_names):
            level = soil_data[nutrient].iloc[0] if hasattr(soil_data, 'iloc') else soil_data[nutrient]
            
            if level >= 60:
                status = 'sufficient'
                recommendation = f"{name} levels are adequate for most crops"
            elif level >= 40:
                status = 'moderate'
                recommendation = f"{name} levels are moderate. Consider light fertilization"
            elif level >= 20:
                status = 'low'
                recommendation = f"{name} levels are low. Fertilization recommended"
            else:
                status = 'deficient'
                recommendation = f"{name} levels are very low. Immediate fertilization required"
            
            analysis[nutrient] = {
                'level': float(level),
                'status': status,
                'recommendation': recommendation
            }
        
        return analysis
    
    def _analyze_ph(self, soil_data):
        """
        Analyze pH levels and provide recommendations.
        """
        ph = soil_data['ph_level'].iloc[0] if hasattr(soil_data, 'iloc') else soil_data['ph_level']
        
        if 6.0 <= ph <= 7.0:
            status = 'optimal'
            recommendation = "pH is in the optimal range for most crops"
        elif 5.5 <= ph < 6.0:
            status = 'slightly_acidic'
            recommendation = "pH is slightly acidic. Consider adding lime for crops requiring neutral pH"
        elif 7.0 < ph <= 7.5:
            status = 'slightly_alkaline'
            recommendation = "pH is slightly alkaline. Consider adding sulfur for acid-loving crops"
        elif ph < 5.5:
            status = 'acidic'
            recommendation = "pH is too acidic. Add lime to raise pH"
        else:
            status = 'alkaline'
            recommendation = "pH is too alkaline. Add sulfur or organic matter to lower pH"
        
        return {
            'level': float(ph),
            'status': status,
            'recommendation': recommendation
        }
    
    def _generate_recommendations(self, soil_data, health_score):
        """
        Generate specific recommendations for soil improvement.
        """
        recommendations = []
        
        # pH recommendations
        ph = soil_data['ph_level'].iloc[0] if hasattr(soil_data, 'iloc') else soil_data['ph_level']
        if ph < 6.0:
            recommendations.append({
                'type': 'pH Adjustment',
                'priority': 'high',
                'action': 'Add agricultural lime to raise pH',
                'amount': f"Apply {max(0, (6.5 - ph) * 1000):.0f} kg/hectare of lime"
            })
        elif ph > 7.5:
            recommendations.append({
                'type': 'pH Adjustment',
                'priority': 'high',
                'action': 'Add sulfur or organic matter to lower pH',
                'amount': f"Apply {max(0, (ph - 6.5) * 500):.0f} kg/hectare of sulfur"
            })
        
        # Nutrient recommendations
        nitrogen = soil_data['nitrogen_level'].iloc[0] if hasattr(soil_data, 'iloc') else soil_data['nitrogen_level']
        phosphorus = soil_data['phosphorus_level'].iloc[0] if hasattr(soil_data, 'iloc') else soil_data['phosphorus_level']
        potassium = soil_data['potassium_level'].iloc[0] if hasattr(soil_data, 'iloc') else soil_data['potassium_level']
        
        if nitrogen < 40:
            recommendations.append({
                'type': 'Fertilization',
                'priority': 'high',
                'action': 'Apply nitrogen fertilizer',
                'amount': f"Apply {max(0, (60 - nitrogen) * 2):.0f} kg/hectare of nitrogen"
            })
        
        if phosphorus < 30:
            recommendations.append({
                'type': 'Fertilization',
                'priority': 'medium',
                'action': 'Apply phosphorus fertilizer',
                'amount': f"Apply {max(0, (50 - phosphorus) * 1.5):.0f} kg/hectare of phosphorus"
            })
        
        if potassium < 40:
            recommendations.append({
                'type': 'Fertilization',
                'priority': 'medium',
                'action': 'Apply potassium fertilizer',
                'amount': f"Apply {max(0, (60 - potassium) * 1.2):.0f} kg/hectare of potassium"
            })
        
        # Organic matter recommendations
        organic_matter = soil_data['organic_matter'].iloc[0] if hasattr(soil_data, 'iloc') else soil_data['organic_matter']
        if organic_matter < 3:
            recommendations.append({
                'type': 'Soil Improvement',
                'priority': 'high',
                'action': 'Add organic matter',
                'amount': f"Apply {max(0, (5 - organic_matter) * 2000):.0f} kg/hectare of compost or manure"
            })
        
        # General recommendations based on health score
        if health_score < 40:
            recommendations.append({
                'type': 'General',
                'priority': 'high',
                'action': 'Conduct comprehensive soil testing',
                'amount': "Test soil every 6 months"
            })
        
        if health_score < 60:
            recommendations.append({
                'type': 'General',
                'priority': 'medium',
                'action': 'Implement crop rotation',
                'amount': "Rotate crops to improve soil structure"
            })
        
        return recommendations
    
    def cluster_soil_types(self, soil_data_list):
        """
        Cluster soil samples into different soil types.
        """
        try:
            # Prepare data
            features = []
            for soil_data in soil_data_list:
                if isinstance(soil_data, dict):
                    soil_data = pd.DataFrame([soil_data])
                
                feature_vector = [
                    soil_data['ph_level'].iloc[0] if hasattr(soil_data, 'iloc') else soil_data['ph_level'],
                    soil_data['nitrogen_level'].iloc[0] if hasattr(soil_data, 'iloc') else soil_data['nitrogen_level'],
                    soil_data['phosphorus_level'].iloc[0] if hasattr(soil_data, 'iloc') else soil_data['phosphorus_level'],
                    soil_data['potassium_level'].iloc[0] if hasattr(soil_data, 'iloc') else soil_data['potassium_level'],
                    soil_data['organic_matter'].iloc[0] if hasattr(soil_data, 'iloc') else soil_data['organic_matter'],
                    soil_data['moisture_content'].iloc[0] if hasattr(soil_data, 'iloc') else soil_data['moisture_content']
                ]
                features.append(feature_vector)
            
            features = np.array(features)
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Apply PCA for visualization
            features_pca = self.pca.fit_transform(features_scaled)
            
            # Determine optimal number of clusters
            best_k = self._find_optimal_clusters(features_scaled)
            
            # Perform clustering
            self.cluster_model = KMeans(n_clusters=best_k, random_state=42)
            cluster_labels = self.cluster_model.fit_predict(features_scaled)
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(features_scaled, cluster_labels)
            
            return {
                'cluster_labels': cluster_labels.tolist(),
                'optimal_clusters': best_k,
                'silhouette_score': silhouette_avg,
                'pca_features': features_pca.tolist(),
                'cluster_centers': self.cluster_model.cluster_centers_.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error in soil clustering: {str(e)}")
            raise
    
    def _find_optimal_clusters(self, features, max_k=10):
        """
        Find optimal number of clusters using elbow method and silhouette score.
        """
        inertias = []
        silhouette_scores = []
        k_range = range(2, min(max_k + 1, len(features)))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(features)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(features, cluster_labels))
        
        # Find elbow point
        if len(inertias) > 1:
            # Calculate second derivative to find elbow
            second_derivatives = np.diff(np.diff(inertias))
            elbow_k = k_range[np.argmax(second_derivatives) + 1]
        else:
            elbow_k = 2
        
        # Find best silhouette score
        best_silhouette_k = k_range[np.argmax(silhouette_scores)]
        
        # Return the average of both methods
        return int((elbow_k + best_silhouette_k) / 2)
    
    def train_model(self, soil_data_list):
        """
        Train the soil analysis model on historical data.
        """
        try:
            # Perform clustering
            clustering_result = self.cluster_soil_types(soil_data_list)
            
            self.is_trained = True
            
            # Save model
            self.save_model()
            
            logger.info("Soil analysis model trained successfully!")
            logger.info(f"Optimal clusters: {clustering_result['optimal_clusters']}")
            logger.info(f"Silhouette score: {clustering_result['silhouette_score']:.4f}")
            
            return clustering_result
            
        except Exception as e:
            logger.error(f"Error training soil analysis model: {str(e)}")
            raise
    
    def save_model(self):
        """
        Save the trained model and preprocessing objects.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save scaler
            scaler_file = os.path.join(self.model_path, f'soil_scaler_{timestamp}.joblib')
            joblib.dump(self.scaler, scaler_file)
            
            # Save PCA
            pca_file = os.path.join(self.model_path, f'soil_pca_{timestamp}.joblib')
            joblib.dump(self.pca, pca_file)
            
            # Save cluster model
            if self.cluster_model is not None:
                cluster_file = os.path.join(self.model_path, f'soil_cluster_{timestamp}.joblib')
                joblib.dump(self.cluster_model, cluster_file)
            
            # Save metadata
            metadata = {
                'is_trained': self.is_trained,
                'timestamp': timestamp,
                'soil_health_thresholds': self.soil_health_thresholds
            }
            
            metadata_file = os.path.join(self.model_path, f'soil_metadata_{timestamp}.joblib')
            joblib.dump(metadata, metadata_file)
            
            logger.info(f"Soil analysis model saved successfully: {scaler_file}")
            
        except Exception as e:
            logger.error(f"Error saving soil analysis model: {str(e)}")
            raise
    
    def load_model(self, timestamp=None):
        """
        Load the most recent trained model.
        """
        try:
            if timestamp is None:
                # Find the most recent model
                scaler_files = [f for f in os.listdir(self.model_path) if f.startswith('soil_scaler_')]
                if not scaler_files:
                    logger.warning("No trained soil analysis model found")
                    return False
                
                # Extract timestamps and get the latest
                timestamps = [f.replace('soil_scaler_', '').replace('.joblib', '') for f in scaler_files]
                timestamp = max(timestamps)
            
            # Load scaler
            scaler_file = os.path.join(self.model_path, f'soil_scaler_{timestamp}.joblib')
            self.scaler = joblib.load(scaler_file)
            
            # Load PCA
            pca_file = os.path.join(self.model_path, f'soil_pca_{timestamp}.joblib')
            self.pca = joblib.load(pca_file)
            
            # Load cluster model if exists
            cluster_file = os.path.join(self.model_path, f'soil_cluster_{timestamp}.joblib')
            if os.path.exists(cluster_file):
                self.cluster_model = joblib.load(cluster_file)
            
            # Load metadata
            metadata_file = os.path.join(self.model_path, f'soil_metadata_{timestamp}.joblib')
            metadata = joblib.load(metadata_file)
            
            self.is_trained = metadata['is_trained']
            self.soil_health_thresholds = metadata['soil_health_thresholds']
            
            logger.info(f"Soil analysis model loaded successfully: {scaler_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading soil analysis model: {str(e)}")
            return False
