from flask_caching import Cache
import redis
import os
import logging

logger = logging.getLogger(__name__)

def init_cache(app):
    """
    Initialize caching for the application.
    """
    try:
        # Configure cache based on environment
        if app.config.get('CACHE_TYPE') == 'redis':
            cache = Cache(app, config={
                'CACHE_TYPE': 'redis',
                'CACHE_REDIS_URL': app.config.get('CACHE_REDIS_URL', 'redis://localhost:6379/0'),
                'CACHE_DEFAULT_TIMEOUT': app.config.get('CACHE_DEFAULT_TIMEOUT', 300)
            })
        else:
            # Fallback to simple cache
            cache = Cache(app, config={
                'CACHE_TYPE': 'simple',
                'CACHE_DEFAULT_TIMEOUT': 300
            })
        
        app.cache = cache
        logger.info("Cache initialized successfully")
        return cache
        
    except Exception as e:
        logger.error(f"Error initializing cache: {str(e)}")
        # Fallback to no cache
        app.cache = None
        return None

def cache_key_prefix(key_prefix):
    """
    Decorator to add prefix to cache keys.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key with prefix
            cache_key = f"{key_prefix}:{func.__name__}:{hash(str(args) + str(kwargs))}"
            return cache_key
        return wrapper
    return decorator

class CacheManager:
    """
    Advanced cache management for ML models and data.
    """
    
    def __init__(self, app):
        self.app = app
        self.cache = app.cache
        self.redis_client = None
        
        if app.config.get('CACHE_TYPE') == 'redis':
            try:
                self.redis_client = redis.from_url(app.config.get('CACHE_REDIS_URL'))
            except Exception as e:
                logger.error(f"Error connecting to Redis: {str(e)}")
    
    def cache_ml_prediction(self, key, prediction_data, timeout=3600):
        """
        Cache ML prediction results.
        """
        if not self.cache:
            return False
        
        try:
            cache_key = f"ml_prediction:{key}"
            self.cache.set(cache_key, prediction_data, timeout=timeout)
            return True
        except Exception as e:
            logger.error(f"Error caching ML prediction: {str(e)}")
            return False
    
    def get_cached_ml_prediction(self, key):
        """
        Get cached ML prediction results.
        """
        if not self.cache:
            return None
        
        try:
            cache_key = f"ml_prediction:{key}"
            return self.cache.get(cache_key)
        except Exception as e:
            logger.error(f"Error getting cached ML prediction: {str(e)}")
            return None
    
    def cache_soil_analysis(self, soil_data_hash, analysis_data, timeout=1800):
        """
        Cache soil analysis results.
        """
        if not self.cache:
            return False
        
        try:
            cache_key = f"soil_analysis:{soil_data_hash}"
            self.cache.set(cache_key, analysis_data, timeout=timeout)
            return True
        except Exception as e:
            logger.error(f"Error caching soil analysis: {str(e)}")
            return False
    
    def get_cached_soil_analysis(self, soil_data_hash):
        """
        Get cached soil analysis results.
        """
        if not self.cache:
            return None
        
        try:
            cache_key = f"soil_analysis:{soil_data_hash}"
            return self.cache.get(cache_key)
        except Exception as e:
            logger.error(f"Error getting cached soil analysis: {str(e)}")
            return None
    
    def cache_crop_recommendations(self, conditions_hash, recommendations, timeout=1800):
        """
        Cache crop recommendations.
        """
        if not self.cache:
            return False
        
        try:
            cache_key = f"crop_recommendations:{conditions_hash}"
            self.cache.set(cache_key, recommendations, timeout=timeout)
            return True
        except Exception as e:
            logger.error(f"Error caching crop recommendations: {str(e)}")
            return False
    
    def get_cached_crop_recommendations(self, conditions_hash):
        """
        Get cached crop recommendations.
        """
        if not self.cache:
            return None
        
        try:
            cache_key = f"crop_recommendations:{conditions_hash}"
            return self.cache.get(cache_key)
        except Exception as e:
            logger.error(f"Error getting cached crop recommendations: {str(e)}")
            return None
    
    def cache_yield_prediction(self, conditions_hash, yield_data, timeout=1800):
        """
        Cache yield predictions.
        """
        if not self.cache:
            return False
        
        try:
            cache_key = f"yield_prediction:{conditions_hash}"
            self.cache.set(cache_key, yield_data, timeout=timeout)
            return True
        except Exception as e:
            logger.error(f"Error caching yield prediction: {str(e)}")
            return False
    
    def get_cached_yield_prediction(self, conditions_hash):
        """
        Get cached yield predictions.
        """
        if not self.cache:
            return None
        
        try:
            cache_key = f"yield_prediction:{conditions_hash}"
            return self.cache.get(cache_key)
        except Exception as e:
            logger.error(f"Error getting cached yield prediction: {str(e)}")
            return None
    
    def invalidate_cache_pattern(self, pattern):
        """
        Invalidate cache entries matching a pattern.
        """
        if not self.redis_client:
            return False
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
            return True
        except Exception as e:
            logger.error(f"Error invalidating cache pattern {pattern}: {str(e)}")
            return False
    
    def get_cache_stats(self):
        """
        Get cache statistics.
        """
        if not self.redis_client:
            return {"error": "Redis not available"}
        
        try:
            info = self.redis_client.info()
            return {
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits"),
                "keyspace_misses": info.get("keyspace_misses"),
                "hit_rate": info.get("keyspace_hits", 0) / max(1, info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0))
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {"error": str(e)}

def generate_cache_key(*args, **kwargs):
    """
    Generate a consistent cache key from arguments.
    """
    import hashlib
    import json
    
    # Convert arguments to string and hash
    key_data = str(args) + str(sorted(kwargs.items()))
    return hashlib.md5(key_data.encode()).hexdigest()

def cache_ml_model_prediction(timeout=3600):
    """
    Decorator to cache ML model predictions.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = generate_cache_key(*args, **kwargs)
            
            # Try to get from cache
            from flask import current_app
            cache_manager = getattr(current_app, 'cache_manager', None)
            
            if cache_manager:
                cached_result = cache_manager.get_cached_ml_prediction(cache_key)
                if cached_result:
                    return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            if cache_manager:
                cache_manager.cache_ml_prediction(cache_key, result, timeout)
            
            return result
        return wrapper
    return decorator
