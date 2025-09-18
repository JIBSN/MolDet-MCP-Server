# detection/performance.py
# Performance monitoring and optimization for detection module

import os
import time
import json
import logging
from functools import lru_cache
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("moldet_performance.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("moldet-performance")

# Performance monitoring
class PerformanceMonitor:
    """Monitor and log performance metrics for the MolDet-MCP-Server"""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, operation_name: str) -> None:
        """Start timing an operation"""
        self.start_times[operation_name] = time.time()
    
    def end_timer(self, operation_name: str) -> float:
        """End timing an operation and return duration"""
        if operation_name not in self.start_times:
            logger.warning(f"No start time found for operation: {operation_name}")
            return 0.0
        
        duration = time.time() - self.start_times[operation_name]
        
        if operation_name not in self.metrics:
            self.metrics[operation_name] = {
                "count": 0,
                "total_time": 0.0,
                "min_time": float('inf'),
                "max_time": 0.0
            }
        
        self.metrics[operation_name]["count"] += 1
        self.metrics[operation_name]["total_time"] += duration
        self.metrics[operation_name]["min_time"] = min(self.metrics[operation_name]["min_time"], duration)
        self.metrics[operation_name]["max_time"] = max(self.metrics[operation_name]["max_time"], duration)
        
        return duration
    
    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all performance metrics"""
        # Calculate average times
        result = {}
        for op_name, metrics in self.metrics.items():
            result[op_name] = metrics.copy()
            if metrics["count"] > 0:
                result[op_name]["avg_time"] = metrics["total_time"] / metrics["count"]
        
        return result
    
    def log_metrics(self) -> None:
        """Log all performance metrics"""
        metrics = self.get_metrics()
        logger.info("Performance Metrics:")
        for op_name, op_metrics in metrics.items():
            logger.info(f"  {op_name}:")
            logger.info(f"    Count: {op_metrics['count']}")
            logger.info(f"    Avg Time: {op_metrics.get('avg_time', 0):.4f}s")
            logger.info(f"    Min Time: {op_metrics['min_time']:.4f}s")
            logger.info(f"    Max Time: {op_metrics['max_time']:.4f}s")
    
    def save_metrics(self, file_path: str) -> None:
        """Save metrics to a JSON file"""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.get_metrics(), f, indent=2)
            logger.info(f"Metrics saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")


# Create a global instance of the performance monitor
performance_monitor = PerformanceMonitor()


# Result caching
class ResultCache:
    """Cache for detection and recognition results"""
    
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a result from the cache"""
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Dict[str, Any]) -> None:
        """Set a result in the cache"""
        # If cache is full, remove least recently used item
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
            self.cache.pop(oldest_key, None)
            self.access_times.pop(oldest_key, None)
        
        self.cache[key] = value
        self.access_times[key] = time.time()
    
    def clear(self) -> None:
        """Clear the cache"""
        self.cache.clear()
        self.access_times.clear()


# Create a global instance of the result cache
result_cache = ResultCache()


# Decorator for timing functions
def time_function(operation_name: str = None):
    """Decorator to time a function and log its performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            performance_monitor.start_timer(op_name)
            result = func(*args, **kwargs)
            duration = performance_monitor.end_timer(op_name)
            logger.debug(f"{op_name} took {duration:.4f}s")
            return result
        return wrapper
    return decorator


# Decorator for caching function results
def cache_result(key_func=None):
    """Decorator to cache function results"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation based on function name and arguments
                arg_str = str(args) + str(sorted(kwargs.items()))
                cache_key = f"{func.__name__}_{hash(arg_str)}"
            
            # Check cache
            cached_result = result_cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            result_cache.set(cache_key, result)
            logger.debug(f"Cached result for {func.__name__}")
            return result
        return wrapper
    return decorator