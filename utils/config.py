# recognition/config_simple.py - Simplified configuration for recognition

import os
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "save_results": True,
    "default_output_dir": "./output",
    "default_models": {
        "general": "moldet_yolo11l_640_general.pt",
        "doc": "moldet_yolo11l_960_doc.pt"
    },
    "molscribe_available": False,  # Will be updated during runtime check
    "model_directories": [
        "./models",
        "./weights"
    ]
}

# Global configuration variable
CONFIG = {}

def load_config(config_path: str = "./mcp-config.json") -> Dict[str, Any]:
    """
    Load configuration from a JSON file or use defaults
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration settings
    """
    global CONFIG
    
    # Start with default configuration
    config = DEFAULT_CONFIG.copy()
    
    # Try to load from file
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                # Update config with file values
                config.update(file_config)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {str(e)}")
            logger.info("Using default configuration")
    else:
        logger.info(f"Configuration file {config_path} not found. Using default configuration.")
    
    # Update global CONFIG
    CONFIG.update(config)
    
    return CONFIG