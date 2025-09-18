# detection/config.py - Simplified configuration for detection

import os
import json
import logging

logger = logging.getLogger('molecular OCSR services')

# Load configuration from mcp-config.json
def load_config(config_path="./config.json"):
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), config_path)
    default_config = {
        "model_dirs": [
            ".",  # Current directory
            "./models",  # Models subdirectory
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"),  # Absolute path to models
        ],
        "default_output_dir": "./output",
        "save_results": True,
        "default_models": {
            "detection": "moldet_yolo11l_640_general.pt"
        }
    }
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Extract model configuration
            if "modelConfig" in config_data:
                model_config = config_data["modelConfig"]
                if "modelDirs" in model_config:
                    default_config["model_dirs"] = model_config["modelDirs"]
                if "defaultModels" in model_config:
                    default_config["default_models"] = model_config["defaultModels"]
            
            # Extract output configuration
            if "outputConfig" in config_data:
                output_config = config_data["outputConfig"]
                if "defaultOutputDir" in output_config:
                    default_config["default_output_dir"] = output_config["defaultOutputDir"]
                if "saveResults" in output_config:
                    default_config["save_results"] = output_config["saveResults"]
            
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {str(e)}")
            logger.info("Using default configuration")
    else:
        logger.warning(f"Configuration file not found: {config_path}")
        logger.info("Using default configuration")
    
    return default_config

# Global configuration
CONFIG = load_config()