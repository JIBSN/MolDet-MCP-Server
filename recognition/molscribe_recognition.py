# molscribe_integration.py
# MolScribe integration for molecular structure recognition and SMILES generation

import os
import base64
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple, Any

import cv2
import numpy as np
from PIL import Image
import io

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("molscribe_recognition")

# Global variables
MOLSCRIBE_AVAILABLE = False

# Try to import MolScribe
try:
    from molscribe import MolScribe
    MOLSCRIBE_AVAILABLE = True
    logger.info("MolScribe successfully imported")
except ImportError:
    logger.warning("MolScribe not available. Please install with 'pip install molscribe'")

# MolScribe model instance
_molscribe_model = None


def get_molscribe_model() -> Any:
    """
    Load and return the MolScribe model instance.
    Uses singleton pattern to avoid loading the model multiple times.
    
    Returns:
        MolScribe model instance or None if MolScribe is not available
    """
    global _molscribe_model
    
    if not MOLSCRIBE_AVAILABLE:
        logger.error("MolScribe is not available. Cannot load model.")
        return None
    
    if _molscribe_model is None:
        try:
            logger.info("Loading MolScribe model...")
            _molscribe_model = MolScribe("swin_base_char_aux_1m.pth")
            logger.info("MolScribe model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading MolScribe model: {str(e)}")
            return None
    
    return _molscribe_model


def load_image_for_molscribe(image_path_or_base64: str) -> Optional[np.ndarray]:
    """
    Load an image from a file path or base64 string for MolScribe processing.
    
    Args:
        image_path_or_base64: File path to the image or base64 encoded image string
        
    Returns:
        numpy.ndarray: Image as a numpy array or None if loading fails
    """
    try:
        # Check if input is a base64 string
        if image_path_or_base64.startswith("data:image") or ";base64," in image_path_or_base64:
            # Extract the base64 part if it's a data URL
            if ";base64," in image_path_or_base64:
                image_path_or_base64 = image_path_or_base64.split(";base64,")[1]
            
            # Decode base64 to bytes
            image_bytes = base64.b64decode(image_path_or_base64)
            
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Convert from BGR to RGB (MolScribe expects RGB)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # Load from file path
            if not os.path.exists(image_path_or_base64):
                logger.error(f"Image file not found: {image_path_or_base64}")
                return None
            
            # Read image using PIL and convert to numpy array
            pil_image = Image.open(image_path_or_base64)
            image = np.array(pil_image.convert('RGB'))
        
        return image
    
    except Exception as e:
        logger.error(f"Error loading image: {str(e)}")
        return None


def recognize_molecule_to_smiles(image_path_or_base64: str, is_path: bool=True) -> Dict[str, Any]:
    """
    Recognize a molecular structure in an image and convert it to SMILES.
    
    Args:
        image_path_or_base64: File path to the image or base64 encoded image string
        
    Returns:
        Dict containing recognition results including SMILES, confidence, etc.
    """
    result = {
        "success": False,
        "smiles": None,
        "molfile": None,
        "confidence": None,
    }
    
    # Check if MolScribe is available
    if not MOLSCRIBE_AVAILABLE:
        result["error"] = "MolScribe is not available. Please install with 'pip install molscribe'"
        return result
    
    # Get MolScribe model
    model = get_molscribe_model()
    if model is None:
        result["error"] = "Failed to load MolScribe model"
        return result
    
    # Load image
    image = load_image_for_molscribe(image_path_or_base64)
    if image is None:
        result["error"] = "Failed to load image"
        return result
    
    try:
        # Recognize molecular structure
        logger.info("Recognizing molecular structure with MolScribe...")
        prediction = model.predict_image(image)
        
        # Extract results
        result["success"] = True
        result["smiles"] = prediction.get("smiles")
        result["molfile"] = prediction.get("molfile")
        result["confidence"] = prediction.get("confidence")
        result["atoms"] = prediction.get("atoms")
        result["bonds"] = prediction.get("bonds")
        
        logger.info(f"Successfully recognized molecule: {result['smiles']}")
        
    except Exception as e:
        logger.error(f"Error recognizing molecule: {str(e)}")
        result["error"] = f"Error recognizing molecule: {str(e)}"
    
    return result


def process_molecule_crops(crop_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Process multiple molecule crops and recognize them to SMILES.
    
    Args:
        crop_paths: List of file paths to molecule crop images
        
    Returns:
        List of dictionaries containing recognition results for each crop
    """
    results = []
    
    for crop_path in crop_paths:
        logger.info(f"Processing molecule crop: {crop_path}")
        result = recognize_molecule_to_smiles(crop_path)
        
        # Add crop path to result
        result["crop_path"] = crop_path
        
        results.append(result)
    
    return results