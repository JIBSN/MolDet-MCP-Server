# detection/detector.py - Updated detector with intermediate file handling

import os
import base64
import cv2
import time
import json
from io import BytesIO
from typing import List, Dict, Any
import numpy as np
from PIL import Image
import sys
import logging
import contextlib
import uuid

from ultralytics import YOLO

from .config import CONFIG

logger = logging.getLogger('molecular OCSR services')

# Global model cache
models = {}

@contextlib.contextmanager
def redirect_stdout_to_stderr():
    old_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = old_stdout

def load_image(image_source, is_path=False):
    """
    Load image from file path or base64 data
    
    Args:
        image_source: File path or base64 encoded image data
        is_path: Whether image_source is a file path
        
    Returns:
        PIL Image object
    """
    try:
        if is_path:
            # Load image from file path
            if os.path.exists(image_source):
                return Image.open(image_source)
            else:
                raise FileNotFoundError(f"Image file not found: {image_source}")
        else:
            # Load image from base64 data
            image_bytes = base64.b64decode(image_source)
            return Image.open(BytesIO(image_bytes))
    except Exception as e:
        raise ValueError(f"Failed to load image: {str(e)}")

def get_model(model_name: str = "moldet_yolo11l_640_general.pt") -> YOLO:
    """Get or load YOLO model from any of the configured model directories"""
    if model_name in models:
        return models[model_name]
    
    # Try to find the model in any of the configured directories
    model_path = None
    for directory in CONFIG["model_dirs"]:
        potential_path = os.path.join(directory, model_name)
        if os.path.exists(potential_path):
            model_path = potential_path
            break
    
    # Load and cache the model - with stdout redirected
    logger.info(f"Loading model: {model_name} from {model_path}")
    with redirect_stdout_to_stderr():
        models[model_name] = YOLO(model_path)
    return models[model_name]

def detect_molecules_with_crops(
    image_data: str,
    model_name: str,
    confidence: float = 0.5,
    save_results: bool = True,
    is_path: bool = False,
    output_dir: str = "./output",
    source_filename: str = None,
    page_number: int = None
) -> Dict[str, Any]:
    """
    Detect molecules in an image and save cropped molecules into a directory
    
    Args:
        image_data: Base64 encoded image or file path (if is_path=True)
        model_name: YOLO model name
        confidence: Detection confidence threshold
        save_results: Whether to save results to disk
        is_path: Whether image_data is a file path
        output_dir: Directory to save results
        source_filename: Original filename for traceability
        page_number: Page number for PDF documents
        
    Returns:
        Dictionary containing detection results and paths to cropped molecules
    """
    try:
        # Load image (supports path or base64)
        image = load_image(image_data, is_path=is_path)
        
        # Create output directory if it doesn't exist
        if save_results and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Generate base name for output files
        if source_filename:
            base_name = os.path.splitext(os.path.basename(source_filename))[0]
        else:
            base_name = f"image_{int(time.time())}"
            
        if page_number:
            base_name = f"{base_name}_page{page_number}"
        
        # Save original image if it's base64 data
        if not is_path and save_results:
            orig_image_path = os.path.join(output_dir, f"{base_name}_original.jpg")
            image.save(orig_image_path)
            logger.info(f"Saved original image to {orig_image_path}")
        
        # Load model and perform detection - with stdout redirected
        model = get_model(model_name)
        with redirect_stdout_to_stderr():  # Ensure all YOLO outputs go to stderr
            results = model.predict(image, conf=confidence, save=False)  # Don't save automatically
        
        # Format results and crop molecules
        formatted_results = []
        cropped_molecules = []
        
        for result_idx, result in enumerate(results):
            boxes = result.boxes
            detections = []
            
            # Convert PIL image to OpenCV format for cropping
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            for i in range(len(boxes)):
                box = boxes[i]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence_score = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                # Create detection object
                detection = {
                    "box": [x1, y1, x2, y2],
                    "confidence": confidence_score,
                    "class_id": class_id,
                    "class_name": class_name
                }
                
                detections.append(detection)
                
                # Crop molecule if it's a molecule and save if requested
                if save_results and class_name == "molecule":
                    # Crop the molecule
                    x1_int, y1_int, x2_int, y2_int = int(x1), int(y1), int(x2), int(y2)
                    molecule_crop = img_cv[y1_int:y2_int, x1_int:x2_int]
                    
                    # Generate crop filename with traceability
                    crop_filename = f"{base_name}_molecule_{result_idx}_{i}.jpg"
                    crop_path = os.path.join(output_dir, crop_filename)
                    
                    # Save the cropped molecule
                    cv2.imwrite(crop_path, molecule_crop)
                    
                    # Add to cropped molecules list
                    cropped_molecules.append({
                        "id": i,
                        "box": [x1, y1, x2, y2],
                        "confidence": confidence_score,
                        "crop_path": crop_path,
                        "source_file": source_filename,
                        "page_number": page_number
                    })
            
            formatted_results.append({
                "detections": detections,
                "image_shape": result.orig_shape
            })
        
        # Save detection results as JSON
        if save_results:
            json_filename = os.path.join(output_dir, f"{base_name}_detection_results.json")
            with open(json_filename, 'w') as f:
                json.dump({
                    "source_file": source_filename,
                    "page_number": page_number,
                    "detections": formatted_results,
                    "cropped_molecules": cropped_molecules,
                    "model_used": model_name,
                    "total_detections": sum(len(r["detections"]) for r in formatted_results),
                    "total_molecules": len([d for r in formatted_results for d in r["detections"] if d["class_name"] == "molecule"])
                }, f, indent=2)
            logger.info(f"Saved detection results to {json_filename}")
        
        return {
            "results": formatted_results,
            "model_used": model_name,
            "total_detections": sum(len(r["detections"]) for r in formatted_results),
            "total_molecules": len([d for r in formatted_results for d in r["detections"] if d["class_name"] == "molecule"]),
            "source": source_filename if source_filename else ("base64_image" if not is_path else image_data),
            "cropped_molecules": cropped_molecules
        }
    except Exception as e:
        logger.error(f"Error in detect_molecules_with_crops: {str(e)}")
        return {
            "error": f"Failed to detect molecules: {str(e)}",
            "source": source_filename if source_filename else ("base64_image" if not is_path else image_data)
        }