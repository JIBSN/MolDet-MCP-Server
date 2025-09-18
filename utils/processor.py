# recognition/processor.py - Updated recognition processor

import os
import json
import base64
import logging
import time
import uuid
from typing import Dict, List, Any

# Import from detection module
from detection.detector import detect_molecules_with_crops

# Import PDF processor
from utils.pdf_preprocessor import process_pdf_to_images

# Import configuration
from .config import CONFIG

# Setup logging
logger = logging.getLogger(__name__)

# MolScribe
from recognition.molscribe_recognition import recognize_molecule_to_smiles


def recognize_molecule_to_smiles_tool(
    image_data: str,
    is_path: bool = False
) -> Dict[str, Any]:
    """
    Recognize a molecule in an image and convert it to SMILES string
    
    Args:
        image_data: Base64 encoded image or file path (if is_path=True)
        is_path: Whether image_data is a file path
        
    Returns:
        Dictionary containing recognition results with SMILES string
    """
   
    try:
        # Call MolScribe integration function
        result = recognize_molecule_to_smiles(image_data, is_path=is_path)
        return result
    except Exception as e:
        logger.error(f"Error in recognize_molecule_to_smiles_tool: {str(e)}")
        return {
            "error": f"Failed to recognize molecule and generate SMILES: {str(e)}",
            "source": image_data if is_path else "base64_image"
        }

def process_molecular_detection_and_recognition(
    image_data: str,
    model_name: str,
    confidence: float = 0.5,
    save_detection_results: bool = True,
    save_recognition_results: bool = True,
    output_dir: str = "./output",
    is_path: bool = True,
    source_filename: str = None,
    is_pdf: bool = False,
    page_number: int = None
) -> Dict[str, Any]:
    """
    Process an image to detect molecules and generate SMILES strings
    
    Args:
        image_data: Base64 encoded image or file path (if is_path=True)
        model_name: Detection model name for molecule detection
        confidence: Detection confidence threshold
        save_detection_results: Whether to save molecular detection results
        save_recognition_results: Whether to save molecular recognition results
        output_dir: Directory to save results
        is_path: Whether image_data is a file path
        source_filename: Original filename for traceability
        
    Returns:
        Dictionary containing detection results with SMILES strings
    """

    try:
        
        # First detect molecules
        detection_result = detect_molecules_with_crops(
            image_data=image_data,
            model_name=model_name,
            confidence=confidence,
            save_results=save_detection_results,
            is_path=is_path,
            output_dir=output_dir,
            source_filename=source_filename,
            page_number=page_number
        )
        
        # Check for errors in detection
        if "error" in detection_result:
            return detection_result
        
        # Get cropped molecules data
        cropped_molecules = detection_result.get("cropped_molecules", [])
        
        if not cropped_molecules:
            return {
                "status": "success",
                "message": "No molecules detected in the image",
                "molecules_with_smiles": [],
                "total_molecules": 0,
                "source": source_filename if source_filename else ("base64_image" if not is_path else image_data)
            }
        
        # Process each detected molecule with MolScribe
        molecules_with_smiles = []
        
        for molecule in cropped_molecules:
            crop_path = molecule.get("crop_path")
            if not crop_path or not os.path.exists(crop_path):
                # Skip if crop path doesn't exist
                molecule["smiles"] = None
                molecule["smiles_error"] = "Molecule crop not available"
                molecules_with_smiles.append(molecule)
                continue
            
            # Recognize SMILES for this molecule crop
            smiles_result = recognize_molecule_to_smiles_tool(crop_path, is_path=True)
            if "error" in smiles_result:
                molecule["smiles"] = None
                molecule["smiles_error"] = smiles_result["error"]
            else:
                molecule["smiles"] = smiles_result.get("smiles")
            
            molecules_with_smiles.append(molecule)

        # Generate base name for output files
        if source_filename:
            base_name = os.path.splitext(os.path.basename(source_filename))[0]
        else:
            base_name = f"image_{int(time.time())}"
        
        # Save results with SMILES if requested
        if save_recognition_results:
            # Save JSON with SMILES
            if page_number is not None:
                json_filename = os.path.join(output_dir, f"{base_name}_page{page_number}_results_with_smiles.json")
            else:
                json_filename = os.path.join(output_dir, f"{base_name}_results_with_smiles.json")
            with open(json_filename, 'w') as f:
                json.dump({
                    "source": source_filename if source_filename else ("base64_image" if not is_path else image_data),
                    "molecules_with_smiles": molecules_with_smiles,
                    "total_molecules": len(molecules_with_smiles)
                }, f, indent=2)
            logger.info(f"Saved SMILES results to {json_filename}")
            
            # Save CSV with SMILES
            if page_number is not None:
                csv_filename = os.path.join(output_dir, f"{base_name}_page{page_number}_results_with_smiles.csv")
            else:
                csv_filename = os.path.join(output_dir, f"{base_name}_results_with_smiles.csv")
            with open(csv_filename, 'w') as f:
                f.write("id,source_file,page_number,x1,y1,x2,y2,confidence,crop_path,smiles\n")
                for mol in molecules_with_smiles:
                    smiles = mol.get("smiles", "").replace('"', '""')  # Escape quotes for CSV
                    box = mol["box"]
                    f.write(f"{mol['id']},{mol.get('source_file', '')},{mol.get('page_number', '')},{box[0]},{box[1]},{box[2]},{box[3]},{mol['confidence']},{mol['crop_path']},\"{smiles}\",\n")
            logger.info(f"Saved SMILES results to {csv_filename}")
        
        return {
            "status": "success",
            "molecules_with_smiles": molecules_with_smiles,
            "total_molecules": len(molecules_with_smiles),
            "source": source_filename if source_filename else ("base64_image" if not is_path else image_data),
            "output_files": {
                "json": json_filename if save_recognition_results else None,
                "csv": csv_filename if save_recognition_results else None
            }
        }
    except Exception as e:
        logger.error(f"Error in process_molecular_detection_and_recognition: {str(e)}")
        return {
            "error": f"Failed to process image for molecules and SMILES: {str(e)}",
            "source": source_filename if source_filename else ("base64_image" if not is_path else image_data)
        }
        
def process_image_for_molecules_and_smiles(
    image_path: str,
    model_name: str = None,
    confidence: float = 0.5,
    save_results: bool = True,
    output_dir: str = "./output",
) -> Dict[str, Any]:
    """
    Process an image file to detect molecules on it and generate SMILES strings
    
    Args:
        image_path: Path to the an image file
        model_name: Detection model name for molecule detection
        confidence: Detection confidence threshold
        save_results: Whether to save results to disk
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing detection results with SMILES strings for all pages
    """
    
    # Check if image file exists
    if not os.path.exists(image_path):
        return {"error": f"image file not found: {image_path}"}
    
    # Use default model if not specified
    if model_name is None:
        model_name = CONFIG.get("default_models", {}).get("general", "moldet_yolo11l_640_general.pt")
    
    # Get PDF filename for traceability
    image_filename = os.path.basename(image_path)
    image_name = os.path.splitext(image_filename)[0]
    
    # Create output directory with PDF name
    image_output_dir = os.path.join(output_dir, image_name)
    if save_results and not os.path.exists(image_output_dir):
        os.makedirs(image_output_dir)
        
    # Process image
    image_result = process_molecular_detection_and_recognition(
        image_data=image_path,
        model_name=model_name,
        confidence=confidence,
        save_detection_results=save_results,
        save_recognition_results=save_results,
        output_dir=image_output_dir,
        source_filename=image_filename,
    )
    
    return image_result    

def process_pdf_for_molecules_and_smiles(
    pdf_path: str,
    model_name: str = None,
    confidence: float = 0.5,
    save_results: bool = True,
    output_dir: str = "./output"
) -> Dict[str, Any]:
    """
    Process a PDF document to detect molecules on all pages and generate SMILES strings
    
    Args:
        pdf_path: Path to the PDF file
        model_name: Detection model name for molecule detection
        confidence: Detection confidence threshold
        save_pdf_results: Whether to save results to disk
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing detection results with SMILES strings for all pages
    """

    try:
        # Check if PDF file exists
        if not os.path.exists(pdf_path):
            return {"error": f"PDF file not found: {pdf_path}"}
        
        # Use default model if not specified
        if model_name is None:
            model_name = CONFIG.get("default_models", {}).get("doc", "moldet_yolo11l_960_doc.pt")
        
        # Get PDF filename for traceability
        pdf_filename = os.path.basename(pdf_path)
        pdf_name = os.path.splitext(pdf_filename)[0]
        
        # Create output directory with PDF name
        pdf_output_dir = os.path.join(output_dir, pdf_name)
        if save_results and not os.path.exists(pdf_output_dir):
            os.makedirs(pdf_output_dir)
        
        # Process PDF to images
        page_images = process_pdf_to_images(pdf_path)
        
        # Process each page
        all_molecules_with_smiles = []
        page_results = []
        
        for page_data in page_images:
            page_num = page_data["page_number"]
            image_data = page_data["image_data"]
            
            # Process page image
            page_result = process_molecular_detection_and_recognition(
                image_data=image_data,
                model_name=model_name,
                confidence=confidence,
                save_detection_results=save_results, # don't save single page intemidate results
                save_recognition_results=False,
                output_dir=pdf_output_dir,
                is_path=False,
                source_filename=pdf_filename,
                is_pdf=True,
                page_number=page_num
            )
            
            # Check for errors
            if "error" in page_result:
                page_results.append({
                    "page_number": page_num,
                    "error": page_result["error"]
                })
                continue
            
            # Add page number to molecules
            molecules = page_result.get("molecules_with_smiles", [])
            for molecule in molecules:
                molecule["page_number"] = page_num
                all_molecules_with_smiles.append(molecule)
            
            # Add to page results
            page_results.append({
                "page_number": page_num,
                "total_molecules": len(molecules),
                "output_files": page_result.get("output_files", {})
            })
        
        # Save combined results if requested
        if save_results:
            # Save JSON with combined results
            combined_json_filename = os.path.join(pdf_output_dir, f"{pdf_name}_combined_results.json")
            with open(combined_json_filename, 'w') as f:
                json.dump({
                    "pdf": pdf_filename,
                    "total_pages": len(page_images),
                    "total_molecules": len(all_molecules_with_smiles),
                    "molecules_with_smiles": all_molecules_with_smiles,
                    "page_results": page_results
                }, f, indent=2)
            logger.info(f"Saved combined results to {combined_json_filename}")
            
            # Save CSV with combined results
            combined_csv_filename = os.path.join(pdf_output_dir, f"{pdf_name}_combined_results.csv")
            with open(combined_csv_filename, 'w') as f:
                f.write("id,pdf_file,page_number,x1,y1,x2,y2,confidence,crop_path,smiles\n")
                for i, mol in enumerate(all_molecules_with_smiles):
                    smiles = mol.get("smiles", "").replace('"', '""')  # Escape quotes for CSV
                    smiles_conf = mol.get("smiles_confidence", "")
                    box = mol["box"]
                    f.write(f"{i},{mol.get('source_file', '')},{mol.get('page_number', '')},{box[0]},{box[1]},{box[2]},{box[3]},{mol['confidence']},{mol['crop_path']},\"{smiles}\",\n")
            logger.info(f"Saved combined results to {combined_csv_filename}")
        
        return {
            "status": "success",
            "pdf": pdf_filename,
            "total_pages": len(page_images),
            "total_molecules": len(all_molecules_with_smiles),
            "molecules_with_smiles": all_molecules_with_smiles,
            "page_results": page_results,
            "output_files": {
                "json": combined_json_filename if save_results else None,
                "csv": combined_csv_filename if save_results else None
            }
        }
    except Exception as e:
        logger.error(f"Error in process_pdf_for_molecules_and_smiles: {str(e)}")
        return {
            "error": f"Failed to process PDF for molecules and SMILES: {str(e)}",
            "pdf_path": pdf_path
        }