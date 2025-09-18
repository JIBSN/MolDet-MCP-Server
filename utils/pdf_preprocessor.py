# pdf_processor.py - PDF processing functionality

import os
import base64
import logging
import tempfile
from typing import Dict, List, Any
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

def process_pdf_to_images(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Process a PDF file and convert each page to an image
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of dictionaries containing page images and metadata
    """
    try:
        # Check if PDF file exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Open the PDF
        pdf_document = fitz.open(pdf_path)
        num_pages = pdf_document.page_count
        
        # Process each page
        page_images = []
        for page_num in range(num_pages):
            # Get the page
            page = pdf_document[page_num]
            
            # Render page to image (300 DPI)
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
            
            # Convert to base64
            img_data = pix.tobytes("png")
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            
            page_images.append({
                "page_number": page_num + 1,
                "image_data": img_base64,
                "width": pix.width,
                "height": pix.height
            })
        
        # Close the PDF
        pdf_document.close()
        
        return page_images
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
        raise