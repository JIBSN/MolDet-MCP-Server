# mcp_server/server.py - Simplified MCP server implementation

import os
import sys
import logging
import anyio
from typing import Dict, List, Any
import threading

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import FastMCP
try:
    from fastmcp import FastMCP
except ImportError:
    logger.error("FastMCP is not installed. Install with: pip install fastmcp")
    sys.exit(1)

# Import detection tools
from detection import detect_molecules_with_crops
from utils.processor import (
    process_image_for_molecules_and_smiles,
    process_pdf_for_molecules_and_smiles,
)

def test_connection() -> Dict[str, Any]:
    """
    Test if MCP service is running properly
    
    Returns:
        Status information and available tools
    """
    return {
        "status": "molecules detection and recognition service is running normally",
        "available_tools": [
            "detect_molecules_with_crops",
            "process_image_for_molecules_and_smiles",
            "process_pdf_for_molecules_and_smiles",
            "test_connection"
        ]
    }


mcp = FastMCP(name="Molecule detection and recognition server")

#Register detection tools
mcp.tool()(detect_molecules_with_crops)

# Register recognition tools
mcp.tool()(process_image_for_molecules_and_smiles)
mcp.tool()(process_pdf_for_molecules_and_smiles)
    
# Register server tools
mcp.tool()(test_connection)

logger.info("Created MCP server with registered tools")


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)