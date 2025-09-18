# MolDet-MCP-Server

A simplified MCP server for chemical structure detection and recognition using deep learning models.

## Project Structure

This project focuses on two core functionalities:

1. **Detection** - Locate molecular structures in images using YOLO models
2. **Recognition** - Convert detected molecular structures to SMILES strings using MolScribe

### Core Modules
- `detection/` - Molecular structure detection using YOLO
- `recognition/` - Molecular recognition and SMILES generation using MolScribe

### Key Files
- `server.py` - MCP server definition

## Features

### Molecular Detection
- Detect and locate molecular structures in single images
- Two pre-trained YOLO models:
  - `moldet_yolo11l_640_general.pt` - For general molecular images
  - `moldet_yolo11l_960_doc.pt` - For document/PDF processing

### Molecular Recognition (with MolScribe)
- Convert detected molecular structures to SMILES strings
- Support for both image and PDF document processing

## Installation

### Basic Dependencies
```bash
uv init --python 3.10
uv venv 
uv pip install -r requirements.txt
```
Note: MolScribe has additional dependencies including PyTorch and RDKit, I have not test it on Windows.

## Usage

### Start the Server (Standalone HTTP server)
```bash
source .venv/bin/activate
python server.py
```

The server will be available at `http://0.0.0.0:8000/mcp`

You can test it with:
```bash
python test_server.py
```

### Use with MCP client (Claude Code, Qwen Code, etc.)
When using with MCP clients like Claude Code or Qwen Code, the server is automatically managed by the client. Add this configuration to your MCP configuration file:

```json
{
"mcpServers": {
    "molecular-detection-recognition-server": {
        "httpUrl": "http://0.0.0.0:8000/mcp",
        "timeout": 600000
        }
    }
}
```

When the MCP client connects to the server, it will automatically start the server process and communicate with it through HTTP.

### Direct HTTP Access
If you want to access the server directly via HTTP (for testing or integration with other services), start the server, then you can make HTTP requests to:

```
POST http://0.0.0.0:8000/mcp
```

The HTTP endpoint accepts JSON-RPC 2.0 requests according to the MCP specification.
### MCP Tools Available
1. `detect_molecules_with_crops` - Detect molecular structures in images and save cropped molecules
2. `process_image_for_molecules_and_smiles` - Detect molecules in an image and convert to SMILES
3. `process_pdf_for_molecules_and_smiles` - Process PDF documents, detect molecules on all pages, and convert to SMILES

## File Naming and Traceability

The system implements a traceable file naming scheme for intermediate files:
- Original files are preserved with their original names
- Cropped molecules are named with the pattern: `{original_filename}[_page{page_number}]_molecule_{detection_index}_{molecule_index}.jpg`
- Results are saved in both JSON and CSV formats with traceability information

## Configuration

The configuration in `config.json` includes:
- Model directories
- Default models for general and document processing
- Output directory settings

## Requirements

See `requirements.txt` for the complete list of dependencies.

Detection model weights are sourced from [MolDet](https://huggingface.co/UniParser/MolDet).
Recognition model is based on [MolScribe](https://github.com/thomas0809/MolScribe).