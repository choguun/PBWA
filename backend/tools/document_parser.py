# backend/tools/document_parser.py
import os
import logging
import json
from typing import List, Dict, Any
import asyncio

try:
    from langchain_upstage import UpstageDocumentParseLoader
    from langchain_core.documents import Document
    UPSTAGE_AVAILABLE = True
except ImportError:
    UpstageDocumentParseLoader = None
    Document = None # type: ignore
    UPSTAGE_AVAILABLE = False

logger = logging.getLogger(__name__)

async def parse_document_upstage(file_path: str) -> str:
    """Parses a local document using Upstage Document AI (OCR forced).

    Args:
        file_path: The relative path to the local document file.

    Returns:
        A JSON string representing the parsed pages (list of dicts with page number and content)
        or an error message string.
    """
    logger.info(f"Attempting to parse document using Upstage: {file_path}")

    if not UPSTAGE_AVAILABLE:
        error_msg = "Error: langchain-upstage library is not installed. Cannot parse document."
        logger.error(error_msg)
        return json.dumps({"error": error_msg})

    if "UPSTAGE_API_KEY" not in os.environ:
        error_msg = "Error: UPSTAGE_API_KEY environment variable not set."
        logger.error(error_msg)
        return json.dumps({"error": error_msg})

    # Basic check for path existence relative to workspace root (adjust if needed)
    # Assumes backend runs from workspace root or has access via relative path
    full_path = os.path.abspath(file_path) # Get absolute path for clarity in logs/errors
    if not os.path.exists(full_path):
        error_msg = f"Error: File not found at path: {file_path} (Resolved to: {full_path})"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})

    try:
        logger.info(f"Initializing UpstageDocumentParseLoader for: {full_path}")
        # Using force OCR as requested
        loader = UpstageDocumentParseLoader(file_path=full_path, ocr="force")

        # Load documents (consider lazy_load for large files if memory is an issue)
        # Note: loader.load() might be blocking; if performance is critical,
        # consider running in a separate thread using asyncio.to_thread (Python 3.9+)
        logger.info("Loading document content via Upstage API...")
        pages: List[Document] = await asyncio.to_thread(loader.load)
        logger.info(f"Successfully parsed {len(pages)} pages from {file_path}.")

        # Format the output as JSON string list [{page: content}, ...]
        output_list = []
        for i, page in enumerate(pages):
            output_list.append({
                "page": i + 1,
                "content": page.page_content,
                # Optionally include metadata if needed
                # "metadata": page.metadata
            })

        return json.dumps(output_list, indent=2)

    except Exception as e:
        error_msg = f"Error parsing document '{file_path}' with Upstage: {e}"
        logger.exception(error_msg) # Log full traceback
        return json.dumps({"error": error_msg}) 