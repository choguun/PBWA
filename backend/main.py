import os
from dotenv import load_dotenv
import logging # Import logging
from fastapi.responses import StreamingResponse # Import StreamingResponse
import json # Import json

# --- Logging Configuration ---
log_file = "agent_log.txt" # Log file in the project root (PBWA/)
logging.basicConfig(
    level=logging.INFO, # Log INFO level messages and above (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file), # Log to a file
        logging.StreamHandler() # Also log to console (optional, remove if you only want file)
    ]
)
logger = logging.getLogger(__name__)
logger.info("Logging configured.")
# --- End Logging Configuration ---

# Load environment variables from .env file
load_dotenv()
logger.info(".env file loaded.")

from fastapi import FastAPI
from .agents import MultiStepAgent
from .schemas import QueryRequest, QueryResponse

app = FastAPI()

logger.info("Instantiating MultiStepAgent...")
agent = MultiStepAgent()
logger.info("MultiStepAgent instantiated.")

@app.post("/invoke") # Remove response_model, it's handled by streaming
async def invoke_agent_stream(request: QueryRequest):
    """Receives a user query and streams back agent progress via SSE."""
    logger.info(f"Received streaming request for query: {request.query}")
    
    async def event_generator():
        try:
            async for sse_event_str in agent.astream_events(request.query):
                yield sse_event_str
        except Exception as e:
            # Log the error during streaming generation
            logger.error(f"Error generating stream for query '{request.query}': {e}", exc_info=True)
            # Yield a final error event to the client
            error_data = {"type": "error", "message": f"Stream generation failed: {e}"}
            yield f"event: error\ndata: {json.dumps(error_data)}\n\n"
            yield f"event: end\ndata: {json.dumps({'message': 'Stream ended due to error.'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/")
async def read_root():
    return {"message": "Planning Agent API is running. Use POST /invoke for streaming."}