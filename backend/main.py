import os
from dotenv import load_dotenv
import logging # Import logging
from fastapi.responses import StreamingResponse # Import StreamingResponse
import json # Import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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

from .agents import ResearchPipeline
from .schemas import UserProfilePayload

app = FastAPI()

# Configure CORS
origins = [
    "http://localhost:3000",  # Allow frontend origin
    # Add any other origins as needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the agent (ensure it handles errors during init)
try:
    agent = ResearchPipeline()
    logger.info("ResearchPipeline initialized successfully.")
except Exception as e:
    logger.critical(f"Fatal error initializing ResearchPipeline: {e}", exc_info=True)
    # Optionally, prevent app startup or set agent to None
    agent = None 

@app.post("/invoke")
async def invoke_agent(payload: UserProfilePayload):
    if agent is None:
        logger.error("/invoke called but agent failed to initialize.")
        raise HTTPException(status_code=500, detail="Agent initialization failed. Check server logs.")
        
    user_query = payload.user_query
    user_profile = payload.user_profile
    
    logger.info(f"Received invoke request. Query: '{user_query}', Profile provided: {bool(user_profile)}")

    async def event_stream():
        try:
            async for event_data in agent.astream_events(user_query, user_profile):
                yield event_data
        except Exception as e:
            logger.error(f"Error during agent stream: {e}", exc_info=True)
            error_msg = json.dumps({"type": "error", "message": str(e)})
            yield f"event: error\ndata: {error_msg}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/")
async def root():
    return {"message": "PBWA Backend is running"}