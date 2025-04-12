import os
from dotenv import load_dotenv
import logging # Import logging

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

@app.post("/invoke", response_model=QueryResponse)
async def invoke_agent(request: QueryRequest):
    """Receives a user query and returns the agent's response."""
    logger.info(f"Received request for query: {request.query}")
    agent_response = await agent.arun(request.query)
    logger.info(f"Agent returned response for query '{request.query}'")
    return {"response": agent_response}

@app.get("/")
async def read_root():
    # Keep the root endpoint for basic checks if needed
    return {"message": "Multi-Step Agent API is running."}