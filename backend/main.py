import os
from dotenv import load_dotenv
import logging # Import logging
from fastapi.responses import StreamingResponse # Import StreamingResponse
import json # Import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.agents import ResearchPipeline  # Import the agent class
from backend.schemas import UserProfilePayload, ResumePayload, AnalysisResult # Add AnalysisResult

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
from .schemas import UserProfilePayload, ResumePayload

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
        logger.info(f"--- Starting agent stream for user query: {payload.user_query} ---")
        # Start the stream by calling the agent's method
        # Pass ONLY the user_query, as astream_events handles initial state
        async for event in agent.astream_events(payload.user_query):
            yield event # Directly yield the formatted SSE string from the agent

        logger.info(f"--- Agent stream finished for user query: {payload.user_query} ---")

    # Return the streaming response
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/resume")
async def resume_agent(payload: ResumePayload):
    """Resumes a paused pipeline instance after user feedback."""
    if agent is None:
        logger.error("/resume called but agent failed to initialize.")
        raise HTTPException(status_code=500, detail="Agent initialization failed.")
        
    thread_id = payload.thread_id
    feedback = payload.feedback
    
    logger.info(f"Received resume request for thread_id: {thread_id}. Feedback provided: {bool(feedback)}")

    config = {"configurable": {"thread_id": thread_id}}

    # Use a single try block for the main resume operation
    try:
        # --- Update state with feedback before resuming --- 
        if feedback:
            logger.debug(f"Attempting to update state for {thread_id} with feedback.")
            await agent.app.update_state(config, {"feedback": feedback})
            logger.info(f"Updated state for {thread_id} with user feedback.")

        # --- Define the async generator for the resumed stream --- 
        async def event_stream_resume():
            try:
                # Pass None as input, config contains thread_id to resume
                async for event_data in agent.app.astream(None, config=config, stream_mode="values"):
                    # Re-process event data similar to invoke_agent\'s stream
                    logger.debug(f"Workflow Event/Value (Resume): {event_data}")
                    last_node = event_data["messages"][-1].name if event_data.get("messages") else None
                    current_state = event_data.get("state")
                    
                    node_name = last_node
                    node_output = current_state

                    sse_data = {} 
                    sse_event_type = None
                    is_interruption = False # Reset interruption flag

                    # --- Map node outputs to SSE events (Refactored for clarity) --- 
                    # TODO: Refactor this mapping logic into a shared helper function 
                    #       to avoid duplication with the /invoke endpoint.
                    if node_name == "load_user_profile":
                        pass # Or send profile loaded event
                    elif node_name == "plan_research":
                        sse_event_type = "plan"
                        sse_data = {"type": "plan", "steps": node_output.get("research_plan", [])}
                    elif node_name == "collect_data":
                        sse_event_type = "tool_result"
                        sse_data = {"type": "tool_result", "result": node_output.get("current_step_output")}
                    elif node_name == "replan_step":
                        sse_event_type = "replan"
                        sse_data = {"type": "replan", "message": "Replanning...", "new_plan": node_output.get("research_plan")}
                    elif node_name == "process_data":
                        sse_event_type = "processing"
                        vector_count = len(node_output.get("processed_data", {}).get("vector", []))
                        ts_count = len(node_output.get("processed_data", {}).get("timeseries", []))
                        sse_data = {"type": "processing", "message": f"Processed {vector_count} vector docs, {ts_count} time-series points."}
                    elif node_name == "update_vector_store":
                        sse_event_type = "storage"
                        sse_data = {"type": "storage", "message": node_output.get("current_step_output") or "Vector store updated."}
                    elif node_name == "update_timeseries_db":
                        sse_event_type = "storage_ts"
                        sse_data = {"type": "storage_ts", "message": node_output.get("current_step_output") or "Time-series DB updated."}
                    elif node_name == "analyze_data":
                        sse_event_type = "analysis"
                        analysis_obj = node_output.get("analysis_results")
                        analysis_text = analysis_obj.analysis_text if analysis_obj and isinstance(analysis_obj, AnalysisResult) else "Analysis error or missing."
                        is_sufficient = analysis_obj.is_sufficient if analysis_obj and isinstance(analysis_obj, AnalysisResult) else False
                        sse_data = {"type": "analysis", "result": analysis_text, "is_sufficient": is_sufficient}
                    elif node_name == "propose_strategy":
                        sse_event_type = "strategy"
                        sse_data = {"type": "strategy", "proposals": node_output.get("strategy_proposals")}
                        is_interruption = True 
                    # No END state check needed here as graph should have ended before resume if it hit END

                    # Yield the mapped event
                    if sse_event_type:
                        yield f"event: {sse_event_type}\ndata: {json.dumps(sse_data)}\n\n"
                        
                    # Check for interruption again (in case graph loops unexpectedly)
                    if is_interruption: 
                        logger.warning(f"Pipeline interrupted AGAIN immediately after resume for {thread_id}. Check graph logic.")
                        feedback_data = {"type": "awaiting_feedback", "message": "Pipeline paused again.", "state_id": thread_id}
                        yield f"event: feedback\ndata: {json.dumps(feedback_data)}\n\n"
                        break # Stop the stream if interrupted again

            # --- Exception handling for the stream itself ---            
            except Exception as stream_err:
                logger.error(f"Error during agent resume stream for {thread_id}: {stream_err}", exc_info=True)
                error_msg = json.dumps({"type": "error", "message": str(stream_err)})
                yield f"event: error\ndata: {error_msg}\n\n"
            # --- Finally block for the stream --- 
            finally:
                # Send final end event when resume stream finishes (unless interrupted again)
                if not is_interruption: # Need to access is_interruption from inner scope
                    logger.info(f"--- Pipeline resume stream finished for thread_id: {thread_id} ---")
                    yield f"event: end\ndata: {json.dumps({'message': 'Resume stream finished.'})}\n\n"

        # Return the streaming response using the defined generator
        return StreamingResponse(event_stream_resume(), media_type="text/event-stream")

    # --- Exception handling for the main /resume operation (e.g., update_state failure) --- 
    except Exception as e:
        logger.error(f"Failed to resume pipeline for thread_id {thread_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to resume pipeline: {e}")

@app.get("/")
async def root():
    return {"message": "PBWA Backend is running"}