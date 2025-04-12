from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig # Import RunnableConfig
from langchain_core.output_parsers.string import StrOutputParser # Import StrOutputParser
from langgraph.graph import END, StateGraph, START
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import logging
import json # Import json for formatting streamed data
from typing import AsyncGenerator, List, Dict, Any, Tuple, Optional # Import AsyncGenerator
import re
import uuid # Import uuid for Qdrant point IDs
import inspect
from urllib.parse import urlparse, parse_qs # <-- Added import
# Qdrant specific imports
from qdrant_client import models # Import Qdrant models for search
from datetime import datetime # For timestamping time-series data
from langgraph.checkpoint.memory import MemorySaver # Import MemorySaver

from .config import GOOGLE_API_KEY, LLM_MODEL, QDRANT_COLLECTION_NAME
from .tools import agent_tools
from .prompts import (
            PLANNER_PROMPT,
    EXECUTOR_PROMPT, 
    REPLANNER_PROMPT,
    ANALYZER_PROMPT,
    STRATEGIST_PROMPT # Import STRATEGIST_PROMPT
)
from .schemas import Plan, ReplannerOutput, PipelineState, AnalysisResult
from .vector_store import qdrant_client_instance, embedding_model_instance # Import DB clients
from qdrant_client.http.models import PointStruct # Import PointStruct for upsert

# InfluxDB specific imports
from influxdb_client import Point

# Import InfluxDB client and config
from .timeseries_db import influxdb_client_instance, get_influxdb_write_api, INFLUXDB_BUCKET, INFLUXDB_ORG

# Import the time-series query function directly
from .tools.timeseries_retriever import query_time_series_data

# Get a logger instance for this module
logger = logging.getLogger(__name__)

# --- Helper --- 
def clean_newlines(text: str) -> str:
    """Removes extra newline characters from a string."""
    return text.replace("\n\n", "\n")

class ResearchPipeline:
    def __init__(self):
        logger.info("Initializing ResearchPipeline...")
        # --- LLM Initialization ---
        try:
            self.llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0)
            logger.info(f"Successfully initialized ChatGoogleGenerativeAI with Model: {LLM_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
            raise
        
        # --- Tools --- 
        self.tools = agent_tools # List of tool objects from tools.py
        self.tool_map = {tool.name: tool for tool in self.tools} # Map tool name to object
        logger.info(f"Loaded tools: {[tool.name for tool in self.tools]}")

        # --- Vector Store Clients (loaded from vector_store.py) ---
        self.qdrant_client = qdrant_client_instance
        self.embedding_model = embedding_model_instance
        if not self.qdrant_client or not self.embedding_model:
            logger.warning("Vector store or embedding model not initialized. RAG features disabled.")
            # Optionally raise error if vector store is critical

        # --- Time Series Client --- 
        self.influxdb_client = influxdb_client_instance
        if not self.influxdb_client:
            logger.warning("InfluxDB client not initialized. Time-series features disabled.")
        
        # --- Prompts ---
        self.planner_prompt = ChatPromptTemplate.from_template(
            PLANNER_PROMPT, # Adjust PLANNER_PROMPT if needed for research focus
        )
        self.replanner_prompt = ChatPromptTemplate.from_template(
            REPLANNER_PROMPT, # Adjust REPLANNER_PROMPT if needed
        )
        # --- New Analyzer Prompt Template ---
        self.analyzer_prompt = ChatPromptTemplate.from_template(ANALYZER_PROMPT)
        # --- New Strategist Prompt Template ---
        self.strategist_prompt = ChatPromptTemplate.from_template(STRATEGIST_PROMPT)

        # --- Chains ---
        self.planner = self.planner_prompt | self.llm.with_structured_output(Plan)
        self.replanner = self.replanner_prompt | self.llm.with_structured_output(ReplannerOutput)
        self.analyzer = self.analyzer_prompt | self.llm.with_structured_output(AnalysisResult)
        self.strategist = self.strategist_prompt | self.llm | StrOutputParser()

        # --- Workflow Definition ---
        self.workflow = StateGraph(PipelineState)

        # --- Nodes --- 
        self.workflow.add_node("load_user_profile", self._user_profile_node)
        self.workflow.add_node("plan_research", self._plan_research_step)
        self.workflow.add_node("collect_data", self._collect_data_step)
        self.workflow.add_node("replan_step", self._replan_step)
        self.workflow.add_node("process_data", self._process_data_step) 
        self.workflow.add_node("update_vector_store", self._update_vector_store_node) 
        self.workflow.add_node("update_timeseries_db", self._timeseries_db_update_node)
        self.workflow.add_node("analyze_data", self._analyze_data_step)
        self.workflow.add_node("propose_strategy", self._propose_strategy_step)
        self.workflow.add_node("refine_strategy", self._refine_strategy_step)

        # --- Edges --- 
        self.workflow.set_entry_point("load_user_profile")
        self.workflow.add_edge("load_user_profile", "plan_research")
        self.workflow.add_edge("plan_research", "collect_data")
        self.workflow.add_conditional_edges(
            "collect_data",
            self._should_replan_or_continue_or_process,
            {
                "replan": "replan_step",
                "continue": "collect_data",
                "process": "process_data"
            }
        )
        self.workflow.add_edge("replan_step", "collect_data")
        
        # Make storage steps sequential to avoid concurrent writes to current_step_output
        self.workflow.add_edge("process_data", "update_vector_store") 
        self.workflow.add_edge("update_vector_store", "update_timeseries_db")
        self.workflow.add_edge("update_timeseries_db", "analyze_data")
        # Remove old parallel edges converging on analyze_data
        # self.workflow.add_edge("process_data", "update_timeseries_db") # Removed
        # self.workflow.add_edge("update_vector_store", "analyze_data")  # Removed

        # Conditional edge after analysis
        self.workflow.add_conditional_edges(
            "analyze_data",
            self._should_propose_strategy_or_replan,
            {
                "propose_strategy": "propose_strategy",
                "replan": "replan_step"
            }
        )
        self.workflow.add_edge("propose_strategy", "refine_strategy")
        self.workflow.add_edge("refine_strategy", END) 
        
        # --- Checkpointer for state persistence --- 
        self.memory = MemorySaver() # Instantiate the checkpointer

        # Compile the workflow
        logger.info("Compiling the LangGraph workflow...")
        # Interrupt after proposing strategy to allow for feedback
        self.app = self.workflow.compile(checkpointer=self.memory, interrupt_after=["propose_strategy"]) 
        logger.info("Workflow compiled successfully.")

    # --- Helper: Load User Profile --- 
    def _load_profile(self, profile_id: str = "default_profile") -> Optional[Dict[str, Any]]:
        """Loads user profile from a JSON file."""
        # Simple file-based loading, could be replaced with DB lookup etc.
        profile_path = os.path.join("user_profiles", f"{profile_id}.json")
        logger.info(f"Attempting to load profile from: {profile_path}")
        try:
            with open(profile_path, 'r') as f:
                profile_data = json.load(f)
            logger.info(f"Successfully loaded profile for user ID: {profile_data.get('user_id', 'N/A')}")
            return profile_data
        except FileNotFoundError:
            logger.warning(f"Profile file not found: {profile_path}. Proceeding without profile.")
            return None
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from profile file: {profile_path}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Error loading profile {profile_path}: {e}", exc_info=True)
            return None

    # --- Node Implementations --- 
    def _user_profile_node(self, state: PipelineState):
        """Loads user profile into the state if not already present."""
        logger.info("--- Loading User Profile Step ---")
        
        # Check if profile already exists in state (passed from API)
        existing_profile = state.get("user_profile")
        
        if existing_profile is not None:
            logger.info("Using user profile provided in initial state.")
            # Profile is already set, no state update needed from this node
            # Return empty dict to signal no changes to other state keys
            return {}
        else:
            logger.info("No user profile in initial state, loading default profile from file.")
            # Load the default profile from file as fallback
            profile_data = self._load_profile("default_profile") 
            # Update state with the loaded default profile
            return {"user_profile": profile_data} 
    
    async def _plan_research_step(self, state: PipelineState):
        logger.info("--- Planning Research Step ---")
        user_query = state.get('user_query')
        user_profile = state.get('user_profile') # Get user profile
        errors = state.get('error_log', [])

        if not user_query:
            logger.error("User query is missing from state for planning.")
            return {"error_log": errors + ["Planning failed: Missing user query."]}
        
        # Format profile for prompt
        user_profile_str = json.dumps(user_profile, indent=2) if user_profile else "Not Provided"

        try:
            plan_obj: Plan = await self.planner.ainvoke({
                "input": user_query, 
                "user_profile": user_profile_str # Pass profile to planner
            })
            logger.info(f"Generated Research Plan: {plan_obj.steps}")
            # Keep profile in state if it was passed in, otherwise it remains None
            return {"research_plan": plan_obj.steps, "user_profile": user_profile}
        except Exception as e:
            logger.error(f"Error during planning: {e}", exc_info=True)
            return {"error_log": errors + [f"Planning failed: {e}"]}
    
    async def _collect_data_step(self, state: PipelineState):
        logger.info("--- Collecting Data Step ---")
        plan = state.get("research_plan")
        collected = state.get("collected_data", [])
        errors = state.get("error_log", []) 
        
        if not plan:
            logger.info("Research plan empty or finished. No collection needed.")
            return {"research_plan": []}

        current_task = plan[0]
        remaining_plan = plan[1:]
        logger.info(f"Executing plan step: {current_task}")

        tool_output: Any = None # Initialize tool_output
        tool_error = False # Flag to indicate if an error occurred
        error_details = "" # Store specific error message
        
        try:
            # --- Refactored Tool ID, Arg Parsing & Validation --- 
            parts = current_task.split()
            potential_tool_name = parts[1] if len(parts) > 1 else None
            found_tool = self.tool_map.get(potential_tool_name)
            
            if not found_tool:
                 error_msg = f"Tool '{potential_tool_name}' not found for task: {current_task}. Available: {list(self.tool_map.keys())}"
                 logger.error(error_msg)
                 errors.append(error_msg)
                 tool_error = True
                 error_details = error_msg
            else:
                tool_name = found_tool.name
                logger.info(f"Identified tool: {tool_name}. Preparing args and validation.")
                
                validated_args_dict = {} 
                can_execute = True

                if hasattr(found_tool, 'args_schema') and found_tool.args_schema:
                    schema = found_tool.args_schema
                    schema_fields = schema.__fields__.keys()
                    logger.debug(f"Tool '{tool_name}' expects args based on schema: {list(schema_fields)}")

                    # Basic Regex to extract potential key=value pairs from the task string
                    # (This is still needed as input isn't guaranteed JSON)
                    raw_args = {}
                    pattern = r'(\w+)\s*=\s*[\'"]?([^\s\'"]+)[\'"]?'
                    prefix = f"Use {tool_name}"
                    args_part = current_task
                    if current_task.lower().startswith(prefix.lower()):
                        args_part = current_task[len(prefix):].strip()
                    
                    matches = re.findall(pattern, args_part)
                    for key, value in matches:
                         # Only consider keys that are expected by the schema
                        if key in schema_fields:
                            raw_args[key] = value
                        else:
                             logger.warning(f"Ignoring extracted arg '{key}' as it's not in schema for tool '{tool_name}'")

                    logger.debug(f"Raw args extracted for validation: {raw_args}")
                    
                    try:
                        validated_model = schema(**raw_args)
                        validated_args_dict = validated_model.dict()
                        logger.info(f"Arguments validated successfully for '{tool_name}': {validated_args_dict}")
                    except Exception as validation_err: 
                        error_msg = f"Argument validation failed for tool '{tool_name}'. Task: '{current_task}'. Extracted raw args: {raw_args}. Error: {validation_err}"
                        logger.error(error_msg)
                        errors.append(error_msg)
                        can_execute = False
                        tool_error = True
                        error_details = f"Argument validation failed: {validation_err}"
                else:
                    # Tool expects no arguments
                    logger.info(f"Tool '{tool_name}' does not have an args_schema. Assuming no arguments.")
                    if len(parts) > 2: # Basic check if extraneous text exists after tool name
                        logger.warning(f"Task string '{current_task}' contained extra text after tool name '{tool_name}', but tool expects no args.")

                # Execute the tool if possible
                if can_execute:
                    try:
                        func = found_tool.func
                        # --- Special Handling for vfat_scraper_tool to inject context --- 
                        if tool_name == "vfat_scraper_tool":
                             user_query_context = state.get('user_query')
                             user_profile_context = state.get('user_profile')
                             logger.info("Injecting query/profile context into vfat_scraper_tool call.")
                             # Add state context to args extracted from task (which is just farm_url)
                             final_args = {
                                 **validated_args_dict, # Contains farm_url
                                 "user_query": user_query_context,
                                 "user_profile": user_profile_context
                             }
                             tool_output = await func(**final_args)
                        # --- Standard tool execution --- 
                        elif inspect.iscoroutinefunction(func):
                            tool_output = await func(**validated_args_dict)
                        else:
                            tool_output = func(**validated_args_dict)
                        
                        logger.info(f"Tool Call Output received for '{tool_name}'.")
                        # Attempt to pretty-print if JSON, otherwise show snippet
                        try:
                             parsed_output = json.loads(tool_output) if isinstance(tool_output, str) else tool_output
                             pretty_output = json.dumps(parsed_output, indent=2, default=str)
                             logger.debug(f"Tool output preview:\n{pretty_output[:500]}{'...' if len(pretty_output) > 500 else ''}")
                        except (json.JSONDecodeError, TypeError):
                             logger.debug(f"Tool output (non-JSON) snippet: {str(tool_output)[:500]}...")

                    except Exception as exec_err:
                        error_msg = f"Error executing tool '{tool_name}' for task '{current_task}': {exec_err}"
                        logger.error(error_msg, exc_info=True)
                        errors.append(error_msg)
                        tool_error = True
                        error_details = f"Tool execution failed: {exec_err}"
                # else: Tool was not executed due to validation failure - error already flagged

        except Exception as e:
            # Catch broader errors during identification/parsing phase
            error_msg = f"Unexpected error during tool processing for task '{current_task}': {e}"
            logger.error(error_msg, exc_info=True)
            errors.append(error_msg)
            tool_error = True
            error_details = f"Unexpected error: {e}"
            
        # --- Update State --- 
        if tool_error:
            step_result = {"error": error_details} 
        else:
            step_result = tool_output
            
        newly_collected = (current_task, step_result)
        updated_collected_data = collected + [newly_collected]

        return {
            "research_plan": remaining_plan, 
            "collected_data": updated_collected_data,
            "current_step_output": step_result, # Pass the actual result (or error dict) 
            "error_log": errors
        }

    def _should_replan_or_continue_or_process(self, state: PipelineState):
        logger.debug("Checking state for replan/continue/process decision...")
        plan = state.get("research_plan")
        collected = state.get("collected_data", [])
        
        # Check the result of the *last* collection step for an error
        if collected:
            last_task, last_result = collected[-1]
            if isinstance(last_result, dict) and 'error' in last_result:
                logger.warning(f"Error detected in last step ('{last_task}'): {last_result['error']}. Triggering replan.")
                return "replan"
        
        # If no error in last step, check if plan exists
        if plan:
            logger.info("No error in last step and plan has remaining steps, continuing collection.")
            return "continue"
        else:
            logger.info("No error in last step and plan finished, moving to process data.")
            return "process"

    def _process_data_step(self, state: PipelineState):
        logger.info("--- Processing Collected Data --- ")
        collected_data = state.get("collected_data", [])
        processed_for_vector = []
        processed_for_timeseries = [] # Will contain InfluxDB Points
        
        if not collected_data:
            logger.warning("No data collected to process.")
            return {"processed_data": {"vector": [], "timeseries": []}}
            
        for i, (task, data) in enumerate(collected_data):
            doc_to_embed = None # Prepare potential doc for embedding
            metadata = { 
                "source_task": task,
                "collection_step": i 
            }
            
            # --- Process for Vector Store (Textual Data) ---
            if isinstance(data, str):
                # Embed raw strings directly (if not empty)
                if data.strip():
                    doc_to_embed = {"page_content": data, "metadata": {**metadata, "data_type": "text"}}
            elif isinstance(data, list) and len(data) > 0:
                 # If data is a list (e.g., from vfat scraper or twitter), create a summary
                 # For now, just indicate the type and number of items
                 summary = f"List data from task '{task}': Contains {len(data)} items. First item type: {type(data[0]).__name__}"
                 doc_to_embed = {"page_content": summary, "metadata": {**metadata, "data_type": "list_summary"}}
            elif isinstance(data, dict):
                if 'error' in data:
                    # Skip embedding errors
                    logger.debug(f"Skipping embedding for error data from task '{task}'")
                else:
                    # Create a more concise summary of dictionary data
                    # Example: "API result for task 'Use coingecko_api_tool ...' with keys: ['id', 'symbol', 'market_data']"
                    summary = f"API result for task '{task}'. Keys: {list(data.keys())}"
                    # Optionally add a snippet of important values if identifiable
                    # if 'name' in data: summary += f" Name: {data['name']}"
                    # if 'symbol' in data: summary += f" Symbol: {data['symbol']}" 
                    doc_to_embed = {"page_content": summary, "metadata": {**metadata, "data_type": "api_summary"}}
            else:
                # Handle other unexpected types if necessary
                logger.warning(f"Skipping vector processing for data from task '{task}' due to unexpected type: {type(data)}")

            if doc_to_embed:
                processed_for_vector.append(doc_to_embed)

            # --- Process for Time Series Store --- 
            # Example: Extracting TVL from DefiLlama result
            if isinstance(data, dict) and task.startswith("Use defi_llama_api_tool") and 'error' not in data:
                protocol_slug = task.split("protocol_slug='")[1].split("'")[0] if "protocol_slug='" in task else "unknown"
                # Check if it's chart data (list of dicts with 'date')
                if isinstance(data, list) and all('date' in item for item in data):
                    points_added = 0
                    for item in data:
                        try:
                            timestamp_sec = int(item['date'])
                            # Check for TVL
                            if 'totalLiquidityUSD' in item:
                                tvl = float(item['totalLiquidityUSD'])
                                point = Point("protocol_metrics") \
                                    .tag("protocol", protocol_slug) \
                                    .tag("source", "defillama") \
                                    .field("tvl_usd", tvl) \
                                    .time(timestamp_sec, write_precision='s')
                                processed_for_timeseries.append(point)
                                points_added += 1
                            # TODO: Check for other common chart metrics like daily volume, fees/revenue if keys exist
                            # Example (assuming keys 'dailyVolumeUSD', 'dailyFeesUSD')
                            # if 'dailyVolumeUSD' in item:
                            #     volume = float(item['dailyVolumeUSD'])
                            #     point_vol = Point("protocol_metrics") \
                            #         .tag("protocol", protocol_slug).tag("source", "defillama") \
                            #         .field("daily_volume_usd", volume) \
                            #         .time(timestamp_sec, write_precision='s')
                            #     processed_for_timeseries.append(point_vol)
                            #     points_added += 1
                        except (ValueError, TypeError, KeyError) as e:
                            logger.warning(f"Skipping DefiLlama point for {protocol_slug} due to parsing error: {e} - Item: {item}")
                    if points_added > 0:
                        logger.info(f"Processed {points_added} time-series points for '{protocol_slug}' from DefiLlama charts.")
            
            # Example: Extracting Market Data from CoinGecko result
            elif isinstance(data, dict) and task.startswith("Use coingecko_api_tool") and 'error' not in data:
                token_id = data.get('id')
                if token_id and 'market_data' in data:
                    md = data['market_data']
                    points_added = 0
                    try:
                        point = Point("token_market_data") \
                            .tag("token_id", token_id) \
                            .tag("source", "coingecko")
                            
                        # Price (already handled, but let's integrate into single point)
                        if 'current_price' in md and isinstance(md['current_price'], dict) and 'usd' in md['current_price']:
                            price_usd = float(md['current_price']['usd'])
                            point = point.field("price_usd", price_usd)
                            
                        # Market Cap
                        if 'market_cap' in md and isinstance(md['market_cap'], dict) and 'usd' in md['market_cap']:
                            market_cap_usd = float(md['market_cap']['usd'])
                            point = point.field("market_cap_usd", market_cap_usd)

                        # Total Volume
                        if 'total_volume' in md and isinstance(md['total_volume'], dict) and 'usd' in md['total_volume']:
                            total_volume_usd = float(md['total_volume']['usd'])
                            point = point.field("total_volume_usd", total_volume_usd)
                        
                        # Check if any fields were added before adding time and appending
                        if point.field_keys:
                            point = point.time(datetime.utcnow()) # Use current time
                            processed_for_timeseries.append(point)
                            points_added += 1
                            
                    except (ValueError, TypeError, KeyError) as e:
                        logger.warning(f"Skipping CoinGecko market data point for {token_id} due to parsing error: {e} - Data: {md}")
                        
                    if points_added > 0:
                         logger.info(f"Processed market data point for '{token_id}' from CoinGecko.")

        logger.info(f"Prepared {len(processed_for_vector)} documents for vector store embedding.")
        logger.info(f"Prepared {len(processed_for_timeseries)} points for time-series store.")
        return {"processed_data": {"vector": processed_for_vector, "timeseries": processed_for_timeseries}}
        
    async def _update_vector_store_node(self, state: PipelineState):
        logger.info("--- Updating Vector Store --- ")
        processed_data = state.get("processed_data", {})
        docs_to_embed = processed_data.get("vector", [])
        
        if not self.qdrant_client or not self.embedding_model:
            logger.error("Cannot update vector store: Qdrant client or embedding model not available.")
            return {"error_log": state.get("error_log", []) + ["Vector store update skipped: Client/Model unavailable."]}
        
        if not docs_to_embed:
            logger.info("No new documents to embed and store.")
            return {}

        try:
            logger.info(f"Embedding {len(docs_to_embed)} documents...")
            contents = [doc["page_content"] for doc in docs_to_embed]
            embeddings = list(self.embedding_model.embed(contents))
            logger.info("Embedding complete.")

            points_to_upsert = []
            for i, doc in enumerate(docs_to_embed):
                point_id = str(uuid.uuid4())
                points_to_upsert.append(PointStruct(
                    id=point_id,
                    vector=embeddings[i].tolist(), # Convert numpy array if needed
                    payload=doc["metadata"] # Store metadata
                ))
            
            if points_to_upsert:
                logger.info(f"Upserting {len(points_to_upsert)} points to Qdrant collection '{QDRANT_COLLECTION_NAME}'...")
                self.qdrant_client.upsert(
                    collection_name=QDRANT_COLLECTION_NAME,
                    points=points_to_upsert,
                    wait=True # Wait for operation to complete
                )
                logger.info("Upsert complete.")
            return {"current_step_output": f"Successfully stored {len(points_to_upsert)} documents."}
            
        except Exception as e:
            logger.error(f"Failed to update vector store: {e}", exc_info=True)
            return {"error_log": state.get("error_log", []) + [f"Vector store update failed: {e}"]}

    # --- New Timeseries DB Update Node --- 
    def _timeseries_db_update_node(self, state: PipelineState):
        logger.info("--- Updating Time-Series Database --- ")
        processed_data = state.get("processed_data", {})
        points_to_write = processed_data.get("timeseries", [])
        errors = state.get('error_log', [])
        
        if not self.influxdb_client:
            logger.error("Cannot update time-series DB: InfluxDB client not available.")
            return {"error_log": errors + ["Time-series update skipped: Client unavailable."]}
        
        if not points_to_write:
            logger.info("No new time-series points to write.")
            return {"current_step_output": "No time-series data to store."}

        write_api = None
        try:
            write_api = get_influxdb_write_api(self.influxdb_client)
            if not write_api:
                 raise ConnectionError("Failed to obtain InfluxDB write API.")
                
            logger.info(f"Writing {len(points_to_write)} points to InfluxDB bucket '{INFLUXDB_BUCKET}'...")
            write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=points_to_write)
            logger.info("InfluxDB write complete.")
            return {"current_step_output": f"Successfully stored {len(points_to_write)} time-series points."}
            
        except Exception as e:
            error_msg = f"Failed to write to InfluxDB: {e}"
            logger.error(error_msg, exc_info=True)
            return {"error_log": errors + [error_msg]}
        finally:
             if write_api:
                 try:
                     write_api.close()
                 except Exception as close_err:
                      logger.error(f"Error closing InfluxDB write API: {close_err}")
                     

    async def _analyze_data_step(self, state: PipelineState):
        logger.info("--- Analyzing Data ---")
        user_query = state.get('user_query')
        user_profile = state.get('user_profile') # Get user profile
        collected_data = state.get('collected_data', [])
        errors = state.get('error_log', [])

        if not user_query:
            logger.error("Analysis step failed: User query is missing.")
            return {"error_log": errors + ["Analysis failed: Missing user query."]}

        retrieved_context_str = "No relevant context found in knowledge base."
        
        # --- RAG - Retrieve from Qdrant ---
        if self.qdrant_client and self.embedding_model:
            try:
                logger.info(f"Generating embedding for user query: '{user_query[:100]}...'")
                query_embedding = list(self.embedding_model.embed([user_query]))[0]
                
                logger.info(f"Searching Qdrant collection '{QDRANT_COLLECTION_NAME}'...")
                search_result = self.qdrant_client.search(
                    collection_name=QDRANT_COLLECTION_NAME,
                    query_vector=query_embedding,
                    limit=3 # Retrieve top 3 relevant documents
                )
                logger.info(f"Found {len(search_result)} relevant points in Qdrant.")

                # Format retrieved context for the prompt
                context_parts = []
                for i, hit in enumerate(search_result):
                    payload = hit.payload or {} # Handle potentially empty payload
                    # Safely get content preview and metadata
                    content_preview = str(payload.get('page_content', 'No content preview available.'))[:200] 
                    source_task = payload.get('source_task', 'N/A')
                    score = hit.score
                    # Format string parts separately
                    header = f"Context {i+1} (Score: {score:.2f}):"
                    meta_line = f"Metadata: Source Task: {source_task}"
                    content_line = f"Content: {content_preview}..."
                    # Combine parts for this hit with newlines
                    hit_str = f"{header}\n{meta_line}\n{content_line}\n---"
                    context_parts.append(hit_str) # Append the formatted string
                
                if context_parts:
                    # Join the already formatted strings
                    retrieved_context_str = "\n".join(context_parts) 
                
            except Exception as e:
                error_msg = f"Error during RAG search in Qdrant: {e}"
                logger.error(error_msg, exc_info=True)
                errors.append(error_msg)
                retrieved_context_str = f"Error retrieving context: {e}"
        else:
            logger.warning("Skipping RAG search: Qdrant client or embedding model not available.")
            retrieved_context_str = "Context retrieval skipped (DB unavailable)."

        # --- Time Series Context Retrieval --- 
        time_series_context_lines = [] # Use a list to build the context string
        # Attempt to identify relevant tokens/protocols from collected data
        # This is a simple heuristic, could be much more sophisticated
        relevant_items = set()
        for task, result in collected_data:
            if task.startswith("Use coingecko_api_tool") and isinstance(result, dict) and result.get('id'):
                relevant_items.add(("token", result['id'] ))
            elif task.startswith("Use defi_llama_api_tool") and isinstance(result, dict):
                 slug_match = re.search(r"protocol_slug='([^\']+)\'", task)
                 if slug_match:
                     relevant_items.add(("protocol", slug_match.group(1)))
        
        if relevant_items:
            logger.info(f"Identified relevant items for time-series query: {relevant_items}")
            for item_type, item_id in relevant_items:
                measurement = None
                tags = None
                fields = None
                if item_type == "token":
                    measurement = "token_market_data"
                    tags = {"token_id": item_id}
                    fields = ["price_usd", "market_cap_usd"] # Query price and market cap
                elif item_type == "protocol":
                    measurement = "protocol_metrics"
                    tags = {"protocol": item_id}
                    fields = ["tvl_usd"] # Query TVL
                   
                if measurement:
                    query_successful = False
                    error_info = None # Store potential error message
                    processed_results = [] # Store processed results for this item
                    
                    try:
                        result_dict = query_time_series_data(
                            measurement=measurement, 
                            tags=tags, 
                            fields=fields, 
                            start_time="-7d", 
                            limit=500 # Increase limit slightly for better change calc
                        )
                        if result_dict and 'results' in result_dict:
                            points = sorted(result_dict['results'], key=lambda p: p.get('time', '')) # Sort by time ascending
                            processed_results = points # Store for later processing
                            query_successful = True
                        elif result_dict and 'error' in result_dict:
                             error_info = result_dict['error']
                             logger.warning(f"Error retrieving time-series for {item_id}: {error_info}")
                        else:
                            # Query succeeded but returned no data
                            query_successful = True 
                             
                    except Exception as ts_err:
                        logger.error(f"Exception querying time-series data for {item_id}: {ts_err}", exc_info=True)
                        error_info = f"Exception occurred during query: {ts_err}"

                    # --- Format the results for the prompt --- 
                    time_series_context_lines.append(f"\n**{item_type.capitalize()} '{item_id}' Time Series (Last 7 Days):**")
                    if error_info:
                        time_series_context_lines.append(f"  - ERROR retrieving data: {error_info}")
                    elif not processed_results:
                        time_series_context_lines.append(f"  - No recent data found.")
                    else:
                        # Calculate stats and format
                        try:
                            # Group points by field
                            points_by_field = {}
                            for p in processed_results:
                                field = p.get('field')
                                if field:
                                    if field not in points_by_field:
                                        points_by_field[field] = []
                                    points_by_field[field].append(p)
                            
                            for field, field_points in points_by_field.items():
                                if len(field_points) > 1:
                                    start_point = field_points[0]
                                    end_point = field_points[-1]
                                    start_val = start_point.get('value')
                                    end_val = end_point.get('value')
                                    start_time_str = start_point.get('time', '').split('T')[0] # Just date
                                    end_time_str = end_point.get('time', '').split('T')[0]
                                    
                                    change_str = "N/A"
                                    if isinstance(start_val, (int, float)) and isinstance(end_val, (int, float)) and start_val != 0:
                                        change_pct = ((end_val - start_val) / start_val) * 100
                                        change_str = f"{change_pct:+.2f}%"
                                    
                                    time_series_context_lines.append(f"  - {field}: Start ({start_time_str}) = {start_val}, End ({end_time_str}) = {end_val}, Change (7d) = {change_str}")
                                elif len(field_points) == 1:
                                     point = field_points[0]
                                     time_str = point.get('time', '').split('T')[0]
                                     time_series_context_lines.append(f"  - {field}: Single point ({time_str}) = {point.get('value')}")
                                else:
                                     time_series_context_lines.append(f"  - {field}: No data points found.")

                            # Optionally add a few raw points for context
                            if len(processed_results) > 0:
                                time_series_context_lines.append(f"    (Recent raw points sample):")
                                for point in processed_results[-3:]: # Last 3 points
                                    ts = point.get('time', '')
                                    field = point.get('field', '')
                                    value = point.get('value', '')
                                    time_series_context_lines.append(f"      - {ts}: {field} = {value}")
                                    
                        except Exception as fmt_err:
                            logger.error(f"Error formatting time-series stats for {item_id}: {fmt_err}", exc_info=True)
                            time_series_context_lines.append(f"  - Error processing time-series data for display.")

        # Join the formatted lines
        if time_series_context_lines:
            time_series_context_str = "\n".join(time_series_context_lines).strip()
        else:
            time_series_context_str = "No relevant time-series data identified or retrieved."
        
        # --- Format Collected Data for Prompt ---
        collected_data_str = ""
        if collected_data:
            formatted_items = []
            for task, result in collected_data:
                # Check if the result is a structured error
                if isinstance(result, dict) and 'error' in result:
                    error_message = result['error']
                    # Format error clearly
                    item_str = f"- Task: {task}\n  Result: ERROR - {error_message}"
                    formatted_items.append(item_str)
                else:
                    # Format successful result (with truncation)
                    result_preview = str(result)[:300] # Limit length
                    ellipsis = "..." if len(str(result)) > 300 else ""
                    task_line = f"- Task: {task}"
                    result_line = f"  Result: {result_preview}{ellipsis}"
                    item_str = f"{task_line}\n{result_line}"
                    formatted_items.append(item_str)
            # Join all items with newlines
            collected_data_str = "\n".join(formatted_items)
        else:
            collected_data_str = "No data was collected in this session."
            
        # Format profile for prompt
        user_profile_str = json.dumps(user_profile, indent=2) if user_profile else "Not Provided"
            
        # --- Invoke Analyzer LLM ---
        logger.info("Invoking Analyzer LLM...")
        analysis_result_obj: Optional[AnalysisResult] = None # Initialize as None
        try:
            # Call analyzer with structured output
            analysis_result_obj = await self.analyzer.ainvoke({
                "user_query": user_query,
                "user_profile": user_profile_str,
                "retrieved_context": retrieved_context_str,
                "time_series_context": time_series_context_str,
                "collected_data": collected_data_str
            })
            logger.info("Analysis generated successfully. Sufficiency: {analysis_result_obj.is_sufficient}")
            logger.debug(f"Analysis Result Text Snippet: {analysis_result_obj.analysis_text[:200]}...")
        except Exception as e:
            error_msg = f"Error during LLM Analysis step: {e}"
            logger.error(error_msg, exc_info=True)
            errors.append(error_msg)
            # Store error indicator in state? For now, just log and proceed (will likely fail sufficiency check)
            analysis_result_obj = AnalysisResult(is_sufficient=False, analysis_text=f"Error during analysis: {e}", reasoning="LLM call failed")
        
        # Return the structured analysis object
        return {
            "analysis_results": analysis_result_obj,
            "error_log": errors
        }

    # --- New Conditional Logic Post-Analysis ---
    def _should_propose_strategy_or_replan(self, state: PipelineState):
        """Decides whether to propose strategy or replan based on analysis sufficiency."""
        logger.debug("Checking analysis result for sufficiency...")
        analysis_result_obj = state.get('analysis_results')
        
        if analysis_result_obj and isinstance(analysis_result_obj, AnalysisResult):
            if analysis_result_obj.is_sufficient:
                logger.info("Analysis deemed sufficient. Proceeding to propose strategy.")
                return "propose_strategy"
            else:
                logger.warning(f"Analysis deemed insufficient. Triggering replan. Reason: {analysis_result_obj.reasoning}")
                return "replan"
        else:
            # Should not happen if analysis step ran, but default to replan if result is missing/invalid
            logger.error("Analysis result object missing or invalid in state. Triggering replan as failsafe.")
            return "replan"

    async def _propose_strategy_step(self, state: PipelineState):
        logger.info("--- Proposing Strategy ---")
        user_query = state.get('user_query')
        user_profile = state.get('user_profile') # Optional
        analysis_result_obj = state.get('analysis_results')
        errors = state.get('error_log', [])

        if not analysis_result_obj:
            logger.error("Strategy step failed: Analysis results are missing.")
            return {"error_log": errors + ["Strategy proposal failed: Missing analysis results."]}

        # Format user profile for prompt (if available)
        user_profile_str = json.dumps(user_profile, indent=2) if user_profile else "Not Provided"

        # Extract the text part for the strategist prompt
        analysis_text = analysis_result_obj.analysis_text if analysis_result_obj and isinstance(analysis_result_obj, AnalysisResult) else "Analysis results unavailable."
        
        if not analysis_result_obj or not analysis_result_obj.is_sufficient: # Double check sufficiency?
            logger.error("Strategy step called but analysis was insufficient or missing.")
            return {"error_log": errors + ["Strategy proposal failed: Analysis missing or insufficient."]}
        
        logger.info("Invoking Strategist LLM...")
        strategy_proposals = "Strategy proposal failed."
        try:
            strategy_proposals = await self.strategist.ainvoke({
                "user_query": user_query or "Not Provided",
                "user_profile": user_profile_str,
                "analysis_results": analysis_text # Pass only the text
            })
            logger.info("Strategy proposals generated successfully.")
            logger.debug(f"Strategy Proposals Snippet: {strategy_proposals[:200]}...")
        except Exception as e:
            error_msg = f"Error during LLM Strategy proposal step: {e}"
            logger.error(error_msg, exc_info=True)
            errors.append(error_msg)
            strategy_proposals = f"Error generating strategies: {e}"
        
        # Add signal for interruption
        logger.info("Strategy proposals generated. Preparing to pause for feedback.")
        # We don't modify the return structure here, the graph interruption handles the pause.
        # The state containing strategy_proposals will be available when the stream pauses.
        return {
            "strategy_proposals": strategy_proposals,
            "error_log": errors
        }

    # --- Updated Replan Node --- 
    async def _replan_step(self, state: PipelineState):
        """Invokes the replanner LLM based on the current state, including errors or analysis feedback."""
        logger.info("--- Replanning Step ---")
        user_query = state.get('user_query')
        plan = state.get("research_plan")
        collected = state.get("collected_data", [])
        errors = state.get('error_log', [])
        
        analysis_result_obj = state.get('analysis_results') # Get potential analysis result object

        # Format collected data including errors for the replanner prompt
        formatted_steps = []
        for task, result in collected:
            if isinstance(result, dict) and 'error' in result:
                formatted_steps.append(f"Step: {task}\nOutcome: FAILED - {result['error']}")
            else:
                # Keep successful results brief for replanner context
                result_preview = str(result)[:150] 
                ellipsis = "..." if len(str(result)) > 150 else ""
                formatted_steps.append(f"Step: {task}\nOutcome: Success - Result Preview: {result_preview}{ellipsis}")
        intermediate_steps_str = "\n\n".join(formatted_steps)

        # --- Truncate intermediate steps if too long to avoid large prompts --- 
        MAX_HISTORY_LEN = 10000 # Characters
        if len(intermediate_steps_str) > MAX_HISTORY_LEN:
            intermediate_steps_str = intermediate_steps_str[-MAX_HISTORY_LEN:]
            logger.warning(f"Truncated intermediate steps history to last {MAX_HISTORY_LEN} chars for replanner prompt.")

        # --- Add context from analysis if replanning was triggered post-analysis ---
        analysis_context = "No analysis performed yet or analysis was sufficient."
        if analysis_result_obj and isinstance(analysis_result_obj, AnalysisResult) and not analysis_result_obj.is_sufficient:
            analysis_context = f"Analysis found data insufficient.\nReason: {analysis_result_obj.reasoning}\nSuggestions: {analysis_result_obj.suggestions_for_next_steps}"

        logger.debug(f"Calling replanner. Context:\nQuery: {user_query}\nPlan: {plan}\nAnalysis Context: {analysis_context}\nHistory: {intermediate_steps_str[:500]}...")
        try:
            replanner_output: ReplannerOutput = await self.replanner.ainvoke({
                "input": user_query,
                "plan": plan,
                "analysis_context": analysis_context, # Pass analysis context
                "intermediate_steps": intermediate_steps_str
            })
            
            # --- Add check for None output from parser --- 
            if replanner_output is None:
                error_msg = "Replanner LLM returned invalid output (parsed as None)."
                logger.error(error_msg)
                # Default to keeping original plan if replanning output is invalid
                return {"research_plan": plan, "error_log": errors + [error_msg]}
            
            # --- Original logic (only runs if replanner_output is not None) ---
            if replanner_output.replan and replanner_output.new_plan is not None:
                logger.info(f"Replanner generated a new plan: {replanner_output.new_plan}")
                # Clear previous analysis result if replanning happens
                return {"research_plan": replanner_output.new_plan, "analysis_results": None, "error_log": errors + ["Replanning triggered."]}
            else:
                logger.info("Replanner decided not to change the plan. Continuing with existing plan (if any).")
                return {"research_plan": plan, "error_log": errors} # Keep existing plan
                
        except Exception as e:
            error_msg = f"Error during replanning step: {e}"
            logger.error(error_msg, exc_info=True)
            # If replanning fails, maybe just keep the original plan or halt?
            # For now, keep original plan and log error.
            return {"research_plan": plan, "error_log": errors + [error_msg]}

    # --- Step: Refine Strategy (using feedback) --- 
    def _refine_strategy_step(self, state: PipelineState) -> Dict[str, Any]:
        """Refines the strategy based on user feedback (if provided)."""
        logger.info("--- Entering Refine Strategy Step ---")
        feedback = state.get("feedback")
        proposals = state.get("strategy_proposals")

        if feedback:
            logger.info(f"Refining strategy based on feedback: {feedback}")
            # TODO: Implement actual refinement logic using LLM
            # For now, just log and keep original proposals
            refined_proposals = proposals # Placeholder
            logger.info(f"Original proposals: {proposals}")
            logger.info(f"Refined proposals (placeholder): {refined_proposals}")
            # Update state if refinement happens (even if placeholder for now)
            return {"strategy_proposals": refined_proposals, "current_step_output": "Strategy refined based on feedback (placeholder)."}
        else:
            logger.info("No feedback provided, skipping refinement.")
            # Return empty dict as state is unchanged
            return {"current_step_output": "No feedback provided, strategy not refined."}

    # --- Streaming Method Update ---
    async def astream_events(self, user_query: str, user_profile: Optional[Dict[str, Any]]) -> AsyncGenerator[str, None]: 
        logger.info(f"--- Starting research pipeline stream for Query: {user_query} ---")
        logger.info(f"--- Using User Profile: {user_profile} ---")
        
        initial_state = PipelineState(
            user_profile=user_profile, 
            user_query=user_query, 
            research_plan=None,
            collected_data=[],
            processed_data={},
            analysis_results=None,
            strategy_proposals=None,
            current_step_output=None,
            error_log=[],
            feedback=None 
        )
        
        # --- Generate a unique thread_id for this run --- 
        thread_id = str(uuid.uuid4())
        logger.info(f"Generated new thread_id for this run: {thread_id}")
        
        # Create start event data payload 
        start_data_payload = {
            'input': user_query,
            'thread_id': thread_id,
            # Optionally include profile summary in start event if needed by frontend?
            # 'profile_summary': {k: v for k, v in user_profile.items() if v} if user_profile else None
        }
        start_data_json = json.dumps(start_data_payload)
        yield f"event: start\ndata: {start_data_json}\n\n"
    
        # --- Explicitly send initial status events before the main stream loop ---
        # 1. Profile Loaded (since it's the first step implied after start)
        # profile_loaded_data = {"type": "progress", "message": "User profile loaded."}
        # yield f"event: progress\ndata: {json.dumps(profile_loaded_data)}\n\n"
        
        # 2. Planning Started (since it's the next logical step)
        planning_started_data = {"type": "progress", "message": "Agent planning research steps..."}
        yield f"event: progress\ndata: {json.dumps(planning_started_data)}\n\n"

        # --- Update config with thread_id for checkpointer --- 
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}
        
        sse_event_type = "progress" # Default type, though less relevant now
        current_state_id = None # Track state ID for potential resume
        is_interruption = False # Initialize before try block
        
        try:
            logger.info(f"Entering self.app.astream loop for thread {thread_id}...")
            # Change stream_mode to "updates"
            async for event in self.app.astream(initial_state, config=config, stream_mode="updates"):
                logger.debug(f"Raw Workflow Event/Update: {event}")

                # --- Event Processing for "updates" stream_mode ---
                # Check if the event is a dictionary and not empty
                if not isinstance(event, dict) or not event:
                    logger.warning(f"Received unexpected event structure from astream (not a non-empty dict): {type(event)}. Skipping.")
                    continue

                # Check if it's a node update event (should have one key: the node name)
                if len(event.keys()) == 1:
                    node_name = list(event.keys())[0]
                    node_output = event[node_name] # This is the value associated with the node name key

                    # Add detailed logging before the type check
                    logger.debug(f"Processing update for Node: {node_name}. Output type: {type(node_output)}")

                    # --- Reinstate Special Handling for load_user_profile ---
                    # If the profile node event comes through (even if output is None due to no state change),
                    # just skip it gracefully as we handle the initial message proactively.
                    if node_name == "load_user_profile":
                        logger.debug(f"Skipping explicit processing for '{node_name}' event in loop (handled proactively).")
                        continue # Skip further processing/warning for this node

                    # --- General Validation for all other nodes ---
                    # Use elif now that the profile node is handled above
                    elif not isinstance(node_output, dict):
                        # Log the problematic output for debugging before skipping
                        logger.warning(f"Node '{node_name}' output is not a dict: Type={type(node_output)}, Value='{node_output}'. Skipping SSE mapping for this update.")
                        continue # Skip SSE mapping for this non-dict output
                    
                    # --- If it's a dict (and not load_user_profile), proceed with SSE mapping ---
                    else: # This 'else' covers the case where node_output *is* a dict for other nodes
                        logger.debug(f"Node '{node_name}' output is a dict. Keys: {list(node_output.keys())}")
                        current_state_id = config["configurable"]["thread_id"] # Update state ID on valid node output

                        sse_data = {}
                        sse_event_type = None
                        is_interruption = False # Reset interruption flag

                        # Map based on node_name, using data from node_output dict
                        # NOTE: The specific mapping for "load_user_profile" is now handled above (removed)
                        if node_name == "plan_research":
                            plan = node_output.get("research_plan", [])
                            sse_event_type = "plan"
                            sse_data = {"type": "plan", "steps": plan}
                        elif node_name == "collect_data":
                            # --- Add logic to potentially send a custom search_scrape_start event ---
                            tool_name = "unknown_tool"
                            task_string = ""
                            search_query = None
                            collected_data_list = node_output.get("collected_data")
                            if collected_data_list:
                                try:
                                    last_task_tuple = collected_data_list[-1]
                                    task_string = last_task_tuple[0] # Get the task string: e.g., "1. Use scrape_tool_direct url='https://google.com/search?q=...'"
                                    parts = task_string.split()
                                    if len(parts) >= 2 and parts[1] == 'scrape_tool_direct':
                                        tool_name = 'scrape_tool_direct'
                                        # Extract URL to check if it's Google search
                                        url_match = re.search(r'url=[\'"]?([^\'"\\s]+)[\'"]?', task_string)
                                        if url_match:
                                            url = url_match.group(1)
                                            parsed_url = urlparse(url)
                                            if "google.com" in parsed_url.netloc and parsed_url.path.startswith("/search"):
                                                query_params = parse_qs(parsed_url.query)
                                                if 'q' in query_params and query_params['q']:
                                                    search_query = query_params['q'][0]
                                                    # Yield the specific start event
                                                    search_start_data = {"type": "search_scrape_start", "query": search_query}
                                                    yield f"event: search_scrape_start\ndata: {json.dumps(search_start_data)}\n\n"
                                                    logger.info(f"Sent search_scrape_start event for query: {search_query}")
                                    elif len(parts) >= 2: # Handle extraction for other tools if needed, simplified here
                                        tool_name = parts[1]

                                # Ensure the exception tuple is correctly closed
                                except (IndexError, TypeError, AttributeError) as e:
                                     logger.warning(f"Could not parse tool/URL from task in collected_data: {e}")

                            # --- Prepare standard tool_result event (runs after the custom one if applicable) --- 
                            current_output = node_output.get("current_step_output")
                            result_display = current_output
                            # Simplified preview logic...
                            if isinstance(current_output, dict) and "error" not in current_output:
                                 result_display = f"Dict with keys: {list(current_output.keys())}" # Less verbose preview
                            elif isinstance(current_output, str) and len(current_output) > 200:
                                 result_display = current_output[:200] + "..."
                            elif not isinstance(current_output, (str, dict, list, tuple, int, float, bool, type(None))): # Handle other complex types
                                 result_display = f"<{type(current_output).__name__}> object"

                            sse_event_type = "tool_result"
                            sse_data = {
                                "type": "tool_result", 
                                "tool_name": tool_name, # Use tool_name extracted above
                                "result": result_display
                            } 
                        elif node_name == "replan_step":
                            new_plan = node_output.get("research_plan")
                            sse_event_type = "replan"
                            sse_data = {"type": "replan", "message": "Replanning...", "new_plan": new_plan}
                        elif node_name == "process_data":
                            vector_count = len(node_output.get("processed_data", {}).get("vector", []))
                            ts_count = len(node_output.get("processed_data", {}).get("timeseries", []))
                            sse_event_type = "processing"
                            sse_data = {"type": "processing", "message": f"Processed {vector_count} vector docs, {ts_count} time-series points."}
                        elif node_name == "update_vector_store":
                            message = node_output.get("current_step_output") or "Vector store updated."
                            sse_event_type = "storage"
                            sse_data = {"type": "storage", "message": message}
                        elif node_name == "update_timeseries_db":
                            message = node_output.get("current_step_output") or "Time-series DB updated."
                            sse_event_type = "storage_ts"
                            sse_data = {"type": "storage_ts", "message": message}
                        elif node_name == "analyze_data":
                            analysis_obj = node_output.get("analysis_results")
                            analysis_text = "Analysis error or missing."
                            is_sufficient = False
                            # Ensure analysis_obj is correctly typed before accessing attributes
                            if isinstance(analysis_obj, AnalysisResult):
                                analysis_text = analysis_obj.analysis_text
                                is_sufficient = analysis_obj.is_sufficient
                            sse_event_type = "analysis"
                            sse_data = {"type": "analysis", "result": analysis_text, "is_sufficient": is_sufficient}
                        elif node_name == "propose_strategy":
                            proposals = node_output.get("strategy_proposals")
                            sse_event_type = "strategy"
                            sse_data = {"type": "strategy", "proposals": proposals}
                            is_interruption = True
                        elif node_name == "refine_strategy":
                            message = node_output.get("current_step_output") or "Strategy refined."
                            sse_event_type = "refinement"
                            sse_data = {"type": "refinement", "message": message}
                        else:
                            # This will now correctly skip "load_user_profile" as it's handled above
                            logger.warning(f"No specific SSE mapping for node: {node_name}. Output keys: {list(node_output.keys())}")

                    # --- Yield Event and Handle Interruption (Common Logic) ---
                    # This part now runs if node_name == 'load_user_profile' OR if it's another node and node_output is a dict
                    if sse_event_type:
                        try:
                            sse_data_json = json.dumps(sse_data, default=str) # Use default=str for broader serialization
                            yield f"event: {sse_event_type}\ndata: {sse_data_json}\n\n"
                        except TypeError as json_err:
                            logger.error(f"Failed to serialize SSE data for event {sse_event_type}: {json_err}. Data snippet: {str(sse_data)[:200]}")
                            # Attempt to send a generic error event
                            try:
                                error_data_json = json.dumps({'type': 'error', 'message': f'Serialization error for event {sse_event_type}'})
                                yield f"event: error\ndata: {error_data_json}\n\n"
                            except Exception: # Fallback if even error serialization fails
                                yield f"event: error\ndata: {json.dumps({'type': 'error', 'message': 'Unserializable error occurred.'})}\n\n"

                    # Handle interruption separately (only relevant for propose_strategy currently)
                    if is_interruption: # is_interruption is False unless explicitly set in the mapping block
                        # Ensure current_state_id is set before interruption
                        # It should be set if we processed any node output dict successfully before this point.
                        # If the *first* node output caused interruption (unlikely), current_state_id might be None. Use thread_id as fallback.
                        state_id_for_feedback = current_state_id or thread_id 
                        feedback_data = {"type": "awaiting_feedback", "message": "Pipeline paused...", "state_id": state_id_for_feedback}
                        yield f"event: feedback\ndata: {json.dumps(feedback_data)}\n\n"
                        logger.info(f"Pipeline interrupted, awaiting feedback. State ID: {state_id_for_feedback}")
                        break # Exit the loop on interruption

                else:
                    # Handle cases where event is a dict but doesn't have exactly one key
                    logger.warning(f"Received unexpected event dictionary structure (keys: {list(event.keys())}). Expected single node update. Skipping: {event}")
                    continue

            logger.info(f"Exiting self.app.astream loop for thread {thread_id}.")
        # Keep outer try/except for stream setup errors
        except Exception as e:
            logger.error(f"Exception during pipeline stream setup or outer loop for thread {thread_id}: {e}", exc_info=True)
            # Attempt to yield a final error event
            try:
                error_data = {"type": "error", "message": f"Stream failed: {e}"}
                yield f"event: error\ndata: {json.dumps(error_data)}\n\n"
            except Exception: # If error serialization fails
                 yield f"event: error\ndata: {json.dumps({'type': 'error', 'message': 'Unserializable stream error.'})}\n\n"
        finally:
            # Send end event if stream finishes *without* interruption (error or normal end)
            if not is_interruption:
                logger.info(f"--- Pipeline stream finished normally for Thread ID: {thread_id} ---")
                # Use thread_id from config if available, otherwise use the initial one
                final_thread_id = config.get("configurable", {}).get("thread_id", thread_id)
                yield f"event: end\ndata: {json.dumps({'message': 'Stream finished.', 'thread_id': final_thread_id})}\n\n"
            # else: Stream ended due to interruption, feedback event sent above

    # --- (arun method likely needs complete rewrite or removal for research pipeline) ---
    # async def arun(self, user_input: str):
    #    ...
