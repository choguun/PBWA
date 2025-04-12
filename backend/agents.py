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
# Qdrant specific imports
from qdrant_client import models # Import Qdrant models for search
from datetime import datetime # For timestamping time-series data

from .config import GOOGLE_API_KEY, LLM_MODEL, QDRANT_COLLECTION_NAME
from .tools import agent_tools
from .prompts import (
    PLANNER_PROMPT, 
    EXECUTOR_PROMPT, 
    REPLANNER_PROMPT, 
    ANALYZER_PROMPT,
    STRATEGIST_PROMPT # Import STRATEGIST_PROMPT
)
from .schemas import Plan, ReplannerOutput, PipelineState
from .vector_store import qdrant_client_instance, embedding_model_instance # Import DB clients
from qdrant_client.http.models import PointStruct # Import PointStruct for upsert

# InfluxDB specific imports
from influxdb_client import Point

# Import InfluxDB client and config
from .timeseries_db import influxdb_client_instance, get_influxdb_write_api, INFLUXDB_BUCKET, INFLUXDB_ORG

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
        # --- New Analyzer Chain ---
        self.analyzer = self.analyzer_prompt | self.llm | StrOutputParser()
        # --- New Strategist Chain ---
        self.strategist = self.strategist_prompt | self.llm | StrOutputParser()

        # --- Workflow Definition ---
        self.workflow = StateGraph(PipelineState)

        # --- Nodes --- 
        self.workflow.add_node("plan_research", self._plan_research_step)
        self.workflow.add_node("collect_data", self._collect_data_step)
        self.workflow.add_node("process_data", self._process_data_step) 
        self.workflow.add_node("update_vector_store", self._update_vector_store_node) 
        self.workflow.add_node("update_timeseries_db", self._timeseries_db_update_node) # Add new node
        self.workflow.add_node("analyze_data", self._analyze_data_step) # Now uses the implemented function
        self.workflow.add_node("propose_strategy", self._propose_strategy_step) # Add strategy node

        # --- Edges --- 
        self.workflow.set_entry_point("plan_research")
        self.workflow.add_edge("plan_research", "collect_data")
        # Loop logic for data collection
        self.workflow.add_conditional_edges(
            "collect_data",
            self._should_continue_or_process, # Renamed check function
            {"continue": "collect_data", "process": "process_data"} 
        )
        # Processing and storage flow
        self.workflow.add_edge("process_data", "update_vector_store")
        self.workflow.add_edge("process_data", "update_timeseries_db")
        # Both storage nodes now lead to analysis
        self.workflow.add_edge("update_vector_store", "analyze_data") 
        self.workflow.add_edge("update_timeseries_db", "analyze_data")
        # --- Updated Edges: Analysis -> Strategy -> End ---
        self.workflow.add_edge("analyze_data", "propose_strategy") 
        self.workflow.add_edge("propose_strategy", END) 
        
        # Compile the graph
        self.app = self.workflow.compile()
        logger.info("ResearchPipeline graph compiled.")

    # --- Node Implementations --- 
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

        tool_output = "Error: Tool execution failed internally."
        tool_name = "Unknown"
        try:
            # --- Refactored Tool ID, Arg Parsing & Validation --- 
            parts = current_task.split()
            potential_tool_name = parts[1] if len(parts) > 1 else None
            found_tool = self.tool_map.get(potential_tool_name)
            
            if not found_tool:
                 error_msg = f"Tool '{potential_tool_name}' not found for task: {current_task}. Available: {list(self.tool_map.keys())}"
                 logger.error(error_msg)
                 tool_output = f"Error: {error_msg}"
                 errors.append(error_msg)
            else:
                tool_name = found_tool.name
                logger.info(f"Identified tool: {tool_name}. Preparing args and validation.")
                
                validated_args_dict = {} # Arguments to pass to the tool function
                can_execute = True

                # Check if the tool expects arguments via args_schema
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
                    
                    # Validate using Pydantic
                    try:
                        validated_model = schema(**raw_args)
                        validated_args_dict = validated_model.dict() # Use validated, typed args
                        logger.info(f"Arguments validated successfully for '{tool_name}': {validated_args_dict}")
                    except Exception as validation_err: # Catches Pydantic validation errors
                        error_msg = f"Argument validation failed for tool '{tool_name}'. Task: '{current_task}'. Extracted raw args: {raw_args}. Error: {validation_err}"
                        logger.error(error_msg)
                        tool_output = f"Error: {error_msg}"
                        errors.append(error_msg)
                        can_execute = False # Prevent calling tool
                else:
                    # Tool expects no arguments
                    logger.info(f"Tool '{tool_name}' does not have an args_schema. Assuming no arguments.")
                    if len(parts) > 2: # Basic check if extraneous text exists after tool name
                        logger.warning(f"Task string '{current_task}' contained extra text after tool name '{tool_name}', but tool expects no args.")

                # Execute the tool if possible
                if can_execute:
                    func = found_tool.func
                    if inspect.iscoroutinefunction(func):
                        tool_output = await func(**validated_args_dict)
                    else:
                        tool_output = func(**validated_args_dict)
                    logger.info(f"Direct Tool Call Output received for '{tool_name}'.")
                    logger.debug(f"Tool output snippet: {str(tool_output)[:200]}...")

        except Exception as e:
            error_msg = f"Unexpected error during tool execution for task '{current_task}': {e}"
            logger.error(error_msg, exc_info=True)
            tool_output = f"Error executing task: {e}"
            errors.append(error_msg)
            
        # Update state
        newly_collected = (current_task, tool_output)
        updated_collected_data = collected + [newly_collected]
        
        return {
            "research_plan": remaining_plan, 
            "collected_data": updated_collected_data,
            "current_step_output": tool_output,
            "error_log": errors
        }

    def _should_continue_or_process(self, state: PipelineState):
        # Continue collecting if plan is not empty, otherwise move to process
        if state.get("research_plan"):
            logger.info("Plan has remaining steps, continuing collection.")
            return "continue"
        else:
            logger.info("Plan finished, moving to process data.")
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
            # --- Process for Vector Store (Textual Data) ---
            if isinstance(data, str) and not data.startswith("Error:"):
                # Simple processing: treat strings as documents
                doc = {
                    "page_content": data, 
                    "metadata": { 
                        "source_task": task,
                        "collection_step": i 
                        # TODO: Add more metadata (URL, tool name, timestamp)
                    }
                }
                processed_for_vector.append(doc)
            elif isinstance(data, dict) and 'error' not in data:
                # If data is a dict (e.g., from API tools) and not an error, 
                # maybe store its JSON string in vector store or just process for timeseries.
                # For now, let's store a summary/JSON string.
                doc_content = json.dumps(data, indent=2, default=str)[:4000] # Limit length
                doc = {
                    "page_content": f"Data from task: {task}\n{doc_content}",
                    "metadata": {
                        "source_task": task,
                        "collection_step": i,
                        "data_type": "api_result"
                    }
                }
                processed_for_vector.append(doc)
            else:
                # Handle errors or unexpected types for vector store
                logger.warning(f"Skipping vector processing for data from task '{task}' due to type/error: {type(data)}")

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

        logger.info(f"Processed {len(processed_for_vector)} documents for vector store.")
        logger.info(f"Processed {len(processed_for_timeseries)} points for time-series store.")
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

        # --- Format Collected Data for Prompt ---
        collected_data_str = ""
        if collected_data:
            formatted_items = []
            for task, result in collected_data:
                 result_preview = str(result)[:300] # Limit length
                 ellipsis = "..." if len(str(result)) > 300 else ""
                 # Construct string parts separately, avoid complex f-string with newline
                 task_line = f"- Task: {task}"
                 result_line = f"  Result: {result_preview}{ellipsis}"
                 # Combine parts for this item
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
        analysis_results = "Analysis failed."
        try:
            analysis_results = await self.analyzer.ainvoke({
                "user_query": user_query,
                "user_profile": user_profile_str, # Pass profile to analyzer
                "retrieved_context": retrieved_context_str,
                "collected_data": collected_data_str
            })
            logger.info("Analysis generated successfully.")
            logger.debug(f"Analysis Result Snippet: {analysis_results[:200]}...")
        except Exception as e:
            error_msg = f"Error during LLM Analysis step: {e}"
            logger.error(error_msg, exc_info=True)
            errors.append(error_msg)
            analysis_results = f"Error generating analysis: {e}"
        
        return {
            "analysis_results": analysis_results,
            "error_log": errors # Pass on any accumulated errors
        }

    # --- New Strategy Proposal Node --- 
    async def _propose_strategy_step(self, state: PipelineState):
        logger.info("--- Proposing Strategy ---")
        user_query = state.get('user_query')
        user_profile = state.get('user_profile') # Optional
        analysis_results = state.get('analysis_results')
        errors = state.get('error_log', [])

        if not analysis_results:
            logger.error("Strategy step failed: Analysis results are missing.")
            return {"error_log": errors + ["Strategy proposal failed: Missing analysis results."]}

        # Format user profile for prompt (if available)
        user_profile_str = json.dumps(user_profile, indent=2) if user_profile else "Not Provided"

        logger.info("Invoking Strategist LLM...")
        strategy_proposals = "Strategy proposal failed."
        try:
            strategy_proposals = await self.strategist.ainvoke({
                "user_query": user_query or "Not Provided",
                "user_profile": user_profile_str,
                "analysis_results": analysis_results
            })
            logger.info("Strategy proposals generated successfully.")
            logger.debug(f"Strategy Proposals Snippet: {strategy_proposals[:200]}...")
        except Exception as e:
            error_msg = f"Error during LLM Strategy proposal step: {e}"
            logger.error(error_msg, exc_info=True)
            errors.append(error_msg)
            strategy_proposals = f"Error generating strategies: {e}"
        
        return {
            "strategy_proposals": strategy_proposals,
            "error_log": errors # Pass on any accumulated errors
        }

    # --- Streaming Method Update ---
    async def astream_events(self, user_query: str, user_profile: Optional[Dict[str, Any]] = None) -> AsyncGenerator[str, None]: # Add user_profile param
        logger.info(f"--- Starting research pipeline stream for Query: {user_query} ---")
        # TODO: Implement actual loading/fetching of user profile if not passed directly
        if user_profile:
             logger.info(f"Using provided User Profile: {list(user_profile.keys())}")
        else:
             logger.info("No user profile provided, proceeding without personalization.")
             # Optionally load a default or guest profile here

        initial_state = PipelineState(
            user_profile=user_profile, # Use passed or loaded profile
            user_query=user_query,
            research_plan=None,
            collected_data=[],
            processed_data={},
            analysis_results=None,
            strategy_proposals=None,
            current_step_output=None,
            error_log=[],
        )
        
        # --- Create start event data payload separately --- 
        start_data_payload = {
            'input': user_query,
            'profile_keys': list(user_profile.keys()) if user_profile else []
        }
        start_data_json = json.dumps(start_data_payload)
        # Yield the correctly formatted SSE string
        yield f"event: start\ndata: {start_data_json}\n\n"
        
        config = {"recursion_limit": 50} 
        sse_event_type = "progress" # Default event type
        
        try:
            async for event in self.app.astream(initial_state, config=config):
                logger.debug(f"Workflow Event: {event}")
                node_name = list(event.keys())[0]
                node_output = event[node_name]
                
                sse_data = {} # Default empty data
                sse_event_type = None # Reset event type
                
                # --- Map node outputs to SSE events --- 
                if node_name == "plan_research":
                    sse_event_type = "plan"
                    sse_data = {"type": "plan", "steps": node_output.get("research_plan", [])}
                elif node_name == "collect_data":
                     sse_event_type = "tool_result"
                     sse_data = {"type": "tool_result", "result": node_output.get("current_step_output")}
                elif node_name == "process_data":
                    sse_event_type = "processing"
                    vector_count = len(node_output.get("processed_data", {}).get("vector", []))
                    ts_count = len(node_output.get("processed_data", {}).get("timeseries", []))
                    sse_data = {"type": "processing", "message": f"Processed {vector_count} vector docs, {ts_count} time-series points."}
                elif node_name == "update_vector_store":
                    sse_event_type = "storage"
                    sse_data = {"type": "storage", "message": node_output.get("current_step_output", "Storage update completed.")}
                elif node_name == "update_timeseries_db":
                    sse_event_type = "storage_ts"
                    sse_data = {"type": "storage_ts", "message": node_output.get("current_step_output", "Time-series update completed.")}
                elif node_name == "analyze_data":
                    sse_event_type = "analysis"
                    sse_data = {"type": "analysis", "result": node_output.get("analysis_results")}
                # --- Handle Strategy Node --- 
                elif node_name == "propose_strategy":
                    sse_event_type = "strategy"
                    sse_data = {"type": "strategy", "proposals": node_output.get("strategy_proposals")}
                # --- Handle End Node --- 
                elif node_name == END:
                     sse_event_type = "end"
                     # Use strategy_proposals as the final response now
                     final_resp = node_output.get("strategy_proposals", "Pipeline finished.") 
                     sse_data = {"type": "final_response", "response": final_resp}

                # Only yield if we have a specific event type assigned
                if sse_event_type:
                    yield f"event: {sse_event_type}\ndata: {json.dumps(sse_data)}\n\n"
                
                if node_name == END:
                    break 

        except Exception as e:
            logger.error(f"Exception during pipeline stream: {e}", exc_info=True)
            error_data = {"type": "error", "message": str(e)}
            yield f"event: error\ndata: {json.dumps(error_data)}\n\n"
        finally:
            logger.info(f"--- Pipeline stream finished for Query: {user_query} ---")
            if sse_event_type != 'end':
                yield f"event: end\ndata: {json.dumps({'message': 'Stream ended.'})}\n\n"

    # --- (arun method likely needs complete rewrite or removal for research pipeline) ---
    # async def arun(self, user_input: str):
    #    ...
