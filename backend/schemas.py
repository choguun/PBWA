from pydantic import BaseModel, Field
from typing import List, Tuple, Optional, Annotated, Dict, Any, Union
from typing_extensions import TypedDict
import operator
from langchain_core.documents import Document
from influxdb_client import Point

# --- Analysis Result Schema (Moved Before PipelineState) ---
class AnalysisResult(BaseModel):
    is_sufficient: bool = Field(..., description="Whether the gathered data (collected, vector store, time-series) was sufficient to comprehensively answer the original user query.")
    analysis_text: str = Field(..., description="The detailed analysis synthesizing all available information.")
    reasoning: Optional[str] = Field(None, description="Brief explanation for why the data is sufficient or insufficient.")
    suggestions_for_next_steps: Optional[str] = Field(None, description="If insufficient, suggested next steps or missing information needed for a better analysis (used for replanning)." )

# --- Tool Input Schemas ---
class SendEthInput(BaseModel):
    to_address: str = Field(description="The recipient Ethereum address.")
    amount_eth: float = Field(description="The amount of ETH to send.")

class ScrapeWebsiteInput(BaseModel):
    url: str = Field(description="The URL of the website to scrape.")

class DefiLlamaInput(BaseModel):
    protocol_slug: str = Field(description="The protocol slug (e.g., 'aave', 'uniswap') as used by DefiLlama.")

class CoinGeckoInput(BaseModel):
    token_id: Optional[str] = Field(None, description="The CoinGecko token ID (e.g., 'bitcoin', 'ethereum', 'aave'). Use this OR contract_address.")
    contract_address: Optional[str] = Field(None, description="The token contract address.")
    asset_platform_id: Optional[str] = Field(None, description="The CoinGecko asset platform ID (e.g., 'ethereum', 'polygon-pos') required if using contract_address.")
    include_market_data: bool = Field(True, description="Whether to include market data (price, market cap, volume) when fetching full coin data.")

class TwitterInput(BaseModel):
    query: str = Field(..., description="The search query for finding recent tweets. Use standard Twitter search operators.")
    max_results: int = Field(10, description="Maximum number of tweets to return (between 10 and 100).", ge=10, le=100)

class OnChainTxHistoryInput(BaseModel):
    address: str = Field(..., description="The blockchain address (e.g., 0x...) for which to fetch transaction history.")

class TimeSeriesInput(BaseModel):
    measurement: str = Field(..., description="The InfluxDB measurement to query (e.g., 'token_market_data', 'protocol_metrics').")
    tags: Optional[Dict[str, str]] = Field(None, description="Dictionary of tag key-value pairs to filter by (e.g., {'token_id': 'bitcoin'}).")
    fields: Optional[List[str]] = Field(None, description="Specific fields to retrieve (e.g., ['price_usd', 'market_cap_usd']). If None, retrieves all fields.")
    start_time: str = Field("-1h", description="Start time for the query range (e.g., '-7d', '-1h', or RFC3339 timestamp '2023-01-01T00:00:00Z'). Defaults to -1h.")
    stop_time: Optional[str] = Field(None, description="End time for the query range (e.g., 'now()', or RFC3339 timestamp). Defaults to now().")
    limit: int = Field(100, description="Maximum number of data points to return.")

class VfatInput(BaseModel):
    farm_url: str = Field(..., description="The specific URL of the vfat.io farm page to scrape.")

# --- New Schema for Document Parsing Tool ---
class DocumentParseInput(BaseModel):
    file_path: str = Field(..., description="The relative path to the local document file (e.g., 'docs/my_report.pdf') within the workspace.")

# --- API Payload Schemas ---
class UserProfilePayload(BaseModel):
    """ Defines the structure for the /invoke endpoint request body. """
    user_query: str = Field(..., description="The main query or research topic from the user.")
    user_profile: Optional[Dict[str, Any]] = Field(None, description="Optional dictionary containing user personalization data (risk, goals, etc.).")

class ResumePayload(BaseModel):
    """ Defines the structure for the /resume endpoint request body. """
    thread_id: str = Field(..., description="The state ID (thread_id) of the paused pipeline instance to resume.")
    feedback: Optional[str] = Field(None, description="Optional user feedback provided during the pause.")

# --- Pipeline State Definition (Moved After Payload Schemas) ---
class PipelineState(TypedDict):
    user_profile: Optional[UserProfilePayload] = None
    user_query: Optional[str] = None
    research_plan: Optional[List[str]] = None
    current_step: Optional[int] = None
    current_step_output: Optional[Any] = None
    collected_data: Optional[List[Tuple[str, Any]]] = None
    processed_data: Optional[Dict[str, List[Union[Document, Point]]]] = None
    analysis_results: Optional[AnalysisResult] = None
    strategy_proposals: Optional[str] = None
    error_log: Optional[List[str]] = None
    feedback: Optional[str] = None

# --- Planning & Replanning Schemas ---
class Plan(BaseModel):
    """Plan to follow."""
    steps: List[str] = Field(description="Steps to follow, in order.")

class ReplannerOutput(BaseModel):
    """Output structure for the Replanner LLM call."""
    replan: bool = Field(..., description="Indicates if replanning is necessary.")
    new_plan: Optional[List[str]] = Field(None, description="The revised plan steps, if replanning occurred.")

# Remove unused QueryRequest and QueryResponse definitions
# class QueryRequest(BaseModel):
#     pass
# class QueryResponse(BaseModel):
#     response: str