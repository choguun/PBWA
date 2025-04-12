from pydantic import BaseModel, Field
from typing import List, Tuple, Optional, Annotated, Dict, Any
from typing_extensions import TypedDict
import operator

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

class PipelineState(TypedDict):
    user_profile: Optional[Dict[str, Any]]
    user_query: str
    research_plan: Optional[List[str]]
    collected_data: List[Tuple[str, Any]]
    processed_data: Dict[str, List[Dict]]
    analysis_results: Optional[str]
    strategy_proposals: Optional[str]
    current_step_output: Optional[Any]
    error_log: List[str]

class Plan(BaseModel):
    """Plan to follow."""
    steps: List[str] = Field(description="Steps to follow, in order.")

class ReplannerOutput(BaseModel):
    """Output structure for the Replanner LLM call."""
    final_answer: Optional[str] = Field(default=None, description="The final answer to the user's query, if determined.")
    plan: Optional[Plan] = Field(default=None, description="A new plan with next steps, if further action is needed.")

class SendEthInput(BaseModel):
    to_address: str = Field(description="The recipient Ethereum address.")
    amount_eth: float = Field(description="The amount of ETH to send.")

class ScrapeWebsiteInput(BaseModel):
    url: str = Field(description="The URL of the website to scrape.")

class UserProfilePayload(BaseModel):
    """ Defines the structure for the /invoke endpoint request body. """
    user_query: str = Field(..., description="The main query or research topic from the user.")
    user_profile: Optional[Dict[str, Any]] = Field(None, description="Optional dictionary containing user personalization data (risk, goals, etc.).")

class Plan(BaseModel):
    steps: List[str] = Field(..., description="List of sequential steps for the agent to execute.")

class ReplannerOutput(BaseModel):
    replan: bool = Field(..., description="Indicates if replanning is necessary.")
    new_plan: Optional[List[str]] = Field(None, description="The revised plan steps, if replanning occurred.")

# --- New Tool Schemas ---
class DefiLlamaInput(BaseModel):
    protocol_slug: str = Field(description="The protocol slug (e.g., 'aave', 'uniswap') as used by DefiLlama.")
    metric: Optional[str] = Field(None, description="Optional specific metric to fetch (e.g., 'tvl', 'volume'). If None, fetches general protocol data.")

class CoinGeckoInput(BaseModel):
    token_id: Optional[str] = Field(None, description="The CoinGecko token ID (e.g., 'bitcoin', 'ethereum', 'aave'). Use this OR contract_address.")
    contract_address: Optional[str] = Field(None, description="The token contract address.")
    asset_platform_id: Optional[str] = Field(None, description="The CoinGecko asset platform ID (e.g., 'ethereum', 'polygon-pos') required if using contract_address.")
    include_market_data: bool = Field(True, description="Whether to include market data (price, market cap, volume) when fetching full coin data.")

    # Add validation later if needed to ensure either token_id OR (contract_address + asset_platform_id) is provided

class TwitterInput(BaseModel):
    query: str = Field(..., description="The search query for finding recent tweets. Use standard Twitter search operators.")
    max_results: int = Field(10, description="Maximum number of tweets to return (between 10 and 100).", ge=10, le=100)

class OnChainTxHistoryInput(BaseModel):
    address: str = Field(..., description="The blockchain address (e.g., 0x...) for which to fetch transaction history.")
    # Optional: Add startblock, endblock, page, offset, sort later if needed

# --- Request/Response Schemas ---
# Remove unused QueryRequest and QueryResponse

class UserProfilePayload(BaseModel):
    """ Defines the structure for the /invoke endpoint request body. """
    user_query: str = Field(..., description="The main query or research topic from the user.")

class Plan(BaseModel):
    steps: List[str] = Field(..., description="List of sequential steps for the agent to execute.")

class ReplannerOutput(BaseModel):
    replan: bool = Field(..., description="Indicates if replanning is necessary.")
    new_plan: Optional[List[str]] = Field(None, description="The revised plan steps, if replanning occurred.")

# --- New Tool Schemas ---
class DefiLlamaInput(BaseModel):
    protocol_slug: str = Field(description="The protocol slug (e.g., 'aave', 'uniswap') as used by DefiLlama.")
    metric: Optional[str] = Field(None, description="Optional specific metric to fetch (e.g., 'tvl', 'volume'). If None, fetches general protocol data.")

class CoinGeckoInput(BaseModel):
    token_id: Optional[str] = Field(None, description="The CoinGecko token ID (e.g., 'bitcoin', 'ethereum', 'aave'). Use this OR contract_address.")
    contract_address: Optional[str] = Field(None, description="The token contract address.")
    asset_platform_id: Optional[str] = Field(None, description="The CoinGecko asset platform ID (e.g., 'ethereum', 'polygon-pos') required if using contract_address.")
    include_market_data: bool = Field(True, description="Whether to include market data (price, market cap, volume) when fetching full coin data.")

    # Add validation later if needed to ensure either token_id OR (contract_address + asset_platform_id) is provided

class TwitterInput(BaseModel):
    query: str = Field(..., description="The search query for finding recent tweets. Use standard Twitter search operators.")
    max_results: int = Field(10, description="Maximum number of tweets to return (between 10 and 100).", ge=10, le=100)