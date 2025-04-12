from langchain_core.tools import Tool, tool
import logging
from web3 import Web3
from ..wallet import EVMWallet # Adjusted relative import
# Remove Playwright import
# from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError 
from ..schemas import SendEthInput, ScrapeWebsiteInput, DefiLlamaInput, CoinGeckoInput, TwitterInput, OnChainTxHistoryInput, TimeSeriesInput, VfatInput # Import TwitterInput, OnChain schema and TimeSeriesInput
# Import implementation functions from sibling modules
from .web_scraper import scrape_website_content 
from .defi_llama import call_defi_llama_api
from .coingecko import call_coingecko_api # Import CoinGecko function
from .twitter import search_recent_tweets # Import Twitter function
from .onchain import get_address_transaction_history # Import onchain function
from .timeseries_retriever import query_time_series_data # Import timeseries function
from .vfat_scraper import scrape_vfat_farm # Import vfat scraper function
import json
import asyncio # Import asyncio
from typing import Optional, Dict, List, Union, Any

logger = logging.getLogger(__name__)

# Instantiate the wallet once
ev_wallet = EVMWallet()
goat_client = ev_wallet.get_client()
w3_instance = ev_wallet.get_web3_instance()
account_address = ev_wallet.get_account().address

# Minimal ABI for ERC20 standard functions needed
ERC20_ABI = [ 
    # ... ABI content ...
]

TOKENS_TO_CHECK = [ 
    # ... Token list ...
]

# --- Tool Implementations (will be moved) ---
# Placeholder for where scrape_website_content was
# Placeholder for where call_defi_llama_api was

# --- Tool Definitions --- 
def get_portfolio_retriever():
    """Fetches the user's current crypto portfolio balances and values."""
    # TODO: Integrate with backend/wallet.py to get actual data
    # For now, return dummy data
    logger.info("Executing dummy portfolio_retriever tool.")
    return {
        "ETH": {"balance": 2.5, "value_usd": 7500.0},
        "USDC": {"balance": 10000, "value_usd": 10000.0},
        "AAVE": {"balance": 50, "value_usd": 4500.0}
    }

portfolio_retriever = Tool(
    name="portfolio_retriever",
    func=get_portfolio_retriever,
    description="Fetches the user's current crypto portfolio balances and values.",
)

# Update Tool definition to use the async function directly
web_scraper = Tool(
    name="web_scraper",
    func=scrape_website_content, # Use the async function directly
    description="Scrapes content from a given website URL. Use for specific information not available via APIs.",
    args_schema=ScrapeWebsiteInput,
    # Modern Langchain/LangGraph often infers coroutine automatically when func is async
    # but specifying can be clearer if needed: coroutine=scrape_website_content 
)

# Wrapper function for DefiLlama (imports are handled at top level now)
def run_defi_llama_tool(protocol_slug: str, metric: Optional[str] = None) -> str:
    """Wrapper to call the DefiLlama API and return a string representation."""
    logger.info(f"DefiLlama Tool called for protocol: '{protocol_slug}', metric: {metric}")
    try:
        result_dict = call_defi_llama_api(protocol_slug=protocol_slug, metric=metric)
        return json.dumps(result_dict, indent=2) 
    except Exception as e:
        logger.error(f"Unexpected error running DefiLlama tool wrapper: {e}", exc_info=True)
        return f"Unexpected error processing DefiLlama request: {e}"

defi_llama_api_tool = Tool(
    name="defi_llama_api_tool",
    func=run_defi_llama_tool,
    description="Fetches data about DeFi protocols (e.g., TVL, volume, general info) from the DefiLlama API. Use the 'protocol_slug' (like 'aave', 'uniswap') and optionally a 'metric' (like 'tvl').",
    args_schema=DefiLlamaInput
)

# --- CoinGecko API Tool ---
def run_coingecko_tool(
    token_id: Optional[str] = None, 
    contract_address: Optional[str] = None, 
    asset_platform_id: Optional[str] = None,
    include_market_data: bool = True
) -> str:
    """Wrapper to call the CoinGecko API and return a string representation."""
    logger.info(f"CoinGecko Tool called with token_id={token_id}, contract_address={contract_address}, asset_platform_id={asset_platform_id}")
    try:
        result_dict = call_coingecko_api(
            token_id=token_id, 
            contract_address=contract_address, 
            asset_platform_id=asset_platform_id, 
            include_market_data=include_market_data
        )
        return json.dumps(result_dict, indent=2)
    except Exception as e:
        logger.error(f"Unexpected error running CoinGecko tool wrapper: {e}", exc_info=True)
        return f"Unexpected error processing CoinGecko request: {e}"

coingecko_api_tool = Tool(
    name="coingecko_api_tool",
    func=run_coingecko_tool,
    description="Fetches token data (price, market cap, etc.) from the CoinGecko API. Use either 'token_id' (e.g., 'bitcoin', 'aave') OR both 'contract_address' and 'asset_platform_id' (e.g., 'ethereum', 'polygon-pos').",
    args_schema=CoinGeckoInput
)

# --- Twitter API Tool ---
def run_twitter_tool(query: str, max_results: int = 10) -> str:
    """Wrapper to call the Twitter search API and return a string representation."""
    logger.info(f"Twitter Tool called with query='{query}', max_results={max_results}")
    try:
        result_dict = search_recent_tweets(query=query, max_results=max_results)
        # Consider summarizing or selecting key info if the list is very long
        # For now, return the full JSON string
        return json.dumps(result_dict, indent=2)
    except Exception as e:
        logger.error(f"Unexpected error running Twitter tool wrapper: {e}", exc_info=True)
        return f"Unexpected error processing Twitter request: {e}"

twitter_api_tool = Tool(
    name="twitter_api_tool",
    func=run_twitter_tool,
    description="Searches for recent tweets matching a query using the Twitter API v2. Requires a Bearer Token environment variable. Use standard Twitter search operators in the 'query'.",
    args_schema=TwitterInput
)

# --- On-Chain Transaction History Tool ---
def run_onchain_tx_history_tool(address: str) -> str:
    """Wrapper to call the on-chain transaction history API."""
    logger.info(f"OnChain Tx History Tool called for address: {address}")
    try:
        result_dict = get_address_transaction_history(address=address)
        return json.dumps(result_dict, indent=2)
    except Exception as e:
        logger.error(f"Unexpected error running OnChain Tx History tool wrapper: {e}", exc_info=True)
        return f"Unexpected error processing OnChain Tx History request: {e}"

onchain_tx_history_tool = Tool(
    name="onchain_tx_history_tool",
    func=run_onchain_tx_history_tool,
    description="Fetches the recent transaction history for a given blockchain address from the RSK Testnet Explorer API.",
    args_schema=OnChainTxHistoryInput
)

# --- Time Series Retriever Tool ---
def run_time_series_retriever_tool(
    measurement: str,
    tags: Optional[Dict[str, str]] = None,
    fields: Optional[List[str]] = None,
    start_time: str = "-1h",
    stop_time: Optional[str] = None,
    limit: int = 100
) -> str:
    """Wrapper to query the time-series database."""
    logger.info(f"Time Series Retriever Tool called for measurement: {measurement}")
    try:
        result_dict = query_time_series_data(
            measurement=measurement,
            tags=tags,
            fields=fields,
            start_time=start_time,
            stop_time=stop_time,
            limit=limit
        )
        # Convert dict to string for agent consumption
        # Consider summarizing or selecting key info if result_dict is very large
        return json.dumps(result_dict, indent=2)
    except Exception as e:
        logger.error(f"Unexpected error running Time Series Retriever tool wrapper: {e}", exc_info=True)
        return f"Unexpected error processing Time Series Retriever request: {e}"

time_series_retriever_tool = Tool(
    name="time_series_retriever",
    func=run_time_series_retriever_tool,
    description="Retrieves historical time-series data (e.g., token prices, protocol TVL) from the database. Specify measurement, optional tags/fields filters, time range (start_time, stop_time), and limit.",
    args_schema=TimeSeriesInput
)

# --- vfat.tools Scraper Tool (Async) ---
@tool
async def vfat_scraper_tool(farm_url: str) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """(Async) Scrapes farm data (APY, pools) from a specific vfat.tools URL.
    Provide the full URL of the specific farm page (e.g., https://vfat.tools/polygon/quickswap-epoch/).
    NOTE: Results depend heavily on website structure and LLM interpretation; may be experimental or fail.
    Args:
        farm_url (str): The full URL of the vfat.tools farm page.
    """
    # This tool needs access to the state (user_query, user_profile) which isn't directly
    # available here when defined traditionally. LangGraph tools usually operate on inputs.
    # Option 1: Modify ResearchPipeline._collect_data_step to manually inject state.
    # Option 2: Refactor tool definition to accept context (more complex).
    # Option 3: Simplification - For now, we call scrape_vfat_farm without query/profile.
    #           The planner *could* include hints in the farm_url if needed, but it's messy.
    #           Let's proceed with Option 3 for now and acknowledge the limitation.
    logger.warning("vfat_scraper_tool currently executes without user query/profile context due to tool definition limitations.")
    
    # Call the implementation function (which now accepts optional args)
    result = await scrape_vfat_farm(farm_url=farm_url, user_query=None, user_profile=None) 
    
    # The implementation now returns dict/list directly. Tool decorator handles serialization implicitly?
    # Let's ensure the return is handled cleanly for the agent.
    # Langchain tools expect a string return usually, unless used within specific agent types.
    # Since our agent parses JSON string outputs, we should dump the result here.
    return json.dumps(result, default=str) # Ensure JSON string output

# --- send_ethereum tool --- 
# Implementation remains here as it uses local wallet instance
@tool(args_schema=SendEthInput)
async def send_ethereum(to_address: str, amount_eth: float) -> str:
    """Sends a specified amount of Ether (ETH) to a given address."""
    logger.info(f"Attempting to send {amount_eth} ETH to {to_address}")
    try:
        amount_wei = Web3.to_wei(amount_eth, 'ether')
        tx_params = {
            'to': Web3.to_checksum_address(to_address),
            'value': amount_wei,
        }
        logger.debug(f"Transaction parameters for goat_client: {tx_params}")
        tx_hash_bytes = await goat_client.transact(tx_params)
        tx_hash = tx_hash_bytes.hex()
        logger.info(f"Transaction submitted with hash: {tx_hash}")
        logger.info(f"Waiting for transaction receipt for {tx_hash}...")
        receipt = await goat_client.wait_for_transaction_receipt(tx_hash_bytes)
        logger.info(f"Transaction confirmed. Receipt: {receipt}")
        return f"Successfully sent {amount_eth} ETH to {to_address}. Transaction Hash: {tx_hash}"
    except Exception as e:
        logger.exception(f"Error sending ETH to {to_address}: {e}")
        return f"Error sending ETH: {e}"

# --- Web Scraper Content Tool (Direct Async - Remove wrapper if not needed?) ---
# The async version defined with @tool can be used directly if the calling agent handles async tool calls.
# If ResearchPipeline._collect_data_step handles calling async tools correctly, 
# we might not need the synchronous `web_scraper` wrapper `run_web_scraper`.
# Keeping both for now, but consider simplifying.

@tool(args_schema=ScrapeWebsiteInput)
async def scrape_tool_direct(url: str) -> str:
    """Direct async tool for scraping website content."""
    return await scrape_website_content(url)

# List of all tools available to the agent
agent_tools = [
    portfolio_retriever, 
    web_scraper,          # Sync wrapper using asyncio.run
    defi_llama_api_tool,  # Sync wrapper
    coingecko_api_tool,   # Add CoinGecko tool
    twitter_api_tool,     # Add Twitter tool
    onchain_tx_history_tool, # Add On-chain tool
    time_series_retriever_tool, # Add TimeSeries retriever tool
    vfat_scraper_tool,    # Add vfat scraper tool
    send_ethereum,        # Async tool
    scrape_tool_direct    # Async tool (potentially redundant with web_scraper wrapper)
] 