from langchain_core.tools import tool
import logging
from web3 import Web3
from .wallet import EVMWallet
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from .schemas import SendEthInput, ScrapeWebsiteInput

logger = logging.getLogger(__name__)

# Instantiate the wallet once
ev_wallet = EVMWallet()
goat_client = ev_wallet.get_client()
w3_instance = ev_wallet.get_web3_instance()
account_address = ev_wallet.get_account().address

# Minimal ABI for ERC20 standard functions needed
ERC20_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "symbol",
        "outputs": [{"name": "", "type": "string"}],
        "type": "function",
    },
]

TOKENS_TO_CHECK = [
    # {
    #     "symbol": "USDC", 
    #     "address": "0x1c7D4B196Cb0C7B01d743Fbc6116a902379C7a08", # Example: Sepolia USDC
    #     "decimals": 6
    # },
    # {
    #     "symbol": "PEPE", 
    #     "address": "0xADDRESS_FOR_PEPE_ON_YOUR_NETWORK", # Replace!
    #     "decimals": 18 # Common default, but verify
    # },
]

@tool
def portfolio_retriever() -> str:
    """
    Retrieves the native token (ETH) balance and balances for a predefined list of ERC20 tokens
    (e.g., USDC, PEPE) using the configured wallet address.
    """
    logger.info(f"Starting portfolio retrieval for address: {account_address}")
    balances = {}
    
    # Check network connection
    if not w3_instance.is_connected():
        error_msg = "Web3 connection error. Cannot retrieve portfolio."
        logger.error(error_msg)
        return error_msg
    
    logger.info(f"Connected to chain ID: {w3_instance.eth.chain_id}")
    
    try:
        # 1. Get Native Token (ETH) Balance
        try:
            eth_balance_wei = w3_instance.eth.get_balance(account_address)
            eth_balance = Web3.from_wei(eth_balance_wei, 'ether')
            balances['ETH'] = f"{eth_balance:.6f}" 
            logger.info(f"  - ETH Balance: {balances['ETH']}")
        except Exception as e:
            logger.error(f"  - Error fetching ETH balance: {e}", exc_info=True)
            balances['ETH'] = "Error"

        # 2. Get ERC20 Token Balances
        for token_info in TOKENS_TO_CHECK:
            token_symbol = token_info.get("symbol", "Unknown")
            token_address_str = token_info.get("address")
            default_decimals = token_info.get("decimals", 18)
            
            if not token_address_str or "0xADDRESS_FOR_" in token_address_str: # Skip if placeholder address
                 logger.warning(f"  - Skipping token {token_symbol} due to missing or placeholder address.")
                 continue

            logger.debug(f"  - Checking balance for {token_symbol} ({token_address_str})")
            try:
                contract_address = Web3.to_checksum_address(token_address_str)
                contract = w3_instance.eth.contract(address=contract_address, abi=ERC20_ABI)

                # Fetch symbol and decimals dynamically, falling back to defined values
                try:
                    fetched_symbol = contract.functions.symbol().call()
                    if fetched_symbol: token_symbol = fetched_symbol # Use fetched symbol if valid
                except Exception as e:
                    logger.warning(f"    - Could not fetch symbol for {token_address_str}, using default '{token_symbol}'. Error: {e}")
                
                try:
                    fetched_decimals = contract.functions.decimals().call()
                    token_decimals = fetched_decimals # Use fetched decimals if valid
                except Exception as e:
                    logger.warning(f"    - Could not fetch decimals for {token_symbol} ({token_address_str}), using default {default_decimals}. Error: {e}")
                    token_decimals = default_decimals

                # Fetch balance
                token_balance_raw = contract.functions.balanceOf(account_address).call()
                token_balance = token_balance_raw / (10 ** token_decimals)
                balances[token_symbol] = f"{token_balance:.6f}" # Format for readability
                logger.info(f"  - {token_symbol} Balance: {balances[token_symbol]}")

            except Exception as e:
                logger.error(f"  - Error fetching balance for token {token_symbol} ({token_address_str}): {e}", exc_info=True)
                balances[token_symbol] = "Error"

        # Format the output string
        if not balances:
            return "Could not retrieve any portfolio balances."

        output_lines = [f"Portfolio for address: {account_address}"]
        for token, balance in balances.items():
            output_lines.append(f"- {token}: {balance}")
        logger.info("Portfolio retrieval finished.")
        return "\n".join(output_lines)

    except Exception as e:
        logger.exception(f"Unexpected error during portfolio retrieval: {e}") # Use logger.exception for unexpected errors
        return f"Unexpected error retrieving portfolio: {e}"

# --- send_ethereum tool (remains the same) ---
@tool(args_schema=SendEthInput)
async def send_ethereum(to_address: str, amount_eth: float) -> str:
    """Sends a specified amount of Ether (ETH) to a given address."""
    logger.info(f"Attempting to send {amount_eth} ETH to {to_address}")
    try:
        # Convert amount to wei before sending
        amount_wei = Web3.to_wei(amount_eth, 'ether')
        logger.debug(f"Amount in wei: {amount_wei}")
        
        # The goat_client.transact method uses web3.py methods internally
        # For simple ETH transfer, it expects a dictionary similar to web3.eth.send_transaction
        tx_params = {
            'to': Web3.to_checksum_address(to_address),
            'value': amount_wei,
            # 'gas': ..., # Optional: Let web3 estimate or set manually
            # 'gasPrice': ..., # Optional: Let web3 estimate or set manually
            # 'nonce': w3_instance.eth.get_transaction_count(account_address) # Optional: Let web3 handle
        }
        logger.debug(f"Transaction parameters for goat_client: {tx_params}")
        
        # Use the goat client to send the transaction
        tx_hash_bytes = await goat_client.transact(tx_params)
        tx_hash = tx_hash_bytes.hex() # Convert bytes to hex string
        logger.info(f"Transaction submitted with hash: {tx_hash}")
        
        # Wait for the transaction receipt for confirmation
        logger.info(f"Waiting for transaction receipt for {tx_hash}...")
        receipt = await goat_client.wait_for_transaction_receipt(tx_hash_bytes)
        logger.info(f"Transaction confirmed. Receipt: {receipt}")
        
        return f"Successfully sent {amount_eth} ETH to {to_address}. Transaction Hash: {tx_hash}"
    except Exception as e:
        logger.exception(f"Error sending ETH to {to_address}: {e}") # Use logger.exception here
        return f"Error sending ETH: {e}"

@tool(args_schema=ScrapeWebsiteInput)
async def scrape_website_content(url: str) -> str:
    """Fetches and returns the main textual content of a given URL using a headless browser."""
    logger.info(f"Attempting to scrape content from URL: {url}")
    html_content = ""
    text_content = "Error: Could not extract text content."
    
    # Input validation
    if not url.startswith(("http://", "https://")):
        return "Error: Invalid URL format. Please provide a full URL starting with http:// or https://"
        
    try:
        async with async_playwright() as p:
            # Launch browser (consider chromium, firefox, or webkit)
            # Headless=True runs without opening a visible browser window
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            logger.debug(f"Navigating to {url}")
            
            # Navigate and wait for page to load (adjust timeout as needed)
            await page.goto(url, wait_until="domcontentloaded", timeout=30000) # 30 second timeout
            logger.debug(f"Page loaded. Extracting content.")
            
            # Extract main content - this often needs site-specific selectors for best results
            # Using body text as a generic fallback
            # For specific sites (like vfat), you'd use more targeted selectors: 
            # e.g., await page.locator('#apy-element-id').text_content()
            html_content = await page.content()
            body_text = await page.locator('body').inner_text()
            
            # Basic cleaning (can be much more sophisticated)
            text_content = '\n'.join([line.strip() for line in body_text.split('\n') if line.strip()])
            
            logger.info(f"Successfully scraped {len(text_content)} characters from {url}")
            await browser.close()
            
    except PlaywrightTimeoutError:
        logger.error(f"Timeout error while scraping {url}")
        return f"Error: Timeout occurred while trying to load or scrape {url}"
    except Exception as e:
        logger.exception(f"Error scraping URL {url}: {e}") # Log full traceback
        return f"Error scraping {url}: {e}" 
        
    # Return extracted text (consider returning structured data or limiting length)
    # Limit length to avoid overwhelming LLM context
    max_len = 4000
    if len(text_content) > max_len:
        logger.warning(f"Scraped content truncated from {len(text_content)} to {max_len} characters.")
        return text_content[:max_len] + "... [Content Truncated]"
    elif not text_content:
        return "Error: Could not extract meaningful text content from the page body."
    else:
        return text_content

# --- List of tools ---
agent_tools = [portfolio_retriever, send_ethereum, scrape_website_content]
