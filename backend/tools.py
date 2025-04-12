from langchain_core.tools import tool
from langchain.prompts import PromptTemplate
from langchain_google_vertexai import ChatVertexAI
from langchain.schema import AIMessage
import os
import json
import re
import logging # Import logging
from web3 import Web3
from pydantic import BaseModel, Field

# Import EVMWallet
from .wallet import EVMWallet
# Import Goat SDK EVM tools
from goat_wallets.evm import send_eth

# Remove goat plugin imports as we define tokens manually
# from goat_plugins.erc20.token import PEPE, USDC
# from goat_plugins.erc20 import erc20, ERC20PluginOptions

from .schemas import SendEthInput

# Get logger instance
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

# --- List of tools ---
agent_tools = [portfolio_retriever, send_ethereum]
