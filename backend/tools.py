from langchain_core.tools import tool
from langchain.prompts import PromptTemplate
from langchain_google_vertexai import ChatVertexAI
from langchain.schema import AIMessage
import os
import json
import re

# Import EVMWallet
from .wallet import EVMWallet
# Import Goat SDK EVM tools
from goat_wallets.evm import send_eth
from pydantic import BaseModel, Field

# Instantiate the wallet once (assuming tools.py is loaded once)
# If tools.py could be loaded multiple times in complex scenarios, consider a singleton pattern or passing the wallet instance.
ev_wallet = EVMWallet()
goat_client = ev_wallet.get_client()

@tool
def portfolio_retriever(prompt: str) -> str:
    """Retrieves portfolio information. Information returned must be information on the portfolio."""
    print("Using Portfolio Retriever tool now")
    return "This tool is a placeholder. Implement actual portfolio logic here."

# Define input schema for the send_ethereum tool for clarity and validation
class SendEthInput(BaseModel):
    to_address: str = Field(description="The recipient Ethereum address.")
    amount_eth: float = Field(description="The amount of ETH to send.")

@tool(args_schema=SendEthInput)
async def send_ethereum(to_address: str, amount_eth: float) -> str:
    """Sends a specified amount of Ether (ETH) to a given address."""
    print(f"Attempting to send {amount_eth} ETH to {to_address}")
    try:
        # The goat_client.transact method from Web3EVMWalletClient expects parameters
        # suitable for the underlying web3.py function. `send_eth` helps format this.
        tx_params = send_eth(to=to_address, value=amount_eth)
        
        # Use the goat client to send the transaction
        tx_hash = await goat_client.transact(tx_params)
        receipt = await goat_client.wait_for_transaction_receipt(tx_hash)
        print(f"Transaction Receipt: {receipt}")
        
        return f"Successfully initiated ETH transfer. Transaction Hash: {tx_hash}"
    except Exception as e:
        print(f"Error sending ETH: {e}")
        return f"Error sending ETH: {e}"

# Add other tools here as needed

# List of tools to be used by the agent
# Make sure to add the new tool here
agent_tools = [portfolio_retriever, send_ethereum]
