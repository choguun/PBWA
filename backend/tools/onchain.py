import requests
import logging
import os
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# RSK Testnet Block Explorer API (Blockscout based)
RSK_TESTNET_EXPLORER_API_URL = "https://backend.explorer.testnet.rsk.co/api/v2"
RSK_EXPLORER_API_KEY = os.getenv("RSK_EXPLORER_API_KEY") # Get optional API key

if not RSK_EXPLORER_API_KEY:
    logger.warning("RSK_EXPLORER_API_KEY environment variable not set. Rate limits may apply.")

def get_address_transaction_history(address: str) -> Dict[str, Any]:
    """Fetches the transaction history for a given address from the RSK Testnet Explorer API.

    Args:
        address: The blockchain address.

    Returns:
        A dictionary containing the transaction list or an error message.
    """
    params = {
        "module": "account",
        "action": "gettxlist",
        "address": address,
        "startblock": 0,
        "endblock": 99999999, # Use a large number for latest
        "page": 1,
        "offset": 50, # Limit the number of transactions returned
        "sort": "desc" # Get most recent first
    }
    # Add API key if available
    if RSK_EXPLORER_API_KEY:
        params['apikey'] = RSK_EXPLORER_API_KEY
        
    url = RSK_TESTNET_EXPLORER_API_URL # Blockscout API v1 uses /api path directly
    # Note: The base URL provided is for v2, which might structure modules differently.
    # Assuming v1 compatibility for module=account&action=gettxlist endpoint pattern.
    # If using v2 strictly, the path might be different, e.g., /addresses/{address}/transactions
    
    # Reconstruct URL for expected v1 style endpoint if needed based on API docs
    # For now, assuming params work with the base URL as if it were v1
    url_v1_style = "https://backend.explorer.testnet.rsk.co/api"

    logger.info(f"Calling RSK Explorer API for transaction history of address {address}")
    logger.debug(f"Request URL (v1 style assumed): {url_v1_style} with params: {params}")

    try:
        response = requests.get(url_v1_style, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()

        if data.get("status") == "1" and data.get("message") == "OK":
            transactions = data.get("result", [])
            logger.info(f"Successfully retrieved {len(transactions)} transactions for address {address}.")
            # Simplify the output slightly
            simplified_txs = []
            for tx in transactions:
                simplified_txs.append({
                    "hash": tx.get("hash"),
                    "from": tx.get("from"),
                    "to": tx.get("to"),
                    "value_rbtc": float(tx.get("value", 0)) / (10**18), # Convert Wei to RBTC
                    "timestamp": tx.get("timeStamp"),
                    "functionName": tx.get("functionName"), # e.g., "transfer(address,uint256)"
                    "isError": tx.get("isError", "0")
                })
            return {"transactions": simplified_txs}
        elif data.get("message") == "No transactions found":
             logger.info(f"No transactions found for address {address}.")
             return {"transactions": []}
        else:
            error_msg = data.get("message", "Unknown error")
            logger.error(f"RSK Explorer API returned an error for {address}: {error_msg}")
            return {"error": f"RSK Explorer API error: {error_msg}", "details": data}

    except requests.exceptions.HTTPError as http_err:
        status_code = response.status_code if 'response' in locals() else None
        logger.error(f"HTTP error occurred calling RSK Explorer API ({status_code}): {http_err}")
        return {"error": f"HTTP error calling RSK Explorer API ({status_code}): {http_err}", "status_code": status_code}
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request exception occurred calling RSK Explorer API: {req_err}")
        return {"error": f"Request exception calling RSK Explorer API: {req_err}"}
    except Exception as e:
        logger.error(f"An unexpected error occurred calling RSK Explorer API: {e}", exc_info=True)
        return {"error": f"An unexpected error occurred calling RSK Explorer API: {e}"}

# Example Usage (for testing)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # Load .env for testing if needed
    from dotenv import load_dotenv
    load_dotenv()
    RSK_EXPLORER_API_KEY = os.getenv("RSK_EXPLORER_API_KEY") # Reload after dotenv
    
    print("--- Testing RSK Explorer API Tool --- ")
    test_address = "0x542b6454668c7aBF9976aF5B5566565631190917" # Example address
    print(f"\nFetching transactions for: {test_address}")
    tx_history = get_address_transaction_history(test_address)
    print(json.dumps(tx_history, indent=2))

    # Test address with potentially no txs
    zero_tx_address = "0x000000000000000000000000000000000000dEaD"
    print(f"\nFetching transactions for: {zero_tx_address}")
    no_txs = get_address_transaction_history(zero_tx_address)
    print(json.dumps(no_txs, indent=2)) 