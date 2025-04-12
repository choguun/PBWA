import requests
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

COINGECKO_API_BASE_URL = "https://api.coingecko.com/api/v3"

# Consider adding your CoinGecko API Key here if using the Pro API
# COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")
# HEADERS = {'x-cg-demo-api-key': COINGECKO_API_KEY} if COINGECKO_API_KEY else {}
HEADERS = {} # Using public API for now

def call_coingecko_api(
    token_id: Optional[str] = None, 
    contract_address: Optional[str] = None, 
    asset_platform_id: Optional[str] = None,
    include_market_data: bool = True
) -> Dict[str, Any]:
    """Calls the CoinGecko API to fetch token data by ID or contract address.

    Args:
        token_id: The CoinGecko token ID (e.g., 'bitcoin').
        contract_address: The token contract address.
        asset_platform_id: The platform ID (e.g., 'ethereum') required with contract_address.
        include_market_data: Whether to fetch detailed market data.

    Returns:
        A dictionary containing the API response data or an error message.
    """
    params = {
        "localization": "false",
        "tickers": "false",
        "community_data": "false",
        "developer_data": "false",
        "sparkline": "false",
    }
    # Only include market_data param if needed (it defaults true in API but explicit is clear)
    if include_market_data:
        params["market_data"] = "true"
    else:
         params["market_data"] = "false"
         
    url = None
    
    # Prioritize fetching by contract address if platform ID is provided
    if contract_address and asset_platform_id:
        url = f"{COINGECKO_API_BASE_URL}/coins/{asset_platform_id}/contract/{contract_address}"
        logger.info(f"Calling CoinGecko API for contract {contract_address} on platform {asset_platform_id}: {url}")
    elif token_id:
        # Fetch by token ID
        url = f"{COINGECKO_API_BASE_URL}/coins/{token_id}"
        logger.info(f"Calling CoinGecko API for token ID '{token_id}': {url}")
    else:
        return {"error": "CoinGecko requires either a token_id or both contract_address and asset_platform_id."}

    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Successfully retrieved data for '{token_id or contract_address}' from CoinGecko.")
        
        # Optionally simplify the response here if needed, CoinGecko responses can be large
        # Example: Extract key fields
        simplified_data = {}
        if 'id' in data: simplified_data['id'] = data['id']
        if 'symbol' in data: simplified_data['symbol'] = data['symbol']
        if 'name' in data: simplified_data['name'] = data['name']
        if include_market_data and 'market_data' in data:
            md = data['market_data']
            simplified_data['market_data'] = {
                'current_price': md.get('current_price'),
                'market_cap': md.get('market_cap'),
                'total_volume': md.get('total_volume')
            }
        if 'links' in data: # Add website link if available
            simplified_data['website'] = data['links'].get('homepage', [None])[0]
            
        return simplified_data if simplified_data else data # Return simplified or full data
        
    except requests.exceptions.HTTPError as http_err:
        status_code = response.status_code if 'response' in locals() else None
        logger.error(f"HTTP error occurred calling CoinGecko API ({status_code}): {http_err}")
        return {"error": f"HTTP error calling CoinGecko ({status_code}): {http_err}", "status_code": status_code}
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request exception occurred calling CoinGecko API: {req_err}")
        return {"error": f"Request exception calling CoinGecko: {req_err}"}
    except Exception as e:
        logger.error(f"An unexpected error occurred calling CoinGecko API: {e}", exc_info=True)
        return {"error": f"An unexpected error occurred calling CoinGecko: {e}"}

# Example Usage (for testing)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("--- Testing CoinGecko API Tool --- ")
    # Test by ID
    btc_data = call_coingecko_api(token_id='bitcoin')
    print("\nBitcoin Data (by ID):")
    print(json.dumps(btc_data, indent=2))
    
    # Test by Contract Address (USDC on Ethereum)
    usdc_eth_data = call_coingecko_api(contract_address='0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48', asset_platform_id='ethereum')
    print("\nUSDC Data (on Ethereum by Contract):")
    print(json.dumps(usdc_eth_data, indent=2))

    # Test missing info
    missing_data = call_coingecko_api(contract_address='0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48') # Missing platform ID
    print("\nMissing Platform ID:")
    print(json.dumps(missing_data, indent=2))

    # Test non-existent token
    fake_data = call_coingecko_api(token_id='fake-coin-12345')
    print("\nFake Coin Data:")
    print(json.dumps(fake_data, indent=2)) 