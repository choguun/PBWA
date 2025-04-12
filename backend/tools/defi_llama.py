import requests
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

DEFI_LLAMA_API_BASE_URL = "https://api.llama.fi"

def call_defi_llama_api(protocol_slug: str, metric: Optional[str] = None) -> Dict[str, Any]:
    """Calls the DefiLlama API to fetch data for a specific protocol.

    Args:
        protocol_slug: The protocol slug (e.g., 'aave', 'uniswap').
        metric: Optional specific metric (e.g., 'tvl', 'volume').

    Returns:
        A dictionary containing the API response data or an error message.
    """
    if metric:
        url = f"{DEFI_LLAMA_API_BASE_URL}/charts/{protocol_slug}"
        logger.info(f"Calling DefiLlama API for metric '{metric}' on protocol '{protocol_slug}': {url}")
    else:
        url = f"{DEFI_LLAMA_API_BASE_URL}/protocol/{protocol_slug}"
        logger.info(f"Calling DefiLlama API for general data on protocol '{protocol_slug}': {url}")

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Successfully retrieved data for '{protocol_slug}' from DefiLlama.")
        return data
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred calling DefiLlama API for '{protocol_slug}': {http_err}")
        # Return status code in error dict if possible
        status_code = response.status_code if 'response' in locals() else None
        return {"error": f"HTTP error calling DefiLlama: {http_err}", "status_code": status_code}
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request exception occurred calling DefiLlama API for '{protocol_slug}': {req_err}")
        return {"error": f"Request exception calling DefiLlama: {req_err}"}
    except Exception as e:
        logger.error(f"An unexpected error occurred calling DefiLlama API for '{protocol_slug}': {e}", exc_info=True)
        return {"error": f"An unexpected error occurred: {e}"} 