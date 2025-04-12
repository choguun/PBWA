import logging
import json
from browser_use import Agent
from langchain_google_genai import ChatGoogleGenerativeAI
from ..config import GOOGLE_API_KEY, LLM_MODEL
from typing import Optional, Dict, Any, Union, List

logger = logging.getLogger(__name__)

async def scrape_vfat_farm(
    farm_url: str,
    user_query: Optional[str] = None,
    user_profile: Optional[Dict[str, Any]] = None
) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """Attempts to scrape farm data (APY, pools) from a vfat.tools URL using browser_use.Agent,
    considering user query and profile for context."""

    logger.info(f"Attempting to scrape vfat.tools farm: {farm_url}")
    logger.info(f"User Query Context: {user_query}")
    logger.info(f"User Profile Context: {user_profile}")

    if not farm_url or not farm_url.startswith("https://vfat.tools/"):
        return {"error": "Invalid vfat.tools URL provided."} 

    # Format profile for prompt
    profile_context = "Not provided."
    if user_profile:
        profile_parts = []
        if user_profile.get("risk_tolerance"):
            profile_parts.append(f"- Risk Tolerance: {user_profile['risk_tolerance']}")
        if user_profile.get("preferred_chains"):
            profile_parts.append(f"- Preferred Chains: {user_profile['preferred_chains']}")
        # Add other relevant profile fields here if needed
        if profile_parts:
            profile_context = "\n".join(profile_parts)

    # Format user query for prompt
    query_context = user_query or "No specific query provided."

    try:
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0)

        # Update task prompt to include context
        task = f"""
        **Context:**
        The user's original query was: "{query_context}"
        User Profile Summary:
        {profile_context}

        **Main Task:**
        Go to the vfat.tools URL: {farm_url}
        Wait for the farm data tables to load completely. These tables usually contain pool names, APY/APR breakdowns, and TVL.
        Iterate through each primary farm/pool row displayed on the page.

        **Data Extraction per Pool:**
        For each row, extract the following information if available:
        1.  **Pool Name:** The name of the liquidity pool or farm (e.g., 'WETH-USDC').
        2.  **APY / APR:** Look for Annual Percentage Yield (APY) or Annual Percentage Rate (APR). Try to break it down into components if shown (e.g., base APR, reward APR, total APY). Note units clearly (%, etc.).
        3.  **TVL:** Total Value Locked in the pool, if displayed. Note currency/units.
        4.  **Associated Tokens/Assets:** List the tokens involved in the pool (e.g., ['WBTC', 'ETH']).

        **Contextual Filtering (Use if helpful based on Context above):**
        - If the user profile mentions risk tolerance, you *could* try to prioritize or flag pools that seem very high/low risk based on APY volatility or token types, but focus primarily on accurate data extraction.
        - If the user query mentions specific tokens or protocols, pay extra attention to pools involving them.

        **Output Format:**
        Structure the extracted data as a list of JSON objects, one for each pool/farm row found. Example object:
        {{ "pool_name": "ABC-XYZ LP", "apr_base": "5.5%", "apr_rewards": "12.1%", "total_apy": "18.2%", "tvl": "$1.2M", "assets": ["ABC", "XYZ"] }}
        If specific values aren't found for a pool, omit the key or set the value to null.
        Return ONLY the JSON list, no other text.

        **Error Handling:**
        If the page fails to load or no farm data tables are found, return an error message in JSON format: {{"error": "Could not find farm data tables."}}
        """

        logger.debug(f"Initializing browser-use Agent for vfat.tools with updated task...")
        agent = Agent(task=task, llm=llm)

        logger.debug(f"Running browser-use Agent for vfat URL: {farm_url}")
        history = await agent.run(max_steps=25) 

        if history.has_errors():
            errors = history.errors()
            logger.error(f"browser-use Agent encountered errors scraping {farm_url}: {errors}")
            return {"error": f"Error during scraping with browser-use: {errors}"} 
            
        final_result_str = history.final_result()
        
        if not final_result_str:
             logger.warning(f"browser-use Agent finished for {farm_url} but returned no final result.")
             return {"error": "browser-use agent did not return structured data."} 

        logger.info(f"Attempted vfat.tools scrape for {farm_url}. Raw result length: {len(final_result_str)}")
        
        # Attempt to parse the result as JSON (expecting a list)
        try:
            parsed_result = json.loads(final_result_str)
            if not isinstance(parsed_result, list):
                 logger.warning(f"vfat scraper result was valid JSON but not a list: {type(parsed_result)}")
                 # Return the potentially valid JSON but wrap in an error structure
                 return {"error": "Scraper returned unexpected data format (not a list).", "data": parsed_result} 
            
            logger.info("Successfully parsed vfat scrape result as list.")
            return parsed_result # Return the list of dicts
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse vfat scraper result as JSON. Raw result:\n{final_result_str[:500]}...")
            # Return the raw string but wrap it in an error structure might be better?
            # For now, return an explicit error dictionary
            return {"error": "Failed to parse scraper output as JSON.", "raw_output": final_result_str[:1000]} # Include snippet of raw output
        except Exception as e:
            # Catch potential errors if result isn't a string? Unlikely but safe.
             logger.exception(f"Unexpected error processing final_result from vfat scrape: {e}")
             return {"error": f"Unexpected error processing vfat scrape result: {e}"} 

    except ImportError:
        logger.critical("ImportError: `browser-use` library not found. Please install it: pip install browser-use")
        return {"error": "browser-use library not installed."} 
    except Exception as e:
        logger.exception(f"Error running browser-use Agent for vfat URL {farm_url}: {e}") 
        return {"error": f"Error running browser-use Agent for {farm_url}: {e}"} 

# Example Usage (for testing)
if __name__ == '__main__':
    import asyncio
    logging.basicConfig(level=logging.INFO)
    from dotenv import load_dotenv
    load_dotenv()
    
    async def run_test():
        print("--- Testing vfat.tools Scraper Tool --- ")
        # Replace with a valid, current vfat.tools URL for testing
        test_url = "https://vfat.tools/polygon/quickswap-epoch/" # Example - URL might change!
        print(f"\nAttempting scrape for: {test_url}")
        result_str = await scrape_vfat_farm(test_url)
        print("\nResult:")
        try:
            # Try parsing the result as JSON for pretty printing
            result_json = json.loads(result_str)
            print(json.dumps(result_json, indent=2))
        except json.JSONDecodeError:
            print("Result was not valid JSON:")
            print(result_str)

    asyncio.run(run_test()) 