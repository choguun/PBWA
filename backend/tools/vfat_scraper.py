import logging
import json
from browser_use import Agent
from langchain_google_genai import ChatGoogleGenerativeAI
from ..config import GOOGLE_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)

async def scrape_vfat_farm(farm_url: str) -> str:
    """Attempts to scrape farm data (APY, pools) from a vfat.tools URL using browser_use.Agent.
    
    NOTE: This is highly dependent on vfat.tools page structure and LLM interpretation.
    May require prompt adjustments and be prone to errors.
    """
    logger.info(f"Attempting to scrape vfat.tools farm using browser-use: {farm_url}")
    
    if not farm_url or not farm_url.startswith("https://vfat.tools/"):
        return json.dumps({"error": "Invalid vfat.tools URL provided."}) 
        
    try:
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0)
        
        # Detailed task instruction for the agent
        task = f"""Go to the vfat.tools URL: {farm_url}
        Wait for the farm data tables to load completely. These tables usually contain pool names, APY breakdowns, and TVL.
        Iterate through each primary farm/pool row displayed on the page.
        For each row, extract the following information if available:
        1.  **Pool Name:** The name of the liquidity pool or farm (e.g., 'WETH-USDC').
        2.  **APY / APR:** Look for Annual Percentage Yield (APY) or Annual Percentage Rate (APR). Try to break it down into components if shown (e.g., base APR, reward APR, total APY).
        3.  **TVL:** Total Value Locked in the pool, if displayed.
        Structure the extracted data as a list of JSON objects, one for each pool/farm row found. Example object: 
        {{ "pool_name": "ABC-XYZ LP", "apr_base": "5.5%", "apr_rewards": "12.1%", "total_apy": "18.2%", "tvl": "$1.2M" }}
        If specific values aren't found for a pool, omit the key or set the value to null. 
        Return ONLY the JSON list, no other text.
        If the page fails to load or no farm data tables are found, return an error message in JSON format: {{"error": "Could not find farm data tables."}}
        """
        
        logger.debug(f"Initializing browser-use Agent for vfat.tools with task... ")
        agent = Agent(task=task, llm=llm)
        
        logger.debug(f"Running browser-use Agent for vfat URL: {farm_url}")
        # Give it more steps due to complexity
        history = await agent.run(max_steps=25) 
        
        if history.has_errors():
            errors = history.errors()
            logger.error(f"browser-use Agent encountered errors scraping {farm_url}: {errors}")
            return json.dumps({"error": f"Error during scraping with browser-use: {errors}"}) 
            
        final_result = history.final_result()
        
        if not final_result:
             logger.warning(f"browser-use Agent finished for {farm_url} but returned no final result.")
             return json.dumps({"error": "browser-use agent did not return structured data. Page might have changed or task failed."})

        logger.info(f"Attempted vfat.tools scrape for {farm_url}. Result length: {len(final_result)}")
        # Assume the agent returns a JSON string based on the prompt
        # We return it directly, expecting it to be a JSON list or an error JSON
        return str(final_result) 

    except ImportError:
        logger.critical("ImportError: `browser-use` library not found. Please install it: pip install browser-use")
        return json.dumps({"error": "browser-use library not installed."})
    except Exception as e:
        logger.exception(f"Error running browser-use Agent for vfat URL {farm_url}: {e}") 
        return json.dumps({"error": f"Error running browser-use Agent for {farm_url}: {e}"})

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