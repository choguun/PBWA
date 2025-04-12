import logging
import asyncio
from typing import Dict, Any
from urllib.parse import urlparse, parse_qs
import json # Import json for potential parsing later if needed

# Import browser-use Agent and necessary LangChain/config components
try:
    from browser_use import Agent
except ImportError:
    Agent = None # Define Agent as None if import fails

from langchain_google_genai import ChatGoogleGenerativeAI
from ..config import GOOGLE_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)

# --- Function called by the scrape_tool_direct Tool ---
async def scrape_website_content(url: str) -> str:
    """Scrapes content from a given URL.
    If the URL is a Google search, it attempts to scrape the top 3 organic results.
    Otherwise, it scrapes the content of the given URL directly.
    Returns a string (either direct text content or a JSON string list for search results).
    """
    logger.info(f"scrape_website_content called for URL: {url}")

    if Agent is None:
         logger.critical("`browser-use` library not found. Please install it: pip install browser-use")
         return "Error: browser-use library not installed."

    # --- Check if it's a Google Search URL ---
    is_google_search = False
    search_query = ""
    try:
        parsed_url = urlparse(url)
        if "google.com" in parsed_url.netloc and parsed_url.path.startswith("/search"):
            query_params = parse_qs(parsed_url.query)
            if 'q' in query_params and query_params['q']:
                 search_query = query_params['q'][0]
                 is_google_search = True
                 logger.info(f"Detected Google Search URL for query: {search_query}")
    except Exception as e:
        logger.warning(f"Could not parse URL '{url}' to check for Google Search: {e}")

    # --- Define the task for browser-use Agent --- 
    task_instruction = ""
    if is_google_search:
        task_instruction = f"""
        Go to the provided Google search results page: '{url}'.
        Wait for the main search results to load completely (allow up to 15 seconds).
        Identify the URLs of the first 2 organic search result links. IMPORTANT: Ignore ads, sponsored links, 'People also ask' boxes, image/video carousels, and map results. Focus only on the standard blue link textual results typically found below the ads.
        If you find fewer than 2 organic links, use only the ones you find.
        For each identified organic result URL:
        1. Navigate to the URL in the current tab.
        2. Wait for the main content of the page to load (allow up to 20 seconds per page). If a page takes too long, skip it and proceed to the next.
        3. Extract the primary textual content of the page. Focus on the main article, blog post, or body text. Aim to exclude headers, footers, navigation bars, sidebars, comment sections, and advertisements. Try to get meaningful content.
        Return the extracted information as a JSON list of objects. Each object MUST have two keys: 'url' (the URL of the scraped page) and 'content' (the extracted text, keep it concise but informative, max ~1500 chars per page).
        Example format: [{{"url": "scraped_url_1", "content": "extracted text 1..."}}, {{"url": "scraped_url_2", "content": "extracted text 2..."}}]
        If you encounter an error loading or scraping a specific page (e.g., timeout, block), include an error entry in the list for that URL, like: {{"url": "failed_url", "error": "Could not load/scrape content."}}
        If no organic results are found at all, return an empty JSON list: []
        Respond ONLY with the valid JSON list string, no other text or explanation.
        """
    else:
        # Standard scraping task for non-search URLs
        task_instruction = f"""
        Go to the provided URL: '{url}'.
        Wait for the page's main content to load (allow up to 20 seconds).
        Extract the primary textual content of the page. Focus on the main article, blog post, or body text, aiming to exclude headers, footers, navigation bars, sidebars, comment sections, and advertisements. Return the most relevant ~1500-2000 characters.
        Return ONLY the extracted text content as a single string. If the page fails to load or content cannot be extracted, return an error message string like 'Error: Could not scrape content from {url}.'.
        """

    # --- Initialize LLM and browser-use Agent --- 
    try:
        # Ensure API key and model are configured
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not set in the environment/config.")
        if not LLM_MODEL:
             raise ValueError("LLM_MODEL is not set in the environment/config.")

        # Use a capable model, GPT-4o recommended for browser-use
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0)
        
        logger.info(f"Initializing browser-use Agent with task instruction (snippet):\n{task_instruction[:250]}...")
        # Consider adding vision=True if using gpt-4o or similar for better element identification
        agent = Agent(task=task_instruction, llm=llm, use_vision=True)
        
        # --- Run the Agent --- 
        logger.info("Running browser-use Agent...")
        # Increase max_steps for multi-page searches
        history = await agent.run(max_steps=75 if is_google_search else 25)
        
        # --- Process Results --- 
        final_result = f"Error: Agent did not produce a final result for {url}."
        if history.has_errors():
            errors = history.errors()
            logger.error(f"browser-use Agent encountered errors scraping {url}: {errors}")
            # Return a structured error if it was a search, otherwise simple error
            if is_google_search:
                 final_result = json.dumps([{"url": url, "error": f"Agent errors during search: {errors}"}])
            else:
                 final_result = f"Error during scraping with browser-use: {errors}"
        else:
            final_result_str = history.final_result()
            if final_result_str is not None:
                logger.info(f"browser-use Agent finished successfully for {url}. Raw output length: {len(final_result_str)}")
                # Validate JSON if it was a search, otherwise return text
                if is_google_search:
                    try:
                        # Test if output is valid JSON list (as requested in prompt)
                        parsed = json.loads(final_result_str)
                        if isinstance(parsed, list):
                            final_result = final_result_str # Return the JSON string
                        else:
                            logger.warning(f"Search scrape for '{search_query}' returned non-list JSON: {type(parsed)}")
                            final_result = json.dumps([{"url": url, "error": "Agent returned non-list JSON.", "raw_output": final_result_str[:500]}])
                    except json.JSONDecodeError:
                         logger.warning(f"Search scrape for '{search_query}' returned non-JSON: {final_result_str[:500]}...")
                         final_result = json.dumps([{"url": url, "error": "Agent returned non-JSON output.", "raw_output": final_result_str[:500]}])
                else:
                    final_result = final_result_str # Return the direct text content

            else:
                 logger.warning(f"browser-use Agent finished for {url} but returned no final result (None).")
                 if is_google_search:
                     final_result = "[]" # Empty list for failed search
                 else:
                    final_result = f"Error: Agent finished but returned no data for {url}."
        
        return final_result

    except ValueError as ve:
         logger.error(f"Configuration error for browser-use: {ve}")
         return f"Error: Configuration missing - {ve}"
    except Exception as e:
        logger.exception(f"Unexpected error running browser-use Agent for {url}: {e}")
        if is_google_search:
            return json.dumps([{"url": url, "error": f"Unexpected error running agent: {e}"}])
        else:
            return f"Error running browser-use Agent: {e}"

# --- Example Usage ---
# async def main():
#     logging.basicConfig(level=logging.INFO)
#     # Make sure GOOGLE_API_KEY and LLM_MODEL are set in your environment or config
#     # Example:
#     # import os
#     # from dotenv import load_dotenv
#     # load_dotenv()
#     # print(f"Using API Key: {os.getenv('GOOGLE_API_KEY')[:5]}...")
#     # print(f"Using LLM Model: {os.getenv('LLM_MODEL')}")
# 
# 
#     test_search_url = "https://www.google.com/search?q=what+is+eigenlayer"
#     test_direct_url = "https://docs.eigenlayer.xyz/eigenlayer/overview/"
#     test_bad_url = "https://thisshouldnotexist12345.xyz"
# 
#     print(f"--- Testing Direct URL: {test_direct_url} ---")
#     direct_result = await scrape_website_content(test_direct_url)
#     print(f"Result (Direct):\n{direct_result[:1000]}...\n")
# 
#     print(f"--- Testing Search URL: {test_search_url} ---")
#     search_result = await scrape_website_content(test_search_url)
#     print(f"Result (Search JSON String):\n{search_result}\n")
# 
#     print(f"--- Testing Bad URL: {test_bad_url} ---")
#     bad_result = await scrape_website_content(test_bad_url)
#     print(f"Result (Bad URL):\n{bad_result}\n")
# 
# 
# if __name__ == '__main__':
#      # Add event loop policy for Windows if needed
#      # if os.name == 'nt':
#      #     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
#      asyncio.run(main()) 