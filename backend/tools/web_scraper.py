import logging
# Remove Playwright import
# from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

# Import browser-use Agent and necessary LLM components
from browser_use import Agent
from langchain_google_genai import ChatGoogleGenerativeAI
from ..config import GOOGLE_API_KEY, LLM_MODEL # Adjusted relative import

logger = logging.getLogger(__name__)

async def scrape_website_content(url: str) -> str:
    """Fetches and returns the main textual content of a given URL using browser_use.Agent.
    
    Handles basic cleaning and limits content length.
    """
    logger.info(f"Attempting to scrape content from URL using browser-use: {url}")
    
    if not url.startswith(("http://", "https://")):
        return "Error: Invalid URL format. Please provide a full URL starting with http:// or https://"
        
    try:
        # Initialize the LLM for the browser-use Agent
        # Consider if this should be shared/passed differently in a larger app
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0)
        
        # Define the task for the agent
        task = f"Go to {url}, wait for the main content to load, and then return the primary textual content of the page. Focus on articles, descriptions, or main body text, excluding headers, footers, ads, and navigation menus."
        
        logger.debug(f"Initializing browser-use Agent with task: '{task[:100]}...'")
        agent = Agent(
            task=task,
            llm=llm,
            # use_vision=True, # Keep vision enabled by default if model supports it (like gpt-4o, potentially Gemini)
            # Add other Agent settings if needed (e.g., browser config)
        )
        
        logger.debug(f"Running browser-use Agent for URL: {url}")
        # Max steps can be adjusted
        history = await agent.run(max_steps=10) 
        
        if history.has_errors():
            errors = history.errors()
            logger.error(f"browser-use Agent encountered errors scraping {url}: {errors}")
            return f"Error during scraping with browser-use: {errors}"
            
        # Extract final result
        final_result = history.final_result()
        
        if not final_result:
             logger.warning(f"browser-use Agent finished for {url} but returned no final result.")
             # You might want to inspect history.extracted_content() or other history attributes here
             return "Error: browser-use agent did not return content."

        logger.info(f"Successfully scraped content (length: {len(final_result)}) from {url} using browser-use.")
        text_content = str(final_result) # Ensure it's a string

    except ImportError:
        logger.critical("ImportError: `browser-use` library not found. Please install it: pip install browser-use")
        return "Error: browser-use library not installed."
    except Exception as e:
        logger.exception(f"Error running browser-use Agent for URL {url}: {e}") 
        return f"Error running browser-use Agent for {url}: {e}" 
        
    # --- Content processing (remains similar) ---
    max_len = 4000 
    cleaned_content = '\n'.join([line.strip() for line in text_content.split('\n') if line.strip()])
    
    if len(cleaned_content) > max_len:
        logger.warning(f"Scraped content truncated from {len(cleaned_content)} to {max_len} characters.")
        return cleaned_content[:max_len] + "... [Content Truncated]"
    elif not cleaned_content:
        return "Error: Could not extract meaningful text content from the page body via browser-use."
    else:
        return cleaned_content 