import requests
import logging
import os
from typing import Dict, Any, List
import json

logger = logging.getLogger(__name__)

TWITTER_API_BASE_URL = "https://api.twitter.com/2"

# --- Authentication --- 
# Load Bearer Token from environment variable
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

if not TWITTER_BEARER_TOKEN:
    logger.warning("TWITTER_BEARER_TOKEN environment variable not set. Twitter tool will not function.")

def search_recent_tweets(query: str, max_results: int = 10) -> Dict[str, Any]:
    """Searches for recent tweets using the Twitter API v2.

    Args:
        query: The search query (supports Twitter standard operators).
        max_results: Number of tweets to return (10-100).

    Returns:
        A dictionary containing a list of tweet texts or an error message.
    """
    if not TWITTER_BEARER_TOKEN:
        return {"error": "Twitter API Bearer Token not configured."}

    # Ensure max_results is within the allowed range (10-100)
    max_results = max(10, min(100, max_results))

    url = f"{TWITTER_API_BASE_URL}/tweets/search/recent"
    headers = {
        "Authorization": f"Bearer {TWITTER_BEARER_TOKEN}",
        "User-Agent": "PBWAgent/1.0" # Good practice to identify your client
    }
    params = {
        "query": query,
        "max_results": max_results,
        "tweet.fields": "created_at,public_metrics", # Request creation time and basic metrics
        # Add expansions=author_id and user.fields=username if you want author info
    }

    logger.info(f"Calling Twitter API v2 recent search. Query: '{query}', Max Results: {max_results}")

    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status() 
        data = response.json()
        
        # Extract relevant information
        tweets_data = data.get("data", [])
        extracted_tweets = []
        for tweet in tweets_data:
            extracted_tweets.append({
                "id": tweet.get("id"),
                "text": tweet.get("text"),
                "created_at": tweet.get("created_at"),
                "metrics": tweet.get("public_metrics")
            })

        logger.info(f"Successfully retrieved {len(extracted_tweets)} tweets for query '{query}'.")
        return {"tweets": extracted_tweets}
        
    except requests.exceptions.HTTPError as http_err:
        status_code = response.status_code if 'response' in locals() else None
        # Log the actual error response from Twitter if possible
        error_detail = "No details available."
        try:
            error_detail = response.json()
        except Exception:
            pass # Ignore if response is not JSON
        logger.error(f"HTTP error occurred calling Twitter API ({status_code}): {http_err}. Details: {error_detail}")
        return {"error": f"HTTP error calling Twitter API ({status_code}): {http_err}", "status_code": status_code, "details": error_detail}
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request exception occurred calling Twitter API: {req_err}")
        return {"error": f"Request exception calling Twitter API: {req_err}"}
    except Exception as e:
        logger.error(f"An unexpected error occurred calling Twitter API: {e}", exc_info=True)
        return {"error": f"An unexpected error occurred calling Twitter API: {e}"}

# Example Usage (for testing - requires TWITTER_BEARER_TOKEN in env)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # Load .env for testing if needed
    from dotenv import load_dotenv
    load_dotenv()
    TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN") # Reload after dotenv
    
    print("--- Testing Twitter API Tool --- ")
    if not TWITTER_BEARER_TOKEN:
        print("Skipping tests: TWITTER_BEARER_TOKEN not set.")
    else:
        test_query = "(#ethereum OR #ETH) lang:en -is:retweet"
        print(f"\nSearching for: {test_query}")
        tweets = search_recent_tweets(test_query, max_results=15)
        print(json.dumps(tweets, indent=2))

        print("\nSearching for non-existent term:")
        fake_tweets = search_recent_tweets("someveryunlikelytermxyzabc123", max_results=10)
        print(json.dumps(fake_tweets, indent=2)) 