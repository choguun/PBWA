import dotenv
import os

dotenv.load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")
WALLET_PRIVATE_KEY=os.getenv("WALLET_PRIVATE_KEY")
RPC_PROVIDER_URL=os.getenv("RPC_PROVIDER_URL")
GOOGLE_CLOUD_PROJECT=os.getenv("GOOGLE_CLOUD_PROJECT")

# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333") # Default to localhost if not set
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None) # Optional API Key
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "defi_research")

UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")