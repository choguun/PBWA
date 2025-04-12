import dotenv
import os

dotenv.load_dotenv()

API_KEY = os.getenv("API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")
WALLET_PRIVATE_KEY=os.getenv("WALLET_PRIVATE_KEY")
RPC_PROVIDER_URL=os.getenv("RPC_PROVIDER_URL")
GOOGLE_CLOUD_PROJECT=os.getenv("GOOGLE_CLOUD_PROJECT")
