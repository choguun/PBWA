import logging
from qdrant_client import QdrantClient, models
from fastembed.embedding import DefaultEmbedding
from .config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME

logger = logging.getLogger(__name__)

def get_embedding_model():
    """Initializes and returns the FastEmbed embedding model."""
    # You can customize the model name if needed, e.g., "BAAI/bge-small-en-v1.5"
    # See FastEmbed documentation for available models.
    logger.info("Initializing embedding model...")
    try:
        # Using DefaultEmbedding provides a reasonable default
        embedding_model = DefaultEmbedding()
        # Attempt a dummy embed to check initialization
        _ = list(embedding_model.embed("test"))
        logger.info(f"Embedding model initialized successfully: {type(embedding_model).__name__}")
        return embedding_model
    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {e}", exc_info=True)
        raise

def get_qdrant_client():
    """Initializes and returns the Qdrant client."""
    logger.info(f"Initializing Qdrant client for URL: {QDRANT_URL}")
    try:
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY, # Pass API key if provided
            timeout=60 # Increase timeout for potentially long operations
        )
        # Check connection by listing collections (updated method name)
        client.get_collections()
        logger.info(f"Successfully connected to Qdrant at {QDRANT_URL}")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize or connect to Qdrant at {QDRANT_URL}: {e}", exc_info=True)
        return None

def initialize_qdrant_collection(client: QdrantClient, embedding_model):
    """Checks if the collection exists and creates it if not."""
    collection_name = QDRANT_COLLECTION_NAME
    try:
        # Get embedding size and distance metric from the FastEmbed model
        # This assumes the model has a convenient way to expose this info, 
        # which might vary slightly between FastEmbed versions or specific models.
        # We might need to embed a dummy text to get the size.
        dummy_embedding = list(embedding_model.embed("dimension_check"))[0]
        vector_size = len(dummy_embedding)
        # Common distance metric for text embeddings
        distance_metric = models.Distance.COSINE 

        logger.info(f"Checking for Qdrant collection: '{collection_name}' with vector size {vector_size}")
        
        collections = client.get_collections().collections
        collection_exists = any(c.name == collection_name for c in collections)

        if not collection_exists:
            logger.warning(f"Collection '{collection_name}' not found. Creating...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=distance_metric)
            )
            logger.info(f"Collection '{collection_name}' created successfully.")
        else:
            # Optionally, verify existing collection parameters
            # config = client.get_collection(collection_name=collection_name).vectors_config
            # if config.params.size != vector_size or config.params.distance != distance_metric:
            #     logger.warning(f"Collection '{collection_name}' exists but parameters might mismatch!")
            logger.info(f"Collection '{collection_name}' already exists.")

    except Exception as e:
        logger.error(f"Error during Qdrant collection initialization ('{collection_name}'): {e}", exc_info=True)
        # Decide if this should raise an error or just log

# --- Initialize clients and model on module load (or use FastAPI lifespan) --- 
# Using global instances here for simplicity in this refactoring step.
# For robustness, consider FastAPI lifespan events to manage these resources.
try:
    embedding_model_instance = get_embedding_model()
    qdrant_client_instance = get_qdrant_client()
    initialize_qdrant_collection(qdrant_client_instance, embedding_model_instance)
except Exception as global_init_e:
    logger.error(f"Fatal error during Vector Store initialization: {global_init_e}", exc_info=True)
    # Handle failure appropriately - maybe exit or disable features
    embedding_model_instance = None
    qdrant_client_instance = None 