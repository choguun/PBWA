import os
import logging
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# --- InfluxDB Configuration ---
INFLUXDB_URL = os.getenv("INFLUXDB_URL")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "defi_metrics") # Default bucket name

def get_influxdb_client() -> InfluxDBClient | None:
    """Initializes and returns an InfluxDB client if configuration is valid."""
    if not all([INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_ORG]):
        logger.warning("InfluxDB environment variables (URL, TOKEN, ORG) not fully configured. Time-series database features disabled.")
        return None
    
    try:
        client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
        # Check connection health
        if client.ping():
            logger.info(f"Successfully connected to InfluxDB at {INFLUXDB_URL}, Org: {INFLUXDB_ORG}")
            return client
        else:
            logger.error(f"Failed to ping InfluxDB at {INFLUXDB_URL}. Check configuration and server status.")
            return None
    except Exception as e:
        logger.error(f"Failed to initialize or connect to InfluxDB: {e}", exc_info=True)
        return None

# --- Global Client Instance ---
# Initialize client once on module load
influxdb_client_instance = get_influxdb_client()

# You might want a separate function to get the write API
def get_influxdb_write_api(client: InfluxDBClient):
    """Returns a synchronous write API from the client."""
    if not client:
        return None
    try:
        # Using SYNCHRONOUS for simplicity in agent steps
        # Consider ASYNCHRONOUS for potentially better performance in high-throughput scenarios
        return client.write_api(write_options=SYNCHRONOUS)
    except Exception as e:
        logger.error(f"Failed to get InfluxDB write API: {e}", exc_info=True)
        return None

# Example Usage (for testing)
if __name__ == '__main__':
    from influxdb_client import Point
    logging.basicConfig(level=logging.INFO)
    
    if influxdb_client_instance:
        write_api = get_influxdb_write_api(influxdb_client_instance)
        if write_api:
            print(f"Attempting to write test point to bucket '{INFLUXDB_BUCKET}'...")
            try:
                point = Point("test_measurement") \
                    .tag("location", "test_lab") \
                    .field("value", 123.45)
                write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)
                print("Test point written successfully.")
                
                # Example query (requires query_api)
                # query_api = influxdb_client_instance.query_api()
                # query = f'from(bucket: "{INFLUXDB_BUCKET}") |> range(start: -1h) |> filter(fn: (r) => r._measurement == "test_measurement")'
                # result = query_api.query(query=query, org=INFLUXDB_ORG)
                # print("\nQuery Result:")
                # for table in result:
                #     for record in table.records:
                #         print(record.values)
                        
            except Exception as e:
                print(f"Error during test write/query: {e}")
            finally:
                 write_api.close()
                 influxdb_client_instance.close()
        else:
            print("Could not get InfluxDB Write API.")
    else:
        print("InfluxDB client could not be initialized. Skipping test.") 