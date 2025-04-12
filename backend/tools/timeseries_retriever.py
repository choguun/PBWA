import logging
import json
from typing import Dict, Any, Optional, List
from influxdb_client import InfluxDBClient
from influxdb_client.client.exceptions import InfluxDBError
from ..timeseries_db import influxdb_client_instance, INFLUXDB_BUCKET, INFLUXDB_ORG

logger = logging.getLogger(__name__)

def query_time_series_data(
    measurement: str,
    tags: Optional[Dict[str, str]] = None,
    fields: Optional[List[str]] = None,
    start_time: str = "-1h",
    stop_time: Optional[str] = None,
    limit: int = 100
) -> Dict[str, Any]:
    """Queries time-series data from InfluxDB using Flux.

    Args:
        measurement: The measurement name.
        tags: Dictionary of tag filters.
        fields: List of fields to select.
        start_time: Start of the time range.
        stop_time: End of the time range (defaults to now()).
        limit: Maximum number of records per series.

    Returns:
        A dictionary containing the query results or an error message.
    """
    if not influxdb_client_instance:
        return {"error": "InfluxDB client not available."}
    if not INFLUXDB_BUCKET or not INFLUXDB_ORG:
         return {"error": "InfluxDB bucket or organization not configured."}

    query_api = None
    try:
        query_api = influxdb_client_instance.query_api()
        
        # Construct Flux Query
        flux_query = f'from(bucket: \"{INFLUXDB_BUCKET}\")\n'
        flux_query += f' |> range(start: {start_time}' 
        if stop_time:
            flux_query += f', stop: {stop_time})' 
        else:
             flux_query += ')\n'
             
        flux_query += f' |> filter(fn: (r) => r._measurement == \"{measurement}\")\n'

        # Add tag filters
        if tags:
            for key, value in tags.items():
                flux_query += f' |> filter(fn: (r) => r[\"{key}\"] == \"{value}\")\n'
                
        # Add field filters
        if fields:
             field_filter_parts = [f'r._field == \"{field}\"' for field in fields]
             flux_query += f' |> filter(fn: (r) => {' or '.join(field_filter_parts)})\n'

        # Add limit (applied per series by default in Flux grouping)
        flux_query += f' |> limit(n: {limit})\n'
        # Optional: Add yield() to flatten results if needed, depending on desired output format
        # flux_query += ' |> yield(name: \"results\")'

        logger.info(f"Executing InfluxDB Flux query (Bucket: {INFLUXDB_BUCKET}):\n{flux_query}")
        
        # Execute query
        result_tables = query_api.query(query=flux_query, org=INFLUXDB_ORG)
        
        # Process results into a more agent-friendly format (e.g., list of dictionaries)
        results_list = []
        for table in result_tables:
            for record in table.records:
                # Flatten the record, include tags, field, value, time
                record_data = {tag_key: record.values.get(tag_key) for tag_key in (tags.keys() if tags else [])}
                record_data['time'] = record.get_time().isoformat()
                record_data['measurement'] = record.get_measurement()
                record_data['field'] = record.get_field()
                record_data['value'] = record.get_value()
                results_list.append(record_data)
                
        logger.info(f"Retrieved {len(results_list)} data points from InfluxDB.")
        return {"results": results_list}

    except InfluxDBError as e:
        logger.error(f"InfluxDB API error occurred during query: {e}", exc_info=True)
        return {"error": f"InfluxDB API error: {e}"}
    except Exception as e:
        logger.error(f"An unexpected error occurred querying InfluxDB: {e}", exc_info=True)
        return {"error": f"An unexpected error occurred querying InfluxDB: {e}"}
    finally:
        # Query API doesn't typically need explicit closing like Write API
        pass 

# Example Usage (for testing)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    from dotenv import load_dotenv
    load_dotenv()
    # Re-initialize client instance after loading .env potentially
    influxdb_client_instance = get_influxdb_client() 
    
    print("--- Testing InfluxDB Time Series Retriever Tool --- ")
    if not influxdb_client_instance:
        print("Skipping tests: InfluxDB client not initialized.")
    else:
        print("\nQuerying test_measurement (written in timeseries_db.py test):")
        test_results = query_time_series_data(measurement="test_measurement", start_time="-5m", tags={"location":"test_lab"})
        print(json.dumps(test_results, indent=2))

        print("\nQuerying bitcoin price (replace with actual data if available):")
        btc_price_results = query_time_series_data(
            measurement="token_market_data", 
            tags={"token_id": "bitcoin"}, 
            fields=["price_usd"],
            start_time="-1d",
            limit=5
        )
        print(json.dumps(btc_price_results, indent=2))
        
        print("\nQuerying non-existent measurement:")
        fake_results = query_time_series_data(measurement="non_existent_measure")
        print(json.dumps(fake_results, indent=2))
        
        # Close client
        influxdb_client_instance.close() 