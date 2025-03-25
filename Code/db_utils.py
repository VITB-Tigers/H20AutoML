# METADATA [db_utils.py] - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Description: This file contains the functions to interact with the database.
# Developed By: 
    # Name: Vansh Raja
    # Role: Intern, PreProd Corp
    # Code ownership rights: Vansh Raja, PreProd Corp

# Version: 1.0 [Date: 12-12-2024]
# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Dependencies:
    # Python 3.11.15
    # Libraries:
        # Redis 5.2.1
        # H2O 3.46.0.6

import redis
import pickle
import h2o
    
def connect_to_redis(host: str, port: int, db: int):
    """
    Establishes a connection to a Redis database with robust error handling.
    
    Purpose:
    - Create a reliable connection to a specific Redis database
    - Provide feedback on connection status
    
    Parameters:
    - host (str): IP address or hostname of the Redis server
    - port (int): Network port number for Redis connection
    - db (int): Specific Redis database number to connect to
    
    Returns:
    - Redis connection object if successful
    - None if connection fails
    
    Connection Strategy:
    - Uses redis.Redis() for direct connection
    - Prints connection status
    - Catches and reports any connection errors
    """
    try: 
        # Attempt to establish Redis connection with specified parameters
        r = redis.Redis(host=host, port=port, db=db)
        
        # Verify connection by pinging the server
        r.ping()
        print("Connected to Redis")
        return r
    
    except Exception as e:
        # Comprehensive error logging for connection issues
        print(f"Error connecting to Redis: {e}")
        return None
    
def write_to_redis(key: str, df, r):
    """
    Serializes and stores a DataFrame in Redis with flexible input handling.
    
    Purpose:
    - Convert different DataFrame types to a storable format
    - Serialize and store data in Redis
    - Provide clear feedback on storage operation
    
    Parameters:
    - key (str): Unique identifier for storing the data in Redis
    - df: Input DataFrame (supports H2OFrame or Pandas DataFrame)
    - r: Active Redis connection object
    
    Workflow:
    1. Convert H2OFrame to Pandas DataFrame if necessary
    2. Delete any existing data with the same key
    3. Serialize DataFrame using pickle
    4. Store serialized data in Redis
    
    Returns:
    - True if successful storage
    - None if an error occurs
    """
    
    # Ensure input is a Pandas DataFrame
    if isinstance(df, h2o.H2OFrame):
        df = df.as_data_frame()
    
    try:
        # Remove any existing data with the same key to prevent conflicts
        r.delete(key)
        
        # Serialize DataFrame using pickle for comprehensive data preservation
        serialized_df = pickle.dumps(df)
        
        # Store serialized data in Redis
        r.set(key, serialized_df)
        
        print(f"DataFrame successfully stored in Redis under key: '{key}'")
        return True
    
    except Exception as e:
        # Log any serialization or storage errors
        print(f"Error storing DataFrame in Redis: {e}")
        return None


def load_from_redis(key: str, r):
    """
    Retrieves and reconstructs a DataFrame from Redis with comprehensive error handling.
    
    Purpose:
    - Fetch serialized data from Redis
    - Deserialize back to a usable DataFrame
    - Convert to H2OFrame for machine learning workflows
    
    Parameters:
    - key (str): Redis key where the DataFrame is stored
    - r: Active Redis connection object
    
    Deserialization Workflow:
    1. Retrieve serialized data from Redis
    2. Unpickle back to Pandas DataFrame
    3. Convert to H2OFrame for further processing
    
    Returns:
    - H2OFrame if successful retrieval and conversion
    - None if any errors occur during the process
    
    Error Handling:
    - Catches issues with key retrieval, deserialization, or conversion
    """
    try:
        # Retrieve serialized DataFrame from Redis
        serialized_df = r.get(key)
        
        # Check if data exists for the given key
        if serialized_df is None:
            print(f"No data found for key: '{key}'")
            return None
        
        # Deserialize to Pandas DataFrame
        pandas_df = pickle.loads(serialized_df)
        
        # Convert to H2OFrame, maintaining data integrity
        h2o_frame = h2o.H2OFrame(pandas_df)
        print(f"DataFrame successfully loaded from Redis key: '{key}'")
        
        return h2o_frame
    
    except Exception as e:
        # Detailed error logging for debugging
        print(f"Error loading DataFrame from Redis: {e}")
        return None
    
def drop_db(r):
    """
    Safely clear all keys from the current Redis database.
    
    Purpose:
    - Provide a method to completely reset the current database
    - Implement with error handling for safety
    
    Parameters:
    - r: Active Redis connection object
    
    Returns:
    - True if database successfully cleared
    - False if an error occurs during clearance
    """
    try:
        # Redis command to remove all keys in the current database
        r.flushdb()
        print("Database successfully cleared")
        return True
    
    except Exception as e:
        # Log any errors preventing database clearance
        print(f"Error clearing database: {e}")
        return False
        
def get_column_names(key: str, r):
    """
    Retrieve column names from a DataFrame stored in Redis.
    
    Purpose:
    - Fetch column structure of a stored DataFrame
    - Provide metadata without full data retrieval
    
    Parameters:
    - key (str): Redis key of the stored DataFrame
    - r: Active Redis connection object
    
    Returns:
    - List of column names if successful
    - None if retrieval fails
    
    Strategy:
    1. Load DataFrame from Redis
    2. Extract column names
    3. Handle potential errors
    """
    try:
        # Load DataFrame from Redis
        df = load_from_redis(key, r)
        
        # Check if DataFrame was successfully retrieved
        if df is not None:
            column_names = df.columns
            print(f"Retrieved {len(column_names)} column names")
            return column_names
        
        return None
    
    except Exception as e:
        # Comprehensive error tracking for column name retrieval
        print(f"Error retrieving column names: {e}")
        return None