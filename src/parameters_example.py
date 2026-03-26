# =========================
# parameters_example.py
# =========================
"""
Example parameter file

IMPORTANT: 
1. Copy this file: cp parameters_example.py parameters.py
2. Edit parameters.py with your actual values
3. Never commit parameters.py (it's in .gitignore)
"""

# DATA CONFIGURATION
DATA_PATH = "data/your_dataset.csv"  # Path to your Amazon reviews CSV

# TEXT CHUNKING PARAMETERS
CHUNK_SIZE = 500      # Characters per chunk
CHUNK_OVERLAP = 100   # Overlap between consecutive chunks

# EMBEDDING MODEL
EMBEDDING_MODEL = "Snowflake/snowflake-arctic-embed-xs"

# PINECONE CONFIGURATION
PINECONE_INDEX = "amazon-reviews"     # Your Pinecone index name
PINECONE_REGION = "us-east-1"         # Your Pinecone region
PINECONE_API_KEY = "your-api-key"     # Replace with your actual Pinecone API key
NAMESPACE = "default"                 # Namespace for organizing vectors

# LOGGING
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR

# OpenAI
OPENAI_API_KEY = "your-openai-api-key" # Replace with your actual OpenAI API key

# top-k results to retrieve from Pinecone
TOP_K = 10