import os
import numpy as np
import logging
import time
import pickle
import pandas as pd
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastembed import TextEmbedding
from pinecone import Pinecone, ServerlessSpec
from parameters import (
    DATA_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    PINECONE_INDEX,
    PINECONE_REGION,
    PINECONE_API_KEY,
    NAMESPACE,
    LOG_LEVEL
)

# ------------------------
# Logging
# ------------------------
# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure logging to both file and console
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(f"logs/main1_{time.strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ------------------------
# Load data
# ------------------------
def load_data(path: str, test_size: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data and split into train and test sets.
    Test set contains complete reviews (not chunked).
    
    Args:
        path: Path to CSV file
        test_size: Number of complete reviews to reserve for testing
        
    Returns:
        train_df: DataFrame for training (will be chunked)
        test_df: DataFrame for testing (complete reviews)
    """
    try:
        logger.info("Loading dataset")
        if not os.path.exists(path):
            logger.error(f"Data file not found: {path}")
            raise FileNotFoundError(f"Data file not found: {path}")

        df = pd.read_csv(path).dropna(subset=["content"])
        df['at'] = pd.to_datetime(df['at'])
        df = df[df['at'] > pd.Timestamp('2025-01-01')]
        df = df.drop_duplicates(subset="reviewId")
        df = df.reset_index(drop=True)
        
        if df.empty:
            logger.warning("Loaded dataset is empty after removing null content")
            raise ValueError("No valid reviews found in dataset")
        
        if len(df) < test_size + 1:
            logger.error(f"Not enough reviews. Need at least {test_size + 1}, got {len(df)}")
            raise ValueError(f"Not enough reviews. Need at least {test_size + 1}, got {len(df)}")
        
        # Split: last test_size reviews for testing, rest for training
        test_df = df.iloc[-test_size:].reset_index(drop=True)
        train_df = df.iloc[:-test_size].reset_index(drop=True)
        
        logger.info(f"Loaded {len(df)} total reviews")
        logger.info(f"Training set: {len(train_df)} reviews")
        logger.info(f"Test set: {len(test_df)} complete reviews (unchunked)")
        
        return train_df, test_df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

# ------------------------
# Chunk reviews
# ------------------------
def chunk_reviews(df: pd.DataFrame) -> List[Dict]:
    try:
        logger.info("Chunking reviews")
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        records = []
        
        for idx, row in df.iterrows():
            try:
                review_id = row["reviewId"]
                score = int(row["score"])
                text = str(row["content"])
                chunks = splitter.split_text(text)
                
                for i, chunk in enumerate(chunks):
                    records.append({
                        "id": f"{review_id}_{i}",
                        "metadata": {"reviewId": review_id, "text": chunk, "score": score}
                    })
            except Exception as e:
                logger.error(f"Error processing review at index {idx}: {str(e)}")
                continue
                
        logger.info(f"Generated {len(records)} chunks from {len(df)} reviews")
        return records
    except Exception as e:
        logger.error(f"Error in chunk_reviews: {str(e)}")
        raise

# ------------------------
# Embedding
# ------------------------
class Embedder:
    def __init__(self):
        try:
            logger.info("Initializing embedding model")
            self.model = TextEmbedding(model_name=EMBEDDING_MODEL)
            logger.info(f"Successfully initialized embedding model: {EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            raise

    def embed(self, texts: List[str]):
        try:
            logger.info(f"Embedding {len(texts)} texts")
            if len(texts) == 0:
                logger.warning("Empty text list provided for embedding")
                return []
            
            embeddings = list(self.model.embed(texts))
            logger.info(f"Successfully generated {len(embeddings)} embeddings")
            
            embeddings = [e.tolist() if isinstance(e, np.ndarray) else e for e in embeddings]

            return embeddings
        except Exception as e:
            logger.error(f"Error during embedding: {str(e)}")
            raise

# ------------------------
# Pinecone client initialization
# ------------------------
def init_pinecone_client():
    try:
        api_key = PINECONE_API_KEY
        if not api_key:
            logger.error("PINECONE_API_KEY not set!")
            raise RuntimeError("PINECONE_API_KEY not set!")
        
        logger.info("Initializing Pinecone client")
        pc = Pinecone(api_key=api_key)
        logger.info("Pinecone client initialized successfully")
        return pc
    except Exception as e:
        logger.error(f"Error initializing Pinecone client: {str(e)}")
        raise

def create_index_if_not_exists(pc: Pinecone, embedding_dim: int):
    """
    Create a standard vector index in Pinecone.
    """
    try:
        # Delete existing index
        if pc.has_index(PINECONE_INDEX):
            logger.warning(f"Index {PINECONE_INDEX} already exists. Deleting...")
            pc.delete_index(PINECONE_INDEX)
            time.sleep(15)
            logger.info(f"Successfully deleted index {PINECONE_INDEX}")

        if not pc.has_index(PINECONE_INDEX):
            logger.info(f"Creating index {PINECONE_INDEX}...")
            # Standard index (precomputed embeddings)
            pc.create_index(
                name=PINECONE_INDEX,
                dimension=embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=PINECONE_REGION
                )
            )
            logger.info(f"Index {PINECONE_INDEX} created successfully")
        
        logger.info(f"Index {PINECONE_INDEX} is ready")
        return pc.Index(PINECONE_INDEX)
    except Exception as e:
        logger.error(f"Error creating/accessing index: {str(e)}")
        raise

# ------------------------
# Upload vectors in batches
# ------------------------
def upload_vectors(index, records: List[Dict], embeddings, batch_size: int = 100):
    """
    Upload all precomputed embeddings to Pinecone.
    """
    try:
        logger.info(f"Uploading {len(records)} vectors to Pinecone")
        
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
            
        vectors = [
            (records[i]["id"], embeddings[i], records[i]["metadata"])
            for i in range(len(records))
        ]

        uploaded_count = 0
        for i in range(0, len(vectors), batch_size):
            try:
                batch = vectors[i:i+batch_size]
                index.upsert(vectors=batch, namespace=NAMESPACE)
                uploaded_count += len(batch)
                
                if (i // batch_size + 1) % 10 == 0:
                    logger.info(f"Uploaded {uploaded_count}/{len(vectors)} vectors")
            except Exception as e:
                logger.error(f"Error uploading batch {i//batch_size + 1}: {str(e)}")
                raise

        logger.info(f"Upload completed: {uploaded_count} vectors uploaded")
        
        # Wait for indexing
        logger.info("Waiting for indexing to complete...")
        time.sleep(20)
        
        stats = index.describe_index_stats()
        logger.info(f"Index stats: {stats}")
        
        if stats.get('total_vector_count', 0) != uploaded_count:
            logger.warning(f"Vector count mismatch: uploaded {uploaded_count}, index has {stats.get('total_vector_count', 0)}")
    except Exception as e:
        logger.error(f"Error in upload_vectors: {str(e)}")
        raise

# ------------------------
# Test retrieval quality with complete reviews
# ------------------------
def test_query(index, test_df: pd.DataFrame, embedder: Embedder, top_k: int = 5):
    """
    Query using complete, unchunked test reviews and validate retrieval quality.
    
    Args:
        index: Pinecone index
        test_df: DataFrame with complete test reviews
        embedder: Embedder instance to generate query embeddings
        top_k: Number of top results to retrieve
    """
    try:
        logger.info(f"\n{'='*80}")
        logger.info(f"TESTING RETRIEVAL QUALITY with {len(test_df)} complete reviews")
        logger.info(f"{'='*80}")
        
        successful_queries = 0
        failed_queries = 0
        
        for idx, row in test_df.iterrows():
            try:
                review_id = row["reviewId"]
                score = int(row["score"])
                query_text = str(row["content"])
                
                logger.info(f"\n{'='*80}")
                logger.info(f"Query {idx+1}/{len(test_df)}")
                logger.info(f"{'='*80}")
                logger.info(f"Review ID: {review_id}")
                logger.info(f"Review Score: {score}")
                logger.info(f"Review Text: {query_text}...")
                logger.info(f"Full text length: {len(query_text)} characters")

                # Embed the complete review
                logger.info("Generating query embedding...")
                query_embedding = embedder.embed([query_text])[0]
                
                if isinstance(query_embedding, np.ndarray):
                    query_embedding = query_embedding.tolist()

                # Query Pinecone
                logger.info("Querying Pinecone...")
                results = index.query(
                    vector=query_embedding, 
                    top_k=top_k, 
                    include_metadata=True, 
                    namespace=NAMESPACE
                )

                if not results.get('matches'):
                    logger.warning(f"No matches found for query {idx+1}")
                    failed_queries += 1
                    continue

                logger.info(f"\nTop {top_k} Retrieved Chunks:")
                for rank, match in enumerate(results["matches"], 1):
                    logger.info(f"\n  Rank [{rank}]")
                    logger.info(f"  Similarity Score: {match['score']:.4f}")
                    logger.info(f"  Chunk ID: {match['id']}")
                    logger.info(f"  Review ID: {match['metadata']['reviewId']}")
                    logger.info(f"  Review Score: {match['metadata']['score']}")
                    logger.info(f"  Text: {match['metadata']['text']}...")

                successful_queries += 1
                
            except Exception as e:
                logger.error(f"Error processing query {idx+1}: {str(e)}")
                failed_queries += 1
                continue

        logger.info(f"\n{'='*80}")
        logger.info(f"TEST SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Total queries: {len(test_df)}")
        logger.info(f"Successful: {successful_queries}")
        logger.info(f"Failed: {failed_queries}")
        logger.info(f"Success rate: {successful_queries/len(test_df)*100:.1f}%")
        
        if failed_queries > 0:
            logger.warning(f"{failed_queries} queries failed during testing")
    except Exception as e:
        logger.error(f"Error in test_query: {str(e)}")
        raise

# ------------------------
# Main
# ------------------------
def main():
    try:
        logger.info("="*80)
        logger.info("Starting pipeline")
        logger.info("="*80)
        
        # Load data and split into train/test
        train_df, test_df = load_data(DATA_PATH, test_size=10)
        
        # Chunk only the training reviews
        records = chunk_reviews(train_df)
        
        # Initialize embedder
        embedder = Embedder()
        
        # Embed training chunks
        logger.info("Embedding training chunks...")
        embeddings = embedder.embed([r["metadata"]["text"] for r in records])

        # Save to pickle
        os.makedirs("data", exist_ok=True)

        # Initialize Pinecone and upload vectors
        pc = init_pinecone_client()
        index = create_index_if_not_exists(pc, embedding_dim=len(embeddings[0]))
        upload_vectors(index, records, embeddings)

        # Test with complete reviews
        test_query(index, test_df, embedder)
        
        logger.info("="*80)
        logger.info("Pipeline finished successfully")
        logger.info("="*80)
    except Exception as e:
        logger.error(f"Pipeline execution failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()