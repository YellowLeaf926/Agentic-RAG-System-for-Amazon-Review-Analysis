import os
import pickle
import logging
import time
from pinecone import Pinecone
from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_core.retrievers import BaseRetriever
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document
from typing import List
from langchain_core.prompts import PromptTemplate
from parameters import (
    EMBEDDING_MODEL,
    PINECONE_INDEX,
    PINECONE_API_KEY,
    OPENAI_API_KEY,
    NAMESPACE,
    TOP_K
)

# ------------------------
# Logging
# ------------------------
os.makedirs("logs", exist_ok=True)

# Configure logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(f"logs/main2_{time.strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY not set")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

# ------------------------
# Load Pinecone VectorStore
# ------------------------
def load_vectorstore() -> PineconeVectorStore:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX)
        embedding_model = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = PineconeVectorStore(index=index, namespace=NAMESPACE, embedding=embedding_model, text_key="text")
        logger.info("Vector store loaded successfully")
        return vectorstore

    except Exception as e:
        logger.error("Failed to load Pinecone VectorStore", exc_info=True)
        raise

# ------------------------
# Simple LLM (no retrieval)
# ------------------------
def simple_llm():
    # llm_chain = (
    #     llm
    #     | StrOutputParser()
    # )
    prompt = PromptTemplate.from_template("{query}")
    llm_chain = prompt | llm | StrOutputParser()
    for query in query_list:
        logger.info("=" * 80)
        logger.info(f"QUESTION: {query}")
        answer = llm_chain.invoke({"query": query})
        logger.info(f"ANSWER:{answer}")
    logger.info("Completed simple LLM testing")
    return 
    

# ------------------------
# Format retrieved docs
# ------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ------------------------
# Build RAG
# ------------------------

# create custom retriever
class CustomRetriever(BaseRetriever):
    vectorstore: PineconeVectorStore

    def _get_relevant_documents(self, query):
        try:
            docs = self.vectorstore.similarity_search(query, k=TOP_K)
            if not docs:
                logger.warning(f"No documents retrieved for query: {query}")
            return docs

        except Exception as e:
            logger.error(f"Retriever failed for query: {query}", exc_info=True)
            return []

# Simple RAG
def simple_rag():

    client = Client()
    prompt = client.pull_prompt("rlm/rag-prompt")

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    for query in query_list:
        logger.info("=" * 80)
        logger.info(f"QUESTION: {query}")
        answer = rag_chain.invoke(query)
        logger.info(f"ANSWER:{answer}")
    logger.info("Completed simple RAG testing")
    return

if __name__ == "__main__":
    logger.info("Starting LLM/RAG Pipeline")
    logger.info("="*80)
    query_list = [
        "Why do many users report that the Amazon Shopping app no longer works on tablets in recent versions?",
        "What are the most common complaints after the latest Amazon Shopping app updates?",
        "What issues are most frequently mentioned in 1-star Amazon Shopping app reviews?",
        "What do users like and dislike the most about the Amazon Shopping app?",
        "Based on recent user reviews, should I install or update the Amazon Shopping app? Why or why not?",
        "Any complaints about the app's haptic feedback calibration?"
    ]

    logger.info(f"Loaded {len(query_list)} queries for testing")

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, seed=0)

    # Load vector store and create retriever
    vectorstore = load_vectorstore()
    retriever = CustomRetriever(vectorstore=vectorstore)
    
    # simple LLM only
    simple_llm()

    # simple RAG
    simple_rag()

    logger.info("\n" + "="*80)
    logger.info("Pipeline completed successfully")
    logger.info("="*80)
