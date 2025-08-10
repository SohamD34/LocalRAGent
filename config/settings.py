from pydantic import BaseSettings
from typing import List

class Settings(BaseSettings):
    # API Keys
    langchain_api_key: str
    firecrawl_api_key: str
    tavily_api_key: str
    
    # Model settings
    llm_model: str = "llama3.1:8b"
    temperature: float = 0.0
    
    # Vector store settings
    chunk_size: int = 256
    chunk_overlap: int = 0
    collection_name: str = "rag-chroma"
    
    # Hybrid search settings
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    rerank_top_k: int = 10
    final_top_k: int = 5
    
    # Paths
    vectorstore_path: str = "./data/vectorstore"
    
    # Default URLs
    default_urls: List[str] = [
        "https://www.ai-jason.com/learning-ai/how-to-reduce-llm-cost",
        "https://www.ai-jason.com/learning-ai/gpt5-llm",
        "https://www.ai-jason.com/learning-ai/how-to-buid-ai-agent-tutorial-3",
    ]
    
    class Config:
        env_file = ".env"
