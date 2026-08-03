from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application configuration"""
    
    # API Keys
    groq_api_key: str = Field(default="", env="GROQ_API_KEY")
    gemini_api_key: str = Field(default="", env="GEMINI_API_KEY")
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    langfuse_public_key: Optional[str] = Field(default="", env="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: Optional[str] = Field(default="", env="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(default="https://cloud.langfuse.com", env="LANGFUSE_HOST")
    langfuse_base_url: str = Field(default="https://cloud.langfuse.com", env="LANGFUSE_BASE_URL")
    
    # Qdrant
    qdrant_host: str = Field(default="localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, env="QDRANT_PORT")
    qdrant_collection_name: str = "enterprise_docs"
    
    # Embeddings
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    embedding_device: str = Field(default="cpu", env="EMBEDDING_DEVICE")
    
    # Chunking
    chunk_size: int = Field(default=500, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, env="CHUNK_OVERLAP")
    
    # Retrieval
    top_k: int = Field(default=3, env="TOP_K")
    context_char_limit: int = Field(default=1200, env="CONTEXT_CHAR_LIMIT")
    
    # LLM
    llm_model: str = Field(default="llama-3.3-70b-versatile", env="LLM_MODEL")
    llm_temperature: float = Field(default=0.2, env="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=300, env="LLM_MAX_TOKENS")
    
    # Paths
    data_dir: str = Field(default="data")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


settings = Settings()
