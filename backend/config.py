"""
Enterprise RAG System — Centralized Configuration

Uses Pydantic Settings to load configuration from environment variables
with validation and type coercion. This is the single source of truth
for all configurable values in the application.

Why Pydantic Settings?
- Type-safe configuration with automatic validation
- Loads from .env files automatically
- Provides defaults while allowing overrides
- Documents all configuration in one place
- Enterprise pattern: 12-factor app methodology
"""

import os
from pathlib import Path

# Fix OpenBLAS / Safetensors Memory Issues on Windows VM
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Google Gemini API ---
    google_api_key: str = "your-gemini-api-key-here"

    # --- LLM Configuration ---
    llm_model_name: str = "gemini-2.5-flash"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 2048

    # --- Embedding Configuration ---
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"

    # --- Chunking Configuration ---
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # --- Retriever Configuration ---
    retrieval_top_k: int = 5
    retrieval_score_threshold: float = 0.15

    # --- ChromaDB Configuration ---
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "enterprise_rag"

    # --- Advanced RAG Settings ---
    parent_store_dir: str = "./parent_store"
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    semantic_cache_db: str = "./semantic_cache.db"
    semantic_cache_threshold: float = 0.08
    jwt_secret: str = "your-local-jwt-secret-key"
    jwt_algorithm: str = "HS256"

    # --- Upload Configuration ---
    upload_dir: str = "./data"
    max_file_size_mb: int = 50

    # --- Server Configuration ---
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    frontend_port: int = 8501

    @property
    def upload_path(self) -> Path:
        """Resolved upload directory path."""
        path = Path(self.upload_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def chroma_path(self) -> Path:
        """Resolved ChromaDB persistence directory path."""
        path = Path(self.chroma_persist_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def parent_store_path(self) -> Path:
        """Resolved Parent Document Store directory path."""
        path = Path(self.parent_store_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def max_file_size_bytes(self) -> int:
        """Maximum file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.

    Uses lru_cache to ensure settings are loaded only once
    and reused across the application (singleton pattern).
    """
    return Settings()
