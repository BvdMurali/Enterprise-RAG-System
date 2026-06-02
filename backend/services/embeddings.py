"""
Embeddings Service for Enterprise RAG System.

This module is responsible for converting text chunks into high-dimensional
dense vector embeddings using open-source models from HuggingFace
(via sentence-transformers).
"""

from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from backend.config import get_settings
from backend.logger import logger


class EmbeddingService:
    """Service for generating vector embeddings from text."""

    def __init__(self, model_name: str = None):
        """
        Initialize the embedding model.

        Args:
            model_name: Name of the HuggingFace model to use. If None, uses the
                        value from centralized configuration.
        """
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model_name
        
        logger.info(f"Initializing HuggingFaceEmbeddings with model: {self.model_name}")
        
        try:
            # We use HuggingFaceEmbeddings which downloads and runs the model locally.
            # all-MiniLM-L6-v2 is an excellent default for fast, accurate semantic search.
            self.embeddings: Embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                # Pass device=cpu, and use_safetensors=False to avoid Windows OOM / Paging file error (os error 1455)
                model_kwargs={
                    'device': 'cpu', 
                    'model_kwargs': {'use_safetensors': False}
                }, 
                encode_kwargs={'normalize_embeddings': True} # Normalize for cosine similarity
            )
            logger.info("Successfully initialized embedding model.")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            raise

    def get_embeddings(self) -> Embeddings:
        """
        Get the initialized LangChain Embeddings interface.
        
        Returns:
            The LangChain Embeddings object ready for use by a Vector Store.
        """
        return self.embeddings
