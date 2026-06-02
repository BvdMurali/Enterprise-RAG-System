"""
Vector Store Service for Enterprise RAG System.

This module provides an interface to ChromaDB using LangChain's Chroma wrapper.
It handles initializing the database, adding document chunks (which automatically
computes and stores their embeddings), and setting up the retriever for semantic search.
"""

from typing import List
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

from backend.config import get_settings
from backend.logger import logger


class ChromaStoreService:
    """Service to manage the ChromaDB vector database."""

    def __init__(self, embeddings: Embeddings):
        """
        Initialize the Chroma Vector Store.

        Args:
            embeddings: The LangChain Embeddings instance used to convert text to vectors.
        """
        settings = get_settings()
        self.persist_dir = str(settings.chroma_path)
        self.collection_name = settings.chroma_collection_name
        self.embeddings = embeddings

        logger.info(f"Connecting to ChromaDB at {self.persist_dir}, collection: {self.collection_name}")
        
        try:
            # Initialize Chroma. If the directory already contains a database, 
            # it will load it. Otherwise, it will create a new one.
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_dir
            )
            logger.info("Successfully connected to ChromaDB.")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add chunked documents to the vector store.
        
        This will compute the embeddings for each chunk and persist them along 
        with the text and metadata.

        Args:
            documents: List of Document chunks to store.

        Returns:
            List of string IDs representing the inserted documents.
        """
        if not documents:
            logger.warning("No documents provided to add to vector store.")
            return []

        logger.info(f"Adding {len(documents)} document chunks to vector store...")
        try:
            ids = self.vector_store.add_documents(documents)
            logger.info(f"Successfully added {len(ids)} document chunks.")
            return ids
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {str(e)}")
            raise

    def get_retriever(self, k: int = None):
        """
        Get a LangChain Retriever interface for the vector store.
        
        This is used later in the RAG pipeline to query the database.

        Args:
            k: Number of top relevant documents to retrieve. 
               If None, uses the default from configuration.
               
        Returns:
            A LangChain VectorStoreRetriever.
        """
        settings = get_settings()
        search_kwargs = {"k": k or settings.retrieval_top_k}
        
        # We can also add score thresholds if needed:
        # search_kwargs["score_threshold"] = settings.retrieval_score_threshold
        
        return self.vector_store.as_retriever(
            search_type="similarity", # Alternatively: "mmr" (Maximal Marginal Relevance)
            search_kwargs=search_kwargs
        )

    def delete_collection(self):
        """Delete the entire collection (useful for resetting/testing)."""
        logger.warning(f"Deleting ChromaDB collection: {self.collection_name}")
        self.vector_store.delete_collection()
