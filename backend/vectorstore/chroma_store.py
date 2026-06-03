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
                persist_directory=self.persist_dir,
                collection_metadata={"hnsw:space": "cosine"}
            )
            logger.info("Successfully connected to ChromaDB.")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise

    def add_documents(self, documents: List[Document], batch_size: int = 200) -> List[str]:
        """
        Add chunked documents to the vector store.
        
        This will compute the embeddings for each chunk and persist them along 
        with the text and metadata. To prevent high memory peaks and potential
        OOM crashes on large files, documents are indexed in batches.

        Args:
            documents: List of Document chunks to store.
            batch_size: The number of documents to embed and insert per batch.

        Returns:
            List of string IDs representing the inserted documents.
        """
        if not documents:
            logger.warning("No documents provided to add to vector store.")
            return []

        total_docs = len(documents)
        logger.info(f"Adding {total_docs} document chunks to vector store in batches of {batch_size}...")
        
        all_ids = []
        try:
            for i in range(0, total_docs, batch_size):
                batch = documents[i : i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (total_docs + batch_size - 1) // batch_size
                
                logger.info(f"Indexing batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
                batch_ids = self.vector_store.add_documents(batch)
                all_ids.extend(batch_ids)
                
            logger.info(f"Successfully added all {len(all_ids)} document chunks to vector store.")
            return all_ids
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
