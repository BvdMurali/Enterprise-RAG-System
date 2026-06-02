"""
Integration tests for RetrieverService and ChromaStoreService.
"""

import pytest
from pathlib import Path
from langchain_core.documents import Document

from backend.services.embeddings import EmbeddingService
from backend.vectorstore.chroma_store import ChromaStoreService
from backend.services.retriever import RetrieverService


@pytest.fixture(scope="module")
def shared_db(app_settings):
    """Set up and tear down a shared test vector database."""
    # Ensure memory settings for Windows VM
    import os
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    
    emb_service = EmbeddingService()
    chroma_service = ChromaStoreService(embeddings=emb_service.embeddings)
    
    # Pre-populate some documents
    docs = [
        Document(
            page_content="FastAPI is a modern web framework for building APIs with Python.",
            metadata={"source": "api.pdf", "page": 1}
        ),
        Document(
            page_content="ChromaDB is a database for storing and querying vector embeddings.",
            metadata={"source": "database.pdf", "page": 1}
        )
    ]
    chroma_service.add_documents(docs)
    
    yield chroma_service
    
    # Tear down database
    try:
        chroma_service.delete_collection()
    except:
        pass


def test_add_and_retrieve(shared_db):
    """Test that documents are successfully retrieved with similarity scores."""
    retriever_service = RetrieverService(chroma_store=shared_db)
    
    # Test query
    results = retriever_service.retrieve("FastAPI framework")
    
    assert len(results) > 0
    assert "FastAPI" in results[0].page_content
    assert results[0].metadata["source"] == "api.pdf"
    assert "relevance_score" in results[0].metadata


def test_formatted_context(shared_db):
    """Test that retrieved context chunks are formatted correctly with citations."""
    retriever_service = RetrieverService(chroma_store=shared_db)
    
    context = retriever_service.retrieve_formatted_context("ChromaDB vector database")
    
    assert "[Source: database.pdf, Page: 1]" in context
    assert "ChromaDB is a database" in context
