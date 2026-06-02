"""
Unit tests for ChunkingService.
"""

from langchain_core.documents import Document
from backend.services.chunking import ChunkingService


def test_chunk_documents():
    """Test that documents are chunked and metadata is preserved."""
    chunk_size = 50
    chunk_overlap = 10
    chunker = ChunkingService(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # 100 character content (should result in at least 2 chunks of size 50)
    docs = [
        Document(
            page_content="This is a very long text sentence designed to trigger chunking splitting mechanisms correctly.",
            metadata={"source": "test.pdf", "page": 1}
        )
    ]
    
    chunks = chunker.chunk_documents(docs)
    
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk.page_content) <= chunk_size
        assert chunk.metadata["source"] == "test.pdf"
        assert chunk.metadata["page"] == 1


def test_chunk_empty_list():
    """Test that chunking an empty list of documents returns empty."""
    chunker = ChunkingService()
    assert chunker.chunk_documents([]) == []
