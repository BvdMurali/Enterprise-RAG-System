"""
Text Chunking Service for Enterprise RAG System.

This module is responsible for splitting large documents into smaller, semantically
meaningful chunks. It uses LangChain's RecursiveCharacterTextSplitter, which tries
to split on paragraphs, then sentences, then words, to keep related text together.
"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import get_settings
from backend.logger import logger


class ChunkingService:
    """Service to split document text into optimal chunks for embedding."""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialize the ChunkingService with specific sizes, or fallback to config defaults.

        Args:
            chunk_size: Maximum size of chunks to return (characters).
            chunk_overlap: Overlap in characters between chunks.
        """
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

        # We use RecursiveCharacterTextSplitter as it's the recommended text splitter
        # for generic text. It is parameterized by a list of characters. It tries to split
        # on them in order until the chunks are small enough.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            # The default separators ["\n\n", "\n", " ", ""] usually work well for general text.
        )
        
        logger.debug(
            f"Initialized ChunkingService (chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap})"
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split a list of documents into smaller chunks.

        The metadata from the original document (e.g., source file, page number)
        is automatically preserved and attached to the resulting chunks.

        Args:
            documents: List of original LangChain Document objects (e.g., from a PDF loader).

        Returns:
            List[Document]: A list of chunked Document objects.
        """
        if not documents:
            logger.warning("No documents provided for chunking.")
            return []

        logger.info(f"Chunking {len(documents)} original document pages...")
        
        try:
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Successfully split {len(documents)} pages into {len(chunks)} chunks.")
            return chunks
        except Exception as e:
            logger.error(f"Error during document chunking: {str(e)}")
            raise
