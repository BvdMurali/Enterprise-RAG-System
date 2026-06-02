"""
PDF Loader Service for Enterprise RAG System.

This module provides functionality to load and extract text from PDF documents.
It utilizes LangChain's PyPDFLoader, which parses the PDF and creates Document objects
with page content and basic metadata (like source file path and page number).
"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

from backend.logger import logger


class PDFLoaderService:
    """Service to load and extract text from PDF files."""

    @staticmethod
    def load_pdf(file_path: str | Path) -> List[Document]:
        """
        Load a PDF file and extract its content into a list of Documents.

        Each Document represents a page in the PDF and contains the text content
        as well as metadata indicating the source file and page number.

        Args:
            file_path: Path to the PDF file.

        Returns:
            List[Document]: A list of LangChain Document objects.

        Raises:
            FileNotFoundError: If the specified PDF file does not exist.
            ValueError: If the file is not a PDF.
            Exception: If PyPDFLoader fails to process the file.
        """
        path = Path(file_path)

        if not path.exists():
            logger.error(f"Failed to load PDF: File not found at {path}")
            raise FileNotFoundError(f"PDF file not found: {path}")
            
        if path.suffix.lower() != ".pdf":
            logger.error(f"Failed to load PDF: File is not a PDF: {path}")
            raise ValueError(f"File must be a PDF, got {path.suffix}")

        logger.info(f"Loading PDF document from {path}")
        
        try:
            loader = PyPDFLoader(str(path))
            documents = loader.load()
            
            logger.info(f"Successfully loaded {len(documents)} pages from {path.name}")
            return documents
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF {path}: {str(e)}")
            raise
