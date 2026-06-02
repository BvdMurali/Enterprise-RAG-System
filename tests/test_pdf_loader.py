"""
Unit tests for PDFLoaderService.
"""

import pytest
from pathlib import Path
from backend.loaders.pdf_loader import PDFLoaderService


from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

def test_load_pdf_success(dummy_pdf):
    """Test that a valid PDF is loaded correctly with metadata."""
    mock_pages = [
        Document(
            page_content="Hello World from PDF!",
            metadata={"source": str(dummy_pdf), "page": 0}
        )
    ]
    with patch("backend.loaders.pdf_loader.PyPDFLoader") as MockPyPDFLoader:
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = mock_pages
        MockPyPDFLoader.return_value = mock_loader_instance
        
        documents = PDFLoaderService.load_pdf(dummy_pdf)
        
        assert len(documents) > 0
        assert isinstance(documents[0].page_content, str)
        assert "Hello" in documents[0].page_content
        
        # Verify metadata
        assert Path(documents[0].metadata["source"]).name == "sample.pdf"
        assert documents[0].metadata["page"] == 0  # 0-indexed page in PyPDFLoader


def test_load_pdf_file_not_found():
    """Test that FileNotFoundError is raised for non-existent path."""
    with pytest.raises(FileNotFoundError):
        PDFLoaderService.load_pdf("non_existent_file.pdf")


def test_load_pdf_invalid_extension(test_dirs):
    """Test that ValueError is raised for non-pdf file paths."""
    txt_path = Path(test_dirs["upload_dir"]) / "sample.txt"
    with open(txt_path, "w") as f:
        f.write("Hello World")
        
    with pytest.raises(ValueError):
        PDFLoaderService.load_pdf(txt_path)
