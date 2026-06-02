"""
Integration tests for FastAPI endpoints using TestClient.
Mocks Gemini LLM and PDF parser file reads to run without API keys and file I/O dependencies.
"""

import pytest
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from backend.main import app
from langchain_core.documents import Document

class MockLLMResponse:
    def __init__(self, content: str):
        self.content = content


@pytest.fixture
def mock_pdf_load():
    """Mock the PDF parsing service to return test documents."""
    mock_pages = [
        Document(
            page_content="Pytest is a framework that makes it easy to write simple and scalable tests.",
            metadata={"source": "testing.pdf", "page": 1}
        )
    ]
    with patch("backend.loaders.pdf_loader.PDFLoaderService.load_pdf", return_value=mock_pages) as mock:
        yield mock


@pytest.fixture
def mock_gemini():
    """Mock Gemini API generator client."""
    with patch("backend.models.llm.ChatGoogleGenerativeAI") as MockLLMClass:
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Pytest is a Python testing framework [Source: testing.pdf, Page: 1]."
        mock_llm_instance.invoke.return_value = mock_response
        MockLLMClass.return_value = mock_llm_instance
        yield mock_llm_instance


def test_api_health():
    """Test that health check route is online."""
    with TestClient(app) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


def test_api_upload_flow(app_settings, mock_pdf_load, mock_gemini):
    """Test full upload, ask, list, and delete workflow via HTTP requests."""
    with TestClient(app) as client:
        # 1. Upload PDF
        files = {"file": ("testing.pdf", b"%PDF-1.4 mock pdf data", "application/pdf")}
        response = client.post("/api/upload", files=files)
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        assert response.json()["filename"] == "testing.pdf"

        # Check file exists on disk in upload directory
        saved_file = Path(app_settings.upload_path) / "testing.pdf"
        assert saved_file.exists()

        # 2. List documents
        response = client.get("/api/documents")
        assert response.status_code == 200
        docs = response.json()
        assert len(docs) == 1
        assert docs[0]["filename"] == "testing.pdf"
        assert docs[0]["chunk_count"] > 0

        # 3. Ask RAG Question
        payload = {"question": "What is pytest?", "filter_document": "testing.pdf"}
        response = client.post("/api/ask", json=payload)
        assert response.status_code == 200
        ans_data = response.json()
        assert "testing framework" in ans_data["answer"]
        assert len(ans_data["sources"]) > 0
        assert ans_data["sources"][0]["source"] == "testing.pdf"

        # 4. Delete document
        response = client.delete("/api/documents/testing.pdf")
        assert response.status_code == 200
        assert response.json()["status"] == "success"

        # Check file was deleted from disk
        assert not saved_file.exists()

        # Verify document list is empty
        response = client.get("/api/documents")
        assert response.status_code == 200
        assert len(response.json()) == 0
