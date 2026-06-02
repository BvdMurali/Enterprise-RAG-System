"""
Pytest configuration and shared fixtures.
"""

import sys
from unittest.mock import MagicMock

# ----------------------------------------------------------------------
# Mock HuggingFaceEmbeddings to prevent loading PyTorch (torch) in tests,
# which avoids WinError 1455 (paging file too small) on Windows VM.
# ----------------------------------------------------------------------
class MockHuggingFaceEmbeddings:
    def __init__(self, *args, **kwargs):
        self.model_name = "mock-model"

    def embed_documents(self, texts):
        return [[0.1] * 384 for _ in texts]

    def embed_query(self, text):
        return [0.1] * 384

import langchain_community.embeddings
langchain_community.embeddings.HuggingFaceEmbeddings = MockHuggingFaceEmbeddings

import os
import shutil
import pytest
from pathlib import Path

from backend.config import get_settings
from backend.vectorstore.chroma_store import ChromaStoreService
from backend.services.retriever import RetrieverService

# Define a minimal valid PDF byte sequence for parsing tests
MINIMAL_PDF_BYTES = (
    b"%PDF-1.4\n"
    b"1 0 obj <</Type /Catalog /Pages 2 0 R>> endobj\n"
    b"2 0 obj <</Type /Pages /Kids [3 0 R] /Count 1>> endobj\n"
    b"3 0 obj <</Type /Page /Parent 2 0 R /Resources <<>> /MediaBox [0 0 612 792] /Contents 4 0 R>> endobj\n"
    b"4 0 obj <</Length 44>> stream\n"
    b"BT /F1 12 Tf 72 712 Td (Hello World from PDF!) Tj ET\n"
    b"endstream\n"
    b"endobj\n"
    b"xref\n"
    b"0 5\n"
    b"0000000000 65535 f\n"
    b"0000000009 00000 n\n"
    b"0000000056 00000 n\n"
    b"0000000111 00000 n\n"
    b"0000000212 00000 n\n"
    b"trailer <</Size 5 /Root 1 0 R>>\n"
    b"startxref\n"
    b"306\n"
    b"%%EOF"
)


@pytest.fixture(scope="session")
def test_dirs():
    """Create and clean up temporary test directories."""
    test_db_dir = Path("./test_run_db").absolute()
    test_upload_dir = Path("./test_run_data").absolute()
    
    # Cleanup before run
    if test_db_dir.exists():
        shutil.rmtree(test_db_dir, ignore_errors=True)
    if test_upload_dir.exists():
        shutil.rmtree(test_upload_dir, ignore_errors=True)
        
    test_upload_dir.mkdir(parents=True, exist_ok=True)
    test_db_dir.mkdir(parents=True, exist_ok=True)
    
    yield {
        "db_dir": str(test_db_dir),
        "upload_dir": str(test_upload_dir)
    }
    
    # Cleanup after run
    if test_db_dir.exists():
        shutil.rmtree(test_db_dir, ignore_errors=True)
    if test_upload_dir.exists():
        shutil.rmtree(test_upload_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def app_settings(test_dirs):
    """Retrieve settings overridden with test directories."""
    settings = get_settings()
    settings.chroma_persist_dir = test_dirs["db_dir"]
    settings.chroma_collection_name = "test_run_collection"
    settings.upload_dir = test_dirs["upload_dir"]
    settings.google_api_key = "test-key"
    return settings


@pytest.fixture(scope="session")
def dummy_pdf(test_dirs):
    """Write a minimal valid PDF to the test upload directory."""
    pdf_path = Path(test_dirs["upload_dir"]) / "sample.pdf"
    with open(pdf_path, "wb") as f:
        f.write(MINIMAL_PDF_BYTES)
    return pdf_path
