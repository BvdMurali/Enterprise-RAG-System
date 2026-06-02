"""
Verification script for Phase 6: FastAPI Backend.
Uses FastAPI TestClient to verify health check, document upload, retrieval, and deletion.
Mocks model inference and file I/O for clean, deterministic local verification.
"""

import os
import sys
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

# Fix OpenBLAS / Safetensors Memory Issues on Windows VM
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

from fastapi.testclient import TestClient
from backend.config import get_settings
from backend.main import app
from langchain_core.documents import Document

def verify_phase6():
    print("=" * 60)
    print("Enterprise RAG System - Phase 6 Verification")
    print("=" * 60)

    settings = get_settings()
    
    # Configure test directories
    settings.chroma_persist_dir = "./test_phase6_db"
    settings.chroma_collection_name = "test_api_endpoints"
    settings.upload_dir = "./test_phase6_data"

    # Cleanup test dirs
    if Path(settings.chroma_persist_dir).exists():
        shutil.rmtree(settings.chroma_persist_dir, ignore_errors=True)
    if Path(settings.upload_dir).exists():
        shutil.rmtree(settings.upload_dir, ignore_errors=True)

    # Mock PDFLoaderService.load_pdf to return dummy documents and avoid actual binary parsing
    mock_pdf_pages = [
        Document(
            page_content="The Enterprise RAG system uses FastAPI for the backend.",
            metadata={"source": "architecture.pdf", "page": 1}
        ),
        Document(
            page_content="LangChain is used to orchestrate the RAG pipeline components.",
            metadata={"source": "architecture.pdf", "page": 2}
        )
    ]

    print("\n[TEST] 1. Initializing FastAPI TestClient and lifespan...")
    
    # We patch both the PDF Loader and LLM invocation to test end-to-end flow without calling external APIs
    with patch("backend.loaders.pdf_loader.PDFLoaderService.load_pdf", return_value=mock_pdf_pages), \
         patch("backend.models.llm.ChatGoogleGenerativeAI") as MockLLMClass:
         
        # Set up mock response for LLM
        mock_llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "FastAPI is the backend framework [Source: architecture.pdf, Page: 1]."
        mock_llm_instance.invoke.return_value = mock_response
        MockLLMClass.return_value = mock_llm_instance

        # TestClient handles startup and shutdown lifespan events
        with TestClient(app) as client:
            print("  [OK] Lifespan started successfully.")

            # Test 1: Health Check
            print("\n[TEST] 2. Checking /api/health...")
            response = client.get("/api/health")
            assert response.status_code == 200
            assert response.json() == {"status": "healthy"}
            print("  [OK] Health check passed.")

            # Test 2: Document Upload
            print("\n[TEST] 3. Testing POST /api/upload...")
            # We pass dummy binary data because load_pdf is patched
            files = {"file": ("architecture.pdf", b"%PDF-1.4...", "application/pdf")}
            response = client.post("/api/upload", files=files)
            assert response.status_code == 200
            json_resp = response.json()
            assert json_resp["status"] == "success"
            assert json_resp["filename"] == "architecture.pdf"
            # Since chunker splits 2 pages (each ~50 chars) with size 1000, we get 2 chunks
            assert json_resp["chunks_count"] > 0
            print(f"  [OK] Upload succeeded. Indexed {json_resp['chunks_count']} chunks.")

            # Test 3: List Documents
            print("\n[TEST] 4. Testing GET /api/documents...")
            response = client.get("/api/documents")
            assert response.status_code == 200
            docs = response.json()
            assert len(docs) == 1
            assert docs[0]["filename"] == "architecture.pdf"
            print(f"  [OK] Documents listed: {docs}")

            # Test 4: RAG Ask
            print("\n[TEST] 5. Testing POST /api/ask...")
            query_payload = {"question": "What backend framework is used?", "filter_document": "architecture.pdf"}
            response = client.post("/api/ask", json=query_payload)
            assert response.status_code == 200
            rag_response = response.json()
            assert "FastAPI" in rag_response["answer"]
            assert len(rag_response["sources"]) > 0
            assert rag_response["sources"][0]["source"] == "architecture.pdf"
            print(f"  [OK] Question answered successfully: '{rag_response['answer']}'")

            # Test 5: Delete Document
            print("\n[TEST] 6. Testing DELETE /api/documents/{filename}...")
            response = client.delete("/api/documents/architecture.pdf")
            assert response.status_code == 200
            assert response.json()["status"] == "success"
            print("  [OK] Document deleted.")

            # Verify deletion listing
            response = client.get("/api/documents")
            assert response.status_code == 200
            assert len(response.json()) == 0
            print("  [OK] Document list verified empty after deletion.")

    print("\n" + "=" * 60)
    print("RESULT: Phase 6 PASSED - FastAPI Backend working perfectly!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    try:
        success = verify_phase6()
        sys.exit(0 if success else 1)
    except AssertionError as ae:
        print(f"\n[ERROR] Assertion failed: {ae}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
