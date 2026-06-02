"""
Verification script for Phase 5: Gemini + RAG Pipeline.
Tests prompt rendering, Retriever integration, and Google Gemini model invocation.
If no valid Gemini API Key is detected, it runs in Mock mode to verify pipeline integrity.
"""

import os
import sys
import shutil
from pathlib import Path
from unittest.mock import MagicMock

# Fix OpenBLAS / Safetensors Memory Issues on Windows VM
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

from backend.config import get_settings
from backend.services.embeddings import EmbeddingService
from backend.vectorstore.chroma_store import ChromaStoreService
from backend.services.retriever import RetrieverService
from backend.models.llm import LLMService
from backend.services.rag_pipeline import RAGPipelineService
from langchain_core.documents import Document

class MockLLMResponse:
    def __init__(self, content: str):
        self.content = content

def verify_phase5():
    print("=" * 60)
    print("Enterprise RAG System - Phase 5 Verification")
    print("=" * 60)

    settings = get_settings()
    
    # 1. Check API Key Status
    is_mock = False
    api_key = settings.google_api_key
    if not api_key or api_key == "your-gemini-api-key-here":
        print("[INFO] Google API Key is not set or is using the placeholder.")
        print("[INFO] Running verification in MOCK mode to verify pipeline logic.")
        is_mock = True
    else:
        print("[INFO] Google API Key detected. Running verification with LIVE Gemini API.")

    # Setup temporary testing database
    settings.chroma_persist_dir = "./test_phase5_db"
    settings.chroma_collection_name = "test_rag_pipeline"
    settings.retrieval_top_k = 2
    settings.retrieval_score_threshold = 0.2

    # Cleanup any previous DB
    if Path(settings.chroma_persist_dir).exists():
        shutil.rmtree(settings.chroma_persist_dir)

    try:
        # 2. Init components
        print("\n[TEST] 1. Initializing Services...")
        emb_service = EmbeddingService()
        chroma_service = ChromaStoreService(embeddings=emb_service.embeddings)
        retriever_service = RetrieverService(chroma_store=chroma_service)
        
        # Initialize LLMService
        if is_mock:
            # Create a mock LLM service
            llm_service = MagicMock(spec=LLMService)
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MockLLMResponse(
                "The Enterprise RAG system uses FastAPI for the backend [Source: architecture.pdf, Page: 1]."
            )
            llm_service.get_llm.return_value = mock_llm
        else:
            llm_service = LLMService()

        # Initialize pipeline
        pipeline = RAGPipelineService(
            retriever_service=retriever_service,
            llm_service=llm_service
        )
        print("  [OK] RAG Pipeline initialized successfully.")

        # 3. Add test documents
        print("\n[TEST] 2. Adding test documents to ChromaDB...")
        test_docs = [
            Document(
                page_content="The Enterprise RAG system uses FastAPI for the backend.",
                metadata={"source": "architecture.pdf", "page": 1}
            ),
            Document(
                page_content="LangChain is used to orchestrate the RAG pipeline components.",
                metadata={"source": "architecture.pdf", "page": 2}
            )
        ]
        chroma_service.add_documents(test_docs)
        print(f"  [OK] Added {len(test_docs)} documents.")

        # 4. Run pipeline query
        query = "What backend framework is used?"
        print(f"\n[TEST] 3. Querying Pipeline: '{query}'")
        result = pipeline.ask(query)

        # 5. Check response
        print("--- RAG PIPELINE RESULT ---")
        print(f"Answer: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print("---------------------------")

        if not result.get("answer"):
            print("  [ERROR] Answer is empty.")
            return False

        if not result.get("sources"):
            print("  [ERROR] Sources are empty.")
            return False

        # Validate structure
        for src in result["sources"]:
            if "source" not in src or "page" not in src or "relevance_score" not in src or "content" not in src:
                print(f"  [ERROR] Invalid source structure: {src}")
                return False

        print("\n" + "=" * 60)
        print("RESULT: Phase 5 PASSED - RAG Pipeline working correctly!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n[ERROR] Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        if Path(settings.chroma_persist_dir).exists():
            print(f"\n[CLEANUP] Removing test database at {settings.chroma_persist_dir}")
            try:
                shutil.rmtree(settings.chroma_persist_dir, ignore_errors=True)
            except:
                pass

if __name__ == "__main__":
    success = verify_phase5()
    sys.exit(0 if success else 1)
