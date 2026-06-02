import os
import sys
from pathlib import Path
import shutil

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
from langchain_core.documents import Document

def verify_phase4():
    print("=" * 60)
    print("Enterprise RAG System - Phase 4 Verification")
    print("=" * 60)

    # 1. Setup temporary testing environment
    settings = get_settings()
    settings.chroma_persist_dir = "./test_phase4_db"
    settings.chroma_collection_name = "test_retrieval"
    settings.retrieval_top_k = 2
    settings.retrieval_score_threshold = 0.2

    # Cleanup any previous test DB
    if Path(settings.chroma_persist_dir).exists():
        shutil.rmtree(settings.chroma_persist_dir)

    try:
        # 2. Init components
        print("\n[TEST] 1. Initializing Services...")
        emb_service = EmbeddingService()
        chroma_service = ChromaStoreService(embeddings=emb_service.embeddings)
        retriever_service = RetrieverService(chroma_store=chroma_service)
        print("  [OK] Services initialized successfully.")

        # 3. Add test documents
        print("\n[TEST] 2. Adding test documents...")
        test_docs = [
            Document(
                page_content="The Enterprise RAG system uses FastAPI for the backend.",
                metadata={"source": "architecture.pdf", "page": 1}
            ),
            Document(
                page_content="LangChain is used to orchestrate the RAG pipeline components.",
                metadata={"source": "architecture.pdf", "page": 2}
            ),
            Document(
                page_content="Apples are usually red or green, and they grow on trees.",
                metadata={"source": "fruits.pdf", "page": 1}
            )
        ]
        chroma_service.add_documents(test_docs)
        print(f"  [OK] Added {len(test_docs)} documents.")

        # 4. Test Retrieval
        print("\n[TEST] 3. Testing Semantic Search (Query: 'What backend framework is used?')")
        results = retriever_service.retrieve("What backend framework is used?")
        
        if not results:
            print("  [ERROR] No results found.")
            return False
            
        print(f"  [OK] Retrieved {len(results)} chunks.")
        print(f"  [OK] Top match: {results[0].page_content}")
        print(f"  [OK] Score: {results[0].metadata.get('relevance_score')}")
        
        if "FastAPI" not in results[0].page_content:
            print("  [ERROR] Did not retrieve the expected chunk.")
            return False

        # 5. Test Formatted Context
        print("\n[TEST] 4. Testing Formatted Context")
        formatted = retriever_service.retrieve_formatted_context("What backend framework is used?")
        print("--- FORMATTED CONTEXT ---")
        print(formatted)
        print("-------------------------")
        
        if "[Source: architecture.pdf, Page: 1]" not in formatted:
            print("  [ERROR] Formatted context missing citation.")
            return False

        print("\n" + "=" * 60)
        print("RESULT: Phase 4 PASSED - Retriever working correctly!")
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
            # Windows workaround for ChromaDB file locking: wait briefly or ignore errors
            try:
                shutil.rmtree(settings.chroma_persist_dir, ignore_errors=True)
            except:
                pass

if __name__ == "__main__":
    success = verify_phase4()
    sys.exit(0 if success else 1)
