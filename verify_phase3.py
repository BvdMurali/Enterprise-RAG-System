"""
Verification script for Phase 3: Embeddings & ChromaDB.
Tests the HuggingFace embedding model and ChromaDB insertion/retrieval.
"""

import sys
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import shutil
from pathlib import Path

from langchain_core.documents import Document

from backend.services.embeddings import EmbeddingService
from backend.vectorstore.chroma_store import ChromaStoreService
from backend.logger import setup_logging, logger

# Initialize logger for stdout
setup_logging("DEBUG")

def verify_phase3():
    print("=" * 60)
    print("Enterprise RAG System - Phase 3 Verification")
    print("=" * 60)
    
    # We will use a temporary chroma DB dir for the test to keep it clean
    test_chroma_dir = "test_chroma_db"
    os.environ["CHROMA_PERSIST_DIR"] = test_chroma_dir
    os.environ["CHROMA_COLLECTION_NAME"] = "test_collection"
    
    try:
        # 1. Test Embedding Service
        print("\n[TEST] 1. Embedding Service (Downloading model if first run...)")
        emb_service = EmbeddingService()
        embeddings_model = emb_service.get_embeddings()
        
        # Test an actual embedding
        test_text = "The quick brown fox jumps over the lazy dog."
        vector = embeddings_model.embed_query(test_text)
        
        if not vector or not isinstance(vector, list) or len(vector) == 0:
            raise ValueError("Embedding generation failed.")
            
        print(f"  [OK] Successfully loaded embedding model: {emb_service.model_name}")
        print(f"  [OK] Generated embedding vector of dimension {len(vector)}")
        
        # 2. Test ChromaDB Store
        print("\n[TEST] 2. ChromaDB Vector Store")
        chroma_service = ChromaStoreService(embeddings=embeddings_model)
        
        # Create some dummy chunks
        dummy_docs = [
            Document(page_content="Apples are red and sweet.", metadata={"source": "doc1.txt", "page": 1}),
            Document(page_content="Bananas are yellow and long.", metadata={"source": "doc1.txt", "page": 2}),
            Document(page_content="The sky is blue today.", metadata={"source": "doc2.txt", "page": 1})
        ]
        
        print("  [ACTION] Adding 3 test documents to ChromaDB...")
        ids = chroma_service.add_documents(dummy_docs)
        print(f"  [OK] Successfully inserted {len(ids)} documents.")
        
        # 3. Test Retrieval
        print("\n[TEST] 3. Semantic Search / Retrieval")
        retriever = chroma_service.get_retriever(k=1)
        
        query = "What color is a banana?"
        print(f"  [ACTION] Querying vector store for: '{query}'")
        
        results = retriever.invoke(query)
        
        if not results:
            raise ValueError("Retriever returned no results!")
            
        top_match = results[0]
        print(f"  [OK] Top match content: '{top_match.page_content}'")
        print(f"  [OK] Top match metadata: {top_match.metadata}")
        
        if "Banana" not in top_match.page_content and "yellow" not in top_match.page_content:
            print("  [WARNING] The top match might not be semantically correct. Expected 'Bananas...'")
        else:
            print("  [OK] Semantic match is correct!")
             
        print("\n" + "=" * 60)
        print("RESULT: Phase 3 PASSED - Services working correctly!")
        print("=" * 60)
        return True
        
    except Exception as e:
        logger.exception("Verification failed!")
        print("\n" + "=" * 60)
        print(f"RESULT: Phase 3 FAILED - {e}")
        print("=" * 60)
        return False
    finally:
        # Cleanup test DB
        if Path(test_chroma_dir).exists():
            print(f"\n[CLEANUP] Removing test database at {test_chroma_dir}")
            # Windows might hold locks on sqlite files, ignore errors during cleanup
            shutil.rmtree(test_chroma_dir, ignore_errors=True)

if __name__ == "__main__":
    success = verify_phase3()
    sys.exit(0 if success else 1)
