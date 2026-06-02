"""
Verification script for Phase 2: PDF Loader and Chunking Service.
Downloads a sample PDF, processes it, and verifies output structures.
"""

import sys
import urllib.request
from pathlib import Path

from backend.loaders.pdf_loader import PDFLoaderService
from backend.services.chunking import ChunkingService
from backend.logger import setup_logging, logger

# Initialize logger for stdout
setup_logging("DEBUG")

def create_sample_pdf(filepath: Path):
    """Download a tiny sample PDF for testing."""
    url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    print(f"\n[DOWNLOAD] Fetching dummy PDF from {url}")
    urllib.request.urlretrieve(url, filepath)
    print(f"[DOWNLOAD] Saved to {filepath}")

def verify_phase2():
    print("=" * 60)
    print("Enterprise RAG System - Phase 2 Verification")
    print("=" * 60)
    
    test_pdf_path = Path("data/dummy.pdf")
    test_pdf_path.parent.mkdir(exist_ok=True)
    
    try:
        # 1. Provide a PDF
        if not test_pdf_path.exists():
            create_sample_pdf(test_pdf_path)
        
        # 2. Test Loader
        print("\n[TEST] 1. PDF Loader Service")
        docs = PDFLoaderService.load_pdf(test_pdf_path)
        
        if not docs:
            raise ValueError("Loader returned empty list.")
            
        print(f"  [OK] Loaded {len(docs)} pages.")
        
        first_page = docs[0]
        print(f"  [OK] Page Content (preview): '{first_page.page_content[:50]}...'")
        print(f"  [OK] Metadata: {first_page.metadata}")
        
        # Validate metadata keys
        if "source" not in first_page.metadata or "page" not in first_page.metadata:
            raise ValueError(f"Missing expected metadata. Got: {first_page.metadata}")
            
        # 3. Test Chunking
        print("\n[TEST] 2. Chunking Service")
        # Initialize with small chunks for this tiny PDF to force splitting
        chunker = ChunkingService(chunk_size=10, chunk_overlap=2)
        
        chunks = chunker.chunk_documents(docs)
        
        if not chunks:
            raise ValueError("Chunking returned empty list.")
            
        print(f"  [OK] Split {len(docs)} pages into {len(chunks)} chunks.")
        
        first_chunk = chunks[0]
        print(f"  [OK] First chunk content: '{first_chunk.page_content}'")
        print(f"  [OK] First chunk metadata: {first_chunk.metadata}")
        
        if "source" not in first_chunk.metadata or "page" not in first_chunk.metadata:
             raise ValueError("Metadata was lost during chunking!")
             
        print("\n" + "=" * 60)
        print("RESULT: Phase 2 PASSED - Services working correctly!")
        print("=" * 60)
        return True
        
    except Exception as e:
        logger.exception("Verification failed!")
        print("\n" + "=" * 60)
        print(f"RESULT: Phase 2 FAILED - {e}")
        print("=" * 60)
        return False
    finally:
        # Cleanup
        if test_pdf_path.exists():
            test_pdf_path.unlink()

if __name__ == "__main__":
    success = verify_phase2()
    sys.exit(0 if success else 1)
