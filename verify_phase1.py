"""Quick verification that Phase 1 setup is correct."""

import sys
from pathlib import Path


def verify_structure():
    """Verify all expected directories and files exist."""
    root = Path(__file__).parent

    expected_dirs = [
        "backend",
        "backend/api",
        "backend/services",
        "backend/vectorstore",
        "backend/loaders",
        "backend/prompts",
        "backend/models",
        "frontend",
        "data",
        "chroma_db",
        "tests",
    ]

    expected_files = [
        "requirements.txt",
        ".env",
        ".gitignore",
        "backend/__init__.py",
        "backend/config.py",
        "backend/logger.py",
        "backend/api/__init__.py",
        "backend/services/__init__.py",
        "backend/vectorstore/__init__.py",
        "backend/loaders/__init__.py",
        "backend/prompts/__init__.py",
        "backend/models/__init__.py",
        "tests/__init__.py",
    ]

    print("=" * 60)
    print("Enterprise RAG System - Phase 1 Verification")
    print("=" * 60)

    all_ok = True

    print("\n[DIRS] Directory Structure:")
    for d in expected_dirs:
        exists = (root / d).is_dir()
        status = "[OK]" if exists else "[MISSING]"
        print(f"  {status} {d}/")
        if not exists:
            all_ok = False

    print("\n[FILES] Files:")
    for f in expected_files:
        exists = (root / f).is_file()
        status = "[OK]" if exists else "[MISSING]"
        print(f"  {status} {f}")
        if not exists:
            all_ok = False

    # Try importing config
    print("\n[CONFIG] Configuration:")
    try:
        sys.path.insert(0, str(root))
        from backend.config import get_settings
        settings = get_settings()
        print(f"  [OK] Config loaded successfully")
        print(f"     LLM Model:       {settings.llm_model_name}")
        print(f"     Embedding Model: {settings.embedding_model_name}")
        print(f"     Chunk Size:      {settings.chunk_size}")
        print(f"     Chunk Overlap:   {settings.chunk_overlap}")
        print(f"     Top-K:           {settings.retrieval_top_k}")
    except Exception as e:
        print(f"  [FAIL] Config error: {e}")
        all_ok = False

    # Check key packages
    print("\n[PACKAGES] Key Dependencies:")
    packages = [
        ("langchain", "langchain"),
        ("chromadb", "chromadb"),
        ("sentence_transformers", "sentence-transformers"),  # installed as sentence_transformers
        ("fastapi", "fastapi"),
        ("streamlit", "streamlit"),
        ("pypdf", "pypdf"),
        ("loguru", "loguru"),
        ("pydantic_settings", "pydantic-settings"),
    ]
    for module, name in packages:
        try:
            __import__(module)
            print(f"  [OK] {name}")
        except ImportError:
            print(f"  [MISSING] {name}")
            all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("RESULT: Phase 1 PASSED - All checks successful!")
    else:
        print("RESULT: Phase 1 FAILED - Some checks did not pass.")
    print("=" * 60)

    return all_ok


if __name__ == "__main__":
    success = verify_structure()
    sys.exit(0 if success else 1)
