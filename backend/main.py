"""
FastAPI Entrypoint for the Enterprise RAG System.

Initializes the API server, configures CORS, registers routes, and sets up
application lifecycle management (lifespan) to load AI models and services
exactly once upon startup.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.logger import setup_logging, logger
from backend.api.routes import router
from backend.services.embeddings import EmbeddingService
from backend.vectorstore.chroma_store import ChromaStoreService
from backend.services.retriever import RetrieverService
from backend.models.llm import LLMService
from backend.services.rag_pipeline import RAGPipelineService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle event handler (lifespan context manager).
    Loads model weights and establishes database connections once at boot.
    """
    settings = get_settings()
    
    # 1. Setup system-wide logging
    setup_logging("INFO")
    logger.info("Initializing Enterprise RAG FastAPI application...")

    try:
        # 2. Initialize Embeddings Service (loads model weights onto CPU)
        logger.info("Loading Embedding Service...")
        embedding_service = EmbeddingService()
        app.state.embedding_service = embedding_service
        
        # 3. Initialize ChromaDB persistent connection
        logger.info("Connecting to ChromaDB Vector Store...")
        chroma_service = ChromaStoreService(embeddings=embedding_service.embeddings)
        app.state.chroma_service = chroma_service
        
        # 4. Initialize Retriever wrapper
        logger.info("Setting up Retriever Service...")
        retriever_service = RetrieverService(chroma_store=chroma_service)
        app.state.retriever_service = retriever_service
        
        # 5. Initialize LLM Service (Gemini API)
        logger.info("Setting up LLM Service...")
        llm_service = None
        try:
            llm_service = LLMService()
            logger.info("LLM Service ready.")
        except ValueError as ve:
            logger.warning(
                f"LLM Service could not be loaded: {ve}. "
                "Questions asked via /api/ask will fail until a valid GOOGLE_API_KEY is provided."
            )
            
        app.state.llm_service = llm_service

        # 6. Initialize full RAG Pipeline
        logger.info("Orchestrating RAG Pipeline...")
        # Even if llm_service is None, we instantiate it so routes won't break on import/boot,
        # but ask() will raise/return appropriate errors at runtime.
        app.state.rag_pipeline = RAGPipelineService(
            retriever_service=retriever_service,
            llm_service=llm_service
        )
        
        logger.info("Enterprise RAG Application initialization complete and ready.")
        yield

    except Exception as e:
        logger.critical(f"Server failed to start during lifespan setup: {str(e)}", exc_info=True)
        raise

    finally:
        logger.info("Shutting down Enterprise RAG FastAPI application...")


# Initialize FastAPI app with lifespan context
app = FastAPI(
    title="Enterprise RAG System API",
    description="Production-grade API for Multi-PDF text extraction, embedding, vector storage, and grounded RAG answer generation.",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS Middleware
# Allows frontends (like Streamlit running locally) to query this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to frontend origin in strict production environments
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(router)
