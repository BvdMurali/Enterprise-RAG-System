"""
FastAPI Routes for Enterprise RAG System.

Defines API endpoints for file uploading, vector store querying, RAG,
and document collection management.
"""

import os
import json
from pathlib import Path
from typing import List
from collections import Counter

from fastapi import APIRouter, File, UploadFile, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse

from backend.config import get_settings
from backend.logger import logger
from backend.api.schemas import (
    QuestionRequest, 
    AnswerResponse, 
    UploadResponse, 
    DocumentInfo,
    SourceCitation,
    ConversationState
)
from backend.loaders.pdf_loader import PDFLoaderService
from backend.services.chunking import ChunkingService

router = APIRouter(prefix="/api")


@router.get("/health", tags=["System"])
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}


@router.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    """
    Upload a PDF document, extract its text, chunk it, embed the chunks,
    and persist them in the ChromaDB vector database.
    """
    settings = get_settings()
    
    # Extract clean basename to prevent directory traversal or absolute path issues
    filename = Path(file.filename).name
    
    # 1. Validate file extension
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400, 
            detail=f"Only PDF files are supported. Got '{filename}'"
        )
        
    logger.info(f"Received upload request for file: {filename}")
    
    # Create upload path if not exists
    upload_dir = settings.upload_path
    temp_file_path = upload_dir / filename

    try:
        # 2. Write file to disk
        # We process chunked writes to prevent high memory usage
        with open(temp_file_path, "wb") as f:
            while content := await file.read(1024 * 1024):  # 1MB chunks
                f.write(content)
                
        logger.info(f"File successfully saved to disk: {temp_file_path}")

        # 3. Load PDF content
        loader = PDFLoaderService()
        documents = loader.load_pdf(temp_file_path)
        
        # 4. Chunk PDF content
        chunker = ChunkingService()
        chunks = chunker.chunk_documents(documents)
        
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail=f"Could not extract any text chunks from PDF: {filename}"
            )

        # 5. Store in vector database
        chroma_service = request.app.state.chroma_service
        # Add the documents to the vector store.
        # This will automatically call the embedding service and persist vectors.
        chroma_service.add_documents(chunks)
        
        logger.info(f"Successfully processed and indexed {filename}")
        
        return UploadResponse(
            status="success",
            filename=filename,
            chunks_count=len(chunks)
        )

    except Exception as e:
        logger.exception(f"Error handling upload for file {filename}")
        # Attempt to cleanup file on failure
        if temp_file_path.exists():
            try:
                os.remove(temp_file_path)
            except:
                pass
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process and index PDF: {str(e)}"
        )


@router.post("/ask", response_model=AnswerResponse, tags=["RAG"])
async def ask_question(request: Request, body: QuestionRequest):
    """
    Submit a natural language question. Retrieves relevant context from the
    vector store and uses Google Gemini to generate a grounded response.
    """
    logger.info(f"Received question: '{body.question}'")
    
    pipeline = request.app.state.rag_pipeline
    
    # Construct filter dictionary if document filter is specified
    filter_dict = None
    if body.filter_document:
        filter_dict = {"source": body.filter_document}
        logger.info(f"Filtering RAG context to document: '{body.filter_document}'")
        
    result = pipeline.ask(body.question, filter_dict=filter_dict)
    
    # Map raw dictionary sources to SourceCitation Pydantic objects
    citations = []
    for src in result.get("sources", []):
        citations.append(
            SourceCitation(
                source=src["source"],
                page=src["page"],
                relevance_score=src["relevance_score"],
                content=src["content"]
            )
        )
        
    return AnswerResponse(
        answer=result.get("answer", "No answer generated."),
        sources=citations
    )


@router.get("/documents", response_model=List[DocumentInfo], tags=["Documents"])
async def list_documents(request: Request):
    """
    Retrieve list of unique documents currently indexed in ChromaDB
    along with their total chunk counts.
    """
    chroma_service = request.app.state.chroma_service
    
    try:
        # Fetch metadata of all documents currently indexed in Chroma
        collection_data = chroma_service.vector_store.get()
        metadatas = collection_data.get("metadatas", [])
        
        if not metadatas:
            return []
            
        # Count chunk occurrences per source file
        counts = Counter()
        for meta in metadatas:
            if meta and "source" in meta:
                # Store only the basename in case absolute paths are saved
                filename = Path(meta["source"]).name
                counts[filename] += 1
                
        doc_list = [
            DocumentInfo(filename=name, chunk_count=count)
            for name, count in counts.items()
        ]
        
        return doc_list

    except Exception as e:
        logger.error(f"Failed to list documents: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch document information from database: {str(e)}"
        )


@router.delete("/documents/{filename}", tags=["Documents"])
async def delete_document(request: Request, filename: str):
    """
    Delete a document's chunks from ChromaDB and remove its source PDF file from disk.
    """
    settings = get_settings()
    chroma_service = request.app.state.chroma_service
    
    logger.warning(f"Request to delete document '{filename}' from system.")
    
    try:
        # 1. Fetch all elements in the collection to match IDs
        collection_data = chroma_service.vector_store.get()
        ids = collection_data.get("ids", [])
        metadatas = collection_data.get("metadatas", [])
        
        ids_to_delete = []
        for doc_id, meta in zip(ids, metadatas):
            if meta:
                meta_source = Path(meta.get("source", "")).name
                if meta_source == filename:
                    ids_to_delete.append(doc_id)
                    
        if not ids_to_delete:
            raise HTTPException(
                status_code=404,
                detail=f"Document '{filename}' not found in database."
            )
            
        # 2. Delete from ChromaDB
        chroma_service.vector_store.delete(ids_to_delete)
        logger.info(f"Deleted {len(ids_to_delete)} vector chunks for {filename}")
        
        # 3. Delete file from upload directory if it exists
        filepath = settings.upload_path / filename
        if filepath.exists():
            os.remove(filepath)
            logger.info(f"Deleted file from local storage: {filepath}")
            
        return {"status": "success", "message": f"Successfully deleted document '{filename}'."}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document '{filename}': {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to complete document deletion: {str(e)}"
        )


@router.get("/conversations", response_model=ConversationState, tags=["Conversations"])
async def get_conversations():
    """Retrieve saved conversation state from disk."""
    settings = get_settings()
    file_path = settings.upload_path / "conversations.json"
    
    if not file_path.exists():
        return ConversationState()
        
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return ConversationState(**data)
    except Exception as e:
        logger.error(f"Failed to load conversation history: {str(e)}")
        # Return empty state if file is corrupt
        return ConversationState()


@router.post("/conversations", tags=["Conversations"])
async def save_conversations(state: ConversationState):
    """Save conversation state to disk."""
    settings = get_settings()
    file_path = settings.upload_path / "conversations.json"
    
    # Ensure upload path exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(state.model_dump(), f, ensure_ascii=False, indent=2)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Failed to save conversation history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save conversations: {str(e)}"
        )
