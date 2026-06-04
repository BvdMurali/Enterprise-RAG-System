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

from fastapi import APIRouter, File, UploadFile, HTTPException, Request, BackgroundTasks, Depends, Query, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from datetime import timedelta
from typing import Optional

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
from backend.services.auth import verify_token, USERS_DB, create_access_token

router = APIRouter(prefix="/api")


# --- Standalone Security and Token Schemas ---
class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_groups: List[str]


security_bearer = HTTPBearer(auto_error=False)


def get_current_user_groups(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_bearer)) -> List[str]:
    """
    FastAPI dependency to verify JWT token and extract role scope groups.
    Defaults to ["public"] if no bearer token is present (backward compatibility with baseline app).
    """
    if not credentials:
        return ["public"]
    
    payload = verify_token(credentials.credentials)
    if not payload:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired access token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload.user_groups


@router.get("/health", tags=["System"])
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}


@router.post("/auth/login", response_model=TokenResponse, tags=["Authentication"])
async def login(body: LoginRequest):
    """
    Authenticate standalone users against local USERS_DB and return a JWT access token.
    """
    user = USERS_DB.get(body.username)
    if not user or user["password_hash"] != body.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = create_access_token(body.username)
    return TokenResponse(
        access_token=token,
        user_groups=user["user_groups"]
    )


@router.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_pdf(
    request: Request, 
    file: UploadFile = File(...),
    access_group: str = Query("public", description="The role access group required to search this document")
):
    """
    Upload a PDF document, extract its text, chunk it, embed the chunks,
    and persist them in the ChromaDB vector database.
    Supports assigning access control roles for multi-tenant isolation.
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
        
    logger.info(f"Received upload request for file: {filename} with access_group: {access_group}")
    
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
        
        if not documents:
            raise HTTPException(
                status_code=400,
                detail=f"Could not extract any text pages from PDF: {filename}"
            )

        # Tag each document page with its access control group
        for doc in documents:
            doc.metadata["access_group"] = access_group

        # 4. Ingest via advanced retriever (handles parent-child splits)
        retriever_service = request.app.state.retriever_service
        child_chunks_count = retriever_service.add_documents(documents)
        
        logger.info(f"Successfully processed and indexed {filename} into {child_chunks_count} child chunks.")
        
        return UploadResponse(
            status="success",
            filename=filename,
            chunks_count=child_chunks_count
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
async def ask_question(
    request: Request, 
    body: QuestionRequest,
    user_groups: List[str] = Depends(get_current_user_groups)
):
    """
    Submit a natural language question. Retrieves relevant context from the
    vector store and uses Google Gemini to generate a grounded response.
    Supports conversational memory history and standalone JWT-based access control.
    """
    logger.info(f"Received question: '{body.question}' for user_groups: {user_groups}")
    
    pipeline = request.app.state.rag_pipeline
    cache_service = request.app.state.semantic_cache
    
    # 1. Check Semantic Cache
    cached_res = cache_service.get(body.question)
    if cached_res:
        cached_resp_text = cached_res["answer"]
        try:
            payload = json.loads(cached_resp_text)
            answer = payload.get("answer", "")
            summary = payload.get("summary", "")
            confidence = payload.get("confidence", 0.85)
            followups = payload.get("follow_up_questions", [])
            doc_type = payload.get("document_type", "Unknown")
            resp_type = payload.get("response_type", "general")
            key_insights = payload.get("key_insights", [])
            confidence_reasons = payload.get("confidence_reasons", [])
            chunks_retrieved = payload.get("chunks_retrieved", 0)
            chunks_used = payload.get("chunks_used", 0)
        except Exception:
            answer = cached_resp_text
            summary = answer[:100] + "..." if len(answer) > 100 else answer
            confidence = 0.85
            followups = []
            doc_type = "Unknown"
            resp_type = "general"
            key_insights = []
            confidence_reasons = []
            chunks_retrieved = 0
            chunks_used = 0

        citations = [
            SourceCitation(
                source=src["source"],
                page=src["page"],
                relevance_score=src["relevance_score"],
                content=src["content"]
            ) for src in cached_res.get("sources", [])
        ]
        return AnswerResponse(
            answer=answer,
            summary=summary,
            confidence=confidence,
            citations=citations,
            follow_up_questions=followups,
            document_type=doc_type,
            response_type=resp_type,
            key_insights=key_insights,
            confidence_reasons=confidence_reasons,
            chunks_retrieved=chunks_retrieved,
            chunks_used=chunks_used
        )
    
    # 2. Construct Metadata Access Control List (ACL) Filter
    acl_filter = {"access_group": {"$in": user_groups}}
    
    if body.filter_document:
        filter_dict = {
            "$and": [
                {"source": body.filter_document},
                acl_filter
            ]
        }
    else:
        filter_dict = acl_filter
        
    result = pipeline.ask(body.question, chat_history=body.chat_history, filter_dict=filter_dict)
    
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
        
    # Store serialized result dictionary in Semantic Cache
    cache_payload = {
        "answer": result.get("answer", ""),
        "summary": result.get("summary", ""),
        "confidence": result.get("confidence", 0.85),
        "follow_up_questions": result.get("followups", []),
        "document_type": result.get("document_type", "Unknown"),
        "response_type": result.get("response_type", "general"),
        "key_insights": result.get("key_insights", []),
        "confidence_reasons": result.get("confidence_reasons", []),
        "chunks_retrieved": result.get("chunks_retrieved", 0),
        "chunks_used": result.get("chunks_used", 0),
        "sources": result.get("sources", [])
    }
    cache_service.set(body.question, json.dumps(cache_payload), result.get("sources", []))
        
    return AnswerResponse(
        answer=result.get("answer", "No answer generated."),
        summary=result.get("summary", ""),
        confidence=result.get("confidence", 0.85),
        citations=citations,
        follow_up_questions=result.get("followups", []),
        document_type=result.get("document_type", "Unknown"),
        response_type=result.get("response_type", "general"),
        key_insights=result.get("key_insights", []),
        confidence_reasons=result.get("confidence_reasons", []),
        chunks_retrieved=result.get("chunks_retrieved", 0),
        chunks_used=result.get("chunks_used", 0)
    )


@router.post("/ask/stream", tags=["RAG"])
async def ask_question_stream(
    request: Request, 
    body: QuestionRequest,
    user_groups: List[str] = Depends(get_current_user_groups)
):
    """
    Submit a natural language question and stream the response chunk-by-chunk
    using Server-Sent Events (SSE). Supports semantic caching and JWT access control.
    """
    logger.info(f"Received streaming question: '{body.question}' for user_groups: {user_groups}")
    
    pipeline = request.app.state.rag_pipeline
    cache_service = request.app.state.semantic_cache
    
    # Construct Metadata Access Control List (ACL) Filter
    acl_filter = {"access_group": {"$in": user_groups}}
    
    if body.filter_document:
        filter_dict = {
            "$and": [
                {"source": body.filter_document},
                acl_filter
            ]
        }
    else:
        filter_dict = acl_filter
        
    from fastapi.responses import StreamingResponse
    import json
    import asyncio
    
    async def event_generator():
        # Check Semantic Cache first
        cached_res = cache_service.get(body.question)
        if cached_res:
            logger.info("Streaming response from Semantic Cache...")
            cached_resp_text = cached_res["answer"]
            try:
                payload = json.loads(cached_resp_text)
                answer = payload.get("answer", "")
                summary = payload.get("summary", "")
                confidence = payload.get("confidence", 0.85)
                followups = payload.get("follow_up_questions", [])
                doc_type = payload.get("document_type", "Unknown")
                resp_type = payload.get("response_type", "general")
                key_insights = payload.get("key_insights", [])
                confidence_reasons = payload.get("confidence_reasons", [])
                chunks_retrieved = payload.get("chunks_retrieved", 0)
                chunks_used = payload.get("chunks_used", 0)
            except Exception:
                answer = cached_resp_text
                summary = answer[:100] + "..." if len(answer) > 100 else answer
                confidence = 0.85
                followups = []
                doc_type = "Unknown"
                resp_type = "general"
                key_insights = []
                confidence_reasons = []
                chunks_retrieved = 0
                chunks_used = 0

            # Yield intent metadata first so UI can adapt immediately
            yield f"data: {json.dumps({'metadata': {'intent': resp_type}})}\n\n"
            await asyncio.sleep(0.005)

            # Yield sources
            yield f"data: {json.dumps({'sources': cached_res['sources']})}\n\n"
            await asyncio.sleep(0.01)

            # Yield token chunks to simulate streaming UI rendering
            chunk_size = 25
            for i in range(0, len(answer), chunk_size):
                token_chunk = answer[i:i+chunk_size]
                yield f"data: {json.dumps({'token': token_chunk})}\n\n"
                await asyncio.sleep(0.001)

            # Yield final parsed dict
            yield f"data: {json.dumps({'parsed': {'answer': answer, 'summary': summary, 'confidence': confidence, 'document_type': doc_type, 'response_type': resp_type, 'followups': followups, 'key_insights': key_insights, 'confidence_reasons': confidence_reasons, 'chunks_retrieved': chunks_retrieved, 'chunks_used': chunks_used, 'sources_text': ''}})}\n\n"
            return

        # Cache miss - Stream from RAG and collect for caching
        collected_tokens = []
        collected_sources = []
        collected_parsed = {}
        try:
            async for chunk in pipeline.ask_async_stream(
                question=body.question,
                chat_history=body.chat_history,
                filter_dict=filter_dict
            ):
                yield f"data: {json.dumps(chunk)}\n\n"
                
                # Aggregate response content
                if "token" in chunk:
                    collected_tokens.append(chunk["token"])
                elif "sources" in chunk:
                    collected_sources = chunk["sources"]
                elif "parsed" in chunk:
                    collected_parsed = chunk["parsed"]
                    
            full_answer = "".join(collected_tokens)
            # Cache the response if it concluded successfully
            if full_answer and not full_answer.startswith("\n[Error"):
                if collected_parsed:
                    cache_payload = {
                        "answer": collected_parsed.get("answer", full_answer),
                        "summary": collected_parsed.get("summary", ""),
                        "confidence": collected_parsed.get("confidence", 0.85),
                        "follow_up_questions": collected_parsed.get("followups", []),
                        "document_type": collected_parsed.get("document_type", "Unknown"),
                        "response_type": collected_parsed.get("response_type", "general"),
                        "key_insights": collected_parsed.get("key_insights", []),
                        "confidence_reasons": collected_parsed.get("confidence_reasons", []),
                        "chunks_retrieved": collected_parsed.get("chunks_retrieved", 0),
                        "chunks_used": collected_parsed.get("chunks_used", 0),
                        "sources": collected_sources
                    }
                else:
                    from backend.services.rag_pipeline import parse_structured_response
                    parsed = parse_structured_response(full_answer)
                    cache_payload = {
                        "answer": parsed["answer"],
                        "summary": parsed["summary"],
                        "confidence": parsed.get("grounding_score", 0.85),
                        "follow_up_questions": parsed["followups"],
                        "document_type": parsed["document_type"],
                        "response_type": parsed["response_type"],
                        "key_insights": parsed["key_insights"],
                        "confidence_reasons": parsed["confidence_reasons"],
                        "chunks_retrieved": len(collected_sources),
                        "chunks_used": 0,
                        "sources": collected_sources
                    }
                cache_service.set(body.question, json.dumps(cache_payload), collected_sources)
                
        except Exception as e:
            logger.error(f"Error in ask_question_stream: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/ask/agentic", response_model=AnswerResponse, tags=["RAG"])
async def ask_question_agentic(
    request: Request, 
    body: QuestionRequest,
    user_groups: List[str] = Depends(get_current_user_groups)
):
    """
    Submit a natural language question. Retrieves context iteratively using ReAct
    loop and function-calling. Supports JWT access control.
    """
    logger.info(f"Received agentic question: '{body.question}' for user_groups: {user_groups}")
    
    agent = request.app.state.agentic_rag
    result = agent.ask(body.question, user_groups=user_groups)
    
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
    Delete a document's parent documents and child chunks from storage,
    and remove its source PDF file from disk.
    """
    settings = get_settings()
    retriever_service = request.app.state.retriever_service
    
    logger.warning(f"Request to delete document '{filename}' from system.")
    
    try:
        # 1. Clean up vectors and parent mappings
        retriever_service.delete_document_by_source(filename)
        
        # 2. Delete file from upload directory if it exists
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
