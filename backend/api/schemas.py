"""
Pydantic API Schemas for the FastAPI backend.
Defines requests, responses, and validation models.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    """Schema for incoming natural language questions."""
    question: str = Field(
        ..., 
        description="The natural language question to ask the RAG system."
    )
    filter_document: Optional[str] = Field(
        None, 
        description="Optional filename to restrict the search to a single document."
    )


class SourceCitation(BaseModel):
    """Schema representing a single source chunk citation."""
    source: str = Field(..., description="The name of the source PDF file.")
    page: int = Field(..., description="The page number inside the PDF.")
    relevance_score: float = Field(..., description="The cosine similarity relevance score.")
    content: str = Field(..., description="The actual text content of the retrieved chunk.")


class AnswerResponse(BaseModel):
    """Schema for the LLM response including grounded answer and source citations."""
    answer: str = Field(..., description="The generated answer from Gemini.")
    sources: List[SourceCitation] = Field(
        default=[], 
        description="List of source chunks used to generate the answer."
    )


class UploadResponse(BaseModel):
    """Schema for document upload responses."""
    status: str = Field(..., description="Status of the upload operation (e.g. success, error).")
    filename: str = Field(..., description="The name of the uploaded file.")
    chunks_count: int = Field(..., description="Number of text chunks successfully indexed.")


class DocumentInfo(BaseModel):
    """Schema representing metadata about a document in the vector store."""
    filename: str = Field(..., description="The name of the PDF file.")
    chunk_count: int = Field(..., description="Number of chunks associated with this file.")
