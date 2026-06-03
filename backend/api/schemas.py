"""
Pydantic API Schemas for the FastAPI backend.
Defines requests, responses, and validation models.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Schema for a single chat message in the conversation history."""
    role: str = Field(..., description="Role of the sender: 'user' or 'assistant'.")
    content: str = Field(..., description="The text content of the message.")
    sources: Optional[List[dict]] = Field(None, description="Optional list of source chunks if role is assistant.")
    intent: Optional[str] = Field(None, description="The classified query intent type.")
    confidence: Optional[float] = Field(None, description="Confidence score percentage as a float.")
    summary: Optional[str] = Field(None, description="A 1-sentence quick summary of the answer.")
    document_type: Optional[str] = Field(None, description="The classified category of the source documents.")
    followups: Optional[List[str]] = Field(None, description="Suggested follow-up questions.")


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
    chat_history: Optional[List[ChatMessage]] = Field(
        None,
        description="Optional past conversation turns for query rewriting and memory."
    )


class SourceCitation(BaseModel):
    """Schema representing a single source chunk citation."""
    source: str = Field(..., description="The name of the source PDF file.")
    page: int = Field(..., description="The page number inside the PDF.")
    relevance_score: float = Field(..., description="The cosine similarity relevance score.")
    content: str = Field(..., description="The actual text content of the retrieved chunk.")


class AnswerResponse(BaseModel):
    """Schema for the LLM response conforming to the enterprise RAG response schema."""
    answer: str = Field(..., description="The main formatted answer markdown text.")
    summary: str = Field(..., description="A 1-sentence quick summary of the answer.")
    confidence: float = Field(..., description="Confidence score percentage as a float between 0.0 and 1.0.")
    citations: List[SourceCitation] = Field(
        default=[], 
        description="List of source document citations used to ground the answer."
    )
    follow_up_questions: List[str] = Field(
        default=[],
        description="Suggested follow-up questions for continuing the conversation."
    )
    document_type: str = Field(..., description="The classified category of the source documents.")
    response_type: str = Field(..., description="The classified query intent type.")


class UploadResponse(BaseModel):
    """Schema for document upload responses."""
    status: str = Field(..., description="Status of the upload operation (e.g. success, error).")
    filename: str = Field(..., description="The name of the uploaded file.")
    chunks_count: int = Field(..., description="Number of text chunks successfully indexed.")


class DocumentInfo(BaseModel):
    """Schema representing metadata about a document in the vector store."""
    filename: str = Field(..., description="The name of the PDF file.")
    chunk_count: int = Field(..., description="Number of chunks associated with this file.")



class ConversationState(BaseModel):
    """Schema for the complete state of the user's conversations."""
    current_chat_title: str = Field("Active Conversation", description="Title of the currently active conversation.")
    chat_history: List[ChatMessage] = Field(default_factory=list, description="Messages in the currently active conversation.")
    past_conversations: dict = Field(default_factory=dict, description="Dictionary mapping past conversation titles to their message lists.")
