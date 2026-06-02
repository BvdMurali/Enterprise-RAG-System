"""
RAG Pipeline Service for Enterprise RAG System.

This module orchestrates the full RAG pipeline:
1. Receives the user question.
2. Retrieves relevant chunks via the RetrieverService.
3. Formats the chunks into a grounded context.
4. Feeds the context and question to the Google Gemini model.
5. Returns a structured response containing the answer and source documents.
"""

from typing import Dict, Any, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage

from backend.models.llm import LLMService
from backend.services.retriever import RetrieverService
from backend.prompts.rag_prompt import get_rag_prompt
from backend.logger import logger


class RAGPipelineService:
    """Orchestrator for the RAG pipeline components."""

    def __init__(self, retriever_service: RetrieverService, llm_service: LLMService):
        """
        Initialize the RAG pipeline.

        Args:
            retriever_service: The initialized RetrieverService.
            llm_service: The initialized LLMService.
        """
        self.retriever = retriever_service
        self.llm_service = llm_service
        self.prompt_template = get_rag_prompt()
        logger.info("RAGPipelineService initialized successfully.")

    def ask(self, question: str, filter_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the RAG flow.

        Args:
            question: The user query/question.
            filter_dict: Optional metadata filters (e.g., {"source": "architecture.pdf"}).

        Returns:
            A dictionary containing:
              - "answer": The grounding-reinforced text response from Gemini.
              - "sources": A list of dictionaries detailing the chunks retrieved,
                           including source filenames, pages, relevance scores, and content.
        """
        logger.info(f"RAG Pipeline invoked for query: '{question}'")

        try:
            # 1. Retrieve the relevant document chunks
            retrieved_docs = self.retriever.retrieve(question, filter_dict)
            
            # If no docs are found, return a fallback early
            if not retrieved_docs:
                return {
                    "answer": "No relevant documents found in the database. Please upload documents first.",
                    "sources": []
                }

            # 2. Format the context for the system prompt
            formatted_context = self.retriever.retrieve_formatted_context(question, filter_dict)

            # 3. Format prompt and invoke LLM
            # Since get_rag_prompt returns ChatPromptTemplate, we format it with context and question
            messages = self.prompt_template.format_messages(
                context=formatted_context,
                question=question
            )
            
            logger.info("Calling Google Gemini model for generation...")
            llm = self.llm_service.get_llm()
            response = llm.invoke(messages)
            answer = response.content

            # 4. Format sources metadata to return to the UI
            sources = []
            seen_sources = set()  # To track unique sources if desired, but returning each chunk is better
            for doc in retrieved_docs:
                sources.append({
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "Unknown"),
                    "relevance_score": doc.metadata.get("relevance_score", 0.0),
                    "content": doc.page_content
                })

            logger.info("RAG pipeline execution finished successfully.")
            return {
                "answer": answer,
                "sources": sources
            }

        except Exception as e:
            logger.error(f"Error during RAG pipeline execution: {str(e)}", exc_info=True)
            return {
                "answer": f"An error occurred while generating the answer: {str(e)}",
                "sources": []
            }
