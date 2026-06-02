import logging
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from backend.config import get_settings
from backend.vectorstore.chroma_store import ChromaStoreService

logger = logging.getLogger(__name__)


class RetrieverService:
    """
    Service for retrieving relevant document chunks from the vector store
    based on semantic similarity to a query.
    """

    def __init__(self, chroma_store: ChromaStoreService):
        """
        Initialize the retriever service with a connected ChromaDB store.
        """
        self.settings = get_settings()
        self.chroma_store = chroma_store
        
        # Configure the LangChain retriever wrapper
        self.retriever = self.chroma_store.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": self.settings.retrieval_top_k,
                "score_threshold": self.settings.retrieval_score_threshold,
            }
        )
        logger.info(
            f"Initialized RetrieverService (top_k={self.settings.retrieval_top_k}, "
            f"threshold={self.settings.retrieval_score_threshold})"
        )

    def retrieve(self, query: str, filter_dict: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Retrieve document chunks relevant to the query.

        Args:
            query: The natural language question or search term.
            filter_dict: Optional metadata filters (e.g., {"source": "doc1.pdf"}).

        Returns:
            List of LangChain Document objects matching the criteria.
        """
        logger.debug(f"Retrieving documents for query: '{query}'")
        
        try:
            # LangChain's retriever wrapper doesn't cleanly expose score threshold filtering 
            # *and* return scores with the standard invoke() method in older versions.
            # We use the underlying vector store's similarity_search_with_relevance_scores 
            # to explicitly get the scores for logging and potential future use.
            results = self.chroma_store.vector_store.similarity_search_with_relevance_scores(
                query=query,
                k=self.settings.retrieval_top_k,
                filter=filter_dict,
                score_threshold=self.settings.retrieval_score_threshold
            )
            
            # results is a list of tuples: (Document, score)
            documents = []
            for doc, score in results:
                # Store the score in the document's metadata for debugging/UI purposes
                doc.metadata["relevance_score"] = round(score, 4)
                documents.append(doc)
                
            logger.info(f"Retrieved {len(documents)} relevant chunks for query.")
            return documents
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}", exc_info=True)
            raise

    def retrieve_formatted_context(self, query: str, filter_dict: Optional[Dict[str, Any]] = None) -> str:
        """
        Retrieve chunks and format them into a single string for prompt injection.
        Includes source and page number citations.
        """
        docs = self.retrieve(query, filter_dict)
        
        if not docs:
            return "No relevant context found in the uploaded documents."
            
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            
            # Format each chunk clearly with its source
            chunk_text = f"[Source: {source}, Page: {page}]\n{doc.page_content}"
            context_parts.append(chunk_text)
            
        return "\n\n---\n\n".join(context_parts)
