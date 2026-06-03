import re
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from langchain_core.prompts import ChatPromptTemplate

from backend.models.llm import LLMService
from backend.services.retriever import RetrieverService
from backend.prompts.rag_prompt import get_rag_prompt
from backend.api.schemas import ChatMessage
from backend.logger import logger


# ---------------------------------------------------------------------------
# Query rewrite prompt
# ---------------------------------------------------------------------------

QUERY_REWRITE_PROMPT = """Given the conversation history and a follow-up user query, analyze if pronouns, acronyms, or implicit context from previous turns are required to understand and search for the query.
If yes, rewrite the query to be a standalone, search-friendly query.
If no context is required (i.e. the query already contains the primary keywords and proper names), output the original query exactly.

Rules:
1. Do NOT over-specify. Do NOT expand simple names (like "Murali") to long, full names (like "Boddeti Veere Durga Murali") unless it is absolutely necessary to resolve ambiguity.
2. Keep the search query keyword-focused and natural.
3. Do NOT carry over typos or speculative information from the user's questions into the rewritten query.
4. Return ONLY the final search query. Do NOT add explanation, greetings, prefix, or punctuation.

Conversation History:
{chat_history_str}

Follow-up Query: {question}
Standalone Query:"""


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

# Keyword-based fast intent classifier (no extra LLM call needed)
_INTENT_PATTERNS: List[Tuple[str, List[str]]] = [
    ("definition",  ["what is a", "what is an", "define", "definition", "meaning of", "what does", "what is meant by"]),
    ("summarize",   ["summarize", "summary", "overview", "describe", "description", "brief", "profile"]),
    ("compare",     ["compare", "difference", "vs", "versus", "better", "contrast", "distinguish", "similarities"]),
    ("list",        ["list", "enumerate", "what are all", "give me all", "show all", "what are the key", "which are"]),
    ("count",       ["how many", "count of", "number of", "total number"]),
    ("extract",     ["extract", "revenue", "financial", "email", "phone", "contact", "address", "gpa", "date", "year", "amount", "salary", "credits"]),
    ("explain",     ["explain", "why", "how do", "how does", "reason for", "cause of", "purpose of", "grounding rules"]),
]


def classify_intent(question: str) -> str:
    """
    Classify the query intent using keyword heuristics.
    Returns one of: definition, summarize, compare, list, count, extract, explain, general.
    """
    q_lower = question.lower().strip()
    for intent, keywords in _INTENT_PATTERNS:
        if any(kw in q_lower for kw in keywords):
            return intent
    return "general"


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def parse_structured_response(raw: str) -> Dict[str, Any]:
    """
    Parse the structured [ANSWER] / [SUMMARY] / [SOURCES] / [CONFIDENCE] / [DOCUMENT_TYPE] / [RESPONSE_TYPE] / [FOLLOWUPS]
    sections from the LLM response text.

    Returns a dict conforming to the response schema.
    Falls back gracefully if the model didn't perfectly follow the format.
    """
    def _extract(tag: str, text: str) -> str:
        pattern = rf"\[{tag}\]\s*(.*?)(?=\n\[|\Z)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    answer = _extract("ANSWER", raw)
    summary = _extract("SUMMARY", raw)
    sources_text = _extract("SOURCES", raw)
    confidence_raw = _extract("CONFIDENCE", raw)
    doc_type = _extract("DOCUMENT_TYPE", raw)
    resp_type = _extract("RESPONSE_TYPE", raw)
    followups_raw = _extract("FOLLOWUPS", raw)

    # If parsing fails entirely, treat the full response as the answer
    if not answer:
        answer = raw.strip()

    # Parse follow-ups: pipe-separated list -> Python list
    followups: List[str] = []
    if followups_raw:
        followups = [q.strip() for q in followups_raw.split("|") if q.strip()][:3]

    # Normalize confidence to float percentage (e.g. "98%" -> 0.98, "High" -> 0.95)
    confidence = 0.85  # default fallback
    if confidence_raw:
        pct_match = re.search(r"(\d+)%", confidence_raw)
        if pct_match:
            confidence = float(pct_match.group(1)) / 100.0
        else:
            conf_str = confidence_raw.lower()
            if "high" in conf_str:
                confidence = 0.95
            elif "low" in conf_str:
                confidence = 0.35
            else:
                confidence = 0.75

    return {
        "answer": answer,
        "summary": summary,
        "sources_text": sources_text,
        "confidence": confidence,
        "document_type": doc_type,
        "response_type": resp_type,
        "followups": followups,
    }


# ---------------------------------------------------------------------------
# RAG Pipeline Service
# ---------------------------------------------------------------------------

class RAGPipelineService:
    """Orchestrator for the advanced conversational RAG pipeline."""

    def __init__(self, retriever_service: RetrieverService, llm_service: LLMService):
        self.retriever = retriever_service
        self.llm_service = llm_service
        logger.info("RAGPipelineService initialized successfully.")

    def _get_unique_files(self) -> List[str]:
        """Fetch unique document filenames from database metadata."""
        try:
            collection_data = self.retriever.chroma_store.vector_store.get()
            metadatas = collection_data.get("metadatas", [])
            return sorted(list(set(Path(meta["source"]).name for meta in metadatas if meta and "source" in meta)))
        except Exception as e:
            logger.error(f"Failed to query database document list: {e}")
            return []

    # ------------------------------------------------------------------
    # Query rewrite helpers
    # ------------------------------------------------------------------

    def rewrite_query(self, question: str, chat_history: List[ChatMessage]) -> str:
        """Rewrite the question to be standalone if history is present."""
        if not chat_history:
            return question

        history_lines = []
        for msg in chat_history[-6:]:
            role = "User" if msg.role == "user" else "Assistant"
            history_lines.append(f"{role}: {msg.content}")
        history_str = "\n".join(history_lines)

        rewrite_template = ChatPromptTemplate.from_messages([
            ("system", "You are a query rewriting assistant. Generate standalone search queries based on history."),
            ("human", QUERY_REWRITE_PROMPT)
        ])
        messages = rewrite_template.format_messages(
            chat_history_str=history_str,
            question=question
        )
        try:
            llm = self.llm_service.get_llm()
            rewrite_response = llm.invoke(messages)
            standalone_query = rewrite_response.content.strip()
            if standalone_query:
                logger.info(f"Query rewritten: '{question}' -> '{standalone_query}'")
                return standalone_query
        except Exception as ex:
            logger.error(f"Failed to rewrite query: {ex}. Using original query.")
        return question

    async def rewrite_query_async(self, question: str, chat_history: List[ChatMessage]) -> str:
        """Asynchronously rewrite the query based on conversation history."""
        if not chat_history:
            return question

        history_lines = []
        for msg in chat_history[-6:]:
            role = "User" if msg.role == "user" else "Assistant"
            history_lines.append(f"{role}: {msg.content}")
        history_str = "\n".join(history_lines)

        rewrite_template = ChatPromptTemplate.from_messages([
            ("system", "You are a query rewriting assistant. Generate standalone search queries based on history."),
            ("human", QUERY_REWRITE_PROMPT)
        ])
        messages = rewrite_template.format_messages(
            chat_history_str=history_str,
            question=question
        )
        try:
            llm = self.llm_service.get_llm()
            rewrite_response = await llm.ainvoke(messages)
            standalone_query = rewrite_response.content.strip()
            if standalone_query:
                logger.info(f"Query rewritten (async): '{question}' -> '{standalone_query}'")
                return standalone_query
        except Exception as ex:
            logger.error(f"Failed to rewrite query (async): {ex}. Using original query.")
        return question

    # ------------------------------------------------------------------
    # Core ask() — synchronous
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        chat_history: Optional[List[ChatMessage]] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute the full RAG flow and return a structured response dict."""
        logger.info(f"RAG Pipeline invoked for query: '{question}'")

        try:
            # 1. Classify intent
            intent = classify_intent(question)
            logger.info(f"Detected intent: {intent}")

            # 2. Query rewrite
            search_query = self.rewrite_query(question, chat_history or [])

            # 3. Retrieve
            retrieved_docs = self.retriever.retrieve(search_query, filter_dict)
            if not retrieved_docs:
                formatted_context = "No relevant document context was found in the uploaded database."
            else:
                formatted_context = self.retriever.retrieve_formatted_context(search_query, filter_dict)

            unique_files = self._get_unique_files()
            files_list_str = ", ".join(unique_files) if unique_files else "None"
            formatted_context += f"\n\n[System Info] Currently indexed PDF documents in database: {files_list_str}"

            # 4. Build intent-aware prompt and call LLM
            prompt_template = get_rag_prompt(intent=intent, include_followups=True)
            messages = prompt_template.format_messages(
                context=formatted_context,
                question=question
            )

            logger.info("Calling Google Gemini model for generation...")
            llm = self.llm_service.get_llm()
            response = llm.invoke(messages)
            raw_answer = response.content

            # 5. Parse structured sections
            parsed = parse_structured_response(raw_answer)

            # 6. Build source metadata list
            sources = []
            for doc in retrieved_docs:
                sources.append({
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "Unknown"),
                    "relevance_score": doc.metadata.get("relevance_score", 0.0),
                    "content": doc.page_content,
                })

            logger.info("RAG pipeline execution finished successfully.")
            return {
                "answer": parsed["answer"],
                "summary": parsed["summary"],
                "sources_text": parsed["sources_text"],
                "confidence": parsed["confidence"],
                "document_type": parsed["document_type"],
                "response_type": parsed["response_type"],
                "followups": parsed["followups"],
                "intent": intent,
                "sources": sources,
            }

        except Exception as e:
            logger.error(f"Error during RAG pipeline execution: {str(e)}", exc_info=True)
            return {
                "answer": f"An error occurred while generating the answer: {str(e)}",
                "summary": "Error during execution.",
                "sources_text": "",
                "confidence": 0.0,
                "document_type": "Unknown",
                "response_type": "general",
                "followups": [],
                "intent": "general",
                "sources": [],
            }

    # ------------------------------------------------------------------
    # ask_async_stream() — streams tokens + metadata
    # ------------------------------------------------------------------

    async def ask_async_stream(
        self,
        question: str,
        chat_history: Optional[List[ChatMessage]] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ):
        """
        Asynchronously streams the RAG response.

        Yields dicts:
          {"metadata": {...}}   — sent first, contains intent/confidence/followups (estimated)
          {"sources": [...]}    — sent before first token
          {"token": "..."}      — streamed tokens
          {"error": "..."}      — on failure
        """
        logger.info(f"RAG Pipeline streaming invoked for query: '{question}'")

        try:
            # 1. Classify intent
            intent = classify_intent(question)
            logger.info(f"Detected intent (stream): {intent}")

            # 2. Query rewrite
            search_query = await self.rewrite_query_async(question, chat_history or [])

            # 3. Retrieve
            retrieved_docs = self.retriever.retrieve(search_query, filter_dict)
            if not retrieved_docs:
                formatted_context = "No relevant document context was found in the uploaded database."
            else:
                formatted_context = self.retriever.retrieve_formatted_context(search_query, filter_dict)

            unique_files = self._get_unique_files()
            files_list_str = ", ".join(unique_files) if unique_files else "None"
            formatted_context += f"\n\n[System Info] Currently indexed PDF documents in database: {files_list_str}"

            # 4. Build prompt
            prompt_template = get_rag_prompt(intent=intent, include_followups=True)
            messages = prompt_template.format_messages(
                context=formatted_context,
                question=question
            )

            # 5. Emit intent metadata first so UI can adapt immediately
            yield {"metadata": {"intent": intent}}

            # 6. Emit sources before streaming starts
            sources = []
            for doc in retrieved_docs:
                sources.append({
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "Unknown"),
                    "relevance_score": doc.metadata.get("relevance_score", 0.0),
                    "content": doc.page_content,
                })
            yield {"sources": sources}

            # 7. Stream tokens
            logger.info("Initiating async stream from Google Gemini...")
            full_text = ""
            llm = self.llm_service.get_llm()
            async for chunk in llm.astream(messages):
                full_text += chunk.content
                yield {"token": chunk.content}

            # 8. Post-stream: parse structured sections and emit them
            parsed = parse_structured_response(full_text)
            yield {
                "parsed": {
                    "summary": parsed["summary"],
                    "confidence": parsed["confidence"],
                    "document_type": parsed["document_type"],
                    "response_type": parsed["response_type"],
                    "followups": parsed["followups"],
                    "sources_text": parsed["sources_text"],
                }
            }

        except Exception as e:
            logger.error(f"Error during async stream generation: {str(e)}", exc_info=True)
            yield {"error": str(e)}
