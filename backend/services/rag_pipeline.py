import re
import logging
import math
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

# Keyword-based fast intent classifier
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
# Citation Superscript Formatter
# ---------------------------------------------------------------------------

def convert_citations_to_superscript(answer_text: str, retrieved_docs: List[Any]) -> Tuple[str, List[Dict[str, Any]], int]:
    """
    Find citation markers like [Doc X, Page Y] or [filename.pdf, Page Y] inside the answer.
    Replaces them with superscript indices (e.g. ¹, ²) and compiles a clean citations list.
    """
    # Regex to find all bracketed content
    bracket_pattern = r"\[([^\]]+)\]"
    
    unique_sources = []
    source_map = {}
    
    superscripts = ["⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹", "¹⁰", "¹¹", "¹²", "¹³", "¹⁴", "¹⁵"]
    
    # Check if a part of bracket content matches citation structure perfectly
    def is_part_citation(part: str) -> bool:
        part = part.strip()
        doc_m = re.match(r"(?i)^Doc\s+(\d+)(?:,\s*Page\s*(\S+))?$", part)
        pdf_m = re.match(r"(?i)^([^\s,]+?\.pdf)(?:,\s*Page\s*(\S+))?$", part)
        return bool(doc_m or pdf_m)

    # We will first scan the entire text to build the complete source_map of VALID citations.
    bracket_matches = re.finditer(bracket_pattern, answer_text)
    for match in bracket_matches:
        content = match.group(1)
        parts = content.split(";")
        
        # All non-empty parts must match citation structure
        non_empty_parts = [p.strip() for p in parts if p.strip()]
        if not non_empty_parts:
            continue
        if not all(is_part_citation(p) for p in non_empty_parts):
            continue
            
        for part in non_empty_parts:
            # Try matching standard doc pattern
            doc_m = re.match(r"(?i)^Doc\s+(\d+)(?:,\s*Page\s*(\S+))?$", part)
            if doc_m:
                doc_idx_str, page_str = doc_m.group(1), doc_m.group(2)
                try:
                    doc_idx = int(doc_idx_str) - 1
                    if 0 <= doc_idx < len(retrieved_docs):
                        doc = retrieved_docs[doc_idx]
                        source_path = doc.metadata.get("source", "Unknown")
                        filename = Path(source_path).name
                        page = page_str.strip() if page_str else str(doc.metadata.get("page", 1))
                        
                        key = (filename, page)
                        if key not in source_map:
                            unique_sources.append({
                                "source": filename,
                                "page": int(page) if page.isdigit() else 1,
                                "relevance_score": doc.metadata.get("relevance_score", 0.0),
                                "content": doc.page_content
                            })
                            source_map[key] = len(unique_sources)
                except Exception:
                    pass
                continue
                
            # Try matching PDF pattern
            pdf_m = re.match(r"(?i)^([^\s,]+?\.pdf)(?:,\s*Page\s*(\S+))?$", part)
            if pdf_m:
                filename_raw, page_str = pdf_m.group(1), pdf_m.group(2)
                filename = Path(filename_raw).name
                page = page_str.strip() if page_str else "1"
                
                key = (filename, page)
                if key not in source_map:
                    # Look up document
                    matching_doc = None
                    for doc in retrieved_docs:
                        if Path(doc.metadata.get("source", "")).name == filename:
                            matching_doc = doc
                            break
                    
                    relevance = matching_doc.metadata.get("relevance_score", 0.0) if matching_doc else 0.5
                    content = matching_doc.page_content if matching_doc else ""
                    
                    if not page_str and matching_doc:
                        page = str(matching_doc.metadata.get("page", 1))
                        key = (filename, page)
                        if key in source_map:
                            continue
                            
                    unique_sources.append({
                        "source": filename,
                        "page": int(page) if page.isdigit() else 1,
                        "relevance_score": relevance,
                        "content": content
                    })
                    source_map[key] = len(unique_sources)

    # Fallback: if no citations are found but documents were retrieved, cite the first retrieved document
    if not unique_sources and retrieved_docs:
        doc = retrieved_docs[0]
        source_path = doc.metadata.get("source", "Unknown")
        filename = Path(source_path).name
        page = doc.metadata.get("page", 1)
        unique_sources.append({
            "source": filename,
            "page": int(page) if str(page).isdigit() else 1,
            "relevance_score": doc.metadata.get("relevance_score", 0.0),
            "content": doc.page_content
        })
        source_map[(filename, str(page))] = 1

    # Now replace the bracketed citation text in the answer with superscripts or empty strings
    def replace_bracket(match):
        content = match.group(1)
        parts = content.split(";")
        non_empty_parts = [p.strip() for p in parts if p.strip()]
        
        # If not all parts match, it's not a citation block, keep it raw
        if not non_empty_parts or not all(is_part_citation(p) for p in non_empty_parts):
            return match.group(0)
            
        resolved_superscripts = []
        for part in non_empty_parts:
            # Match standard doc pattern
            doc_m = re.match(r"(?i)^Doc\s+(\d+)(?:,\s*Page\s*(\S+))?$", part)
            if doc_m:
                doc_idx_str, page_str = doc_m.group(1), doc_m.group(2)
                try:
                    doc_idx = int(doc_idx_str) - 1
                    if 0 <= doc_idx < len(retrieved_docs):
                        doc = retrieved_docs[doc_idx]
                        source_path = doc.metadata.get("source", "Unknown")
                        filename = Path(source_path).name
                        page = page_str.strip() if page_str else str(doc.metadata.get("page", 1))
                        
                        idx = source_map.get((filename, page))
                        if idx is not None:
                            superscript = superscripts[idx] if idx < len(superscripts) else f"[{idx}]"
                            resolved_superscripts.append(superscript)
                except Exception:
                    pass
                continue
                
            # Match PDF pattern
            pdf_m = re.match(r"(?i)^([^\s,]+?\.pdf)(?:,\s*Page\s*(\S+))?$", part)
            if pdf_m:
                filename_raw, page_str = pdf_m.group(1), pdf_m.group(2)
                filename = Path(filename_raw).name
                page = page_str.strip() if page_str else "1"
                
                matching_doc = None
                if not page_str:
                    for doc in retrieved_docs:
                        if Path(doc.metadata.get("source", "")).name == filename:
                            matching_doc = doc
                            break
                    if matching_doc:
                        page = str(matching_doc.metadata.get("page", 1))
                        
                idx = source_map.get((filename, page))
                if idx is not None:
                    superscript = superscripts[idx] if idx < len(superscripts) else f"[{idx}]"
                    resolved_superscripts.append(superscript)

        if resolved_superscripts:
            return "".join(resolved_superscripts)
        else:
            return ""

    # Replace citations
    clean_text = re.sub(bracket_pattern, replace_bracket, answer_text)
    
    # Clean up double spaces or spaces before punctuation caused by stripped citations.
    clean_text = re.sub(r" +", " ", clean_text)
    clean_text = re.sub(r"\s+([.,;:?!])", r"\1", clean_text)
    
    # Clean up spacing before superscripts
    superscript_chars = "".join(superscripts)
    clean_text = re.sub(rf"\s+([{superscript_chars}]+)", r"\1", clean_text)

    # Format and append footnotes at the end of the answer block
    if unique_sources:
        footnotes = "\n\n**Sources**\n"
        for i, src in enumerate(unique_sources, 1):
            footnotes += f"*{i}. {src['source']}, Page {src['page']}*\n"
        clean_text += footnotes

    return clean_text, unique_sources, len(unique_sources)


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def parse_structured_response(raw: str) -> Dict[str, Any]:
    """
    Parse the structured RAG templates from LLM tags.
    """
    def _extract(tag: str, text: str) -> str:
        pattern = rf"\[{tag}\]\s*(.*?)(?=\n\[|\Z)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    answer = _extract("ANSWER", raw)
    summary = _extract("SUMMARY", raw)
    sources_text = _extract("SOURCES", raw)
    grounding_score_raw = _extract("CONFIDENCE_GROUNDING_SCORE", raw)
    confidence_reasons_raw = _extract("CONFIDENCE_REASONS", raw)
    doc_type = _extract("DOCUMENT_TYPE", raw)
    resp_type = _extract("RESPONSE_TYPE", raw)
    followups_raw = _extract("FOLLOWUPS", raw)
    key_insights_raw = _extract("KEY_INSIGHTS", raw)

    # Fallback to legacy tag
    if not grounding_score_raw:
        grounding_score_raw = _extract("CONFIDENCE", raw)

    # If parsing fails entirely, treat the full response as the answer
    if not answer:
        answer = raw.strip()

    # Parse follow-ups: pipe-separated list -> Python list
    followups: List[str] = []
    if followups_raw:
        followups = [q.strip() for q in followups_raw.split("|") if q.strip()][:3]

    # Parse key insights: pipe-separated list -> Python list
    key_insights: List[str] = []
    if key_insights_raw:
        key_insights = [ins.strip() for ins in key_insights_raw.split("|") if ins.strip()][:3]

    # Parse confidence reasons: pipe-separated list -> Python list
    confidence_reasons: List[str] = []
    if confidence_reasons_raw:
        confidence_reasons = [reason.strip() for reason in confidence_reasons_raw.split("|") if reason.strip()][:3]

    # Normalize grounding score to float (0.0 to 1.0)
    grounding_score = 0.85
    if grounding_score_raw:
        pct_match = re.search(r"(\d+)%", grounding_score_raw)
        if pct_match:
            grounding_score = float(pct_match.group(1)) / 100.0
        else:
            try:
                clean_val = re.sub(r"[^\d\.]", "", grounding_score_raw)
                if clean_val:
                    val = float(clean_val)
                    if val > 1.0:
                        val = val / 100.0
                    grounding_score = max(0.0, min(val, 1.0))
            except ValueError:
                conf_str = grounding_score_raw.lower()
                if "high" in conf_str:
                    grounding_score = 0.95
                elif "low" in conf_str:
                    grounding_score = 0.35
                else:
                    grounding_score = 0.75

    return {
        "answer": answer,
        "summary": summary,
        "sources_text": sources_text,
        "grounding_score": grounding_score,
        "confidence_reasons": confidence_reasons,
        "document_type": doc_type or "Unknown",
        "response_type": resp_type or "general",
        "followups": followups,
        "key_insights": key_insights,
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

    def compute_confidence_score(
        self,
        retrieved_docs: List[Any],
        citation_count: int,
        grounding_score: float
    ) -> float:
        """
        Calculate confidence score based on the formula:
        confidence = (retrieval_score + reranker_score + citation_count_score + grounding_score) / 4
        """
        # 1. Retrieval Score: average of dense search vector similarities
        dense_scores = [doc.metadata.get("dense_score", 0.0) for doc in retrieved_docs if "dense_score" in doc.metadata]
        retrieval_score = sum(dense_scores) / len(dense_scores) if dense_scores else 0.80

        # 2. Reranker Score: average of sigmoid-scaled rerank logits
        rerank_scores = [doc.metadata.get("relevance_score", 0.0) for doc in retrieved_docs]
        def sigmoid(x):
            return 1.0 / (1.0 + math.exp(-x)) if x is not None else 0.5
        reranker_score = sum(sigmoid(s) for s in rerank_scores) / len(rerank_scores) if rerank_scores else 0.80

        # 3. Citation density score: min(citations / 3.0, 1.0)
        citation_score = min(citation_count / 3.0, 1.0)

        # Compute average
        final_score = (retrieval_score + reranker_score + citation_score + grounding_score) / 4.0
        return float(max(0.1, min(final_score, 1.0)))

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

            # 6. Format superscript inline citations and footnotes list
            clean_answer, citations_list, citation_count = convert_citations_to_superscript(
                parsed["answer"], 
                retrieved_docs
            )

            # 7. Compute formulaic confidence score
            confidence = self.compute_confidence_score(retrieved_docs, citation_count, parsed["grounding_score"])

            # 8. Clean up confidence reasons & key insights
            reasons = []
            parsed_reasons = parsed["confidence_reasons"]
            if parsed_reasons:
                for r in parsed_reasons:
                    cleaned = re.sub(r"^[•\-\*\s]+", "", r).strip()
                    if cleaned:
                        reasons.append(cleaned)
            if not reasons:
                if parsed["grounding_score"] >= 0.85:
                    reasons.append("No conflicting evidence detected in source texts")
                else:
                    reasons.append("Answer contains assertions not fully supported by context")
                if citation_count >= 2:
                    reasons.append("Verified across multiple matching document sections")
                elif citation_count == 1:
                    reasons.append("Grounded on a single matching document page")
                dense_scores = [doc.metadata.get("dense_score", 0.0) for doc in retrieved_docs if "dense_score" in doc.metadata]
                avg_dense = sum(dense_scores) / len(dense_scores) if dense_scores else 0.8
                if avg_dense >= 0.75:
                    reasons.append("Strong semantic alignment with database index")

            key_insights = []
            if parsed["key_insights"]:
                for ins in parsed["key_insights"]:
                    cleaned_ins = re.sub(r"^[•\-\*\s]+", "", ins).strip()
                    if cleaned_ins:
                        key_insights.append(cleaned_ins)
            else:
                key_insights.append(parsed["summary"])

            # 9. Build raw source metadata list
            sources = []
            for doc in retrieved_docs:
                sources.append({
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "Unknown"),
                    "relevance_score": doc.metadata.get("relevance_score", 0.0),
                    "content": doc.page_content,
                    "matched_child_text": doc.metadata.get("matched_child_text", "")
                })

            logger.info("RAG pipeline execution finished successfully.")
            return {
                "answer": clean_answer,
                "summary": parsed["summary"],
                "sources_text": parsed["sources_text"],
                "confidence": confidence,
                "confidence_reasons": reasons,
                "document_type": parsed["document_type"],
                "response_type": parsed["response_type"],
                "followups": parsed["followups"],
                "key_insights": key_insights,
                "intent": intent,
                "citations": citations_list,
                "sources": sources,
                "chunks_retrieved": len(retrieved_docs),
                "chunks_used": citation_count
            }

        except Exception as e:
            logger.error(f"Error during RAG pipeline execution: {str(e)}", exc_info=True)
            return {
                "answer": f"An error occurred while generating the answer: {str(e)}",
                "summary": "Error during execution.",
                "sources_text": "",
                "confidence": 0.0,
                "confidence_reasons": ["Execution failure encountered"],
                "document_type": "Unknown",
                "response_type": "general",
                "followups": [],
                "key_insights": ["Failed to extract insights"],
                "intent": "general",
                "citations": [],
                "sources": [],
                "chunks_retrieved": 0,
                "chunks_used": 0
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
                    "matched_child_text": doc.metadata.get("matched_child_text", "")
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
            
            # Format superscript citations and footnotes
            clean_answer, citations_list, citation_count = convert_citations_to_superscript(
                parsed["answer"], 
                retrieved_docs
            )

            # Compute formulaic confidence score
            confidence = self.compute_confidence_score(retrieved_docs, citation_count, parsed["grounding_score"])

            # Clean up confidence reasons & key insights
            reasons = []
            parsed_reasons = parsed["confidence_reasons"]
            if parsed_reasons:
                for r in parsed_reasons:
                    cleaned = re.sub(r"^[•\-\*\s]+", "", r).strip()
                    if cleaned:
                        reasons.append(cleaned)
            if not reasons:
                if parsed["grounding_score"] >= 0.85:
                    reasons.append("No conflicting evidence detected in source texts")
                else:
                    reasons.append("Answer contains assertions not fully supported by context")
                if citation_count >= 2:
                    reasons.append("Verified across multiple matching document sections")
                elif citation_count == 1:
                    reasons.append("Grounded on a single matching document page")
                dense_scores = [doc.metadata.get("dense_score", 0.0) for doc in retrieved_docs if "dense_score" in doc.metadata]
                avg_dense = sum(dense_scores) / len(dense_scores) if dense_scores else 0.8
                if avg_dense >= 0.75:
                    reasons.append("Strong semantic alignment with database index")

            key_insights = []
            if parsed["key_insights"]:
                for ins in parsed["key_insights"]:
                    cleaned_ins = re.sub(r"^[•\-\*\s]+", "", ins).strip()
                    if cleaned_ins:
                        key_insights.append(cleaned_ins)
            else:
                key_insights.append(parsed["summary"])

            yield {
                "parsed": {
                    "answer": clean_answer,
                    "summary": parsed["summary"],
                    "confidence": confidence,
                    "confidence_reasons": reasons,
                    "document_type": parsed["document_type"],
                    "response_type": parsed["response_type"],
                    "followups": parsed["followups"],
                    "key_insights": key_insights,
                    "sources_text": parsed["sources_text"],
                    "citations": citations_list,
                    "chunks_retrieved": len(retrieved_docs),
                    "chunks_used": citation_count
                }
            }

        except Exception as e:
            logger.error(f"Error during async stream generation: {str(e)}", exc_info=True)
            yield {"error": str(e)}
