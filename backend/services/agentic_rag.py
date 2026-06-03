import logging
import json
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, FunctionMessage

from backend.models.llm import LLMService
from backend.services.retriever import RetrieverService

logger = logging.getLogger(__name__)


class AgenticRAGService:
    """
    Agentic RAG Service (Phase 10) leveraging Gemini function-calling
    to execute multi-step planning, retrieval, and self-reflection.
    """

    def __init__(self, retriever_service: RetrieverService, llm_service: LLMService):
        self.retriever = retriever_service
        self.llm_service = llm_service

    def ask(self, question: str, user_groups: List[str] = None) -> Dict[str, Any]:
        """
        Runs a ReAct loop:
        1. Queries LLM with retriever search tool.
        2. If tool is called, runs search and feeds result back to LLM.
        3. Validates and returns final grounded answer.
        """
        logger.info(f"Agentic RAG triggered for query: '{question}'")
        llm = self.llm_service.get_llm()
        
        # Define access control filter based on user groups
        filter_dict = {"access_group": {"$in": user_groups}} if user_groups else None

        # System instructions outlining agent roles and self-reflection checks
        system_instruction = (
            "You are a Senior Agentic RAG Analyst. Your goal is to answer the user query accurately.\n"
            "You have access to a vector search tool `search_documents(query: str)` which returns corporate text snippets.\n"
            "Follow these steps:\n"
            "1. Analyze the user question. If it requires multiple facts, generate sub-queries and call `search_documents` for each.\n"
            "2. Read the retrieved document contexts carefully.\n"
            "3. If you need more information, call `search_documents` with a revised query.\n"
            "4. Before outputting the final answer, run a self-reflection step: check if every statement in your answer is strictly supported by the retrieved text. If not, delete the ungrounded statement.\n"
            "5. Cite the source and page in the format: [Doc X, Page Y].\n"
        )

        messages = [
            SystemMessage(content=system_instruction),
            HumanMessage(content=question)
        ]

        # Bind retrieval tool to Gemini
        tools = [
            {
                "name": "search_documents",
                "description": "Search the local corporate PDF database for policy, finance, or HR text snippets matching a query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search terms or question to execute in the vector store"
                        }
                    },
                    "required": ["query"]
                }
            }
        ]
        
        llm_with_tools = llm.bind(functions=tools)
        
        max_steps = 3
        sources = []
        seen_contents = set()
        
        for step in range(max_steps):
            logger.info(f"Agentic step {step + 1}/{max_steps}")
            try:
                response = llm_with_tools.invoke(messages)
            except Exception as e:
                logger.error(f"Error invoking Gemini in agentic loop: {e}")
                break
            
            # Check for function/tool call
            function_call = response.additional_kwargs.get("function_call")
            if function_call:
                tool_name = function_call.get("name")
                tool_args = self._json_parse(function_call.get("arguments", "{}"))
                query_param = tool_args.get("query", "")
                
                logger.info(f"Agent calling tool {tool_name} with args: {tool_args}")
                
                if tool_name == "search_documents":
                    # Execute hybrid retrieval
                    retrieved_docs = self.retriever.retrieve(query_param, filter_dict)
                    
                    # Accumulate sources for citations response mapping
                    for doc in retrieved_docs:
                        doc_content = doc.page_content
                        if doc_content not in seen_contents:
                            seen_contents.add(doc_content)
                            sources.append({
                                "source": doc.metadata.get("source", "Unknown"),
                                "page": doc.metadata.get("page", "Unknown"),
                                "relevance_score": doc.metadata.get("relevance_score", 0.0),
                                "content": doc_content
                            })
                    
                    context_str = "\n\n".join([d.page_content for d in retrieved_docs])
                    if not context_str:
                        context_str = "No documents found matching query."
                        
                    messages.append(response)
                    messages.append(FunctionMessage(name=tool_name, content=context_str))
                else:
                    messages.append(response)
                    messages.append(FunctionMessage(name=tool_name, content=f"Error: Tool {tool_name} not found."))
            else:
                logger.info("Agentic execution completed.")
                return {
                    "answer": response.content,
                    "sources": sources
                }
                
        logger.warning("Agentic RAG reached maximum step limit or failed.")
        return {
            "answer": response.content if 'response' in locals() else "Failed to generate answer in agentic mode.",
            "sources": sources
        }

    def _json_parse(self, args_str: str) -> dict:
        if isinstance(args_str, dict):
            return args_str
        try:
            return json.loads(args_str)
        except Exception:
            return {}
