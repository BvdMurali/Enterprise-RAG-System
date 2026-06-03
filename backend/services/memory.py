import logging
from typing import List, Dict, Any
from langchain.memory import ConversationSummaryBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.config import get_settings

logger = logging.getLogger(__name__)


class ConversationalMemoryService:
    """
    Enterprise Conversational Memory Service.
    Implements a ConversationSummaryBufferMemory to:
    - Maintain a rolling window of recent raw messages.
    - Compile a cumulative summary of older dialogue turns beyond the window limit.
    - Avoid token waste and context drift.
    """

    def __init__(self, llm: ChatGoogleGenerativeAI, max_token_limit: int = 1500):
        self.settings = get_settings()
        self.memory = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=max_token_limit,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",  # Maps response content to LLM output
            input_key="question"   # Maps query content to User input
        )
        logger.info(f"Initialized ConversationalMemoryService with max_token_limit={max_token_limit}")

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load history variables from memory."""
        return self.memory.load_memory_variables(inputs)

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Save dialogue context (user input and model output) to memory history."""
        self.memory.save_context(inputs, outputs)
        logger.debug("Successfully saved conversation context to memory.")

    def get_summary(self) -> str:
        """Retrieve the compiled running summary of the conversation."""
        try:
            return self.memory.moving_summary_buffer
        except Exception:
            return ""

    def clear(self) -> None:
        """Clear memory history."""
        self.memory.clear()
        logger.info("Conversational memory cleared.")
