"""
LLM Model Service for Enterprise RAG System.

This module initializes and manages the Google Gemini API client via LangChain's
ChatGoogleGenerativeAI interface, configuring model defaults, temperature, and keys.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from backend.config import get_settings
from backend.logger import logger


class LLMService:
    """Service to manage LLM interactions using Google Gemini API."""

    def __init__(self):
        """Initialize the Google Gemini Chat model."""
        settings = get_settings()
        
        # Verify the API key is set and not a placeholder
        api_key = settings.google_api_key
        if not api_key or api_key == "your-gemini-api-key-here":
            logger.error("Google API Key is missing or using default placeholder.")
            raise ValueError(
                "Google API Key (GOOGLE_API_KEY) is not set in the environment or .env file. "
                "Please get an API key from Google AI Studio and configure it."
            )

        # Define the priority model rank order based on user criteria (balanced RPM, TPM, RPD)
        model_rank = [
            settings.llm_model_name,
            "gemini-3.1-flash-lite",
            "gemma-4-31b",
            "gemma-4-26b",
            "gemma-2-27b-it",
            "gemma-2-9b-it",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-1.5-flash"
        ]
        
        # Remove duplicates while preserving priority order
        seen = set()
        unique_model_names = []
        for m in model_rank:
            if m and m not in seen:
                seen.add(m)
                unique_model_names.append(m)

        logger.info(f"Establishing Resilient LLM Gateway with model priority: {unique_model_names}")

        # Instantiate all chat models in the rank order
        self.models = []
        for model_name in unique_model_names:
            try:
                chat_model = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=api_key,
                    temperature=settings.llm_temperature,
                    max_output_tokens=settings.llm_max_tokens,
                    transport="rest",
                )

                self.models.append(chat_model)
            except Exception as e:
                logger.warning(f"Could not prepare model client for '{model_name}': {e}")

        if not self.models:
            raise ValueError("Failed to initialize any of the Gemini/Gemma models.")

        # Chain them with LangChain's native with_fallbacks mechanism to handle runtime errors (429s, timeouts, etc.)
        self.llm = self.models[0]
        if len(self.models) > 1:
            self.llm = self.llm.with_fallbacks(self.models[1:])
            logger.info("Successfully established resilient LLM gateway with native LangChain fallbacks.")
        else:
            logger.info("Successfully initialized single LLM model.")



    def get_llm(self) -> ChatGoogleGenerativeAI:
        """
        Get the resilient LLM gateway (with fallbacks) for standard invocations.
        Use this for non-streaming calls: /api/ask and /api/ask/agentic.

        Returns:
            The LangChain fallback-chained ChatGoogleGenerativeAI object.
        """
        return self.llm

    def get_primary_llm(self) -> ChatGoogleGenerativeAI:
        """
        Get the unwrapped primary model (no fallback chain) for streaming.

        LangChain's with_fallbacks() wrapper does NOT support .astream() — it
        silently falls back to .ainvoke() and blocks until the full response is
        ready, causing frontend timeout errors in Hugging Face Spaces.
        Use this method in ask_async_stream() to get true token-by-token streaming
        from the primary model. Fallbacks on stream failure are handled manually.

        Returns:
            The raw ChatGoogleGenerativeAI object for the primary (highest-rank) model.
        """
        return self.models[0]
