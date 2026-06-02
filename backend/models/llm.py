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

        logger.info(f"Initializing ChatGoogleGenerativeAI with model: {settings.llm_model_name}")
        
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=settings.llm_model_name,
                google_api_key=api_key,
                temperature=settings.llm_temperature,
                max_output_tokens=settings.llm_max_tokens,
                # Safe defaults for enterprise apps: disable dangerous content safety blocks if necessary,
                # or rely on defaults. Let's stick to standard parameters.
            )
            logger.info("Successfully initialized Gemini LLM model.")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini LLM model: {str(e)}")
            raise

    def get_llm(self) -> ChatGoogleGenerativeAI:
        """
        Get the initialized LangChain Chat Model interface.
        
        Returns:
            The LangChain ChatGoogleGenerativeAI object ready to run chains.
        """
        return self.llm
