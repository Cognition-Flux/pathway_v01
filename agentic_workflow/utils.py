"""Utility functions for the agentic workflow."""

import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import AzureChatOpenAI


load_dotenv(override=True)


def get_llm(
    provider: str = "azure",
    model: str = "gpt-4.1-mini",
) -> AzureChatOpenAI | ChatAnthropic | ChatGoogleGenerativeAI | ChatGroq:
    """Get a language model instance based on the specified provider.

    Args:
        provider: The LLM provider to use (defaults to 'azure')
        model: The specific model to use for the provider

    Returns:
        An instance of the appropriate LLM class depending on the provider

    Raises:
        ValueError: If an unsupported provider is specified

    """
    if provider == "azure":
        # Registrar el modelo en una variable de entorno para que otros
        # componentes (p.ej. streamer) puedan acceder a él rápidamente sin
        # pasar explícitamente la instancia del LLM.
        os.environ["LAST_LLM_MODEL"] = model
        return AzureChatOpenAI(
            azure_deployment=model,
            api_version=os.getenv("AZURE_API_VERSION"),
            temperature=0 if model != "o3-mini" else None,
            max_tokens=None,
            timeout=None,
            max_retries=5,
            streaming=True,
            api_key=os.getenv("AZURE_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
    elif provider == "anthropic":
        # Use provided model or default to claude-3-5-sonnet-latest
        anthropic_model = (
            model if model != "gpt-4.1-mini" else "claude-3-5-sonnet-latest"
        )
        os.environ["LAST_LLM_MODEL"] = anthropic_model
        return ChatAnthropic(
            model=anthropic_model,
            temperature=0,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            streaming=True,
            max_retries=5,
        )
    elif provider == "google":
        # Use provided model or default to gemini-2.5-flash-preview-05-20
        google_model = (
            model if model != "gpt-4.1-mini" else "gemini-2.5-flash-preview-05-20"
        )
        os.environ["LAST_LLM_MODEL"] = google_model
        return ChatGoogleGenerativeAI(
            model=google_model,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=5,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
    elif provider == "groq":
        # Use provided model or default to llama-3.3-70b-versatile
        groq_model = model if model != "gpt-4.1-mini" else "llama-3.3-70b-versatile"
        os.environ["LAST_LLM_MODEL"] = groq_model
        return ChatGroq(
            model=groq_model,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=5,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )
    else:
        raise ValueError(
            f"Unsupported provider: {provider}. Supported providers are: "
            "azure, anthropic, google, groq"
        )
