"""Utility functions for the agentic workflow."""

import os
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings


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


def merge_reasoning(
    existing: str | list[str] | None, new: str | list[str]
) -> str | list[str]:
    """Merge reasoning strings from multiple nodes.

    This function concatenates new reasoning information with existing reasoning,
    adding a delimiter between them to maintain readability. It can handle both
    string and list of strings.

    Args:
        existing: The existing reasoning string or list of strings, if any
        new: The new reasoning to add, which can be a string or list of strings

    Returns:
        A concatenated string with all reasoning information, or a list of messages

    """
    # Convert lists to strings to avoid showing raw Python list representation
    if isinstance(new, list):
        new = "\n".join(new)

    if isinstance(existing, list):
        existing = "\n".join(existing)

    # Both are now strings or None, handle as before
    if not existing:
        return new
    if not new:
        return existing

    # Add a delimiter to separate different reasoning chunks
    return f"{existing}\n\n---\n\n{new}"


def reduce_docs(
    existing: list[Document] | None,
    new: list[Document] | list[dict[str, Any]] | list[str] | str | Literal["delete"],
) -> list[Document]:
    """Reduce and process documents based on the input type."""
    if new == "delete":
        return []

    existing_list = list(existing) if existing else []

    # Function to convert metadata to a hashable representation for comparison
    def metadata_signature(doc):
        metadata = getattr(doc, "metadata", {})
        if not isinstance(metadata, dict):
            return None
        # Sort items to ensure consistent comparison
        return tuple(sorted(metadata.items()))

    # Track metadata signatures for deduplication
    existing_signatures = {metadata_signature(doc) for doc in existing_list}

    if isinstance(new, str):
        # For strings, create a Document with empty metadata
        new_doc = Document(page_content=new, metadata={})
        return existing_list + [new_doc]

    new_list = []
    if isinstance(new, list):
        for item in new:
            if isinstance(item, str):
                # Create a new Document with empty metadata
                new_doc = Document(page_content=item, metadata={})
                new_list.append(new_doc)

            elif isinstance(item, dict):
                # Create Document from dict
                metadata = item.get("metadata", {})
                doc = Document(**item)
                sig = metadata_signature(doc)

                if sig not in existing_signatures:
                    new_list.append(doc)
                    existing_signatures.add(sig)

            elif isinstance(item, Document):
                # Check if document with identical metadata already exists
                sig = metadata_signature(item)

                if sig not in existing_signatures:
                    new_list.append(item)
                    existing_signatures.add(sig)

    return existing_list + new_list


def get_azure_embeddings(
    model: str = "text-embedding-3-large",
) -> AzureOpenAIEmbeddings:
    """Instantiate Azure OpenAI embeddings helper with env-configured keys."""
    return AzureOpenAIEmbeddings(
        model=model,
        azure_endpoint=os.getenv("AZUREOPENAIEMBEDDINGS_AZURE_ENDPOINT"),
        api_key=os.getenv("AZUREOPENAIEMBEDDINGS_API_KEY"),
    )
