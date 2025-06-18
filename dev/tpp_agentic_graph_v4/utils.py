"""Utility functions for the agentic workflow."""

# %%
import os
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import (
    AzureAIDocumentIntelligenceLoader,
)
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Initialise environment variables.
load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Request timeout configuration
# ---------------------------------------------------------------------------


def _parse_timeout(env_value: str | None, default: float = 1200.0) -> float:
    """Parse *env_value* into a float timeout.

    Falls back to *default* if the environment variable is not set or cannot
    be converted to ``float``.
    """

    if env_value is None:
        return default

    try:
        return float(env_value)
    except (TypeError, ValueError):  # pragma: no cover – defensive
        return default


# Global constant reused by all LLM helper constructors.
LLM_TIMEOUT_SEC: float = _parse_timeout(os.getenv("LLM_TIMEOUT"), default=1200.0)


def get_pinecone_client() -> Pinecone:
    """Return an initialised Pinecone client using ``PINECONE_API_KEY``."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise OSError("PINECONE_API_KEY is missing from environment.")

    return Pinecone(api_key=api_key)


def ensure_pinecone_index(
    index_name: str,
    *,
    dimension: int = 3072,
    metric: str = "cosine",
    client: Pinecone | None = None,
) -> None:
    """Create *index_name* in Pinecone if it does not exist."""
    pc = client or get_pinecone_client()

    try:
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"Error ensuring Pinecone index '{index_name}': {exc}",
        ) from exc


# Embeddings ----------------------------------------------------------------


def get_azure_embeddings(
    model: str = "text-embedding-3-large",
) -> AzureOpenAIEmbeddings:
    """Instantiate Azure OpenAI embeddings helper with env-configured keys."""
    return AzureOpenAIEmbeddings(
        model=model,
        azure_endpoint=os.getenv("AZUREOPENAIEMBEDDINGS_AZURE_ENDPOINT"),
        api_key=os.getenv("AZUREOPENAIEMBEDDINGS_API_KEY"),
    )


def analyze_document_sizes(docs):
    """Analyze and display size statistics for document content.

    Args:
        docs: List of Document objects to analyze

    """
    sizes = []
    for idx, d in enumerate(docs):
        sz = len(d.page_content.encode("utf-8"))
        sizes.append(sz)
        print(f"[Pre-Check] Doc {idx} size: {sz} bytes")

    if sizes:
        print(
            (
                "[Pre-Check] Stats - "
                f"count: {len(sizes)} | min: {min(sizes)} | "
                f"max: {max(sizes)} | avg: {sum(sizes) / len(sizes):.2f} bytes"
            ),
        )


def normalize_dict_keys(docs_list):
    """Normaliza las claves de los metadatos en una lista de documentos.

    Esta función encuentra todas las claves únicas en los metadatos de todos los documentos
    y asegura que cada documento tenga todas las claves, asignando 'null' a las faltantes.

    Args:
        docs_list: Lista de documentos con atributo metadata

    Returns:
        La misma lista de documentos con los metadatos normalizados

    """
    if not docs_list:
        print("La lista de documentos está vacía.")
        return docs_list

    # Extraer todos los metadatos primero
    metadata_list = [getattr(doc, "metadata", {}) for doc in docs_list]

    # Encontrar todas las claves únicas
    all_keys = set()
    for metadata in metadata_list:
        if not isinstance(metadata, dict):
            print(f"Advertencia: metadata no es diccionario: {type(metadata)}")
            continue
        all_keys.update(metadata.keys())
        print(f"Encontradas claves: {metadata.keys()}")

    print(f"Total de claves únicas encontradas: {len(all_keys)}")
    print(f"Claves únicas: {all_keys}")

    # Completar los metadatos que no tengan todas las claves
    changes_made = 0
    for i, metadata in enumerate(metadata_list):
        if not isinstance(metadata, dict):
            continue

        for key in all_keys:
            if key not in metadata:
                metadata[key] = "null"  # Usar 'null' en lugar de None
                changes_made += 1
                print(
                    f"Agregada clave '{key}' con valor 'null' al documento {i}",
                )

        # Actualizar el metadata del documento original
        try:
            docs_list[i].metadata = metadata
        except AttributeError as e:
            print(f"Error al actualizar metadata del documento {i}: {e}")

    print(f"Total de cambios realizados: {changes_made}")
    return docs_list


def add_metadata_to_content(docs_list):
    """Prepend header-level metadata (header1-header6) to each document's content.

    The function extracts keys that start with ``header`` from each document's
    metadata and **omits** those whose value is the string ``"null"``. The
    resulting *non-null* header values are concatenated with ``" | "`` as a
    separator and inserted at the beginning of ``page_content`` in square
    brackets. If a document contains no valid header metadata the content is
    left unchanged.

    Example:
    -------
    >>> doc.metadata
    {"header1": "Title", "header2": "Section", "header3": "null"}
    >>> add_metadata_to_content([doc])
    doc.page_content starts with "[Title | Section]".

    Args:
        docs_list: List of ``Document`` instances whose ``metadata`` has been
            normalised with :pyfunc:`normalize_dict_keys`.

    Returns:
        The *same* list instance with modified ``page_content`` fields.

    """
    if not docs_list:
        # Nothing to process.
        return docs_list

    updated_count = 0

    for doc in docs_list:
        # Basic sanity check – skip objects lacking required attributes.
        if not hasattr(doc, "metadata") or not hasattr(doc, "page_content"):
            continue

        # Ensure we have the source path inside metadata.
        source_path = doc.metadata.get("source")
        if not source_path:
            # Attempt to infer from existing keys or leave placeholder.
            source_path = doc.metadata.get("file_path", "source_desconocida")
            doc.metadata["source"] = source_path

        # Collect non-null header values in numerical order (header1, header2 ...).
        header_values: list[str] = []
        for i in range(1, 7):
            key = f"header{i}"
            value = doc.metadata.get(key)
            if value and value != "null":
                header_values.append(str(value))

        # Compose prefix parts: first the source, then any header values.
        prefix_parts = [f"Fuente: {source_path}"] + header_values

        prefix_str = " | ".join(prefix_parts)
        doc.page_content = f"[{prefix_str}]\n\n{doc.page_content}"
        updated_count += 1

    print(f"Cabeceras añadidas al contenido de {updated_count} documentos")
    return docs_list


def truncate_metadata_values(
    docs_list: list[Document],
    max_length: int = 10,
) -> list[Document]:
    """Truncate each metadata value to *max_length* characters.

    Args:
        docs_list: List of LangChain ``Document`` objects.
        max_length: Maximum length for each metadata value.

    Returns:
        The list of documents with truncated metadata values.

    """
    if not docs_list:
        return docs_list

    for doc in docs_list:
        metadata = getattr(doc, "metadata", {})
        if isinstance(metadata, dict):
            new_meta = {}
            for key, value in metadata.items():
                # Convert non-string values to str, then truncate
                str_value = str(value) if not isinstance(value, str) else value
                new_meta[key] = str_value[:max_length]
            doc.metadata = new_meta

    return docs_list


def upsert_documents(
    docs: list[Document],
    index_name: str,
    *,
    batch_size: int = 1,
    embeddings: AzureOpenAIEmbeddings | None = None,
    show_progress: bool = True,
) -> None:
    """Embed and upsert ``docs`` into Pinecone in batches.

    This wraps ``PineconeVectorStore.from_documents`` ensuring index existence
    and providing a single point to configure batch behaviour.

    Args:
        docs: List of Document objects to upload to Pinecone
        index_name: Name of the Pinecone index
        batch_size: Number of documents to process in each batch
        embeddings: AzureOpenAIEmbeddings instance or None to create one
        show_progress: Whether to display progress information

    """
    if not docs:
        print("No documents to upsert")
        return

    pc = get_pinecone_client()
    ensure_pinecone_index(index_name, client=pc)

    emb = embeddings or get_azure_embeddings()
    total_docs = len(docs)

    print(
        f"Beginning upsert of {total_docs} documents to index '{index_name}'",
    )

    # Try to use tqdm if available, otherwise fall back to simple progress output
    try:
        from tqdm import tqdm

        use_tqdm = show_progress
    except ImportError:
        use_tqdm = False

    # Create progress bar if using tqdm
    if use_tqdm:
        pbar = tqdm(total=total_docs, desc="Upserting documents", unit="doc")

    # Process in batches
    for start in range(0, total_docs, batch_size):
        end = min(start + batch_size, total_docs)
        batch = docs[start:end]
        batch_size_actual = len(batch)

        # Show progress information if not using tqdm
        if show_progress and not use_tqdm:
            percent_complete = (start / total_docs) * 100 if total_docs > 0 else 0
            print(
                f"Upserting batch {start + 1}-{end} of {total_docs} ({percent_complete:.1f}% complete)",
            )

        # Perform the actual upsert
        PineconeVectorStore.from_documents(batch, emb, index_name=index_name)

        # Update progress bar if using tqdm
        if use_tqdm:
            pbar.update(batch_size_actual)

    # Close progress bar if using tqdm
    if use_tqdm:
        pbar.close()

    print(
        f"Completed upsert of {total_docs} documents to index '{index_name}'",
    )


def pdf_to_chunks(pdf_path: str) -> list[Document]:
    """Extract text from *pdf_path* and return processed LangChain chunks."""
    file_path_obj = Path(pdf_path)
    if not file_path_obj.exists():
        raise FileNotFoundError(pdf_path)

    loader = AzureAIDocumentIntelligenceLoader(
        api_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        file_path=pdf_path,
        # api_model="prebuilt‑read",  # prebuilt-layout",
        api_model="prebuilt-read",  #
        analysis_features=None,  # ["ocrHighResolution"],
        mode="markdown",
    )

    raw_docs = loader.load()

    # Attach source to each document
    for doc in raw_docs:
        doc.metadata["source"] = str(file_path_obj)

    # First split by headers
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "header1"),
            ("##", "header2"),
            ("###", "header3"),
            ("####", "header4"),
            ("#####", "header5"),
            ("######", "header6"),
        ],
    )

    header_chunks: list[Document] = []
    for doc in raw_docs:
        pieces = header_splitter.split_text(doc.page_content)
        for piece in pieces:
            piece.metadata["source"] = str(file_path_obj)
        header_chunks.extend(pieces)

    # Second split by chars
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10_000,
        chunk_overlap=200,
    )
    chunks = char_splitter.split_documents(header_chunks)

    # Normalise + enrich
    chunks = normalize_dict_keys(chunks)
    chunks = add_metadata_to_content(chunks)
    chunks = truncate_metadata_values(chunks, max_length=200)

    return chunks


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
            # Increase request timeout to 1200 seconds (20 minutes) to prevent
            # premature termination on long-running calls.  The value can be
            # overridden at runtime via the environment variable ``LLM_TIMEOUT``.
            # If the variable is not set or is invalid (non-numeric), the
            # default of 1200 seconds is used.
            timeout=LLM_TIMEOUT_SEC,
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
            # Increase request timeout to 1200 seconds (20 minutes) to prevent
            # premature termination on long-running calls.  The value can be
            # overridden at runtime via the environment variable ``LLM_TIMEOUT``.
            # If the variable is not set or is invalid (non-numeric), the
            # default of 1200 seconds is used.
            timeout=LLM_TIMEOUT_SEC,
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
            # Increase request timeout to 1200 seconds (20 minutes) to prevent
            # premature termination on long-running calls.  The value can be
            # overridden at runtime via the environment variable ``LLM_TIMEOUT``.
            # If the variable is not set or is invalid (non-numeric), the
            # default of 1200 seconds is used.
            timeout=LLM_TIMEOUT_SEC,
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


# %%
