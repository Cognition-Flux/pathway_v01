"""Upsert index module."""

# %%
import os
import pickle
import sys
from pathlib import Path

from dotenv import load_dotenv

# pylint: disable=import-error,wrong-import-position
load_dotenv(override=True)

MODULE_NAME = os.getenv("AGENTS_MODULE")
ALSO_MODULE = True
if ALSO_MODULE:
    current_file = Path(__file__).resolve()
    package_root = current_file.parent
    while package_root.name != MODULE_NAME:
        package_root = package_root.parent

    if str(package_root.parent) not in sys.path:
        sys.path.insert(0, str(package_root.parent))

from agentic_workflow.utils import (  # noqa: E402
    analyze_document_sizes,
    ensure_pinecone_index,
    get_azure_embeddings,
    upsert_documents,
)

os.chdir(package_root)


if __name__ == "__main__":
    DOCS_PATH = (
        "documents/collections/normalized_docs_from_AzureAIDocumentIntelligence.pkl"
    )
    INDEX_NAME = os.getenv("INDEX_NAME")
    BATCH_SIZE = 10

    if not os.path.exists(DOCS_PATH):
        raise FileNotFoundError(f"File not found: {DOCS_PATH}")

    with open(DOCS_PATH, "rb") as f:
        normalized_docs = pickle.load(f)

    print(f"Loaded {len(normalized_docs)} documents from {DOCS_PATH}")

    # Load pre-processed documents

    # Analyze document sizes
    analyze_document_sizes(normalized_docs)
    # %%
    # Ensure index exists
    ensure_pinecone_index(INDEX_NAME)
    # %%
    # Get embeddings model
    embeddings = get_azure_embeddings()

    # Upsert documents
    # %%
    upsert_documents(
        normalized_docs,
        index_name=INDEX_NAME,
        embeddings=embeddings,
        batch_size=BATCH_SIZE,
        show_progress=True,
    )

    print(
        f"Successfully indexed {len(normalized_docs)} documents in '{INDEX_NAME}' index.",
    )

# %%
