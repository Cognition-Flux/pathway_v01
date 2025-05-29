#!/usr/bin/env python
"""Module for creating and managing vector indexes in Pinecone.

This script loads pre-processed documents from a pickle file and
upserts them into a Pinecone vector index.

Usage
-----
uv run python pathway/vectorstore/indexing.py [--index pathway] [--batch-size 10]

Prerequisites
-------------
• Pre-processed documents must exist in the path defined by DEFAULT_DOCS_PATH
• Environment variables must be set for Pinecone and Azure OpenAI
"""

from __future__ import annotations

import sys
from pathlib import Path


# Ensure repository root is in path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from pathway.utils import (  # noqa: E402
    add_batch_size_arg,
    add_pinecone_args,
    create_base_parser,
    ensure_pinecone_index,
    get_azure_embeddings,
    load_documents,
    setup_environment,
    upsert_documents,
)


# Default path for pre-processed documents
DEFAULT_DOCS_PATH = (
    "pathway/documents_parsing/normalized_docs_from_AzureAIDocumentIntelligence.pkl"
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


def main():
    """Execute the document indexing process."""
    setup_environment()

    # Parse command-line arguments
    parser = create_base_parser(
        "Upload pre-processed documents to Pinecone index",
    )
    add_pinecone_args(parser)
    add_batch_size_arg(parser, default=10)
    parser.add_argument(
        "--docs-path",
        default=DEFAULT_DOCS_PATH,
        help=(
            f"Path to the pre-processed documents pickle file "
            f"(default: {DEFAULT_DOCS_PATH})"
        ),
    )
    args = parser.parse_args()

    # Load pre-processed documents
    normalized_docs = load_documents(args.docs_path)

    # Analyze document sizes
    analyze_document_sizes(normalized_docs)

    # Ensure index exists
    ensure_pinecone_index(args.index)

    # Get embeddings model
    embeddings = get_azure_embeddings()

    # Upsert documents
    upsert_documents(
        normalized_docs,
        index_name=args.index,
        embeddings=embeddings,
        batch_size=args.batch_size,
        show_progress=True,
    )

    print(
        f"Successfully indexed {len(normalized_docs)} documents "
        f"in '{args.index}' index.",
    )


if __name__ == "__main__":
    main()
