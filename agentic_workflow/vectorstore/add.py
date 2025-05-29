#!/usr/bin/env python
"""Add (upsert) all chunks of a **single PDF** to the Pinecone vector store.

The script replicates the preprocessing pipeline used during bulk indexing:

1. Extract text with ``AzureAIDocumentIntelligenceLoader``.
2. Split by Markdown headers → ``header1`` … ``header6`` metadata.
3. Further split by characters preserving overlap.
4. Normalize metadata keys, prepend headers + source info to content.
5. Truncate metadata values for Pinecone limits.
6. Embed with ``text-embedding-3-large`` (Azure OpenAI) and upsert.

Usage
-----
uv run python pathway/vectorstore/add.py --pdf-path <ruta_pdf> \
    [--index pathway] [--batch-size 5]

Prerequisites
-------------
• Environment variables:
  - ``PINECONE_API_KEY``
  - ``AZUREOPENAIEMBEDDINGS_API_KEY``
  - ``AZUREOPENAIEMBEDDINGS_AZURE_ENDPOINT``
  - ``AZURE_OPENAI_ENDPOINT``
  - ``AZURE_OPENAI_API_KEY``.
• The target Pinecone index must exist or will be created automatically.
"""

# Standard library imports
from __future__ import annotations

import sys
from pathlib import Path


# Ensure repository root is in path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Local imports
from pathway.utils import (  # noqa: E402
    add_batch_size_arg,
    add_pdf_path_arg,
    add_pinecone_args,
    create_base_parser,
    pdf_to_chunks,
    setup_environment,
    upsert_documents,
)


def main() -> None:
    """Run the PDF processing and Pinecone indexing process."""
    setup_environment()

    # Parse command-line arguments
    parser = create_base_parser(
        "Preprocess a PDF and add its chunks to Pinecone.",
    )
    add_pdf_path_arg(parser)
    add_pinecone_args(parser)
    add_batch_size_arg(parser)
    args = parser.parse_args()

    # Process the PDF
    pdf_path = str(Path(args.pdf_path).resolve())
    print(f"Processing PDF: {pdf_path}")

    chunks = pdf_to_chunks(pdf_path)
    print(f"Generated {len(chunks)} chunks. Beginning upsert…")

    # Upsert documents with progress reporting
    upsert_documents(
        chunks,
        args.index,
        batch_size=args.batch_size,
        show_progress=True,
    )

    print(f"PDF '{pdf_path}' successfully indexed in '{args.index}' index.")


if __name__ == "__main__":
    main()
