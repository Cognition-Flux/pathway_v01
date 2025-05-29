#!/usr/bin/env python
"""Delete all vectors belonging to a PDF from the Pinecone index.

This utility removes *all* chunks that were generated from a specific PDF.
Each vector stored in Pinecone contains a ``metadata`` entry called
``source`` that holds the absolute (or project-relative) path of the
originating PDF.  We leverage that key to build a deletion filter and purge
the corresponding vectors.

Usage
-----
uv run python pathway/vectorstore/delete.py --pdf-path <ruta_pdf> \
    [--index pathway]

Environment variables (see ``.env.example``):
    PINECONE_API_KEY   - Your Pinecone API key (required)

Notes:
-----
* The filter-based deletion API is **not** supported on *serverless* Pinecone
  indexes.  In that case you must instead query for vector IDs and delete them
  explicitly.

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
    add_pdf_path_arg,
    add_pinecone_args,
    create_base_parser,
    get_pinecone_client,
    setup_environment,
)


def delete_pdf_vectors(pdf_path: str, index_name: str) -> None:
    """Delete every vector whose ``metadata.source`` matches *pdf_path*.

    Parameters
    ----------
    pdf_path : str
        Exact path (string equality) stored in ``metadata['source']``.
    index_name : str
        Target Pinecone index name.

    """
    pc = get_pinecone_client()

    if not pc.has_index(index_name):
        sys.exit(f"Index '{index_name}' does not exist.")

    index = pc.Index(index_name)

    # Build deletion filter - simple equality on source.
    deletion_filter = {"source": {"$eq": pdf_path}}

    print(
        (
            "Deleting vectors from index '"
            f"{index_name}' where source == '{pdf_path}' ..."
        ),
    )

    try:
        index.delete(filter=deletion_filter)
    except Exception as exc:  # pragma: no cover
        print("Error during deletion:", exc)
        sys.exit(1)

    print("Deletion request submitted. Check Pinecone dashboard for status.")


def main() -> None:
    """Execute the PDF vector deletion process."""
    setup_environment()

    # Parse command-line arguments
    parser = create_base_parser(
        "Delete all vectors originating from a given PDF path.",
    )
    add_pdf_path_arg(parser)
    add_pinecone_args(parser)
    args = parser.parse_args()

    delete_pdf_vectors(args.pdf_path, args.index)


if __name__ == "__main__":
    main()
