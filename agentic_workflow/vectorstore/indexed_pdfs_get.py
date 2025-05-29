#!/usr/bin/env python
"""Script to list all indexed PDFs in the Pinecone vector store.

This script extracts unique source paths from document metadata in the
Pinecone vector store and saves them to a YAML file.

Usage
-----
uv run python pathway/vectorstore/indexed_pdfs_get.py [--index pathway]

Prerequisites
-------------
• Environment variables must be set for Pinecone:
  - ``PINECONE_API_KEY``
"""

# %%
import sys
from collections import defaultdict
from pathlib import Path

import yaml


# Ensure repository root is in path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from pathway.utils import (  # noqa: E402
    add_pinecone_args,
    create_base_parser,
    get_pinecone_client,
    setup_environment,
)


def get_indexed_pdfs(index_name: str) -> dict:
    """Retrieve the list of indexed PDFs from a Pinecone index.

    Args:
        index_name: Name of the Pinecone index to query

    Returns:
        Dictionary with source paths as keys and counts as values

    """
    pc = get_pinecone_client()

    # Check if index exists
    if index_name not in pc.list_indexes().names():
        print(f"Index '{index_name}' does not exist.")
        return {}

    index = pc.Index(index_name)

    # Get all indexed items (limit 10000 should be enough for most use cases)
    results = index.query(
        vector=[0.0] * 3072,  # Dummy vector for metadata-only query
        top_k=10000,
        include_metadata=True,
    )

    # Extract unique source paths
    pdf_counts = defaultdict(int)

    for match in results.matches:
        if match.metadata and "source" in match.metadata:
            source = match.metadata["source"]
            pdf_counts[source] += 1

    return dict(pdf_counts)


def save_to_yaml(pdf_data: dict, output_path: str) -> None:
    """Save PDF paths to a YAML file.

    Args:
        pdf_data: Dictionary with PDF paths and counts
        output_path: Path where the YAML file will be saved

    """
    # Create directory if it doesn't exist
    output_file_path = Path(output_path)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Format the data for YAML
    yaml_data = {
        "indexed_pdfs": [
            {"name": Path(path).name, "chunks": count}
            for path, count in pdf_data.items()
        ],
        "total_pdfs": len(pdf_data),
        "total_chunks": sum(pdf_data.values()),
    }

    # Write to YAML file
    with output_file_path.open("w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    print(f"Indexed PDFs information saved to {output_path}")


def main() -> None:
    """Run the main script to extract and save indexed PDF information."""
    setup_environment()

    # Parse command-line arguments
    parser = create_base_parser(
        "List PDFs indexed in Pinecone and save to YAML",
    )
    add_pinecone_args(parser)
    args = parser.parse_args()

    # Get indexed PDFs
    pdf_data = get_indexed_pdfs(args.index)

    if not pdf_data:
        print(f"No indexed PDFs found in index '{args.index}'.")
        return

    # Print summary
    print(
        f"Found {len(pdf_data)} unique PDFs with {sum(pdf_data.values())} "
        f"total chunks.",
    )

    # Save to YAML
    output_path = "pathway/vectorstore/indexed_pdfs.yaml"
    save_to_yaml(pdf_data, output_path)


if __name__ == "__main__":
    main()
