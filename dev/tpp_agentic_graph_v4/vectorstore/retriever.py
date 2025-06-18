#!/usr/bin/env python
"""Script to create a retriever from a Pinecone vector store.

This script creates a retriever from a Pinecone vector store using the
PineconeVectorStore class.

Usage
-----
uv run python pathway/vectorstore/retriever.py [--index pathway]

Prerequisites
-------------
• Environment variables must be set for Pinecone:
  - ``PINECONE_API_KEY``
"""

# %%
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore

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

from agentic_workflow.utils import get_azure_embeddings  # noqa: E402

vectorstore = PineconeVectorStore.from_existing_index(
    index_name=os.getenv("INDEX_NAME"),
    embedding=get_azure_embeddings(),
)

search_kwargs = {"k": 3}
retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

if __name__ == "__main__":
    docs = retriever.invoke("comuna de los estudio")
    print(len(docs))

# %%
