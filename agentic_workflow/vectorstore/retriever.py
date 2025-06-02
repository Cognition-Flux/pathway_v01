#!/usr/bin/env python
"""Script to create a retriever from a Pinecone vector store.

This script creates a retriever from a Pinecone vector store using the
PineconeVectorStore class.

Usage
-----
uv run python agentic_workflow/vectorstore/retriever.py [--index pathway]

Prerequisites
-------------
• Environment variables must be set for Pinecone:
  - ``PINECONE_API_KEY``
"""

# %%
import os

from langchain_pinecone import PineconeVectorStore

from agentic_workflow.utils import get_azure_embeddings


vectorstore = PineconeVectorStore.from_existing_index(
    index_name=os.getenv("INDEX_NAME"),
    embedding=get_azure_embeddings(),
)

search_kwargs = {"k": 10}
retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

if __name__ == "__main__":
    docs = retriever.get_relevant_documents("que es el envejecimiento")
    print(len(docs))

# %%
