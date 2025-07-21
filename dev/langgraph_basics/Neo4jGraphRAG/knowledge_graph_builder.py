# %%
"""
Knowledge Graph Builder example that constructs a Neo4j knowledge graph
from plain text using the experimental `SimpleKGPipeline` provided by
`neo4j_graphrag`.

The pipeline extracts entities and relationships from the `TEXT` constant
according to the predefined `NODE_TYPES`, `RELATIONSHIP_TYPES`, and
`PATTERNS` schema, and then writes the resulting graph to the Neo4j
instance configured via environment variables.

Run locally with:
    uv run python dev/langgraph_basics/Neo4jGraphRAG/knowledge_graph_builder.py
"""

import os

import dotenv
from langchain_core.documents import Document
from neo4j import Driver, GraphDatabase
from neo4j_graphrag.embeddings.cohere import CohereEmbeddings
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import AzureOpenAILLM

dotenv.load_dotenv(override=True)

NEO4J_USERNAME = os.getenv("NEO4J_USERNAME_UPGRADED")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD_UPGRADED")
URI = os.getenv("NEO4J_CONNECTION_URI_UPGRADED")
AUTH = (NEO4J_USERNAME, NEO4J_PASSWORD)
with GraphDatabase.driver(URI, auth=AUTH) as driver:
    driver.verify_connectivity()
neo4j_driver = GraphDatabase.driver(URI, auth=AUTH)


llm = AzureOpenAILLM(
    model_name="gpt-4.1-mini",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

embedder = CohereEmbeddings(
    model="embed-v4.0",
    api_key=os.getenv("COHERE_API_KEY"),
)

# --------------------------------------------------------------------------- #
# Text splitter to ensure chunks fit within the LLM context window
# --------------------------------------------------------------------------- #


# Split the text into 500-token chunks with 100-token overlap
text_splitter = FixedSizeSplitter(chunk_size=500, chunk_overlap=100)

# --------------------------------------------------------------------------- #
# Example input texts to be transformed into a Knowledge Graph
# --------------------------------------------------------------------------- #
TEXT_EINSTEIN = (
    "It is known that Maxwell’s electrodynamics—as usually understood at the present time—"  # noqa: E501
    "when applied to moving bodies, leads to asymmetries that do not seem to agree with observed phenomena. "
    "We shall raise and discuss this apparent conflict in the following."
)

TEXT_NEWTON = (
    "Every body perseveres in its state of rest or of uniform motion in a straight line, "
    "except insofar as it is compelled to change its state by force impressed. "
    "This law is foundational for the motion of bodies."
)

TEXT_BOHR = (
    "The spectrum of hydrogen is found to consist of a series of lines whose wavelengths "
    "can be represented very accurately by Balmer’s formula. "
    "The present paper seeks to show that this spectrum may be explained on the basis of Planck’s quantum theory."
)

DOCS: list[Document] = [
    Document(
        page_content=TEXT_EINSTEIN, metadata={"author": "Albert Einstein", "year": 1905}
    ),
    Document(
        page_content=TEXT_NEWTON, metadata={"author": "Isaac Newton", "year": 1687}
    ),
    Document(page_content=TEXT_BOHR, metadata={"author": "Niels Bohr", "year": 1913}),
]

ALLOWED_PHYSICISTS = [
    "Albert Einstein",
    "Isaac Newton",
    "Niels Bohr",
]

# --------------------------------------------------------------------------- #
# Guided schema for the extraction process
# --------------------------------------------------------------------------- #
NODE_TYPES = [
    "Person",
    {"label": "Concept", "description": "Scientific concept or theory."},
    {"label": "University", "description": "Academic institution."},
]

RELATIONSHIP_TYPES = [
    "PROPOSED_BY",
    "WORKS_AT",
]

PATTERNS = [
    ("Concept", "PROPOSED_BY", "Person"),
    ("Person", "WORKS_AT", "University"),
]


def clear_graph(driver: Driver) -> None:
    """Remove all nodes and relationships from the connected Neo4j database."""
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("🧹 Graph cleared — starting from an empty database.")


# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #


def prune_non_physicists(driver: Driver, allowed_names: list[str]) -> None:
    """Remove Person nodes whose name property is not in *allowed_names*."""
    cypher = """
        MATCH (p:Person)
        WHERE NOT p.name IN $names
        DETACH DELETE p
    """
    with driver.session() as session:
        session.run(cypher, names=allowed_names)
    print("🗑️  Removed non-physicist Person nodes from the graph.")


async def build_kg_from_docs(docs: list[Document]) -> None:
    """Clear the graph, build KG for each document, and keep only physicists."""

    # Start from a clean slate
    clear_graph(neo4j_driver)

    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=neo4j_driver,
        embedder=embedder,
        text_splitter=text_splitter,
        schema={
            "node_types": NODE_TYPES,
            "relationship_types": RELATIONSHIP_TYPES,
            "patterns": PATTERNS,
            "additional_node_types": False,
        },
        from_pdf=False,
    )

    for doc in docs:
        await kg_builder.run_async(text=doc.page_content)
        print(f"✅ Processed {doc.metadata['author']} paragraph → KG updated.")

    # Prune any Person nodes that are not the main physicists
    prune_non_physicists(neo4j_driver, ALLOWED_PHYSICISTS)
    print("🎉 Knowledge Graph creation completed and pruned!")


if __name__ == "__main__":
    _ = await build_kg_from_docs(DOCS)
