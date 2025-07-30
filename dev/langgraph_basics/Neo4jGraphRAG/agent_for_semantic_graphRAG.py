# %%
from __future__ import annotations

import os
from typing import Annotated, Literal, Optional, Union

from dotenv import load_dotenv
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, Send
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.cohere import CohereEmbeddings
from neo4j_graphrag.generation import GraphRAG, RagTemplate
from neo4j_graphrag.indexes import create_fulltext_index, create_vector_index
from neo4j_graphrag.llm import AzureOpenAILLM
from neo4j_graphrag.retrievers import HybridCypherRetriever
from pydantic import BaseModel, Field

from dev.langgraph_basics.Neo4jGraphRAG.llm_chains_cypher import (
    chain_for_questions_generation,
)

# --------------------------------------------------------------------------- #
# 1) Entorno e índices
# --------------------------------------------------------------------------- #

load_dotenv(override=True)

NEO4J_USERNAME = os.getenv("NEO4J_USERNAME_UPGRADED")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD_UPGRADED")
NEO4J_URI = os.getenv("NEO4J_CONNECTION_URI_UPGRADED")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
# Verificamos conectividad sin cerrar el driver prematuramente.
driver.verify_connectivity()

# Embeddings Cohere (mismo modelo que en la construcción del KG)
embedder = CohereEmbeddings(model="embed-v4.0", api_key=os.getenv("COHERE_API_KEY"))

# Nombre de índices usados para nodos :Chunk creados por SimpleKGPipeline
vector_index_name = "chunkEmbedding"
fulltext_index_name = "chunkFulltext"

# Dimensionalidad inferida dinámicamente (solo una vez)
try:
    VECTOR_DIM = len(embedder.embed_query("test"))
except Exception:
    VECTOR_DIM = 1024

# Crear índices si faltan ----------------------------------------------------
try:
    create_vector_index(
        driver,
        name=vector_index_name,
        label="Chunk",
        embedding_property="embedding",
        dimensions=VECTOR_DIM,
        similarity_fn="cosine",
    )
except Exception:
    pass  # Puede existir

try:
    create_fulltext_index(
        driver,
        name=fulltext_index_name,
        label="Chunk",
        node_properties=["text"],
    )
except Exception:
    pass  # Puede existir

# Índices de propiedades para acelerar búsquedas por nombre ------------------
with driver.session() as _idx_sess:
    _idx_sess.run("CREATE INDEX enzyme_name IF NOT EXISTS FOR (e:Enzyme) ON (e.name)")
    _idx_sess.run(
        "CREATE INDEX metabolite_name IF NOT EXISTS FOR (m:Metabolite) ON (m.name)"
    )
# --------------------------------------------------------------------------- #
# 2) Cypher Retrieval Query
# --------------------------------------------------------------------------- #

RETRIEVAL_QUERY = """
// --- 1) Traverse from retrieved Chunk to enzyme and subsystem ----------------
MATCH (node)<-[:FROM_CHUNK]-(enzyme:Enzyme)                                      // Each Chunk must belong to an Enzyme
MATCH (enzyme)-[:PERTENECE_A]->(subsystem:Subsystem)                             // Enzyme must belong to a Subsystem
OPTIONAL MATCH (met_c:Metabolite)-[:SUBSTRATO_DE]->(enzyme)                      // Metabolites that are substrates of the enzyme
OPTIONAL MATCH (enzyme)-[:PRODUCE]->(met_p:Metabolite)                           // Metabolites that are products of the enzyme

// --- 2) Identify neighbouring enzymes sharing any metabolite -----------------
OPTIONAL MATCH (enzyme)-[:PRODUCE|SUBSTRATO_DE]-(shared_met:Metabolite)
            -[:PRODUCE|SUBSTRATO_DE]-(neighbor:Enzyme)                           // Any enzyme connected to the same metabolite (excluding pure cofactors) is considered a neighbour
WHERE neighbor <> enzyme                                                         // Exclude the enzyme itself
  AND NOT shared_met.name IN ['NAD+', 'NADH', 'ATP', 'ADP']                     // Ignore matches via cofactors only

// --- 3) Aggregate lists and project desired fields ---------------------------
WITH
  enzyme,                                                                        // Current enzyme node
  subsystem,                                                                     // Subsystem node linked to the enzyme
  collect(DISTINCT met_c.name) AS substrates,                                    // Unique substrate names
  collect(DISTINCT met_p.name) AS products,                                      // Unique product names
  collect(DISTINCT neighbor.name) AS neighbor_enzymes                            // Unique neighbour enzyme names
ORDER BY enzyme.name                                                             // Deterministic ordering of result rows
RETURN
  enzyme.name       AS enzyme_name,                                              // Enzyme identifier
  coalesce(products, [])   AS products,                                          // List of products (empty list if none)
  coalesce(substrates, []) AS substrates,                                        // List of substrates (empty list if none)
  subsystem.name    AS subsystem,                                                // Name of the subsystem
  enzyme.reversible AS reversible,                                               // Boolean: reaction reversibility
  enzyme.flux       AS flux,                                                     // Relative flux value
  coalesce(neighbor_enzymes, []) AS neighbor_enzymes;                            // Enzymes sharing metabolites (empty if none)
"""

# --------------------------------------------------------------------------- #
# 3) Configuración HybridCypherRetriever
# --------------------------------------------------------------------------- #

retriever = HybridCypherRetriever(
    driver=driver,
    vector_index_name=vector_index_name,
    fulltext_index_name=fulltext_index_name,
    retrieval_query=RETRIEVAL_QUERY,
    embedder=embedder,
)

# --------------------------------------------------------------------------- #
# 4) LLM y plantilla
# --------------------------------------------------------------------------- #

llm = AzureOpenAILLM(
    model_name="gpt-4.1",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

rag_template = RagTemplate(
    template="""You are a metabolic pathway expert. Answer the **Question** ONLY
 using the **Context** provided.
 
 NEVER add NOR inject information or data that is not in the context.
 
 # Question:
 {query_text}
 
 # Context:
 {context}
 
 # Answer:
 """,
    expected_inputs=["query_text", "context"],
)

# --------------------------------------------------------------------------- #
# 5) GraphRAG pipeline
# --------------------------------------------------------------------------- #

graph_rag = GraphRAG(retriever=retriever, llm=llm, prompt_template=rag_template)
# USER_QUESTION = "¿Cuántas enzimas tiene la glucólisis?"
# USER_QUESTION = "cuantas dehydrogenase enzimas hay, y dame sus nombres"
# USER_QUESTION = "Ketoglutarate en cual subsistema está?"
# USER_QUESTION = "como funciona la glucolisis?"
# USER_QUESTION = "cuales son las enzimas vecinas de PGI?"

# response = graph_rag.search(
#     USER_QUESTION,
#     retriever_config={"top_k": 5},
#     return_context=False,
# )


# print("\nPregunta:", USER_QUESTION)
# print("Respuesta:", response.answer)
class OneQuery(BaseModel):
    """One query."""

    query_str: str


class GeneratedQueries(BaseModel):
    """Generated queries."""

    queries_list: list[OneQuery]


class CypherQuery(BaseModel):
    """Cypher query agent."""

    cypher_query: str = Field(description="The Cypher query ready to be executed.")


def reduce_lists(
    existing: Optional[list[str]],
    new: Union[list[str], str, Literal["delete"], None],
) -> list[str]:
    """Combine two lists of strings in a robust way.

    Behaviour
    ---------
    • If *new* is the literal ``"delete"`` → returns an empty list (reset).
    • If either argument is ``None`` → treats it as an empty list.
    • Accepts *new* as a single string or list of strings.
    • Ensures the returned list has **unique items preserving order**.
    """

    # Reset signal
    if new == "delete":
        return []

    # Normalise inputs
    if existing is None:
        existing = []

    if new is None:
        new_items: list[str] = []
    elif isinstance(new, str):
        new_items = [new]
    else:
        new_items = list(new)

    combined: list[str] = existing + new_items

    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for item in combined:
        if item not in seen:
            seen.add(item)
            deduped.append(item)

    return deduped


class Neo4jQueryState(MessagesState):
    """State of the Neo4j Graph RAG."""

    question: str = Field(default_factory=lambda: "")
    generated_questions: GeneratedQueries = Field(
        default_factory=lambda: GeneratedQueries(queries_list=[])
    )
    query: str = Field(default_factory=lambda: "")
    # cypher_query: CypherQuery = Field(
    #     default_factory=lambda: CypherQuery(cypher_query="")
    # )
    cypher_query: str = Field(default_factory=lambda: "")
    cypher_queries: Annotated[list[str], reduce_lists] = Field(default_factory=list)
    results: Annotated[list[str], reduce_lists] = Field(default_factory=list)


async def generate_questions(
    state: Neo4jQueryState,
) -> Command[Literal["send_queries_in_parallel"]]:
    """Node that generates queries."""
    generated_questions = await chain_for_questions_generation.ainvoke(
        {"input": state["question"]}
    )
    return Command(
        goto="send_queries_in_parallel",
        update={"generated_questions": generated_questions},
    )


async def send_queries_in_parallel(
    state: Neo4jQueryState,
) -> Command[list[Send]]:
    """Node that generates Cypher queries in parallel."""
    lista_de_queries = [
        query.query_str for query in state["generated_questions"].queries_list
    ]

    # lista_de_queries = [
    #     "Todos los nombres de las enzimas",
    #     "Todos los nombres de los metabolitos",
    # ]
    print(f"lista_de_queries: {lista_de_queries}")
    sends = [
        Send(
            "generate_answer",
            {"query": query},
        )
        for query in lista_de_queries
    ]
    return Command(goto=sends)


async def generate_answer(
    state: Neo4jQueryState,
) -> Command[Literal[END]]:
    """Node that generates a Cypher query."""
    query_str = state["query"]

    response = graph_rag.search(
        query_str,
        retriever_config={"top_k": 5},
        return_context=False,
    )

    print("\nPregunta:", query_str)
    print("Respuesta:", response.answer)

    return Command(goto=END, update={"results": [response.answer]})


builder = StateGraph(Neo4jQueryState)

builder.add_node("generate_questions", generate_questions)
builder.add_node("send_queries_in_parallel", send_queries_in_parallel)
builder.add_node("generate_answer", generate_answer)
builder.add_edge(START, "generate_questions")

graph = builder.compile()

async for chunk in graph.astream(
    {"question": "cuales son las enzimas vecinas de PGI?"},
    stream_mode="updates",
    subgraphs=True,
    debug=True,
):
    pass
