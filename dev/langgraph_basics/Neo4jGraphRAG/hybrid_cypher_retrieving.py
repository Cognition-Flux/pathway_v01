# %%
"""hybrid_cypher_retrieving.py
+-----------------------------------------------------------
Ejemplo de RAG híbrido (vector + full-text) para consultar el
**Knowledge Graph de enzimas metabólicas** creado con
`knowledge_graph_builder.py`.

El grafo contiene nodos:
  • Enzyme  (prop: name, subsystem, substrates, products, reversible, flux)
  • Metabolite (prop: name)
  • Subsystem (prop: name)

Relaciones principales:
  • (Metabolite)-[:CONSUMIDO_POR]->(Enzyme)
  • (Metabolite)-[:GENERADO_POR]->(Enzyme)
  • (Enzyme)-[:EN]->(Subsystem)

El script prepara índices vectoriales y de texto completo sobre los nodos
`Chunk` creados automáticamente por *SimpleKGPipeline* y define un
`HybridCypherRetriever` que, tras recuperar los `Chunk`s relevantes,
traversa hasta las enzimas, metabolitos y subsistemas asociados para
construir un contexto rico.

Preguntas de ejemplo (se pueden personalizar):
  • "¿Cuántas enzimas hay en total?"
  • "¿Cuántas enzimas tiene la glucólisis?"
  • "¿Cuántas enzimas tiene el TCA?"
  • "Dame las enzimas que producen ATP"
  • "¿Cuáles enzimas producen NADH?"
  • "¿Cuántas enzimas son reversibles en el TCA?"
  • "¿Cuáles son los pasos irreversibles de la glucólisis?"
  • "Dame los nombres de las enzimas que están asociadas a piruvato"
  • "Dame un resumen de las funciones de las deshidrogenasas"

Para lanzar una consulta rápida, ejecuta este archivo y edita la variable
`USER_QUESTION` al final.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.cohere import CohereEmbeddings
from neo4j_graphrag.generation import GraphRAG, RagTemplate
from neo4j_graphrag.indexes import create_fulltext_index, create_vector_index
from neo4j_graphrag.llm import AzureOpenAILLM
from neo4j_graphrag.retrievers import HybridCypherRetriever

# --------------------------------------------------------------------------- #
# 1) Entorno e índices
# --------------------------------------------------------------------------- #

load_dotenv(override=True)

NEO4J_USERNAME = os.getenv("NEO4J_USERNAME_UPGRADED")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD_UPGRADED")
NEO4J_URI = os.getenv("NEO4J_CONNECTION_URI_UPGRADED")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
# Verificamos conectividad para evitar sorpresas.
with driver as _drv:
    _drv.verify_connectivity()

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

# --------------------------------------------------------------------------- #
# 2) Cypher Retrieval Query
# --------------------------------------------------------------------------- #

RETRIEVAL_QUERY = """
// Starting from retrieved Chunk node → traverse to Enzyme and related info
OPTIONAL MATCH (node)<-[:FROM_CHUNK]-(enzyme:Enzyme)
OPTIONAL MATCH (enzyme)-[:PERTENECE_A]->(subsystem:Subsystem)
OPTIONAL MATCH (met_c:Metabolite)-[:SUBSTRATO_DE]->(enzyme)
OPTIONAL MATCH (enzyme)-[:PRODUCE]->(met_p:Metabolite)

// Global stats per query execution
CALL () {
  MATCH (ss:Subsystem)<-[:PERTENECE_A]-(e2:Enzyme)
  WITH ss, collect(DISTINCT e2.name) AS enz_list
  RETURN collect({subsystem:ss.name, enzymes:enz_list, enzyme_count: size(enz_list)}) AS subsystems_stats
}

WITH
  node,
  enzyme,
  subsystem,
  collect(DISTINCT met_c.name)        AS substrates,
  collect(DISTINCT met_p.name)        AS products,
  subsystems_stats,
  size(subsystems_stats)              AS total_subsystems
 RETURN
   enzyme.name            AS enzyme_name,
   subsystem.name         AS subsystem,
   enzyme.reversible      AS reversible,
   enzyme.flux            AS flux,
   substrates             AS substrates,
   products               AS products,
   node.text              AS chunk_text,
   subsystems_stats       AS subsystems_stats,
   total_subsystems       AS total_subsystems;
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
 
 Counting rules:
 • When asked "¿Cuántas enzimas ...?" count ONLY unique `enzyme_name` entries.
 • If the question specifies a subsystem (e.g. TCA, ciclo TCA, (glyco)lisis, glucólisis), count only those rows where `subsystem` matches that subsystem.
 • Provide the count as an integer with no additional text.
 
 Listing rules:
 • If a list is requested, reply with a comma-separated list of unique names.
 
 NEVER add information that is not in the context.
 
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

# --------------------------------------------------------------------------- #
# 5) Ejecución de ejemplo
# --------------------------------------------------------------------------- #
# %%
if __name__ == "__main__":
    # USER_QUESTION = "¿Cuántas enzimas tiene la glucólisis?"
    USER_QUESTION = "cuantos  subsistemas hay"

    response = graph_rag.search(
        USER_QUESTION,
        retriever_config={"top_k": 1},
        return_context=False,
    )

    print("\nPregunta:", USER_QUESTION)
    print("Respuesta:", response.answer)
