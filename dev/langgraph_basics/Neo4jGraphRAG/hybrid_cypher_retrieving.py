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


# %%
if __name__ == "__main__":
    # --------------------------------------------------------------------------- #
    # 5) Ejecución de ejemplo
    # --------------------------------------------------------------------------- #
    # USER_QUESTION = "¿Cuántas enzimas tiene la glucólisis?"
    # USER_QUESTION = "cuantas dehydrogenase enzimas hay, y dame sus nombres"
    # USER_QUESTION = "Ketoglutarate en cual subsistema está?"
    # USER_QUESTION = "como funciona la glucolisis?"
    USER_QUESTION = "cuales son las enzimas vecinos de HK?"

    response = graph_rag.search(
        USER_QUESTION,
        retriever_config={"top_k": 10},
        return_context=False,
    )

    print("\nPregunta:", USER_QUESTION)
    print("Respuesta:", response.answer)
