# %%
"""hybrid_cypher_retrieving.py
+-----------------------------------------------------------
Ejemplo **auto-contenible** que demuestra cómo combinar búsqueda
semántica y búsqueda por texto completo (Hybrid Search) con
consultas Cypher en un grafo Neo4j para implementar un flujo RAG
(*Retrieval-Augmented Generation*).

Basado en la documentación oficial de Neo4j GraphRAG:

* Construcción del KG –
  https://neo4j.com/docs/neo4j-graphrag-python/current/user_guide_kg_builder.html
* Configuración del retriever –
  https://neo4j.com/docs/neo4j-graphrag-python/current/user_guide_rag.html#retriever-configuration
* Opciones de búsqueda GraphRAG –
  https://neo4j.com/docs/neo4j-graphrag-python/current/user_guide_rag.html#graphrag-search-options

El guion se divide en 4 secciones principales:

1. Conexión a Neo4j y carga de variables de entorno.
2. Creación (si faltan) de índices vectoriales y de texto completo.
3. Definición del *retrieval_query* Cypher para traer contexto rico.
4. Ejecución de GraphRAG con un LLM de Azure OpenAI.

Cada línea está comentada en español para facilitar el aprendizaje.
"""

# ---------------------------
# 1) Importaciones y entorno
# ---------------------------

import os  # Acceso a variables de entorno

# Carga el archivo .env para traer credenciales y endpoints
from dotenv import load_dotenv
from neo4j import GraphDatabase  # Driver oficial de Neo4j

# Componentes GraphRAG ↴
from neo4j_graphrag.embeddings.cohere import CohereEmbeddings
from neo4j_graphrag.generation import GraphRAG, RagTemplate
from neo4j_graphrag.indexes import create_fulltext_index, create_vector_index
from neo4j_graphrag.llm import AzureOpenAILLM
from neo4j_graphrag.retrievers import HybridCypherRetriever

# Cargar variables de entorno (sobrescribe si ya existen en el proceso)
load_dotenv(override=True)

# # Demo database credentials
# URI = "neo4j+s://demo.neo4jlabs.com"
# AUTH = ("recommendations", "recommendations")
# # Connect to Neo4j database
# driver = GraphDatabase.driver(URI, auth=AUTH)

# ---------------------------
# 2) Conexión a Neo4j
# ---------------------------

# Credenciales obtenidas de variables de entorno.
# Mantenerlas fuera del código respeta las buenas prácticas de seguridad.
NEO4J_USERNAME: str | None = os.getenv("NEO4J_USERNAME_UPGRADED")  # Usuario Neo4j
NEO4J_PASSWORD: str | None = os.getenv("NEO4J_PASSWORD_UPGRADED")  # Contraseña
URI: str | None = os.getenv(
    "NEO4J_CONNECTION_URI_UPGRADED"
)  # bolt://... o neo4j+s://...

# Tuple con credenciales → requerido por el driver
AUTH = (NEO4J_USERNAME, NEO4J_PASSWORD)

# Verificamos la conectividad antes de seguir — lanza excepción si el servidor no responde.
with GraphDatabase.driver(URI, auth=AUTH) as driver:
    driver.verify_connectivity()

# Se mantiene un segundo driver para uso posterior (fuera del *with*).
driver = GraphDatabase.driver(URI, auth=AUTH)

# ---------------------------
# 3) Embeddings: Cohere
# ---------------------------

embedder = CohereEmbeddings(
    model="embed-v4.0",  # Modelo recomendado por Neo4j para Cohere
    api_key=os.getenv("COHERE_API_KEY"),  # Clave leída de variables de entorno
)

# --------------------------------------------------------------------------- #
# Ensure required indexes exist
# --------------------------------------------------------------------------- #
# Nombres de índices usados en el ejemplo (pueden cambiarse libremente)
vector_index_name = "chunkEmbedding"
fulltext_index_name = "chunkFulltext"

# Se infiere dimensionalidad de los embeddings de manera dinámica (solo una vez)
try:
    VECTOR_DIM = len(embedder.embed_query("test"))
except Exception:
    VECTOR_DIM = 1024  # fallback for Cohere v4.0

# Create indexes if they don’t already exist
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
    # Si ya existe, se ignora la excepción para mantener idempotencia
    pass

try:
    create_fulltext_index(
        driver,
        name=fulltext_index_name,
        label="Chunk",
        node_properties=["text"],
    )
except Exception:
    pass  # Índice fulltext ya existe

# --------------------------------------------------------------------------- #
# 4) Consulta Cypher con enriquecimiento de contexto
# --------------------------------------------------------------------------- #
#
# La siguiente cadena es una consulta Cypher multilinea que Neo4j ejecutará
# *después* de recuperar los nodos `Chunk` más relevantes (por vector o texto
# completo). Sirve para navegar el grafo y recopilar información complementaria
# (personas, universidades, etc.).

RETRIEVAL_QUERY = """
// From the retrieved Chunk node, traverse to concepts and related scientists.
OPTIONAL MATCH (node)<-[:FROM_CHUNK]-(concept:Concept)
OPTIONAL MATCH (concept)-[:PROPOSED_BY]->(person:Person)
// Also capture Person nodes that are directly linked to the Chunk (e.g. when the author is mentioned)
OPTIONAL MATCH (node)<-[:FROM_CHUNK]-(direct_person:Person)
OPTIONAL MATCH (person)-[:WORKS_AT]->(university:University)
WITH
  node,
  collect(DISTINCT person.name) + collect(DISTINCT direct_person.name) AS people_names,
  collect(DISTINCT university.name) AS universities,
  coalesce(concept.name, node.text, concept.id) AS main_entity,
  labels(concept) AS entity_labels
RETURN
  main_entity                                  AS entity_name,
  entity_labels                                AS entity_labels,
  people_names                                 AS scientists,
  size(people_names)                           AS person_count,
  universities                                 AS universities,
  node.text                                    AS chunk_text;
"""

############################
# 5) Configuración Retriever
############################

# El `HybridCypherRetriever` combina
#  • búsqueda vectorial (aprox. semántica)    → índice `vector_index_name`
#  • búsqueda full-text (sparce BM25)         → índice `fulltext_index_name`
# y posteriormente ejecuta la consulta Cypher anterior para regresar un
# `RetrievalResult` con nodos/propiedades listos para el LLM.

retriever = HybridCypherRetriever(
    driver=driver,
    vector_index_name=vector_index_name,
    fulltext_index_name=fulltext_index_name,
    retrieval_query=RETRIEVAL_QUERY,
    embedder=embedder,
)

# Pregunta de ejemplo. Descomente la línea que desee probar.
# query_text = "how many people are in the db, and how are they?"
QUERY_TEXT = "dame las fechas en que se publicaron los papers"

# Ejecutamos la búsqueda con un `top_k` pequeño para acelerar la demo
retriever_result = retriever.search(query_text=QUERY_TEXT, top_k=3)
# Imprimimos el resultado crudo para depuración
print(retriever_result.items)
# %%

############################################
# 6) Modelo LLM (Azure OpenAI)
############################################

llm = AzureOpenAILLM(
    model_name="gpt-4.1-mini",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

############################################
# 7) Plantilla (prompt) RAG
############################################

rag_template = RagTemplate(
    template="""Answer the Question using ONLY the information in the Context. NEVER INJECT ANY SPECULATIVE INFORMATION NOT IN THE CONTEXT.
If the question asks "how many" people are present, use the `person_count` value provided in the Context (if available) or count the unique names yourself and list their names.
If the question asks for publication dates (e.g. "fechas" / "dates"), extract every year or full date mentioned in the Context and present them in ascending order, avoiding duplicates.

# Question:
{query_text}

# Context:
{context}

# Answer:
""",
    expected_inputs=["query_text", "context"],
)

############################################
# 8) Ejecución de GraphRAG
############################################

graph_rag = GraphRAG(
    llm=llm,  # LLM configurado arriba
    retriever=retriever,  # Recuperador híbrido + Cypher
    prompt_template=rag_template,  # Plantilla de sistema
)

# La respuesta final del LLM se imprime en consola.
print(graph_rag.search(QUERY_TEXT, retriever_config={"top_k": 5}).answer)
# %%
