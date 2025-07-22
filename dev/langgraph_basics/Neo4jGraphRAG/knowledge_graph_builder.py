# %%
"""knowledge_graph_builder.py
====================================================================
Este script demuestra **paso a paso** cómo generar un *Knowledge Graph*
en Neo4j a partir de texto plano empleando la versión experimental de
`SimpleKGPipeline` incluida en la librería **neo4j-graphrag**.

Guías oficiales consultadas
---------------------------
* User Guide – KG Builder  <https://neo4j.com/docs/neo4j-graphrag-python/current/user_guide_kg_builder.html>
* API Reference – SimpleKGPipeline  <https://neo4j.com/docs/neo4j-graphrag-python/current/api.html>

Estructura del flujo
+--------------------
1. **Carga de entorno**: se leen credenciales y endpoints desde el fichero
   `.env`, manteniendo buenas prácticas de seguridad.
2. **Conexión a Neo4j**: se verifica conectividad antes de avanzar.
3. **Preparación de componentes**: LLM (Azure OpenAI), embedder (Cohere) y
   *text splitter*.
4. **Definición de documentos**: tres fragmentos clásicos de
   física con metadatos *author* y *year*.
5. **Diseño del *schema***: tipos de nodos, relaciones y patrones que sirven
   de guía para la extracción.
6. **Ejecución del pipeline**: para cada documento se crean los nodos y
   relaciones correspondientes.
7. **Limpieza y pos-proceso**: se eliminan nodos de personas que no forman
   parte de la lista blanca `ALLOWED_PHYSICISTS`.

El código incluye comentarios exhaustivos en español pensados para ayudar
a entender *qué* hace cada línea y *por qué* se hace así, siempre en
alineación con la documentación oficial de Neo4j.
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
# Cargamos las variables definidas en `.env`. Usamos *override=True* para
# que los valores del archivo sobrescriban posibles variables ya presentes
# en el entorno de la sesión.
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME_UPGRADED")  # Usuario administrado
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD_UPGRADED")  # Contraseña segura
URI = os.getenv("NEO4J_CONNECTION_URI_UPGRADED")  # bolt:// o neo4j+s://
AUTH = (NEO4J_USERNAME, NEO4J_PASSWORD)
# Abrimos un *driver* temporal para verificar que el servidor responde.
# `verify_connectivity()` lanza excepción si las credenciales o la URI
# son inválidas, evitando fallos más adelante en tiempo de ejecución.
with GraphDatabase.driver(URI, auth=AUTH) as driver:
    driver.verify_connectivity()
neo4j_driver = GraphDatabase.driver(URI, auth=AUTH)


# -------------------------------------------------------------------
# 1) LLM: Azure OpenAI – será usado por el KG Builder para *parsing*
# -------------------------------------------------------------------

llm = AzureOpenAILLM(
    model_name="gpt-4.1-mini",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

# *Embedder* basado en Cohere. Se utiliza internamente por SimpleKGPipeline
# para calcular embeddings de cada *Chunk* y almacenarlos en Neo4j (si se
# desea un flujo RAG posterior).
embedder = CohereEmbeddings(
    model="embed-v4.0",
    api_key=os.getenv("COHERE_API_KEY"),
)

# --------------------------------------------------------------------------- #
# Text splitter to ensure chunks fit within the LLM context window
# --------------------------------------------------------------------------- #


# Divisor de texto fijo: 500 tokens por trozo con 100 de solapamiento.
# Estos valores aseguran que cada *chunk* cabe con holgura en modelos GPT-4
# (contexto de 8 k tokens) y mejoran la calidad de extracción.
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

# Lista de documentos de entrada junto con metadatos. `langchain_core`
# modela un documento como (`page_content`, `metadata`).
DOCS: list[Document] = [
    Document(
        page_content=TEXT_EINSTEIN, metadata={"author": "Albert Einstein", "year": 1905}
    ),
    Document(
        page_content=TEXT_NEWTON, metadata={"author": "Isaac Newton", "year": 1687}
    ),
    Document(page_content=TEXT_BOHR, metadata={"author": "Niels Bohr", "year": 1913}),
]

# Lista blanca de personas que deseamos conservar en el grafo. Facilita
# la demostración sin ruido de entidades irrelevantes.
ALLOWED_PHYSICISTS = [
    "Albert Einstein",
    "Isaac Newton",
    "Niels Bohr",
]

# --------------------------------------------------------------------------- #
# Guided schema for the extraction process
# --------------------------------------------------------------------------- #
# ---------------------------
# *Schema* que guía la extracción
# ---------------------------

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


# ---------------------------------------------------------------------------
# Funciones utilitarias
# ---------------------------------------------------------------------------


def clear_graph(driver: Driver) -> None:
    """Vacía por completo el grafo.

    Eliminamos nodos y relaciones para garantizar un entorno limpio antes de
    cada ejecución. Esta estrategia simplifica la reproducibilidad del
    ejemplo sin preocuparnos de residuos de ejecuciones previas.
    """
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("🧹 Graph cleared — starting from an empty database.")


# noqa: D401 – mantener frase en infinitivo por consistencia
def prune_non_physicists(driver: Driver, allowed_names: list[str]) -> None:
    """Elimina nodos *Person* cuyo atributo `name` no está en `allowed_names`."""
    cypher = """
        MATCH (p:Person)
        WHERE NOT p.name IN $names
        DETACH DELETE p
    """
    with driver.session() as session:
        session.run(cypher, names=allowed_names)
    print("🗑️  Removed non-physicist Person nodes from the graph.")


# noqa: D401
async def build_kg_from_docs(docs: list[Document]) -> None:
    """Construir el KG desde `docs` siguiendo el flujo descrito arriba."""

    # Start from a clean slate
    clear_graph(neo4j_driver)

    # Instanciamos **SimpleKGPipeline** con todos sus componentes. Esta clase
    # orquesta internamente: división del texto → extracción de entidades ↔
    # relaciones → (opcional) embeddings → inserción en Neo4j.
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
        # Añadimos una línea que menciona explícitamente al autor y el año para
        # facilitar que el extractor identifique la entidad *Person* y la propiedad
        # `year` como dato contextual.
        augmented_text = (
            f"{doc.page_content}\n\n"
            f"Author: {doc.metadata['author']} ({doc.metadata['year']}) wrote this piece."
        )

        await kg_builder.run_async(text=augmented_text)
        print(f"✅ Processed {doc.metadata['author']} paragraph → KG updated.")

    # Eliminamos los nodos *Person* que no estén en `ALLOWED_PHYSICISTS` para
    # mantener la claridad del ejemplo.
    prune_non_physicists(neo4j_driver, ALLOWED_PHYSICISTS)
    print("🎉 Knowledge Graph creation completed and pruned!")


# Punto de entrada del script: se invoca la corrutina `build_kg_from_docs`.
# `asyncio.run` gestiona el *event loop* automáticamente.
if __name__ == "__main__":
    _ = await build_kg_from_docs(DOCS)
