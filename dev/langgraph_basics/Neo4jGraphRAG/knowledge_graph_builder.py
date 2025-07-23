# %%
"""knowledge_graph_builder.py
====================================================================
Construye un *Knowledge Graph* en Neo4j que representa enzimas de los
sub-sistemas **glucólisis** y **ciclo TCA**.  Para cada enzima se
capturan propiedades clave (subsystem, substrates, products,
reversibility, flux) y se generan relaciones:

• (Metabolito)-[:CONSUMIDO_POR]->(Enzyme)
• (Metabolito)-[:GENERADO_POR]->(Enzyme)
• (Enzyme)-[:EN]->(Subsystem)

El flujo de trabajo sigue estos pasos:
  1. Carga de credenciales desde `.env` y verificación de conectividad.
  2. Definición de documentos – un `Document` por enzima con metadatos
     estructurados.
  3. Configuración de componentes (LLM, embedder, splitter).
  4. Ejecución de `SimpleKGPipeline` guiado por *schema* y *patterns*.
  5. Limpieza previa de la base y lanzamiento asíncrono del pipeline.

Para ejecutar el script:
$ uv run python dev/langgraph_basics/Neo4jGraphRAG/knowledge_graph_builder.py
"""

from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.cohere import CohereEmbeddings
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.indexes import create_fulltext_index, create_vector_index
from neo4j_graphrag.llm import AzureOpenAILLM

# --------------------------------------------------------------------------- #
# 1) Entorno y conexión a Neo4j
# --------------------------------------------------------------------------- #

load_dotenv(override=True)

NEO4J_USERNAME: str | None = os.getenv("NEO4J_USERNAME_UPGRADED")
NEO4J_PASSWORD: str | None = os.getenv("NEO4J_PASSWORD_UPGRADED")
NEO4J_URI: str | None = os.getenv("NEO4J_CONNECTION_URI_UPGRADED")

if not (NEO4J_USERNAME and NEO4J_PASSWORD and NEO4J_URI):
    raise EnvironmentError(
        "⚠️  Variables de entorno de Neo4j incompletas. Revisa `.env`."
    )

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
# Verificamos conectividad antes de proseguir.
with driver as _tmp_driver:
    _tmp_driver.verify_connectivity()

# --------------------------------------------------------------------------- #
# 2) Componentes auxiliares: LLM, Embeddings, Splitter
# --------------------------------------------------------------------------- #

llm = AzureOpenAILLM(
    model_name="gpt-4.1-mini",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

embedder = CohereEmbeddings(model="embed-v4.0", api_key=os.getenv("COHERE_API_KEY"))

# Chunks de 400 tokens con 50 de solapamiento – suficiente para los textos
# descriptivos de cada enzima.
text_splitter = FixedSizeSplitter(chunk_size=400, chunk_overlap=50)

# --------------------------------------------------------------------------- #
# 2.1) Crear índices vectoriales y full-text para los nodos Chunk
# --------------------------------------------------------------------------- #

VECTOR_INDEX_NAME = "chunkEmbedding"
FULLTEXT_INDEX_NAME = "chunkFulltext"

# Intentamos inferir la dimensión automáticamente.
try:
    VECTOR_DIM = len(embedder.embed_query("test"))
except Exception:
    VECTOR_DIM = 1024  # fallback razonable

# Crear índices si no existen (idempotente)
try:
    create_vector_index(
        driver,
        name=VECTOR_INDEX_NAME,
        label="Chunk",
        embedding_property="embedding",
        dimensions=VECTOR_DIM,
        similarity_fn="cosine",
        fail_if_exists=False,
    )
except Exception:
    pass  # ya existe o no es crítico

try:
    create_fulltext_index(
        driver,
        name=FULLTEXT_INDEX_NAME,
        label="Chunk",
        node_properties=["text"],
        fail_if_exists=False,
    )
except Exception:
    pass

# --------------------------------------------------------------------------- #
# 3) Documentos: enzimas + metadatos
# --------------------------------------------------------------------------- #

# Utilizamos la carpeta actual para leer, por simplicidad los definimos aquí.
# Cada párrafo describe la enzima y añade las claves de su metadata.

documents: List[Document] = [
    # ------------------- Glucólisis -------------------
    Document(
        page_content=(
            "Hexokinase (HK) catalyzes the phosphorylation of glucose to "
            "glucose-6-phosphate — first committed step of glycolysis."
        ),
        metadata={
            "enzyme": "HK",
            "subsystem": "glycolysis",
            "substrates": ["Glc", "ATP"],
            "products": ["G6P", "ADP"],
            "reversible": False,
            "flux": 1.5,
        },
    ),
    Document(
        page_content=(
            "Phosphofructokinase-1 (PFK-1) is the main rate-limiting enzyme of "
            "glycolysis, converting F6P to F1,6BP in an ATP-dependent reaction."
        ),
        metadata={
            "enzyme": "PFK1",
            "subsystem": "glycolysis",
            "substrates": ["F6P", "ATP"],
            "products": ["F1,6BP", "ADP"],
            "reversible": False,
            "flux": 1.2,
        },
    ),
    Document(
        page_content=(
            "Pyruvate kinase (PK) transfers the phosphate from PEP to ADP, "
            "yielding ATP and pyruvate — final step of glycolysis."
        ),
        metadata={
            "enzyme": "PK",
            "subsystem": "glycolysis",
            "substrates": ["PEP", "ADP"],
            "products": ["Pyr", "ATP"],
            "reversible": False,
            "flux": 2.0,
        },
    ),
    Document(
        page_content=(
            "Glyceraldehyde-3-phosphate dehydrogenase (GAPDH) produces 1,3-BPG "
            "and NADH from G3P, connecting glycolysis with redox balance."
        ),
        metadata={
            "enzyme": "GAPDH",
            "subsystem": "glycolysis",
            "substrates": ["G3P", "NAD+", "Pi"],
            "products": ["1,3-BPG", "NADH"],
            "reversible": True,
            "flux": 1.8,
        },
    ),
    # ------------------- Ciclo TCA -------------------
    Document(
        page_content=(
            "Citrate synthase (CS) condenses acetyl-CoA and oxaloacetate to "
            "form citrate, imposing directionality on the TCA cycle."
        ),
        metadata={
            "enzyme": "CS",
            "subsystem": "TCA",
            "substrates": ["AcCoA", "OAA"],
            "products": ["Cit"],
            "reversible": False,
            "flux": 0.8,
        },
    ),
    Document(
        page_content=(
            "Isocitrate dehydrogenase (IDH) converts isocitrate to α-ketoglutarate "
            "with NADH production; mutations generate 2-hydroxyglutarate."
        ),
        metadata={
            "enzyme": "IDH",
            "subsystem": "TCA",
            "substrates": ["IsoCit", "NAD+"],
            "products": ["aKG", "CO2", "NADH"],
            "reversible": True,
            "flux": 0.7,
        },
    ),
    Document(
        page_content=(
            "α-Ketoglutarate dehydrogenase (AKGDH) transforms aKG to succinyl-CoA, "
            "linking carbon flux to oxidative phosphorylation."
        ),
        metadata={
            "enzyme": "AKGDH",
            "subsystem": "TCA",
            "substrates": ["aKG", "CoA", "NAD+"],
            "products": ["SucCoA", "CO2", "NADH"],
            "reversible": True,
            "flux": 0.6,
        },
    ),
    Document(
        page_content=(
            "Succinate dehydrogenase (SDH) participates in both the TCA cycle "
            "and the electron transport chain, oxidising succinate to fumarate."
        ),
        metadata={
            "enzyme": "SDH",
            "subsystem": "TCA",
            "substrates": ["Suc", "Q"],
            "products": ["Fum", "QH2"],
            "reversible": True,
            "flux": 0.9,
        },
    ),
    Document(
        page_content=(
            "Malate dehydrogenase (MDH) interconverts malate and oxaloacetate "
            "with concomitant NAD+/NADH cycling."
        ),
        metadata={
            "enzyme": "MDH",
            "subsystem": "TCA",
            "substrates": ["Mal", "NAD+"],
            "products": ["OAA", "NADH"],
            "reversible": True,
            "flux": 0.5,
        },
    ),
]
num_enzymes, num_subsystems, num_substrates, num_products = (
    len({d.metadata["enzyme"] for d in documents}),
    len({d.metadata["subsystem"] for d in documents}),
    len({s for d in documents for s in d.metadata["substrates"]}),
    len({p for d in documents for p in d.metadata["products"]}),
)
print(
    f"Enzymes: {num_enzymes}, Subsystems: {num_subsystems}, Substrates: {num_substrates}, Products: {num_products}"
)
# %%
# --------------------------------------------------------------------------- #
# 4) Schema dirigido (enzimas, metabolitos, subsistema)
# --------------------------------------------------------------------------- #

NODE_TYPES = [
    {
        "label": "Enzyme",
        "description": "Metabolic enzyme with key biochemical properties.",
        "properties": [
            {"name": "name", "type": "STRING"},
            {"name": "subsystem", "type": "STRING"},
            {"name": "substrates", "type": "STRING"},
            {"name": "products", "type": "STRING"},
            {"name": "reversible", "type": "BOOLEAN"},
            {"name": "flux", "type": "FLOAT"},
        ],
    },
    "Metabolite",  # metabolite names captured from substrates/products
    "Subsystem",  # e.g. glycolysis, TCA
]

RELATIONSHIP_TYPES = [
    "PRODUCE",  # (Enzyme)-[:PRODUCE]->(Metabolite)
    "SUBSTRATO_DE",  # (Metabolite)-[:SUBSTRATO_DE]->(Enzyme)
    "PERTENECE_A",  # (Enzyme)-[:PERTENECE_A]->(Subsystem)
]

# Aseguramos que los patrones usen EXACTAMENTE los mismos labels.
PATTERNS = [
    ("Enzyme", "PRODUCE", "Metabolite"),
    ("Metabolite", "SUBSTRATO_DE", "Enzyme"),
    ("Enzyme", "PERTENECE_A", "Subsystem"),
]

# --------------------------------------------------------------------------- #
# 5) Funciones utilitarias
# --------------------------------------------------------------------------- #


def clear_graph(_driver: GraphDatabase.driver) -> None:
    """Vacía completamente la base antes de cada corrida."""
    with _driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("🧹  Graph cleared.")


async def build_kg_from_docs(docs: List[Document]) -> None:  # noqa: D401
    """Construye el KG a partir de la lista de documentos proporcionada."""

    clear_graph(driver)

    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,
        text_splitter=text_splitter,
        schema={
            "node_types": NODE_TYPES,
            "relationship_types": RELATIONSHIP_TYPES,
            "patterns": PATTERNS,
            "additional_node_types": True,  # Metabolite y Subsystem surgirán dinámicamente
        },
        from_pdf=False,
    )

    for doc in docs:
        # Creamos una representación textual enriquecida con los metadatos para
        # facilitar la extracción por parte del LLM.
        meta = doc.metadata  # type: ignore[attr-defined]
        # Sentencias adicionales para guiar al LLM y crear correctamente las
        # relaciones definidas en PATTERNS.
        substrates_str = ", ".join(meta["substrates"])
        products_str = ", ".join(meta["products"])
        reversibility_text = (
            "La reacción es reversible."
            if meta["reversible"]
            else "La reacción es irreversible."
        )

        augmented_text = (
            f"{doc.page_content}\n\n"  # texto original
            "---\n"
            f"Nombre de la enzima: {meta['enzyme']}.\n"
            f"Subsistema metabólico: {meta['subsystem']}.\n"
            # Relaciones con metabolitos (sustratos ↔ productos)
            f"Los sustratos {substrates_str} son SUBSTRATO_DE la enzima {meta['enzyme']}.\n"
            f"La enzima {meta['enzyme']} PRODUCE los metabolitos {products_str}.\n"
            f"La enzima {meta['enzyme']} PERTENECE_A {meta['subsystem']}.\n"
            f"{reversibility_text}\n"
            f"Flujo relativo: {meta['flux']}.\n"
        )
        await kg_builder.run_async(text=augmented_text)
        print(f"✅  Processed {meta['enzyme']} → KG updated.")

    print("🎉  Knowledge graph creation finished!")


# --------------------------------------------------------------------------- #
# 6) Punto de entrada
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import asyncio

    asyncio.run(build_kg_from_docs(documents))
