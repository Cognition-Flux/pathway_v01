"""
Simple parallel retriever with metadata filtering using Hybrid Search.
"""

# %%
import os
import time
from uuid import uuid4

from dotenv import load_dotenv
from langchain_community.retrievers import (
    PineconeHybridSearchRetriever,
)
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse.splade_encoder import SpladeEncoder

load_dotenv(override=True)

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
INDEX_NAME = "metadata-filtering-hybrid"
index_list = pc.list_indexes()

# Re-create the index if it already exists to ensure a clean slate
names = [inx["name"] for inx in index_list]
if INDEX_NAME in names:
    pc.delete_index(INDEX_NAME)
    # Wait for the index to be deleted
    while INDEX_NAME in [inx["name"] for inx in pc.list_indexes()]:
        time.sleep(1)


# For single-index hybrid search, Pinecone requires the 'dotproduct' metric.
pc.create_index(
    name=INDEX_NAME,
    dimension=3072,
    metric="dotproduct",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)

index = pc.Index(INDEX_NAME)

embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-large",
    azure_endpoint=os.getenv("AZUREOPENAIEMBEDDINGS_AZURE_ENDPOINT"),
    api_key=os.getenv("AZUREOPENAIEMBEDDINGS_API_KEY"),
)

sparse_encoder = SpladeEncoder()

# The retriever handles the logic of creating dense and sparse vectors
# and combining them for the query.
retriever = PineconeHybridSearchRetriever(
    embeddings=embeddings, sparse_encoder=sparse_encoder, index=index
)

document_1 = Document(
    page_content=(
        "Hexokinase catalyzes the phosphorylation of glucose to "
        "glucose-6-phosphate, the first step in glycolysis."
    ),
    metadata={
        "enzyme": "HK",
        "subsystem": "glycolysis",
        "substrates": ["Glc", "ATP"],
        "products": ["G6P", "ADP"],
        "reversible": False,
        "flux": 1.5,  # assumed flux in μmol/min/g in human liver cell
    },
)

document_2 = Document(
    page_content=(
        "Phosphofructokinase-1 catalyzes the phosphorylation of "
        "fructose-6-phosphate to fructose-1,6-bisphosphate, a key "
        "regulatory step in glycolysis."
    ),
    metadata={
        "enzyme": "PFK1",
        "subsystem": "glycolysis",
        "substrates": ["F6P", "ATP"],
        "products": ["F1,6BP", "ADP"],
        "reversible": False,
        "flux": 1.2,  # assumed flux in μmol/min/g in human liver cell
    },
)

document_3 = Document(
    page_content=(
        "Pyruvate kinase catalyzes the conversion of "
        "phosphoenolpyruvate to pyruvate, generating ATP in the "
        "final step of glycolysis."
    ),
    metadata={
        "enzyme": "PK",
        "subsystem": "glycolysis",
        "substrates": ["PEP", "ADP"],
        "products": ["Pyr", "ATP"],
        "reversible": False,
        "flux": 2.0,  # assumed flux in μmol/min/g in human liver cell
    },
)

document_4 = Document(
    page_content=(
        "Citrate synthase catalyzes the condensation of acetyl-CoA "
        "and oxaloacetate to form citrate in the first step of the "
        "TCA cycle."
    ),
    metadata={
        "enzyme": "CS",
        "subsystem": "TCA",
        "substrates": ["AcCoA", "OAA"],
        "products": ["Cit"],
        "reversible": False,
        "flux": 0.8,  # assumed flux in μmol/min/g in human liver cell
    },
)

document_5 = Document(
    page_content=(
        "Isocitrate dehydrogenase catalyzes the oxidative "
        "decarboxylation of isocitrate to alpha-ketoglutarate, "
        "producing NADH."
    ),
    metadata={
        "enzyme": "IDH",
        "subsystem": "TCA",
        "substrates": ["IsoCit", "NAD+"],
        "products": ["aKG", "CO2", "NADH"],
        "reversible": True,
        "flux": 0.7,  # assumed flux in μmol/min/g in human liver cell
    },
)

document_6 = Document(
    page_content=(
        "Alpha-ketoglutarate dehydrogenase complex catalyzes the "
        "conversion of alpha-ketoglutarate to succinyl-CoA, "
        "producing NADH and CO2."
    ),
    metadata={
        "enzyme": "AKGDH",
        "subsystem": "TCA",
        "substrates": ["aKG", "CoA", "NAD+"],
        "products": ["SucCoA", "CO2", "NADH"],
        "reversible": False,
        "flux": 0.6,  # assumed flux in μmol/min/g in human liver cell
    },
)

document_7 = Document(
    page_content=(
        "Succinate dehydrogenase catalyzes the oxidation of "
        "succinate to fumarate, coupled to the reduction of "
        "ubiquinone in the TCA cycle."
    ),
    metadata={
        "enzyme": "SDH",
        "subsystem": "TCA",
        "substrates": ["Suc", "Q"],
        "products": ["Fum", "QH2"],
        "reversible": True,
        "flux": 0.9,  # assumed flux in μmol/min/g in human liver cell
    },
)

document_8 = Document(
    page_content=(
        "Glyceraldehyde-3-phosphate dehydrogenase (GAPDH) catalyzes "
        "the conversion of glyceraldehyde 3-phosphate to "
        "1,3-bisphosphoglycerate, producing NADH in glycolysis."
    ),
    metadata={
        "enzyme": "GAPDH",
        "subsystem": "glycolysis",
        "substrates": ["G3P", "NAD+", "Pi"],
        "products": ["1,3-BPG", "NADH"],
        "reversible": True,
        "flux": 1.8,  # assumed flux in μmol/min/g in human liver cell
    },
)


documents = [
    document_1,
    document_2,
    document_3,
    document_4,
    document_5,
    document_6,
    document_7,
    document_8,
]
uuids = [str(uuid4()) for _ in range(len(documents))]
texts = [doc.page_content for doc in documents]
metadatas = [doc.metadata for doc in documents]
retriever.add_texts(texts=texts, metadatas=metadatas, ids=uuids)

# %%

if __name__ == "__main__":
    results = retriever.invoke(
        "biología",
        filter={
            "$and": [
                {"subsystem": "TCA"},
                {"reversible": True},
                {"$or": [{"substrates": "NADH"}, {"products": "NADH"}]},
            ]
        },
    )
    for res in results:
        score = res.metadata.get("score", "N/A")
        print(f"* [score: {score:.3f}] {res.page_content} [{res.metadata}]")
