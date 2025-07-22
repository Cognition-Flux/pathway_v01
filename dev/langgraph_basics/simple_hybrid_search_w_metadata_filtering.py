"""
Simple parallel retriever with metadata filtering using Hybrid Search.
"""

# %%
import os
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
# if INDEX_NAME in names:
#     pc.delete_index(INDEX_NAME)
#     # Wait for the index to be deleted
#     while INDEX_NAME in [inx["name"] for inx in pc.list_indexes()]:
#         time.sleep(1)


# For single-index hybrid search, Pinecone requires the 'dotproduct' metric.
if INDEX_NAME not in names:
    pc.create_index(
        name=INDEX_NAME,
        dimension=3072,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
else:
    print(f"Index {INDEX_NAME} already exists")

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
        "Hexokinase (HK) is a ubiquitous cytosolic enzyme that catalyzes the first committed step of glycolysis: the phosphorylation of glucose to form glucose-6-phosphate (G6P). "
        "This reaction traps glucose inside the cell and primes it for further metabolism. "
        "Human hexokinase exists in four isoforms (HK I-IV) that differ in tissue distribution, kinetic properties, and regulation. "
        "HK activity is tightly regulated by its product G6P, cellular energy charge (ATP/ADP ratio), and interaction with the outer mitochondrial membrane, linking glycolytic flux with oxidative phosphorylation."
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
        "Phosphofructokinase-1 (PFK-1) catalyzes the ATP-dependent phosphorylation of fructose-6-phosphate (F6P) to fructose-1,6-bisphosphate (F1,6BP), committing the sugar to glycolysis. "
        "PFK-1 is considered the major rate-limiting and regulatory step of glycolysis. "
        "It is an allosteric tetramer whose activity integrates a variety of metabolic signals: it is activated by AMP, ADP and fructose-2,6-bisphosphate, while ATP and citrate act as potent inhibitors, thereby synchronizing glycolytic throughput with cellular energy status and anabolic needs."
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
        "Pyruvate kinase (PK) catalyzes the final step of glycolysis, transferring a phosphate group from phosphoenolpyruvate (PEP) to ADP to generate pyruvate and ATP. "
        "Mammalian cells express distinct isoforms (PKM1, PKM2, PKL, PKR) that support tissue-specific metabolic demands. "
        "PK activity is allosterically activated by fructose-1,6-bisphosphate (feed-forward regulation) and inhibited by phosphorylation via protein kinase A during gluconeogenic conditions. "
        "The enzyme thereby serves as a metabolic gatekeeper controlling carbon entry into the TCA cycle or lactate production."
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
        "Citrate synthase (CS) initiates the tricarboxylic acid (TCA) cycle by condensing acetyl-CoA and oxaloacetate to form citrate and CoA-SH. "
        "This irreversible reaction is highly exergonic and therefore imposes directionality on the cycle. "
        "CS activity reflects mitochondrial health and is modulated by substrate availability and feedback inhibition by NADH, succinyl-CoA and ATP. "
        "Beyond energy metabolism, citrate exported to the cytosol serves as a precursor for fatty-acid and cholesterol biosynthesis, linking the TCA cycle with anabolic pathways."
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
        "Isocitrate dehydrogenase (IDH) catalyzes the oxidative decarboxylation of isocitrate to α-ketoglutarate (aKG), producing CO₂ and reducing NAD⁺ (or NADP⁺) to NADH (or NADPH). "
        "Mammals possess three isoforms: IDH3 (NAD-dependent, mitochondrial) functions in the TCA cycle, whereas IDH1 (cytosolic) and IDH2 (mitochondrial) are NADP-dependent and contribute to redox balance via NADPH generation. "
        "Gain-of-function mutations in IDH1/2, frequent in gliomas and acute myeloid leukemia, produce the oncometabolite 2-hydroxyglutarate, illustrating the enzyme’s clinical relevance."
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
        "The α-ketoglutarate dehydrogenase complex (AKGDH) is a large multienzyme assembly analogous to pyruvate dehydrogenase that catalyzes the conversion of α-ketoglutarate to succinyl-CoA and CO₂ while reducing NAD⁺ to NADH. "
        "Because the reaction is highly exergonic, it acts as another control point of the TCA cycle and is inhibited by its products (succinyl-CoA and NADH) as well as high ratios of ATP/ADP and Ca²⁺. "
        "AKGDH links carbon flux with oxidative phosphorylation and the production of biosynthetic precursors derived from succinyl-CoA."
    ),
    metadata={
        "enzyme": "AKGDH",
        "subsystem": "TCA",
        "substrates": ["aKG", "CoA", "NAD+"],
        "products": ["SucCoA", "CO2", "NADH"],
        "reversible": True,
        "flux": 0.6,  # assumed flux in μmol/min/g in human liver cell
    },
)

document_7 = Document(
    page_content=(
        "Succinate dehydrogenase (SDH), also known as Complex II of the electron transport chain, uniquely participates in both the TCA cycle and oxidative phosphorylation. "
        "It oxidizes succinate to fumarate while reducing ubiquinone (CoQ) to ubiquinol. "
        "SDH is a membrane-bound flavoprotein whose activity integrates TCA flux with respiratory chain function; inherited defects lead to mitochondrial encephalomyopathies and tumorigenesis (paragangliomas, pheochromocytomas)."
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
        "Glyceraldehyde-3-phosphate dehydrogenase (GAPDH) catalyzes the reversible oxidative phosphorylation of glyceraldehyde-3-phosphate to 1,3-bisphosphoglycerate, producing NADH. "
        "Besides its central metabolic role, GAPDH is a multifunctional ‘moonlighting’ protein involved in nuclear RNA transport, DNA repair, and programmed cell death. "
        "Its high abundance makes it a common loading control in biochemical assays, yet its regulation by oxidative modifications underscores its participation in redox signaling pathways."
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

# New reversible NAD-dependent TCA cycle enzymes

document_9 = Document(
    page_content=(
        "Malate dehydrogenase (MDH) catalyzes the reversible conversion of malate to oxaloacetate while reducing NAD⁺ to NADH. "
        "The reaction has a high positive ΔG°′ under standard conditions, but proceeds in vivo due to continual oxaloacetate removal by citrate synthase. "
        "MDH operates in both mitochondrial and cytosolic isoforms, enabling the malate-aspartate shuttle that transfers reducing equivalents across the mitochondrial membrane for oxidative phosphorylation."
    ),
    metadata={
        "enzyme": "MDH",
        "subsystem": "TCA",
        "substrates": ["Mal", "NAD+"],
        "products": ["OAA", "NADH"],
        "reversible": True,
        "flux": 0.5,  # assumed flux in μmol/min/g in human liver cell
    },
)

document_10 = Document(
    page_content=(
        "Succinate semialdehyde dehydrogenase (SSADH) participates in the γ-aminobutyric acid (GABA) shunt by oxidizing succinate semialdehyde to succinate with concomitant reduction of NAD⁺ to NADH. "
        "This pathway connects neurotransmitter metabolism with central carbon metabolism and provides an anaplerotic route into the TCA cycle. "
        "Inherited SSADH deficiency leads to elevated GABA and neurological dysfunctions such as ataxia and epilepsy, highlighting the enzyme’s physiological importance."
    ),
    metadata={
        "enzyme": "SSADH",
        "subsystem": "TCA",
        "substrates": ["SSA", "NAD+"],
        "products": ["Suc", "NADH"],
        "reversible": True,
        "flux": 0.4,  # assumed flux in μmol/min/g in human liver cell
    },
)

# New glycolytic enzymes ------------------------------------------------------

document_11 = Document(
    page_content=(
        "Phosphoglucose isomerase (PGI), also known as glucose-6-phosphate isomerase, catalyzes the reversible aldose-ketose isomerization of glucose-6-phosphate (G6P) to fructose-6-phosphate (F6P). "
        "Beyond its metabolic role, PGI functions extracellularly as neuroleukin and autocrine motility factor, underscoring its moonlighting nature."
    ),
    metadata={
        "enzyme": "PGI",
        "subsystem": "glycolysis",
        "substrates": ["G6P"],
        "products": ["F6P"],
        "reversible": True,
        "flux": 1.4,
    },
)

document_12 = Document(
    page_content=(
        "Fructose-1,6-bisphosphate aldolase (ALDO) cleaves fructose-1,6-bisphosphate to yield two triose phosphates—glyceraldehyde-3-phosphate (G3P) and dihydroxyacetone phosphate (DHAP). "
        "In mammals, three isoforms (A, B, C) support tissue-specific glycolytic and gluconeogenic demands."
    ),
    metadata={
        "enzyme": "ALDO",
        "subsystem": "glycolysis",
        "substrates": ["F1,6BP"],
        "products": ["G3P", "DHAP"],
        "reversible": True,
        "flux": 1.3,
    },
)

document_13 = Document(
    page_content=(
        "Triose-phosphate isomerase (TPI) interconverts dihydroxyacetone phosphate and glyceraldehyde-3-phosphate, ensuring that both products of aldolase continue through the pay-off phase of glycolysis. "
        "TPI is often cited as a ‘catalytically perfect’ enzyme due to its diffusion-limited turnover rate."
    ),
    metadata={
        "enzyme": "TPI",
        "subsystem": "glycolysis",
        "substrates": ["DHAP"],
        "products": ["G3P"],
        "reversible": True,
        "flux": 1.6,
    },
)

document_14 = Document(
    page_content=(
        "Phosphoglycerate kinase (PGK) performs the first substrate-level phosphorylation of glycolysis, transferring the high-energy phosphate from 1,3-bisphosphoglycerate to ADP to generate ATP and 3-phosphoglycerate. "
        "Human PGK1 deficiency is a rare X-linked disorder leading to hemolytic anemia and myopathy."
    ),
    metadata={
        "enzyme": "PGK",
        "subsystem": "glycolysis",
        "substrates": ["1,3-BPG", "ADP"],
        "products": ["3PG", "ATP"],
        "reversible": True,
        "flux": 1.7,
    },
)

document_15 = Document(
    page_content=(
        "Phosphoglycerate mutase (PGM) catalyzes the intramolecular transfer of the phosphate group from the 3- to the 2-position of phosphoglycerate, converting 3-phosphoglycerate to 2-phosphoglycerate. "
        "In mammals, the reaction proceeds via a 2,3-bisphosphoglycerate intermediate."
    ),
    metadata={
        "enzyme": "PGM",
        "subsystem": "glycolysis",
        "substrates": ["3PG"],
        "products": ["2PG"],
        "reversible": True,
        "flux": 1.1,
    },
)

document_16 = Document(
    page_content=(
        "Enolase catalyzes the dehydration of 2-phosphoglycerate to phosphoenolpyruvate (PEP), a high-energy enol phosphate poised for ATP generation. "
        "Three human isoforms (α, β, γ) support ubiquitous, muscle, and neuronal glycolysis respectively, and enolase autoantibodies are biomarkers in several cancers."
    ),
    metadata={
        "enzyme": "ENO",
        "subsystem": "glycolysis",
        "substrates": ["2PG"],
        "products": ["PEP", "H2O"],
        "reversible": True,
        "flux": 1.0,
    },
)

# Additional TCA enzymes -------------------------------------------------------

document_17 = Document(
    page_content=(
        "Aconitase (ACO) catalyzes the reversible isomerization of citrate to isocitrate via the intermediate cis-aconitate. "
        "The enzyme employs a [4Fe–4S] cluster for substrate binding and catalysis and functions as both a metabolic enzyme and an iron-regulatory protein in its cytosolic form."
    ),
    metadata={
        "enzyme": "ACO",
        "subsystem": "TCA",
        "substrates": ["citrate"],
        "products": ["isocitrate"],
        "reversible": True,
        "flux": 0.9,
    },
)

document_18 = Document(
    page_content=(
        "Succinyl-CoA synthetase (SCS, also called succinate-CoA ligase) converts succinyl-CoA to succinate while simultaneously phosphorylating GDP or ADP to GTP/ATP—one of the few substrate-level phosphorylation steps outside glycolysis. "
        "Human mitochondria express both ADP- and GDP-dependent isoforms to tailor nucleotide output to tissue demands."
    ),
    metadata={
        "enzyme": "SCS",
        "subsystem": "TCA",
        "substrates": ["succinyl-CoA", "GDP/ADP", "Pi"],
        "products": ["succinate", "GTP/ATP", "CoA"],
        "reversible": True,
        "flux": 0.85,
    },
)

document_19 = Document(
    page_content=(
        "Fumarase (FUM, fumarate hydratase) catalyzes the stereospecific hydration of fumarate to L-malate, sustaining TCA cycle flux and furnishing intermediates for biosynthetic pathways. "
        "Cytosolic fumarase also moonlights in DNA damage response, and germline loss-of-function mutations underlie hereditary leiomyomatosis and renal cell carcinoma."
    ),
    metadata={
        "enzyme": "FUM",
        "subsystem": "TCA",
        "substrates": ["fumarate", "H2O"],
        "products": ["malate"],
        "reversible": True,
        "flux": 0.95,
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
    document_9,
    document_10,
    document_11,
    document_12,
    document_13,
    document_14,
    document_15,
    document_16,
    document_17,
    document_18,
    document_19,
]

# ---------------------------------------------------------------------------
# Ensure each page_content includes its metadata string representation.
# This makes the metadata searchable in the text content itself.
# ---------------------------------------------------------------------------

for _doc in documents:
    _doc.page_content += f"\n\nMETADATA: {_doc.metadata}"  # type: ignore[attr-defined]

uuids = [str(uuid4()) for _ in range(len(documents))]
texts = [doc.page_content for doc in documents]  # type: ignore[attr-defined]
metadatas = [doc.metadata for doc in documents]
if INDEX_NAME not in names:
    retriever.add_texts(texts=texts, metadatas=metadatas, ids=uuids)
else:
    print(f"Index {INDEX_NAME} already exists, not adding documents")

# %%

if __name__ == "__main__":
    print(f"INDEX: {index}")
    results = retriever.invoke(
        # " Succinate semialdehyde",
        # "Isocitrate",
        "Malate",
        # {
        #     "$and": [
        #         {"subsystem": "TCA"},
        #         {"reversible": True},
        #         {"$or": [{"substrates": "NADH"}, {"products": "NADH"}]},
        #     ]
        # },
        filter={
            "$and": [
                {"subsystem": {"$eq": "TCA"}},
                {"substrates": {"$in": ["NAD+"]}},
                {"reversible": {"$eq": True}},
            ]
        },
    )
    print(f"results: {results}")
    for res in results:
        score = res.metadata.get("score", "N/A")
        print(f"* [score: {score:.3f}] {res.page_content} [{res.metadata}]")
