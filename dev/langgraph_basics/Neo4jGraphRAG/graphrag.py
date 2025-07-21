# %%
import os
from pathlib import Path

import dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.openai import AzureOpenAIEmbeddings
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
llm.invoke("say something")
embedder = AzureOpenAIEmbeddings(
    model="text-embedding-3-large",
    api_version=os.getenv("AZURE_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

# Graph Schema Setup
basic_node_labels = ["Object", "Entity", "Group", "Person", "Organization", "Place"]

academic_node_labels = ["ArticleOrPaper", "PublicationOrJournal"]

medical_node_labels = [
    "Anatomy",
    "BiologicalProcess",
    "Cell",
    "CellularComponent",
    "CellType",
    "Condition",
    "Disease",
    "Drug",
    "EffectOrPhenotype",
    "Exposure",
    "GeneOrProtein",
    "Molecule",
    "MolecularFunction",
    "Pathway",
]

node_labels = basic_node_labels + academic_node_labels + medical_node_labels
# define prompt template
prompt_template = """
You are a medical researcher tasks with extracting information from papers
and structuring it in a property graph to inform further medical and research Q&A.

Extract the entities (nodes) and specify their type from the following Input text.
Also extract the relationships between these nodes. the relationship direction goes from the start node to the end node.


Return result as JSON using the following format:
{{"nodes": [ {{"id": "0", "label": "the type of entity", "properties": {{"name": "name of entity" }} }}],
  "relationships": [{{"type": "TYPE_OF_RELATIONSHIP", "start_node_id": "0", "end_node_id": "1", "properties": {{"details": "Description of the relationship"}} }}] }}

...

Use only fhe following nodes and relationships:
{schema}

Assign a unique ID (string) to each node, and reuse it to define relationships.
Do respect the source and target node types for relationship and the relationship direction.

Do not return any additional information other than the JSON in it.

Examples:
{examples}

Input text:

{text}
"""

# Knowledge Graph Builder

# kg_builder_pdf = SimpleKGPipeline(
#     llm=llm,
#     driver=driver,
#     text_splitter=FixedSizeSplitter(chunk_size=500, chunk_overlap=100),
#     embedder=embedder,
#     entities=node_labels,
#     relations=rel_types,
#     prompt_template=prompt_template,
#     from_pdf=True,
# )
# %%

root_dir = Path(__file__).parents[0]
file_path = root_dir / "pdfs" / "Harry Potter and the Chamber of Secrets Summary.pdf"
# Instantiate NodeType and RelationshipType objects. This defines the
# entities and relations the LLM will be looking for in the text.
NODE_TYPES = ["Person", "Organization", "Location"]
RELATIONSHIP_TYPES = ["SITUATED_AT", "INTERACTS", "LED_BY"]
PATTERNS = [
    ("Person", "SITUATED_AT", "Location"),
    ("Person", "INTERACTS", "Person"),
    ("Organization", "LED_BY", "Person"),
]

kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=neo4j_driver,
    embedder=embedder,
    schema={
        "node_types": NODE_TYPES,
        "relationship_types": RELATIONSHIP_TYPES,
        "patterns": PATTERNS,
    },
    neo4j_database=os.getenv("NEO4J_DATABASE"),
)
pdf_result = await kg_builder.run_async(file_path=file_path)
"""
API credentials
Authentication provider name

0a4h7ltw
Authentication provider ID
6c696386-3063-4f8d-a0bb-0e89be7f34a8

Authentication provider key
aLKncOB7aqz395YuoRmRFnZXXU3s1b5C

curl --location https://194e9aac-graphql.production-orch-0386.neo4j.io/graphql \
  --header 'Content-Type: application/json' \
  --header 'x-api-key: <YOUR_API_KEY>' \
  --data '{"query": "<YOUR_GRAPHQL_QUERY>"}'

"""
# %%
import requests

endpoint = os.getenv(
    "xNEO4J_GRAPHQL_ENDPOINT",
    "https://194e9aac-graphql.production-orch-0386.neo4j.io/graphql",
)
api_key = os.getenv("xNEO4J_API_KEY", "aLKncOB7aqz395YuoRmRFnZXXU3s1b5C")

query = """
    { __schema { types { name } } }
    """
headers = {
    "Content-Type": "application/json",
    "x-api-key": api_key,
}
response = requests.post(
    endpoint,
    json={"query": query},
    headers=headers,
    timeout=30,
)
response.raise_for_status()
print(response.json())
