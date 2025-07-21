# %%

import os

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.cohere import CohereEmbeddings
from neo4j_graphrag.generation import GraphRAG, RagTemplate
from neo4j_graphrag.llm import AzureOpenAILLM
from neo4j_graphrag.retrievers import HybridCypherRetriever

load_dotenv(override=True)

# Demo database credentials
URI = "neo4j+s://demo.neo4jlabs.com"
AUTH = ("recommendations", "recommendations")
# Connect to Neo4j database
driver = GraphDatabase.driver(URI, auth=AUTH)
embedder = CohereEmbeddings(
    model="embed-v4.0",
    api_key=os.getenv("COHERE_API_KEY"),
)

retrieval_query = """
MATCH
(actor:Actor)-[:ACTED_IN]->(node)
RETURN
node.title AS movie_title,
node.plot AS movie_plot, 
collect(actor.name) AS actors;
"""

retriever = HybridCypherRetriever(
    driver=driver,
    vector_index_name="moviePlotsEmbedding",
    fulltext_index_name="movieFulltext",
    retrieval_query=retrieval_query,
    embedder=embedder,
)

query_text = (
    "What are the names of the actors in the movie set in 1375 in Imperial China?"
)
retriever_result = retriever.search(query_text=query_text, top_k=3)
print(retriever_result.items)
# %%


llm = AzureOpenAILLM(
    model_name="gpt-4.1-mini",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)
llm.invoke("say something")
# %%

rag_template = RagTemplate(
    template="""Answer the Question using the following Context. Only respond with information mentioned in the Context. Do not inject any speculative information not mentioned.

# Question:
{query_text}

# Context:
{context}

# Answer:
""",
    expected_inputs=["query_text", "context"],
)

graph_rag = GraphRAG(llm=llm, retriever=retriever, prompt_template=rag_template)
q = "What are the names of the actors in the movie set in 1375 in Imperial China?."
graph_rag.search(q, retriever_config={"top_k": 5}).answer
