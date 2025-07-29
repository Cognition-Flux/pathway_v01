# %%

import os
from pathlib import Path
from typing import Annotated, Dict, List, Literal, Optional, Union

import yaml
from dotenv import load_dotenv
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, Send
from pydantic import BaseModel, Field

from dev.langgraph_basics.Neo4jGraphRAG.cypher_runner import run_cypher

load_dotenv(override=True)

yaml_path = Path(__file__).with_name("sample_queries.yaml")
sample_queries = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

# Construir ejemplos a partir del YAML
examples = [
    {"input": item["pregunta"], "output": item["cypher_query"].strip()}
    for item in sample_queries
]

to_vectorize = [" ".join(example.values()) for example in examples]
embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-large")


vectorstore = InMemoryVectorStore.from_texts(
    to_vectorize, embeddings, metadatas=examples
)
example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=1,
)

# Define the few-shot prompt.
few_shot_prompt = FewShotChatMessagePromptTemplate(
    # The input variables select the values to pass to the example_selector
    input_variables=["input"],
    example_selector=example_selector,
    # Define how each example will be formatted.
    # In this case, each example will become 2 messages:
    # 1 human, and 1 ai
    example_prompt=ChatPromptTemplate.from_messages(
        [("human", "{input}"), ("ai", "{output}")]
    ),
)


system_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert Cypher query writer."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)


llm = AzureChatOpenAI(
    azure_deployment="gpt-4.1-mini",
    api_version=os.getenv("AZURE_API_VERSION"),
    temperature=0,
    max_tokens=None,
    timeout=1200,
    max_retries=5,
    streaming=True,
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)


def sanitise_query(query: str) -> str:
    """Sanitise the query in case the LLM returned it inside markdown fences."""
    if query.startswith("```"):
        # Remove leading/trailing code fences
        stripped = query.strip("`").strip()
        # If language identifier present (e.g. ```cypher), drop first line
        if "\n" in stripped:
            first_line, rest = stripped.split("\n", 1)
            if first_line.lower().startswith("cypher"):
                query = rest
            else:
                query = stripped
        else:
            query = stripped
    return query


def safe_run_cypher(query: str) -> Union[str, List[Dict[str, any]]]:
    """Devuelve el resultado de la consulta o un string de error en formato de lista."""
    try:
        return run_cypher(query)
    except Exception as exc:  # noqa: BLE001
        return [f"ERROR: {exc}"]


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


class OneQuery(BaseModel):
    """One query."""

    query_str: str


class GeneratedQueries(BaseModel):
    """Generated queries."""

    queries_list: list[OneQuery]


class CypherQuery(BaseModel):
    """Cypher query agent."""

    cypher_query: str = Field(description="The Cypher query ready to be executed.")


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


PROMPT_FOR_GENERATE_QUERIES = """
Based on the user question, return three (3) queries useful to retrieve documents in parallel.
Queries should expand/enrich the semantic space of the user question.
User question: {user_question}
"""
TEMPLATE_FOR_GENERATE_QUERIES = ChatPromptTemplate.from_template(
    PROMPT_FOR_GENERATE_QUERIES
)
chain_for_generate_queries = TEMPLATE_FOR_GENERATE_QUERIES | llm.with_structured_output(
    GeneratedQueries, method="function_calling"
)


async def generate_questions(
    state: Neo4jQueryState,
) -> Command[Literal["generate_cypher_query"]]:
    """Node that generates queries."""
    generated_questions = chain_for_generate_queries.invoke(
        {"user_question": state["question"]}
    )
    return Command(
        goto="generate_cypher_queries_in_parallel",
        update={"generated_questions": generated_questions},
    )


async def generate_cypher_queries_in_parallel(
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
            "generate_cypher_query",
            {"query": query},
        )
        for query in lista_de_queries
    ]
    return Command(goto=sends)


async def generate_cypher_query(
    state: Neo4jQueryState,
) -> Command[Literal["run_cypher_query_in_parallel"]]:
    """Node that generates a Cypher query."""
    query_str = state["query"]

    chain_for_cypher_query = system_prompt | llm.with_structured_output(CypherQuery)
    response = await chain_for_cypher_query.ainvoke({"input": query_str})
    raw_query = response.cypher_query.strip()

    cypher_query = sanitise_query(raw_query)

    return Command(
        goto="run_cypher_query_in_parallel", update={"cypher_queries": [cypher_query]}
    )


async def run_cypher_query_in_parallel(
    state: Neo4jQueryState,
) -> Command[list[Send]]:
    """Node that runs a Cypher query."""
    lista_de_cypher_queries = [cypher_query for cypher_query in state["cypher_queries"]]
    print(f"lista_de_cypher_queries: {lista_de_cypher_queries}")
    sends = [
        Send(
            "run_cypher_query",
            {"cypher_query": query},
        )
        for query in lista_de_cypher_queries
    ]
    return Command(goto=sends)


async def run_cypher_query(state: Neo4jQueryState) -> Command[Literal[END]]:
    """Node that runs a Cypher query."""
    cypher_query = state["cypher_query"]
    print(f"################## cypher_query: {cypher_query}")
    results = str(safe_run_cypher(cypher_query))
    print(f"################## results: {results}")

    return Command(goto=END, update={"results": [results]})


builder = StateGraph(Neo4jQueryState)

builder.add_node("generate_questions", generate_questions)
builder.add_node(
    "generate_cypher_queries_in_parallel", generate_cypher_queries_in_parallel
)
builder.add_node("generate_cypher_query", generate_cypher_query)
builder.add_node("run_cypher_query_in_parallel", run_cypher_query_in_parallel)
builder.add_node("run_cypher_query", run_cypher_query)
builder.add_edge(START, "generate_questions")
graph = builder.compile()

async for chunk in graph.astream(
    {"question": "Todos los nombres de los enzimas"},
    stream_mode="updates",
    subgraphs=True,
    debug=True,
):
    pass


# %%
async def generate_cypher_query(
    state: Neo4jQueryState,
) -> Command[Literal["retrieve_in_parallel"]]:
    """Node that generates a Cypher query."""
    chain_for_cypher_query = system_prompt | llm.with_structured_output(CypherQuery)

    response = chain_for_cypher_query.invoke(
        {"input": "Todos los nombres de los enzimas"}
    )

    raw_query = response.cypher_query.strip()

    cypher_query = sanitise_query(raw_query)

    return Command(goto="retrieve_in_parallel", update={"cypher_query": cypher_query})


async def retrieve_in_parallel(state: Neo4jQueryState) -> Command[list[Send]]:
    """Node that retrieves documents in parallel."""
    lista_de_queries = [
        query.query_str for query in state["generated_questions"].queries_list
    ]
    print(f"lista_de_queries: {lista_de_queries}")
    sends = [
        Send(
            "metadata_filtering_and_hybrid_search_node",
            {"query": query, "num_pending": len(lista_de_queries)},
        )
        for query in lista_de_queries
    ]
    return Command(goto=sends)


# Ejecutar la consulta
results_or_error = safe_run_cypher(cypher_query)
for row in results_or_error:
    print(row)
# %%
