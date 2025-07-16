"""
This workflow is a parallel workflow that uses the chains_for_hybrid_search_and_metadata_filtering.py file to create a workflow that can be used to search for information in the vectorstore.
"""

# %%
import asyncio
import os
from typing import Annotated, Literal
from uuid import uuid4

import aiosqlite
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Send
from pinecone import PineconeApiException  # For catching filter errors
from pydantic import BaseModel, Field, ValidationError

from dev.langgraph_basics.chains_for_hybrid_search_and_metadata_filtering import (
    chain_for_filter_fields,
    chain_for_filter_generation,
)
from dev.langgraph_basics.simple_hybrid_search_w_metadata_filtering import retriever
from dev.langgraph_basics.simple_ReAct import (
    LastLLMResponse,
    reduce_docs,
)

load_dotenv(override=True)


# Create a Runnable that exposes both sync and async retrieval


def _sync_retrieve(query: str, filter: dict | None = None) -> list[Document]:
    """Synchronous wrapper around the Pinecone retriever invoke method."""
    if filter is None:
        return retriever.invoke(query)
    return retriever.invoke(query, filter=filter)


async def _async_retrieve(query: str, filter: dict | None = None) -> list[Document]:
    """Execute the synchronous ``retriever.invoke`` call in a thread so it can
    be awaited from asyncio code. This avoids the issues with the built-in
    ``retriever.ainvoke`` method, which currently does not accept a *filter*
    argument.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync_retrieve, query, filter)


# Expose the runnable with both sync and async behaviour.
async_retriever = RunnableLambda(_sync_retrieve, afunc=_async_retrieve)


class OneQuery(BaseModel):
    """One query."""

    query_str: str


class GeneratedQueries(BaseModel):
    """Generated queries."""

    queries_list: list[OneQuery]


class RetrievalGraphState(MessagesState):
    """State of the graph."""

    rag_input: Annotated[list[HumanMessage], add_messages] = Field(
        default_factory=lambda: [HumanMessage(content="")]
    )
    ai_generated_response: LastLLMResponse = Field(
        default_factory=lambda: LastLLMResponse(response="")
    )
    documents: Annotated[list[Document], reduce_docs] = Field(
        default_factory=lambda: []
    )
    generated_queries: GeneratedQueries = Field(
        default_factory=lambda: GeneratedQueries(queries_list=[])
    )
    query: str = Field(default_factory=lambda: "")


llm = AzureChatOpenAI(
    azure_deployment="gpt-4.1-mini",
    api_version=os.getenv("AZURE_API_VERSION"),
    temperature=0,
    max_tokens=None,
    timeout=1200,
    max_retries=5,
    streaming=True,
    api_key=os.getenv("AZURE_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)
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


async def generate_queries(
    state: RetrievalGraphState,
) -> Command[Literal["retrieve_in_parallel"]]:
    """Node that generates queries."""
    generated_queries = chain_for_generate_queries.invoke(
        {"user_question": state["rag_input"][-1].content}
    )
    return Command(
        goto="retrieve_in_parallel",
        update={"generated_queries": generated_queries},
    )


async def retrieve_in_parallel(state: RetrievalGraphState) -> Command[list[Send]]:
    """Node that retrieves documents in parallel."""
    lista_de_queries = [
        query.query_str for query in state["generated_queries"].queries_list
    ]
    print(f"lista_de_queries: {lista_de_queries}")
    sends = [
        Send(
            "metadata_filtering_and_hybrid_search_node",
            {"query": query},
        )
        for query in lista_de_queries
    ]
    return Command(goto=sends)


async def metadata_filtering_and_hybrid_search_node(
    state: RetrievalGraphState,
) -> Command[Literal["metadata_filtering_and_hybrid_search_node", END]]:
    """Node that extracts fields and generates the final metadata filter.

    If the structured output from the LLM cannot be parsed (for example, the
    returned JSON does not contain the required ``filter`` key), we gracefully
    fall back to an *empty* filter to avoid hard failures in the workflow.
    """

    query_str = state["query"]

    extracted_fields = await chain_for_filter_fields.ainvoke({"query": query_str})
    extracted_json = extracted_fields.model_dump_json(indent=2)

    # Try to generate the final filter with structured output.
    try:
        final_filter_obj = await chain_for_filter_generation.ainvoke(
            {
                "query": query_str,
                "extracted_fields": extracted_json,
            }
        )
        filter_dict = final_filter_obj.filter  # type: ignore[attr-defined]

        docs = await async_retriever.ainvoke(query_str, filter=filter_dict)
        if len(docs) == 0:
            # retry this node.
            return Command(goto="metadata_filtering_and_hybrid_search_node")

    except (ValidationError, PineconeApiException):
        # retry this node.
        return Command(goto="metadata_filtering_and_hybrid_search_node")
    print(f"query_str: {query_str}")
    for res in docs:
        score = res.metadata.get("score", "N/A")
        print(f"* [score: {score:.3f}] {res.page_content} [{res.metadata}]")
    # Keeping the top score document
    top_score_doc = docs[0]
    print(f"top_score_doc: {top_score_doc}")
    return Command(
        goto=END,
        update={"documents": [top_score_doc]},
    )


builder = StateGraph(RetrievalGraphState)
builder.add_node(
    "metadata_filtering_and_hybrid_search_node",
    metadata_filtering_and_hybrid_search_node,
)
# builder.add_node("hybrid_async_retriever", hybrid_async_retriever)
builder.add_node("retrieve_in_parallel", retrieve_in_parallel)
builder.add_node("generate_queries", generate_queries)
builder.add_edge(START, "generate_queries")


def get_memory():
    """Get a memory."""
    conn = aiosqlite.connect(":memory:")
    return AsyncSqliteSaver(conn=conn)


def get_graph():
    """Get a graph."""
    memory = get_memory()
    return builder.compile(checkpointer=memory, debug=True)


async def aget_next_state(
    compiled_graph: CompiledStateGraph, config: dict
) -> RetrievalGraphState:
    """Get the next state of the graph."""
    latest_checkpoint = await compiled_graph.aget_state(config=config)
    return latest_checkpoint.next


if __name__ == "__main__":
    USER_QUERY = "enzimas del ciclo de Krebs que usan NAD y son reversibles"
    thread_config = {"configurable": {"thread_id": str(uuid4())}}
    graph = get_graph()

    next_state = aget_next_state(graph, thread_config)
    print(f"graph state: {next_state}")
    async for chunk in graph.astream(
        {"rag_input": [HumanMessage(content=USER_QUERY)]},
        config=thread_config,
        stream_mode="updates",
        subgraphs=True,
    ):
        pass
