"""
This workflow is a parallel workflow that uses the chains_for_hybrid_search_and_metadata_filtering.py file to create a workflow that can be used to search for information in the vectorstore.
"""

# %%

import ast
import asyncio
from typing import Literal
from uuid import uuid4

import aiosqlite
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Send
from pinecone import PineconeApiException  # For catching filter errors
from pydantic import ValidationError

from dev.langgraph_basics.chains_for_hybrid_search_and_metadata_filtering import (
    chain_for_filter_fields,
    chain_for_filter_generation,
)
from dev.langgraph_basics.simple_hybrid_search_w_metadata_filtering import retriever
from dev.langgraph_basics.simple_ReAct import HumanLastQuestion, State

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


async def metadata_filtering_node(
    state: State,
) -> Command[Literal["retrieve_in_parallel"]]:
    """Node that extracts fields and generates the final metadata filter.

    If the structured output from the LLM cannot be parsed (for example, the
    returned JSON does not contain the required ``filter`` key), we gracefully
    fall back to an *empty* filter to avoid hard failures in the workflow.
    """
    extracted_fields = await chain_for_filter_fields.ainvoke(
        {"query": state["human_messages"][-1].content}
    )
    extracted_json = extracted_fields.model_dump_json(indent=2)

    # Try to generate the final filter with structured output.
    try:
        final_filter_obj = await chain_for_filter_generation.ainvoke(
            {
                "query": state["human_messages"][-1].content,
                "extracted_fields": extracted_json,
            }
        )
        filter_dict = final_filter_obj.filter  # type: ignore[attr-defined]
    except ValidationError:
        # If parsing fails, ask the workflow to retry this node.
        return Command(goto="metadata_filtering_node")

    return Command(
        goto="retrieve_in_parallel",
        update={
            "human_last_question": HumanLastQuestion(
                last_question=state["human_messages"][-1].content
            ),
            "tool_messages": [
                ToolMessage(
                    content=filter_dict,  # type: ignore[arg-type]
                    tool_call_id=str(uuid4()),
                )
            ],
        },
    )


async def retrieve_in_parallel(state: State) -> Command[list[Send]]:
    """Node that retrieves documents in parallel."""
    lista_de_queries = [
        # state["human_last_question"].last_question,
        # state["human_last_question"].last_question,
        "Succinate semialdehyde",
        "Isocitrate",
        "Malate",
    ]
    sends = [
        Send(
            "hybrid_async_retriever",
            State(
                human_last_question=HumanLastQuestion(last_question=query),
                tool_messages=[
                    ToolMessage(
                        content=state["tool_messages"][-1].content,
                        tool_call_id=str(uuid4()),
                    )
                ],
            ),
        )
        for query in lista_de_queries
    ]
    return Command(goto=sends)
    # return sends


async def hybrid_async_retriever(
    state: State,
) -> Command[Literal[END, "metadata_filtering_node"]]:
    """Node that performs a hybrid search using the async retriever."""
    query_str = state["human_last_question"].last_question
    print(f"filter_dict_str: {state['tool_messages'][-1].content}")
    filter_dict_str = state["tool_messages"][-1].content
    filter_dict = ast.literal_eval(filter_dict_str)  # ahora es un dict
    # Retrieve top documents asynchronously
    try:
        docs = await async_retriever.ainvoke(query_str, filter=filter_dict)
    except PineconeApiException:
        # If Pinecone rejects the filter, retry without it.
        return Command(goto="metadata_filtering_node", update={})
    if len(docs) == 0:
        return Command(goto="metadata_filtering_node", update={})

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


builder = StateGraph(State)
builder.add_node("metadata_filtering_node", metadata_filtering_node)
builder.add_node("hybrid_async_retriever", hybrid_async_retriever)
builder.add_node("retrieve_in_parallel", retrieve_in_parallel)
builder.add_edge(START, "metadata_filtering_node")


def get_memory():
    """Get a memory."""
    conn = aiosqlite.connect(":memory:")
    return AsyncSqliteSaver(conn=conn)


def get_graph():
    """Get a graph."""
    memory = get_memory()
    return builder.compile(checkpointer=memory, debug=True)


async def aget_next_state(compiled_graph: CompiledStateGraph, config: dict) -> State:
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
        {"human_messages": [HumanMessage(content=USER_QUERY)]},
        config=thread_config,
        stream_mode="updates",
        subgraphs=True,
    ):
        pass
