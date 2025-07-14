"""
This workflow is a parallel workflow that uses the chains_for_hybrid_search_and_metadata_filtering.py file to create a workflow that can be used to search for information in the vectorstore.
"""

# %%

import asyncio
from typing import Literal

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END
from langgraph.types import Command

from dev.langgraph_basics.chains_for_hybrid_search_and_metadata_filtering import (
    chain_for_filter_fields,
    chain_for_filter_generation,
)
from dev.langgraph_basics.simple_hybrid_search_w_metadata_filtering import retriever
from dev.langgraph_basics.simple_ReAct import HumanLastQuestion, State

load_dotenv(override=True)
# import chains

# %%


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


async def metadata_filtering_node(state: State) -> Command[Literal[END]]:
    """Node that adds a new message to the state."""
    extracted_fields = await chain_for_filter_fields.ainvoke(
        {"query": state["human_messages"][-1].content}
    )
    extracted_json = extracted_fields.model_dump_json(indent=2)
    final_filter_obj = await chain_for_filter_generation.ainvoke(
        {
            "query": state["human_messages"][-1].content,
            "extracted_fields": extracted_json,
        }
    )

    return Command(
        goto=END,
        update={
            "human_last_question": HumanLastQuestion(
                last_question=state["human_messages"][-1].content
            ),
            "tool_messages": [ToolMessage(content=final_filter_obj.filter)],
        },
    )


async def hybrid_search_node(state: State) -> Command[Literal[END]]:
    """Node that performs a hybrid search using the async retriever."""
    query = state["human_messages"][-1].content
    # Retrieve top documents asynchronously
    results = await async_retriever.ainvoke(query)
    return Command(
        goto=END,
        update={"ai_messages": [AIMessage(content=str(results))]},
    )


USER_QUERY = "enzimas del ciclo de Krebs que usan NAD y son reversibles"
