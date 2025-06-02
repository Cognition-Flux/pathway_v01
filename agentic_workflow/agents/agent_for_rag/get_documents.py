"""RAG agent."""

# %%
import logging
from typing import Literal

from langgraph.graph import END
from langgraph.types import Command

from agentic_workflow.agents.agent_for_rag.multi_query_retriever import (
    MultiRetrieverGraph,
)
from agentic_workflow.schemas import PathwayGraphState


logger = logging.getLogger(__name__)


def rag(
    state: PathwayGraphState,
) -> Command[Literal[END, "check_if_plan_is_done"]]:
    """RAG.

    This function conducts a RAG using the chain_for_rag.
    """
    current_step = state["current_step"]
    # print(f"RAG: {current_step.step=}")
    next_node = "check_if_plan_is_done"

    docs = MultiRetrieverGraph.invoke({"retrieval_query": current_step.step})
    # print(f"RAG: {len(docs)=}")
    return Command(
        goto=next_node,
        update={
            "documents": docs["documents"],
            "messages": [f"RAG recuperó {len(docs['documents'])} documentos."],
            "current_agent": "rag",
            "next_node": next_node,
            "llm_model": "MultiRetrieverGraph",  # Uses specialized RAG retrieval system
        },
    )
