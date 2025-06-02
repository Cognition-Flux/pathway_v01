"""Report generation agent."""

# %%
from typing import Literal

from dotenv import load_dotenv
from langgraph.graph import END
from langgraph.types import Command

from agentic_workflow.schemas import (
    PathwayGraphState,
)


load_dotenv(override=True)


def report_generator(
    state: PathwayGraphState,
) -> Command[Literal[END]]:
    """Report generator.

    This function generates a report using the chain_for_report_generator.
    """
    return Command(goto=END, update={"messages": state["messages"]})
