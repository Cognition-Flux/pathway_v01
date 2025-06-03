"""Tables analysis agent."""

# %%
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.graph import END
from langgraph.types import Command

from agentic_workflow.agents.agent_for_tables.pandas_agent import agent as tables_agent
from agentic_workflow.schemas import (
    PathwayGraphState,
)


load_dotenv(override=True)


def tables(
    state: PathwayGraphState,
) -> Command[Literal[END, "check_if_plan_is_done"]]:
    """Tables.

    This function analyzes data in tables using the chain_for_tables.
    """
    current_step = state["current_step"]
    prompt = (
        "Genera una respuesta muy detallada con datos cuantitativos. "
        "Debes entregar tablas extensas detallando los todos los datos."
        "SIEMPRE: muestra al menos 20 filas de cada tabla."
        "El requerimiento del usuario es: " + current_step.step + "\n\n\n\n"
        "Estos son resultados previos:" + state.get("tables_results", "")
    )
    print(f"-------------------------------------Prompt: {prompt}")
    tables_results = tables_agent.invoke(prompt)
    next_node = "check_if_plan_is_done"
    return Command(
        goto=next_node,
        update={
            "tables_results": tables_results["output"],
            "reasoning": [tables_results["output"]],
            "scratchpad": [
                AIMessage(content=tables_results["output"]),
            ],
            "current_agent": "tables",
            "next_node": next_node,
            "llm_model": "gpt-4.1-mini",  # pandas_agent uses get_llm() default model
        },
    )
