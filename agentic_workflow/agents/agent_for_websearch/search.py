"""Websearch agent."""

# %%

from pathlib import Path
from typing import Literal

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.types import Command

from agentic_workflow.agents.agent_for_websearch.websearch_tool import (
    search_multiple_queries,
)
from agentic_workflow.llm_chains import chain_for_queries_to_websearch
from agentic_workflow.schemas import PathwayGraphState


with Path("agentic_workflow/prompts/system_prompts.yaml").open("r") as f:
    prompts = yaml.safe_load(f)


def websearch(
    state: PathwayGraphState,
) -> Command[Literal[END, "check_if_plan_is_done"]]:
    """Websearch.

    This function conducts a websearch using the chain_for_websearch.
    """
    # Pass only the text of the current step to the LLM, not the full `OneStep` object.
    # The model only needs the human-readable instruction contained in `current_step.step`.
    current_step = state["current_step"]
    queries = chain_for_queries_to_websearch.invoke(
        [
            SystemMessage(content=prompts["queries_to_websearch"]),
            HumanMessage(content=current_step.step),
        ]
    )

    web_search_results = search_multiple_queries(queries.queries)

    # Crear una muestra corta de resultados (primeros 100 caracteres) para el razonamiento
    sample_snippets = []
    for item in web_search_results:
        # Cada item puede ser dict con 'results' o un dict directo con 'content'
        # Intentamos extraer el contenido textual más representativo
        if isinstance(item, dict):
            # Si viene del MultiQuery util probablemente tiene 'results'
            if (
                "results" in item
                and isinstance(item["results"], list)
                and item["results"]
            ):
                # Tomamos el contenido del primer resultado de esa consulta
                first_res = item["results"][0]
                text = (
                    first_res.get("content") or first_res.get("title") or str(first_res)
                )
            else:
                # Un único resultado directo
                text = item.get("content") or item.get("title") or str(item)
        else:
            text = str(item)

        sample_snippets.append(text[:100])

    # Dar formato markdown en lista con viñetas
    sample_preview = "\n".join(f"- {s}" for s in sample_snippets[:3])

    next_node = "check_if_plan_is_done"
    return Command(
        goto=next_node,
        update={
            "web_search_results": web_search_results,
            "messages": [
                f"Se encontraron {len(web_search_results)} resultados de búsqueda web."
            ],
            "reasoning": [
                f"Muestras de resultados de búsqueda web (primeros 100 caracteres):\n{sample_preview}",
                *(
                    [f"proximo agente: {next_node.replace('_', ' ').title()}"]
                    if isinstance(next_node, str) and next_node != END
                    else []
                ),
            ],
            "current_agent": "websearch",
            "next_node": next_node,
            "llm_model": "gpt-4.1-mini",  #
        },
    )
