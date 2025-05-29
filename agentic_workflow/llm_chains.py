"""llm_chains.py.

This module contains the LLM chains for the agentic workflow.
"""

# %%
from pathlib import Path
from typing import Any

import yaml
from langchain_core.prompts import ChatPromptTemplate

from agentic_workflow.fewshot.retriever import retriever
from agentic_workflow.schemas import PlanRespond, TemporalSeriesChecker
from agentic_workflow.utils import get_llm


with Path("agentic_workflow/prompts/system_prompts.yaml").open("r") as f:
    prompts = yaml.safe_load(f)


def get_planning_chain():
    """get_planning_chain.

    Creates a planning chain that uses fewshot examples to determine whether
    to plan or respond directly.

    Returns:
        A chain that processes user input and returns a structured PlanRespond
        object.
    """

    def format_prompt_with_fewshot(
        input_vars: dict[str, Any], k_top_examples: int = 3
    ) -> dict[str, Any]:
        """Formats the prompt with relevant examples based on the user's query."""
        # Extract the last message from the inpur messages
        query = input_vars["input"][-1].content
        # Get top 2 relevant examples
        top_examples = retriever.invoke(query)[:k_top_examples]
        examples_str = "\n".join([doc.page_content for doc in top_examples])

        # Create a new variables dict with the examples added
        formatted_vars = input_vars.copy()
        formatted_vars["examples"] = examples_str

        return formatted_vars

    prompt_for_planning = ChatPromptTemplate.from_messages(
        [("system", prompts["planner"]), ("human", "{input}")],
    )

    # Create the chain with example retrieval
    planning_chain = (
        format_prompt_with_fewshot
        | prompt_for_planning
        | get_llm(provider="azure", model="gpt-4.1-mini").with_structured_output(
            PlanRespond
        )
    )

    return planning_chain


# Initialize the planning chain
chain_for_planning = get_planning_chain()


temporal_info_series_checker_prompt = ChatPromptTemplate.from_messages(
    [("system", prompts["temporal_info_series_checker"]), ("human", "{input}")]
)

chain_for_temporal_series_info = temporal_info_series_checker_prompt | get_llm(
    provider="anthropic", model="claude-sonnet-4-20250514"
).with_structured_output(TemporalSeriesChecker)


ask_for_temporal_series_information_prompt = ChatPromptTemplate.from_messages(
    [("system", prompts["ask_for_temporal_series_information"]), ("human", "{input}")]
)

chain_for_ask_for_temporal_series_information = (
    ask_for_temporal_series_information_prompt
    | get_llm(provider="anthropic", model="claude-sonnet-4-20250514")
)

# Example usage
if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    temporal_series_parameters = chain_for_temporal_series_info.invoke(
        {"input": [HumanMessage(content="forecast de la serie temporal biomarcador")]}
    )
    print(temporal_series_parameters)
    missing_fields = []
    present_fields = []
    if not temporal_series_parameters.nombre_de_la_serie_temporal:
        missing_fields.append("nombre de la serie temporal")
    else:
        present_fields.append(
            f"nombre de la serie temporal: "
            f"{temporal_series_parameters.nombre_de_la_serie_temporal}"
        )
    if not temporal_series_parameters.nombre_de_la_tabla:
        missing_fields.append("nombre de la tabla")
    else:
        present_fields.append(
            f"nombre de la tabla: {temporal_series_parameters.nombre_de_la_tabla}"
        )
    if not temporal_series_parameters.ventana_contexto:
        missing_fields.append("ventana de contexto")
    else:
        present_fields.append(
            f"ventana de contexto: {temporal_series_parameters.ventana_contexto}"
        )
    if not temporal_series_parameters.ventana_prediccion:
        missing_fields.append("ventana de predicción")
    else:
        present_fields.append(
            f"ventana de predicción: {temporal_series_parameters.ventana_prediccion}"
        )

    missing_fields_str_parts = []
    if missing_fields:
        missing_fields_str_parts.append(
            f"Faltan los siguientes campos: {', '.join(missing_fields)}"
        )
    if present_fields:
        missing_fields_str_parts.append(
            f"Campos presentes: {', '.join(present_fields)}"
        )

    if not missing_fields_str_parts:
        current_fields_state = (
            "Todos los campos requeridos están presentes y completos."
        )
    else:
        current_fields_state = ". ".join(missing_fields_str_parts)

    ask_for_temporal_series_information = (
        chain_for_ask_for_temporal_series_information.invoke(
            {"input": "sigue las instrucciones", "fields_state": current_fields_state}
        )
    )
    print(ask_for_temporal_series_information)
