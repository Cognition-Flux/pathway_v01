# %%

import logging
from pathlib import Path
from typing import Literal

import yaml
from langgraph.graph import END
from langgraph.types import Command, interrupt

from agentic_workflow.agents.agent_for_visualizations.plotly_builder.plot_generator import (
    ChatHistory,
    instantiate_model_with_prompt_and_plotly_schema,
    llm_json_to_plot_from_text,
)
from agentic_workflow.llm_chains import chain_for_ask_what_to_plot
from agentic_workflow.schemas import PathwayGraphState


logger = logging.getLogger(__name__)

with Path("agentic_workflow/prompts/system_prompts.yaml").open("r") as f:
    prompts = yaml.safe_load(f)


def ask_if_plot_is_needed(
    state: PathwayGraphState,
) -> Command[Literal["plot_generator", END]]:
    """Ask if plot is needed.

    This function asks if the plot is needed.
    """
    if_plot_is_needed = interrupt(
        "¿Necesitas visualizar los resultados en un gráfico? (si/no)",
    )
    affirmative_responses = [
        "si",
        "sí",
        "yes",
        "y",
        "ok",
        "okay",
    ]
    if if_plot_is_needed.strip().lower() in affirmative_responses:
        return Command(
            goto="plot_generator",
            update={
                "current_agent": "ask_if_plot_is_needed",
                "next_node": "plot_generator",
                "llm_model": "logic",  # Pure logic, no LLM used
            },
        )
    else:
        return Command(
            goto=END,
            update={
                "messages": ["Okey, puedes seguir investigando."],
                "current_agent": "ask_if_plot_is_needed",
                "next_node": END,
                "llm_model": "logic",  # Pure logic, no LLM used
            },
        )


def plot_generator(
    state: PathwayGraphState,
) -> Command[Literal["ask_for_plot_edits"]]:
    """Plot generator.

    This function generates a plot using the chain_for_plot_generator.
    """
    # Si no existe ``plot`` en el estado significa que es la **primera generación**
    check_if_first_plot = state.get("plot", None) is None
    if check_if_first_plot:
        print("Generando primer gráfico")

        ask_what_to_plot = chain_for_ask_what_to_plot.invoke(
            {
                "input": "\n".join(
                    [
                        f"{m.type.capitalize()}: {m.content}"
                        for m in state["messages"]
                        if m.type != "tool"
                    ]
                )
            }
        )
        what_to_plot = interrupt(ask_what_to_plot.content)

        plot_generator_prompt = prompts["plot_generator"].format(
            current_plot=None,
            question=f"AI assistant: {ask_what_to_plot.content}"
            + "\n\n\n\n"
            + f"Usuario: {what_to_plot}",
            context=state.get("report") or state.get("tables_results", ""),
            tables_results=state.get("tables_results", None),
            scratchpad=state["scratchpad"],
        )
        chat_history_plot_gen = ChatHistory()
        llm_plot_gen = instantiate_model_with_prompt_and_plotly_schema()
        fig = llm_json_to_plot_from_text(
            input_instructions=plot_generator_prompt,
            model_with_structure_and_prompt=llm_plot_gen,
            chat_history=chat_history_plot_gen,
        )
        return Command(
            goto="ask_for_plot_edits",
            update={
                "plot": fig.to_json(),
                "reasoning": [
                    f"proximo agente: {'ask_for_plot_edits'.replace('_', ' ').title()}"
                ],
                "current_agent": "plot_generator",
                "next_node": "ask_for_plot_edits",
                "llm_model": "gpt-4.1",  # Uses specialized plotly model
            },
        )
    else:
        print("editando gráfico")
        plot_edits = interrupt("¿Qué deseas editar respecto al gráfico?")
        plot_generator_prompt = prompts["plot_generator"].format(
            current_plot=state["plot"],
            question=plot_edits,
            context=state.get("report") or state.get("tables_results", ""),
            tables_results=state.get("tables_results", None),
            scratchpad=state["scratchpad"],
        )

        chat_history_plot_gen = ChatHistory()
        llm_plot_gen = instantiate_model_with_prompt_and_plotly_schema()
        # Informar al usuario antes de la operación costosa
        updating_message = "Voy a actualizar el gráfico, espera un momento..."

        # Generar nuevo gráfico
        fig = llm_json_to_plot_from_text(
            input_instructions=plot_generator_prompt,
            model_with_structure_and_prompt=llm_plot_gen,
            chat_history=chat_history_plot_gen,
        )
        return Command(
            goto="ask_for_plot_edits",
            update={
                "plot": fig.to_json(),
                "messages": [updating_message],
                "reasoning": [
                    f"proximo agente: {'ask_for_plot_edits'.replace('_', ' ').title()}"
                ],
                "current_agent": "plot_generator",
                "next_node": "ask_for_plot_edits",
                "llm_model": "gpt-4.1",  # Uses specialized plotly model
            },
        )


def ask_for_plot_edits(
    state: PathwayGraphState,
) -> Command[Literal[END, "plot_generator"]]:
    """Ask for plot edits.

    This function asks for plot edits.
    """
    plot_edits = interrupt(
        "¿Necesitas editar el gráfico? (si/no)",
    )
    if plot_edits.strip().lower() == "si":
        return Command(
            goto="plot_generator",
            update={
                "reasoning": [
                    f"proximo agente: {'plot_generator'.replace('_', ' ').title()}"
                ],
                "current_agent": "ask_for_plot_edits",
                "next_node": "plot_generator",
                "llm_model": "logic",  # Pure logic, no LLM used
            },
        )
    else:
        return Command(
            goto=END,
            update={
                "messages": ["Okey, puedes seguir investigando."],
                "reasoning": [f"proximo agente: {'END'.replace('_', ' ').title()}"],
                "current_agent": "ask_for_plot_edits",
                "next_node": END,
                "llm_model": "logic",  # Pure logic, no LLM used
                "plot": None,
            },
        )


# %%
