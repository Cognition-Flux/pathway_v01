"""Graph definition for the multiturn conversational agent."""

# %%
import sqlite3
import uuid
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.func import task
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command, interrupt

from agentic_workflow.llm_chains import (
    chain_for_ask_for_temporal_series_information,
    chain_for_planning,
    chain_for_temporal_series_info,
)
from agentic_workflow.schemas import PathwayGraphState
from agentic_workflow.utils import get_llm


load_dotenv(override=True)
with Path("agentic_workflow/prompts/system_prompts.yaml").open("r") as f:
    prompts = yaml.safe_load(f)


@tool
def string_to_uuid(input_string: str) -> str:
    """Convert a string to a UUID5."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, input_string))


@tool(return_direct=True)
def transfer_to_hotel_advisor():
    """Ask hotel advisor agent for help."""
    return "Successfully transferred to hotel advisor"


# define an agent
travel_advisor_tools = [transfer_to_hotel_advisor]
travel_advisor = create_react_agent(get_llm(), travel_advisor_tools)


# define a task that calls an agent
@task
def call_travel_advisor(messages: list) -> list:
    """Call the travel advisor agent with the given messages."""
    response = travel_advisor.invoke({"messages": messages})
    return response["messages"]


def multi_turn_planner(
    state: PathwayGraphState,
) -> Command[Literal["conduct_plan"]]:
    """Entrypoint for the multiturn conversational graph."""
    while True:
        response = chain_for_planning.invoke({"input": state["messages"]})
        if response.what_to_do != "plan":
            user_input = interrupt(value=response.response)
            state["messages"] = add_messages(state["messages"], [user_input])
            continue
        elif response.what_to_do == "plan":
            next_node = "conduct_plan"
            return Command(
                goto=next_node,
                update={
                    "plan_respond": response if next_node == "conduct_plan" else None,
                    "steps": response.steps if next_node == "conduct_plan" else None,
                    "messages": [response.response]
                    if next_node != "conduct_plan"
                    else [
                        response.response,
                        "Ejecutaré las siguientes acciones:",
                        *[f"- {s.step}" for idx, s in enumerate(response.steps)],
                    ],
                    "user_question": state["messages"][-1].content,
                    "reasoning": [
                        response.reasoning,
                        *(
                            [f"proximo agente: {next_node.replace('_', ' ').title()}"]
                            if isinstance(next_node, str) and next_node != END
                            else []
                        ),
                    ],
                    "current_agent": "multi_turn_planner",
                    "next_node": next_node,
                    "llm_model": "gpt-4.1-mini",
                },
            )
        else:
            raise ValueError(
                f"multi_turn_planner: Expected valid response.what_to_do, got "
                f"'{response.what_to_do=}'"
            )


def conduct_plan(
    state: PathwayGraphState,
) -> Command[
    Literal[
        "forecast_information_checker",
        END,
    ]
]:
    """Conduct plan.

    This function conducts a plan using the chain_for_planning.
    """
    current_step = state["steps"][0]

    if current_step.agent == "forecast_information_checker":
        next_node = "forecast_information_checker"
    else:
        raise ValueError(f"conduct_plan: Expected agent, got '{current_step.agent}'")

    return Command(
        goto=next_node,
        update={
            "steps": state["steps"][1:],
            "current_step": current_step,
            "messages": [f"Ejecutando: {current_step.step}"],
            "reasoning": [
                current_step.reasoning,
                *(
                    [f"proximo agente: {next_node.replace('_', ' ').title()}"]
                    if isinstance(next_node, str) and next_node != END
                    else []
                ),
            ],
            "current_agent": "conduct_plan",
            "next_node": next_node,
            "llm_model": "logic",
            "user_parameters_for_forecast": state["messages"]
            if next_node == "forecast_information_checker"
            else None,
        },
    )


def forecast_information_checker(state: PathwayGraphState) -> Command[Literal[END]]:
    """Conduct the plan."""
    print(f"user_parameters_for_forecast: {state['user_parameters_for_forecast']}")
    print(f"temporal_series_info: {state.get('temporal_series_info', None)}")

    if state["user_parameters_for_forecast"]:
        temporal_series_parameters = chain_for_temporal_series_info.invoke(
            {
                "input": state["user_parameters_for_forecast"]
                + [
                    f"Si la siguiente información existe, usa esto para completar "
                    f"los campos: {state.get('temporal_series_info', None)}"
                ]
            }
        )
    else:
        forcast_update = interrupt(
            f"Confirma que parámetros del forecast quieres cambiar: \n"
            f"{state.get('temporal_series_info', None)}"
        )
        temporal_series_parameters = chain_for_temporal_series_info.invoke(
            {
                "input": f"Estos son los parámetros anteriores "
                f"{state.get('temporal_series_info', None)},  debe aplicar estos "
                f"cambios o actualizar los campos (solo si existen): {forcast_update}"
            }
        )

    print(f"temporal_series_parameters: {temporal_series_parameters}")
    if (
        temporal_series_parameters.nombre_de_la_serie_temporal
        and temporal_series_parameters.nombre_de_la_tabla
        and temporal_series_parameters.ventana_contexto
        and temporal_series_parameters.ventana_prediccion
    ):
        next_node = "extract_temporal_series_from_tables"
        return Command(
            goto="extract_temporal_series_from_tables",
            update={
                "temporal_series_info": temporal_series_parameters,
                "messages": [
                    (
                        "✅ Información completa verificada: Serie "
                        f"'{temporal_series_parameters.nombre_de_la_serie_temporal}' "
                        f"de tabla "
                        f"'{temporal_series_parameters.nombre_de_la_tabla}' "
                        f"con ventana contexto "
                        f"{temporal_series_parameters.ventana_contexto} "
                        f"y predicción "
                        f"{temporal_series_parameters.ventana_prediccion}"
                    )
                ],
                "user_parameters_for_forecast": [
                    (
                        "✅ Información completa verificada: Serie "
                        f"'{temporal_series_parameters.nombre_de_la_serie_temporal}' "
                        f"de tabla "
                        f"'{temporal_series_parameters.nombre_de_la_tabla}' "
                        f"con ventana contexto "
                        f"{temporal_series_parameters.ventana_contexto} "
                        f"y predicción "
                        f"{temporal_series_parameters.ventana_prediccion}"
                    )
                ],
                "reasoning": [
                    (
                        "✅ Verificación completa de parámetros para forecast:"
                        f"• Serie temporal identificada:"
                        f"{temporal_series_parameters.nombre_de_la_serie_temporal}"
                    ),
                    (
                        f"• Fuente de datos: tabla "
                        f"{temporal_series_parameters.nombre_de_la_tabla}"
                    ),
                    (
                        f"• Lag temporal: "
                        f"{temporal_series_parameters.ventana_contexto} "
                        f"puntos de contexto histórico"
                    ),
                    (
                        f"• Horizonte de predicción: "
                        f"{temporal_series_parameters.ventana_prediccion} "
                        f"puntos futuros"
                    ),
                    (
                        "• Estado: Todos los parámetros requeridos están presentes y "
                        "validados"
                    ),
                    ("• Siguiente paso: Proceder con extracción de datos temporales"),
                    f"proximo agente: {next_node.replace('_', ' ').title()}",
                ],
                "current_agent": "forecast_information_checker",
                "next_node": next_node,
                "llm_model": "claude-sonnet-4",
            },
        )
    else:
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
                f"ventana de predicción: "
                f"{temporal_series_parameters.ventana_prediccion}"
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

            ask_message = chain_for_ask_for_temporal_series_information.invoke(
                {"input": state["messages"], "fields_state": current_fields_state}
            )
        user_response = interrupt(ask_message.content)
        temporal_series_parameters = chain_for_temporal_series_info.invoke(
            {
                "input": [
                    user_response,
                    f"Si existe complementa con la siguiente información: "
                    f"{state.get('temporal_series_info', None)}",
                ]
            }
        )

        print(f"temporal_series_parameters: {temporal_series_parameters}")
        return Command(
            goto="forecast_information_checker",
            update={
                "temporal_series_info": temporal_series_parameters,
            },
        )


builder = StateGraph(PathwayGraphState)
builder.add_node("planner", multi_turn_planner)
builder.add_node("forecast_information_checker", forecast_information_checker)
builder.add_node("conduct_plan", conduct_plan)

builder.add_edge(START, "planner")


conn = sqlite3.connect(":memory:", check_same_thread=False)
memory = SqliteSaver(conn)
multi_turn_graph = builder.compile(memory)


if __name__ == "__main__":
    thread_config = {"configurable": {"thread_id": uuid.uuid4()}}

    mensaje_del_usuario = (
        "necesito un forecast de la variable biomarcador"
        # "si"
        # "no"
    )
    for chunk in multi_turn_graph.stream(
        (
            {"messages": mensaje_del_usuario}
            if not multi_turn_graph.get_state(thread_config).next
            else Command(resume=mensaje_del_usuario)
        ),
        config=thread_config,
        stream_mode="updates",
        subgraphs=True,
    ):
        print(chunk)
