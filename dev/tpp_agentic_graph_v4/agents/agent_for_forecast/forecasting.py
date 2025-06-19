"""Forecasting agent: validates and processes temporal series forecasting parameters."""

# %%
import os
import sys
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Command, interrupt

# pylint: disable=import-error,wrong-import-position

MODULE_NAME = os.getenv("AGENTS_MODULE")
ALSO_MODULE = True
if ALSO_MODULE:
    current_file = Path(__file__).resolve()
    package_root = current_file.parent
    while package_root.name != MODULE_NAME:
        package_root = package_root.parent

    if str(package_root.parent) not in sys.path:
        sys.path.insert(0, str(package_root.parent))
os.chdir(package_root)

load_dotenv(override=True)

from tpp_agentic_graph_v4.llm_chains import (  # noqa: E402
    chain_for_ask_for_temporal_series_information,
    chain_for_if_another_forecast_is_needed,
    chain_for_temporal_series_info,
)
from tpp_agentic_graph_v4.schemas import PathwayGraphState  # noqa: E402
from tpp_agentic_graph_v4.tools.get_moving_avg_forecast import (  # noqa: E402
    get_forecast_moving_avg,
)


def forecast_information_checker(
    state: PathwayGraphState,
) -> Command[Literal["make_forecast"]]:
    """Valida y normaliza los parámetros de entrada usando `TemporalSeriesChecker`."""

    print(f"user_parameters_for_forecast: {state['user_parameters_for_forecast']}")
    print(f"temporal_series_info: {state.get('temporal_series_info', None)}")

    # Paso 1 ─ Normalizar entrada usando la *chain* dedicada
    if state["user_parameters_for_forecast"]:
        temporal_series_parameters = chain_for_temporal_series_info.invoke(
            {
                "input": state["user_parameters_for_forecast"]
                + [
                    "Si la siguiente información existe, úsala para completar los campos: "
                    f"{state.get('temporal_series_info', None)}",
                ]
            }
        )
    else:
        forecast_update = interrupt(
            "Confirma qué parámetros del forecast deseas cambiar:\n"
            f"{state.get('temporal_series_info', None)}"
        )
        temporal_series_parameters = chain_for_temporal_series_info.invoke(
            {
                "input": (
                    f"Estos son los parámetros anteriores {state.get('temporal_series_info', None)}. "
                    "Debes aplicar estos cambios o actualizar los campos (solo si existen): "
                    f"{forecast_update}"
                )
            }
        )

    print(f"temporal_series_parameters: {temporal_series_parameters}")

    # Paso 2 ─ Verificar completitud
    if (
        temporal_series_parameters.nombre_de_la_oficina
        and temporal_series_parameters.fecha_del_dia_de_hoy
        and temporal_series_parameters.fecha_inicio_de_la_proyeccion
        and temporal_series_parameters.numero_de_dias_a_proyectar is not None
    ):
        next_node = "make_forecast"
        confirmation_msg = (
            "✅ Información completa verificada: "
            f"Oficina '{temporal_series_parameters.nombre_de_la_oficina}' | "
            f"Hoy {temporal_series_parameters.fecha_del_dia_de_hoy} | "
            f"Proyección desde {temporal_series_parameters.fecha_inicio_de_la_proyeccion} "
            f"por {temporal_series_parameters.numero_de_dias_a_proyectar} días."
        )
        return Command(
            goto=next_node,
            update={
                "temporal_series_info": temporal_series_parameters,
                "messages": [confirmation_msg],
                "user_parameters_for_forecast": [confirmation_msg],
                "reasoning": [
                    "✅ Parámetros de forecast verificados:",
                    f"• Oficina: {temporal_series_parameters.nombre_de_la_oficina}",
                    f"• Fecha (hoy): {temporal_series_parameters.fecha_del_dia_de_hoy}",
                    (
                        "• Fecha de inicio de proyección: "
                        f"{temporal_series_parameters.fecha_inicio_de_la_proyeccion}"
                    ),
                    (
                        "• Número de días a proyectar: "
                        f"{temporal_series_parameters.numero_de_dias_a_proyectar}"
                    ),
                    "• Estado: Todos los parámetros requeridos están presentes y validados.",
                    "• Siguiente paso: Generar forecast mediante media móvil.",
                    "proximo agente: Make Forecast",
                ],
                "current_agent": "forecast_information_checker",
                "next_node": next_node,
                "llm_model": "claude-sonnet-4",
            },
        )

    # Paso 3 ─ Construir lista de campos faltantes / presentes
    missing_fields: list[str] = []
    present_fields: list[str] = []

    if not temporal_series_parameters.nombre_de_la_oficina:
        missing_fields.append("nombre de la oficina")
    else:
        present_fields.append(
            f"nombre de la oficina: {temporal_series_parameters.nombre_de_la_oficina}"
        )

    if not temporal_series_parameters.fecha_del_dia_de_hoy:
        missing_fields.append("fecha del día de hoy")
    else:
        present_fields.append(
            f"fecha del día de hoy: {temporal_series_parameters.fecha_del_dia_de_hoy}"
        )

    if not temporal_series_parameters.fecha_inicio_de_la_proyeccion:
        missing_fields.append("fecha inicio de la proyección")
    else:
        present_fields.append(
            "fecha inicio de la proyección: "
            f"{temporal_series_parameters.fecha_inicio_de_la_proyeccion}"
        )

    if temporal_series_parameters.numero_de_dias_a_proyectar is None:
        missing_fields.append("número de días a proyectar")
    else:
        present_fields.append(
            "número de días a proyectar: "
            f"{temporal_series_parameters.numero_de_dias_a_proyectar}"
        )

    missing_fields_str_parts: list[str] = []
    if missing_fields:
        missing_fields_str_parts.append(
            f"Faltan los siguientes campos: {', '.join(missing_fields)}"
        )
    if present_fields:
        missing_fields_str_parts.append(
            f"Campos presentes: {', '.join(present_fields)}"
        )

    current_fields_state = ". ".join(missing_fields_str_parts)

    # Paso 4 ─ Solicitar información faltante al usuario
    ask_message = chain_for_ask_for_temporal_series_information.invoke(
        {"input": state["messages"], "fields_state": current_fields_state}
    )
    user_response = interrupt(ask_message.content)
    temporal_series_parameters = chain_for_temporal_series_info.invoke(
        {
            "input": [
                user_response,
                "Si existe, complementa con la siguiente información: "
                f"{state.get('temporal_series_info', None)}",
            ]
        }
    )

    return Command(
        goto="forecast_information_checker",
        update={"temporal_series_info": temporal_series_parameters},
    )


def make_forecast(
    state: PathwayGraphState,
) -> Command[Literal["ask_if_another_forecast_is_needed"]]:
    """Genera el pronóstico utilizando la herramienta de media móvil."""

    params = state["temporal_series_info"]

    office_name = params.nombre_de_la_oficina
    today_date_str = params.fecha_del_dia_de_hoy
    start_date = params.fecha_inicio_de_la_proyeccion
    num_days = params.numero_de_dias_a_proyectar

    try:
        forecast_txt: str = get_forecast_moving_avg.invoke(
            {
                "office_name": office_name,
                "today_date_str": today_date_str,
                "start_date": start_date,
                "num_days": num_days,
            }
        )

        next_node = "ask_if_another_forecast_is_needed"
        return Command(
            goto=next_node,
            update={
                "forecast_generated_response": forecast_txt,
                "messages": [forecast_txt],
                "scratchpad": [AIMessage(content=forecast_txt)],
                "reasoning": [
                    "🎯 Forecast generado correctamente usando media móvil y simulación.",
                    "proximo agente: Check If Plan Is Done",
                ],
                "current_agent": "make_forecast",
                "next_node": next_node,
                "llm_model": "moving-average",
            },
        )
    except Exception as e:  # pylint: disable=broad-except
        next_node = "ask_if_another_forecast_is_needed"
        return Command(
            goto=next_node,
            update={
                "messages": [f"❌ Error al generar forecast: {e}"],
                "reasoning": [
                    "⚠️ Se produjo un error al ejecutar el forecast.",
                    "proximo agente: Check If Plan Is Done",
                ],
                "current_agent": "make_forecast",
                "next_node": next_node,
                "llm_model": "moving-average",
            },
        )


def ask_if_another_forecast_is_needed(
    state: PathwayGraphState,
) -> Command[Literal["check_if_plan_is_done", "forecast_information_checker"]]:
    """Ask if forecast is needed."""
    if_another_forecast_is_needed = interrupt("¿Necesitas hacer otro forecast?")

    if_another_forecast_is_needed_response = (
        chain_for_if_another_forecast_is_needed.invoke(if_another_forecast_is_needed)
    )

    if if_another_forecast_is_needed_response.if_another_forecast_is_needed:
        next_node = "forecast_information_checker"
        return Command(
            goto=next_node,
            update={
                "current_agent": "ask_if_another_forecast_is_needed",
                "next_node": next_node,
                "llm_model": "logic",
                "user_parameters_for_forecast": None,  # [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
                # + (
                #     [if_another_forecast_is_needed_response.extra_information]
                #     if if_another_forecast_is_needed_response.extra_information
                #     else []
                # ),
                "reasoning": [
                    "El usuario pide hacer otro forecast, por lo que se debe verificar la información adicional que el usuario puede haber proporcionado",
                    "proximo agente: Forecast Information Checker",
                ],
                # "messages": state["messages"][-1],
            },
        )
    else:
        next_node = "check_if_plan_is_done"
        return Command(
            goto=next_node,
            update={
                "current_agent": "ask_if_another_forecast_is_needed",
                "next_node": next_node,
                "llm_model": "gemini-2.5",
                # "user_parameters_for_forecast": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
                "reasoning": [
                    "El usuario no pide hacer otro forecast, por lo que se debe verificar si el plan está completo",
                    "proximo agente: Check If Plan Is Done",
                ],
                "plot": None,
            },
        )


if __name__ == "__main__":
    from time import perf_counter

    t0 = perf_counter()
    result = get_forecast_moving_avg.invoke(
        {
            "office_name": "160 - Ñuñoa",
            # "office_name": "159 - Providencia",
            "today_date_str": "2025-05-08",
            "start_date": "2025-06-01",
            "num_days": 3,
        }
    )
    elapsed_min = (perf_counter() - t0) / 60.0
    print(result)

    print(
        f"Tiempo de ejecución de get_forecast_moving_avg.invoke: {elapsed_min:.2f} minutos"
    )
