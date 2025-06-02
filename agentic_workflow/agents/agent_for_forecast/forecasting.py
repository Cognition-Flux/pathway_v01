"""Forecasting agent: validates and processes temporal series forecasting parameters."""

# %%
from typing import Literal

import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command, interrupt

from agentic_workflow.agents.agent_for_forecast.forecast_model.inference import (
    forecast_series_no_plot,
)
from agentic_workflow.agents.agent_for_forecast.visualization.forecast_plot import (
    build_plotly_forecast_figure,
)
from agentic_workflow.agents.agent_for_tables.pandas_agent import agent as tables_agent
from agentic_workflow.llm_chains import (
    chain_for_ask_for_temporal_series_information,
    chain_for_temporal_series_info,
)
from agentic_workflow.schemas import (
    ForecastInput,
    IfAnotherForecastIsNeeded,
    PathwayGraphState,
)
from agentic_workflow.utils import get_llm


load_dotenv(override=True)


def forecast_information_checker(
    state: PathwayGraphState,
) -> Command[Literal["extract_temporal_series_from_tables"]]:
    """Forecast information checker."""
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
                f"cambios o actualizar los campos (solo si existen):"
                # f"{state['messages'][-1]}"
                f"{forcast_update}"
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


def extract_temporal_series_from_tables(
    state: PathwayGraphState,
) -> Command[Literal["make_forecast"]]:
    """Extract temporal series from tables."""
    # Calculate total points needed (context window + prediction window)
    total_points_needed = (
        state["temporal_series_info"].ventana_contexto
        + state["temporal_series_info"].ventana_prediccion
    )

    # Create detailed extraction prompt for the tables agent
    extraction_prompt = """Extraer la serie temporal de la tabla.
    Nombre de la tabla: {nombre_de_la_tabla}
    Nombre de la serie temporal: {nombre_de_la_serie_temporal}
    Debes extraer la serie temporal (valores de la variable temporal)
    IMPORTANTE: Debes entregar/responder al usuario los {total_points} últimos registros de la serie temporal.
    """
    extraction_prompt = extraction_prompt.format(
        nombre_de_la_tabla=state["temporal_series_info"].nombre_de_la_tabla,
        nombre_de_la_serie_temporal=state[
            "temporal_series_info"
        ].nombre_de_la_serie_temporal,
        total_points=total_points_needed,
    )

    # Execute data extraction using tables agent
    extraction = tables_agent.invoke(extraction_prompt)

    # Format the extracted data into forecast input structure
    forcast_input_formatter = get_llm(
        provider="groq", model="deepseek-r1-distill-llama-70b"
    ).with_structured_output(ForecastInput)
    forcast_input = forcast_input_formatter.invoke(extraction["output"])

    return Command(
        goto="make_forecast",
        update={
            "forecast_input": forcast_input,
            "messages": [
                f"📊 Extracción completada: {len(forcast_input.valores)} puntos de la serie '{state['temporal_series_info'].nombre_de_la_serie_temporal}' desde tabla '{state['temporal_series_info'].nombre_de_la_tabla}'"
            ],
            "reasoning": [
                "📊 Proceso de extracción de serie temporal completado exitosamente:",
                f"• Fuente consultada: Base de datos, tabla '{state['temporal_series_info'].nombre_de_la_tabla}'",
                f"• Variable objetivo: Serie temporal '{state['temporal_series_info'].nombre_de_la_serie_temporal}'",
                f"• Requerimiento de datos: {total_points_needed} registros más recientes",
                f"  - Para contexto histórico: {state['temporal_series_info'].ventana_contexto} puntos",
                f"  - Para validación de predicción: {state['temporal_series_info'].ventana_prediccion} puntos",
                f"• Resultado obtenido: {len(forcast_input.valores)} registros extraídos correctamente",
                f"• Rango de valores: {min(forcast_input.valores):.4f} (mínimo) → {max(forcast_input.valores):.4f} (máximo)",
                "• Estado de procesamiento: Datos estructurados y listos para inferencia",
                "• Siguiente paso: Iniciar proceso de predicción con modelo de deep learning",
                "proximo agente: Make Forecast",
            ],
            "current_agent": "extract_temporal_series_from_tables",
            "next_node": "make_forecast",
            "llm_model": "deepseek-r1 + gpt-4.1",
        },
    )


def make_forecast(
    state: PathwayGraphState,
) -> Command[Literal["ask_if_another_forecast_is_needed"]]:
    """Forecast.

    This function generates predictions and forecasts using time series models.
    """
    forecast_input = state["forecast_input"]
    data_length = len(forecast_input.valores)

    ts_history_len = state["temporal_series_info"].ventana_contexto
    ts_target_len = state["temporal_series_info"].ventana_prediccion

    # Validate that we have enough data
    required_data_points = ts_history_len + ts_target_len
    if data_length < required_data_points:
        next_node = "ask_if_another_forecast_is_needed"
        return Command(
            goto=next_node,
            update={
                "messages": [
                    f"❌ Datos insuficientes para forecast: Se necesitan {required_data_points} puntos, disponibles {data_length}"
                ],
                "reasoning": [
                    "❌ Validación de datos fallida - insuficientes registros históricos:",
                    "• Análisis de disponibilidad:",
                    f"  - Registros disponibles: {data_length} puntos en la serie temporal",
                    f"  - Registros requeridos: {required_data_points} puntos mínimos",
                    f"  - Déficit identificado: {required_data_points - data_length} registros faltantes",
                    "• Configuración solicitada para el forecast:",
                    f"  - Contexto histórico necesario: {ts_history_len} puntos",
                    f"  - Horizonte de predicción deseado: {ts_target_len} puntos",
                    f"  - Total combinado requerido: {required_data_points} puntos",
                    "• Recomendación: Solicitar más datos históricos o reducir parámetros de ventana",
                    "• Estado: Proceso de forecast abortado por datos insuficientes",
                    "• Siguiente paso: Reportar limitación al usuario",
                    "proximo agente: Check If Plan Is Done",
                ],
                "current_agent": "make_forecast",
                "next_node": next_node,
                "llm_model": "validation-logic",
            },
        )

    # Convert to numpy array for easier manipulation
    all_values = np.array(forecast_input.valores)

    # Split data correctly: first ts_history_len points for training, last ts_target_len for validation
    training_data = all_values[:ts_history_len]  # First 15 points for context
    validation_data = all_values[
        ts_history_len : ts_history_len + ts_target_len
    ]  # Last 5 points for validation

    # Detailed logging for verification
    print("FORECAST DEBUG INFO:")
    print(f"Total data points: {data_length}")
    print(f"Training data (índices 0-{ts_history_len - 1}): {training_data.tolist()}")
    print(
        f"Validation data (índices {ts_history_len}-{ts_history_len + ts_target_len - 1}): {validation_data.tolist()}"
    )

    try:
        # Run inference using only the training data (first 15 points)
        predictions = forecast_series_no_plot(
            training_data, history_len=ts_history_len, target_len=ts_target_len
        )
        print(f"Generated predictions: {predictions.tolist()}")

        # Calculate performance metrics
        mse = float(np.mean((predictions - validation_data) ** 2))
        mae = float(np.mean(np.abs(predictions - validation_data)))
        rmse = float(np.sqrt(mse))
        mape = float(
            np.mean(np.abs((validation_data - predictions) / validation_data)) * 100
        )

        # Create the forecast figure with proper data alignment
        fig = build_plotly_forecast_figure(
            training_data=training_data,
            validation_data=validation_data,
            forecasted_values=predictions,
            history_len=ts_history_len,
            target_len=ts_target_len,
            original_data=all_values,  # Pass the original data for verification
        )

        next_node = "ask_if_another_forecast_is_needed"
        return Command(
            goto=next_node,
            update={
                "plot": fig.to_json(),
                "scratchpad": [
                    AIMessage(
                        content=f"🎯 Forecast generado exitosamente. Predicción de {ts_target_len} puntos usando {ts_history_len} puntos históricos. Métricas: MSE={mse:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%"
                    ),
                ],
                "messages": [
                    f"🎯 Forecast completado: {ts_target_len} predicciones generadas con MSE={mse:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}"
                ],
                "reasoning": [
                    "🎯 Proceso de forecasting ejecutado con éxito usando inteligencia artificial:",
                    "• Tecnología empleada: Red neuronal profunda pre-entrenada para series temporales",
                    f"• Datos procesados: {data_length} registros temporales validados y preparados",
                    "• Estrategia de validación: División temporal para evaluación robusta",
                    f"  - Entrenamiento: {ts_history_len} puntos históricos (contexto del modelo)",
                    f"  - Validación: {ts_target_len} puntos reales (para medir precisión)",
                    "• Características de los datos de entrada:",
                    f"  - Rango histórico: {training_data.min():.4f} → {training_data.max():.4f}",
                    f"  - Rango de validación: {validation_data.min():.4f} → {validation_data.max():.4f}",
                    "• Resultados de predicción generados:",
                    f"  - Rango de predicciones: {predictions.min():.4f} → {predictions.max():.4f}",
                    "• Métricas de rendimiento del modelo:",
                    f"  - Error cuadrático medio (MSE): {mse:.4f}",
                    f"  - Error absoluto medio (MAE): {mae:.4f}",
                    f"  - Raíz del error cuadrático (RMSE): {rmse:.4f}",
                    f"  - Error porcentual absoluto medio (MAPE): {mape:.2f}%",
                    "• Visualización: Gráfico interactivo generado con líneas continuas",
                    "• Estado final: Forecast completado y listo para análisis",
                    "proximo agente: Check If Plan Is Done",
                ],
                "current_agent": "make_forecast",
                "next_node": next_node,
                "llm_model": "deep-neural-network + plotly-visualization",
            },
        )
    except Exception as e:
        # If there's still an error, provide a helpful message
        next_node = "ask_if_another_forecast_is_needed"
        return Command(
            goto=next_node,
            update={
                "messages": [
                    f"❌ Error en forecast: {str(e)[:100]}{'...' if len(str(e)) > 100 else ''}"
                ],
                "reasoning": [
                    "⚠️ Error crítico durante el proceso de forecasting:",
                    "• Diagnóstico del error:",
                    f"  - Tipo de excepción: {type(e).__name__}",
                    f"  - Descripción técnica: {str(e)[:200]}{'...' if len(str(e)) > 200 else ''}",
                    "• Contexto del fallo:",
                    f"  - Datos disponibles al momento del error: {data_length} registros",
                    f"  - Configuración aplicada: {ts_history_len} contexto + {ts_target_len} predicción",
                    "  - Modelo objetivo: Red neuronal profunda para series temporales",
                    "• Posibles causas:",
                    "  - Problemas en el modelo de deep learning",
                    "  - Incompatibilidad de formato en los datos",
                    "  - Recursos insuficientes del sistema",
                    "• Acción tomada: Proceso abortado y error reportado al usuario",
                    "• Siguiente paso: Notificar fallo y sugerir reintentar",
                    "proximo agente: Check If Plan Is Done",
                ],
                "current_agent": "make_forecast",
                "next_node": next_node,
                "llm_model": "deep-neural-network",
            },
        )


def ask_if_another_forecast_is_needed(
    state: PathwayGraphState,
) -> Command[Literal["check_if_plan_is_done", "forecast_information_checker"]]:
    """Ask if forecast is needed."""
    if_another_forecast_is_needed = interrupt("¿Necesitas hacer otro forecast?")

    chain_for_if_another_forecast_is_needed = get_llm(
        provider="azure", model="gpt-4.1-mini"
    ).with_structured_output(IfAnotherForecastIsNeeded)
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
                "user_parameters_for_forecast": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
                "reasoning": [
                    "El usuario no pide hacer otro forecast, por lo que se debe verificar si el plan está completo",
                    "proximo agente: Check If Plan Is Done",
                ],
                "plot": None,
            },
        )
