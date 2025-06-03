"""This module defines Pydantic models for structuring the agentic workflow."""

# %%
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator

from agentic_workflow.reducers import (
    merge_reasoning,
    reduce_docs,
    steps_reducer,
    web_results_reducer,
)


with Path("agentic_workflow/prompts/system_prompts.yaml").open("r") as f:
    system_prompts = yaml.safe_load(f)

# Load vectorstore themes
with Path("agentic_workflow/vectorstore/vectorstore_theme.yaml").open("r") as f:
    vectorstore_themes = yaml.safe_load(f)

# Format the themes as a comma-separated string
formatted_themes = ", ".join(vectorstore_themes["theme"])

# Format the RAG agent prompt with the actual themes
if "rag" in system_prompts["agents"]:
    system_prompts["agents"]["rag"] = system_prompts["agents"]["rag"].replace(
        "{vectorstore_theme}", formatted_themes
    )

# Create a string with each agent prompt on a separate line
agent_prompts_string = "\n".join(
    [
        system_prompts["agents"][agent_type]
        for agent_type in [
            "rag",
            "websearch",
            "tables",
            "plot_generator",
            "forecast_information_checker",
        ]
    ]
)


class ResearchPlannerState(MessagesState):
    """Base state for step-by-step research planning workflow.

    Contains attributes needed for the research planning and retrieval.
    """

    documents: Annotated[list[Document], reduce_docs]
    final_documents: list[Document]  # Field for storing final documents without reducer
    query: str
    steps: Annotated[list[str], steps_reducer]
    retrieval_query: str
    reasoning: Annotated[str | list[str], merge_reasoning] = ""


class OneStep(BaseModel):
    """One step of the plan."""

    step: str = Field(
        description=(
            "El paso a realizar en forma de instrucción. "
            "El pase debe ser exclusivamente una instrucción que un agente "
            "pueda realizar. "
            f"Los las funciones de los agentes son: {agent_prompts_string}"
        ),
    )
    agent: Literal[
        "rag", "websearch", "tables", "plot_generator", "forecast_information_checker"
    ] = Field(
        description=(
            "El agente que debe realizar el paso. "
            "Debes elegir un único agente que es capaz de realizar el paso. "
            f"Los las funciones de los agentes son: {agent_prompts_string}"
        ),
    )
    reasoning: str = Field(
        description=(
            "Explica el razonamiento detrás de tu decisión sobre elegir el agente que "
            "debe realizar el paso."
        ),
    )


class PlanRespond(BaseModel):
    """Generate plan or respond."""

    what_to_do: Literal["plan", "respond"] = Field(
        description=(
            "En base a la conversación con el usuario, "
            "determina la acción correcta:\n"
            "• 'plan': Cuando el requerimiento requiere varios pasos de "
            "investigación o interacción con los agentes (rag, websearch, tables).\n"
            "• 'respond': Cuando la consulta puede ser respondida de inmediato sin "
            "planificación.\n"
            "Debes responder 'plan' únicamente si el requerimiento está "
            "asociado a alguna de estas temáticas/agentes:\n"
            f"{agent_prompts_string}"
        ),
    )
    steps: list[OneStep] = Field(
        description=(
            "Este atributo debe ser llenado únicamente si what_to_do es plan. "
            "Debes resolver el requerimiento del usuario, para esto: "
            "Genera pasos secuenciales step-by-step (mínimo 1, máximo 5), "
            "prioriza planes cortos de 1 o 2 pasos. "
            "solo si la pregunta es compleja, puedes generar más pasos (3-5). "
            "Enumerar los pasos. "
            "Asegurate que cada paso resuelva incrementamente el problema. "
            "Y que al completar los pasos, el problema quede resuelto. "
            "IMPORTANTE: En cada paso DEBES mencionar explícitamente la "
            "temática central (envejecimiento, longevidad) "
            "ya que cada paso entra independientemente a una query retrieval, "
            "por lo tanto cada paso por sí solo debe mantener el contexto completo."
        ),
    )
    response: str = Field(
        description=(
            "Este atributo debe ser llenado explicando brevemente qué es lo que vas a "
            "hacer (what_to_do)."
        ),
    )
    reasoning: str = Field(
        description=(
            "Explica el razonamiento detrás de tu decisión sobre elegir plan o "
            "responder."
        ),
    )


class TemporalSeriesChecker(BaseModel):
    """Temporal series checker."""

    nombre_de_la_serie_temporal: str | None = Field(
        default=None,
        description="El nombre de la serie temporal, si no está explícita, "
        "rellenar con None",
    )
    nombre_de_la_tabla: str | None = Field(
        default=None,
        description="El nombre de la tabla donde se encuentra la serie temporal, "
        "si no está explícita, rellenar con None",
    )
    ventana_contexto: int | None = Field(
        default=None,
        description="El largo de la ventana de contexto de la serie temporal "
        "(cuántos puntos históricos usar), si no está explícita, rellenar con None",
    )
    ventana_prediccion: int | None = Field(
        default=None,
        description="El largo de la ventana de predicción del forecast (cuántos puntos "
        "predecir), si no está explícita, rellenar con None",
    )

    @field_validator("*", mode="before", check_fields=False)
    @classmethod
    def parse_none_string(cls, v: Any) -> Any:
        """Parse 'none' string to None before validation."""
        if isinstance(v, str) and v.strip().lower() == "none":
            return None
        return v


class ForecastInput(BaseModel):
    """Forecast input.

    This class is used to convert DataFrame structure to forecast input.
    """

    valores: list[float] = Field(description="Lista de valores")


class PathwayGraphState(MessagesState):
    """State model for the graph."""

    plan_respond: PlanRespond
    plan: str
    user_question: str
    question_route: str
    reasoning: Annotated[str | list[str], merge_reasoning]
    steps: Annotated[list[OneStep], steps_reducer]
    next_node: str
    current_step: OneStep
    retrieval_query: str
    documents: Annotated[list[Document], reduce_docs]
    final_documents: list[Document]  # Field for storing final documents without reducer
    web_search_results: Annotated[list[dict[str, Any]], web_results_reducer]
    tables_results: str  # Annotated[list[str], add]
    report: str
    plot: str | None = None
    scratchpad: Annotated[Sequence[BaseMessage], add_messages]
    # Modelo LLM usado por el nodo que realizó la última actualización. Permite
    # que el streamer muestre en el panel de razonamiento qué modelo se empleó.
    llm_model: str | None = None
    temporal_series_info: TemporalSeriesChecker | None = None
    forecast_input: ForecastInput | None = None
    user_parameters_for_forecast: (
        Annotated[Sequence[BaseMessage], add_messages] | None
    ) = None


class IfAnotherForecastIsNeeded(BaseModel):
    """If another forecast is needed."""

    if_another_forecast_is_needed: bool = Field(
        description=(
            "Si el usuario pide hacer otro forecast, devuelve True, si no, devuelve False"
        )
    )
    extra_information: str | None = Field(
        description=(
            "Si el usuario pide hacer otro forecast, aquí escribe la información adicional que el usuario puede haber proporcionado, si no, devuelve None"
        )
    )


class ResponseAfterPlan(BaseModel):
    """Response after plan.

    This class generates a response after plan using the chain_for_plan_response.
    """

    response: str = Field(
        description=("La respuesta que debes dar al usuario."),
    )
    reasoning: str = Field(
        description=(
            "Explica por que la respuesta responde a la pregunta del usuario."
            "Explica como aseguraste que no hay alucinaciones."
            "Explica como te basaste exclusivamente en las fuentes "
            "(documentos, websearch y/o tablas) para generar la respuesta."
        )
    )


class IfReportIsNeeded(BaseModel):
    """If report is needed.

    This class defines if the user is asking for a report or not.
    """

    if_report_is_needed: bool = Field(
        description=(
            "Si el usuario pide un reporte, devuelve True, si no, devuelve False"
        )
    )


class ReportGenerator(BaseModel):
    """Report generator.

    This class generates a report using the chain_for_report_generator.
    """

    report: str = Field(description=("El reporte que debes generar."))
    reasoning: str = Field(
        description=(
            "Explica por que el reporte responde a la pregunta del usuario."
            "Explica como aseguraste que no hay alucinaciones."
            "Explica como te basaste exclusivamente en las fuentes "
            "(documentos, websearch y/o tablas) para generar el reporte."
        )
    )


class QueriesToWebsearch(BaseModel):
    """Queries to websearch.

    This class is used to convert a list of queries to a list of websearch results.
    """

    queries: list[str]


class MultiQueryResponse(BaseModel):
    """Response model for structured output from query generation."""

    queries: list[str]


class MultiQueryRetrieverState(MessagesState):
    """State for the multi-query retriever workflow.

    Contains the essential attributes needed for parallel query retrieval.
    """

    documents: Annotated[list[Document], reduce_docs]
    retrieval_query: str
    queries: list[str]
    reasoning: Annotated[str | list[str], merge_reasoning] = ""


from dataclasses import dataclass


@dataclass(kw_only=True)
class QueryState:
    """Private state for the retrieve_documents node in the researcher graph."""

    query: str


# %%
