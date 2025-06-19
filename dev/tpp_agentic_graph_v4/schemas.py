"""This module defines Pydantic models for structuring the agentic workflow."""

# %%
from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator

# pylint: disable=import-error,wrong-import-position
load_dotenv(override=True)

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

from tpp_agentic_graph_v4.reducers import (  # noqa: E402
    executed_steps_reducer,
    merge_reasoning,
    reduce_docs,
    scratchpad_reducer,
    steps_reducer,
    web_results_reducer,
)

os.chdir(package_root)

with Path("prompts/system_prompts.yaml").open("r", encoding="utf-8") as f:
    SYSTEM_PROMPTS = yaml.safe_load(f)

# Load vectorstore themes
with Path("vectorstore/vectorstore_theme.yaml").open("r", encoding="utf-8") as f:
    VECTORSTORE_THEMES = yaml.safe_load(f)

# Format the themes as a comma-separated string
FORMATTED_THEMES = ", ".join(VECTORSTORE_THEMES["theme"])

# Format the RAG agent prompt with the actual themes
if "rag" in SYSTEM_PROMPTS["agents"]:
    SYSTEM_PROMPTS["agents"]["rag"] = SYSTEM_PROMPTS["agents"]["rag"].replace(
        "{vectorstore_theme}", FORMATTED_THEMES
    )

# Create a string with each agent prompt on a separate line


ACTIVE_AGENTS = ["forecast_information_checker", "reasoner"]

AGENT_PROMPTS_STRING = "\n".join(
    [SYSTEM_PROMPTS["agents"][agent_type] for agent_type in ACTIVE_AGENTS]
)

# Explicit literal enumeration for agent types to ensure correct JSON schema generation
# Using a runtime list (ACTIVE_AGENTS) inside Literal leads to an array literal which
# breaks Gemini structured output expectations (it interprets the field as an array
# without specifying its item type). We instead define the allowed agent strings
# explicitly via a Literal.

# NOTE: If you add new agents to ACTIVE_AGENTS, remember to update AgentLiteral.

# Allowed agent names literal type
AgentLiteral = Literal["forecast_information_checker", "reasoner"]


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
            f"Los las funciones de los agentes son: {AGENT_PROMPTS_STRING}"
        ),
    )
    agent: AgentLiteral = Field(
        description=(
            "El agente que debe realizar el paso. "
            "Debes elegir un único agente que es capaz de realizar el paso. "
            f"Los las funciones de los agentes son: {AGENT_PROMPTS_STRING}"
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
            f"investigación o interacción con los agentes ({', '.join(ACTIVE_AGENTS)}).\n"
            "• 'respond': Cuando la consulta puede ser respondida de inmediato sin "
            "planificación.\n"
            "Debes responder 'plan' únicamente si el requerimiento está "
            "asociado a alguna de estas temáticas/agentes:\n"
            f"{AGENT_PROMPTS_STRING}"
        ),
    )
    steps: list[OneStep] = Field(
        description=(
            "Este atributo debe ser llenado únicamente si what_to_do es plan. "
            "Debes resolver el requerimiento del usuario, para esto: "
            "Genera pasos secuenciales step-by-step (mínimo 1, máximo 7), "
            "prioriza planes cortos de 1 o 4 pasos. "
            "solo si la pregunta es compleja, puedes generar más pasos (3-10). "
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

    nombre_de_la_oficina: str | None = Field(
        description=(
            "El nombre de la oficina, por ejemplo: '159 - Providencia' o '160 - Ñuñoa', "
            "si no está explícita, rellenar con None"
        ),
    )
    fecha_del_dia_de_hoy: str | None = Field(
        description=(
            "La fecha del día de hoy, en formato YYYY-MM-DD, por ejemplo: '2025-05-08'"
        ),
    )
    fecha_inicio_de_la_proyeccion: str | None = Field(
        description=(
            "La fecha de inicio de la proyección, en formato YYYY-MM-DD, por ejemplo: '2025-06-01'"
        ),
    )
    numero_de_dias_a_proyectar: int | None = Field(
        description=("El número de días a proyectar, por ejemplo: 3"),
    )

    @field_validator("*", mode="before", check_fields=False)
    @classmethod
    def parse_none_string(cls, v: Any) -> Any:
        """Convert strings like 'none', 'unknown', etc. to actual None before validation."""
        if isinstance(v, str) and v.strip().lower() in {
            "none",
            "null",
            "n/a",
            "na",
            "desconocido",
            "unknown",
            "<unknown>",
            "",
        }:
            return None
        return v


class ForecastInput(BaseModel):
    """Forecast input.

    This class is used to convert DataFrame structure to forecast input.
    """

    valores: list[float] = Field(description="Lista de valores")


class OneSection(BaseModel):
    """Modelo para una sección del informe/reporte.

    Este modelo define una sección del informe/reporte, con su nombre y breve descripción.
    IMPORTANTE: Usar formato markdown bien estructurado para el contenido.
    """

    name: str = Field(
        description="Título/Encabezado para esta sección del informe/reporte.",
    )
    description: str = Field(
        description="Brevísimo/conciso resumen del tema principal cubierto en esta sección.",
    )


class ReportSections(BaseModel):
    """Modelo para el informe/reporte.

    Este modelo define el informe/reporte, con sus secciones y solicitud de aprobación para el usuario.
    """

    sections: list[OneSection] = Field(
        description="Secciones del informe/reporte.",
    )
    sections_approval_request: str = Field(
        description=(
            "Pregunta para consultar al usuario si es q aprueba las secciones propuestas, o si desea hacer modificaciones."
        ),
    )


class PathwayGraphState(MessagesState):
    """State model for the graph."""

    plan_respond: PlanRespond
    plan: str
    user_question: str
    question_route: str
    reasoning: Annotated[str | list[str], merge_reasoning]
    steps: Annotated[list[OneStep], steps_reducer]
    executed_steps: Annotated[list[OneStep], executed_steps_reducer] | None = None
    next_node: str
    current_step: OneStep
    retrieval_query: str
    documents: Annotated[list[Document], reduce_docs]
    final_documents: list[Document]  # Field for storing final documents without reducer
    web_search_results: Annotated[list[dict[str, Any]], web_results_reducer]
    tables_results: str  # Annotated[list[str], add]
    report: str
    plot: str | None = None
    scratchpad: Annotated[list[BaseMessage], scratchpad_reducer] | None = None
    # Modelo LLM usado por el nodo que realizó la última actualización. Permite
    # que el streamer muestre en el panel de razonamiento qué modelo se empleó.
    llm_model: str | None = None
    temporal_series_info: TemporalSeriesChecker | None = None
    forecast_input: ForecastInput | None = None
    user_parameters_for_forecast: (
        Annotated[Sequence[BaseMessage], add_messages] | None
    ) = None
    report_sections: ReportSections | None = None
    sections_user_feedback: str | None = None
    web_search_generated_response: str | None = None
    rag_generated_response: str | None = None
    forecast_generated_response: str | None = None


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
            "(documentos, websearch o scratchpad) para generar la respuesta."
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
            "(documentos, websearch o scratchpad) para generar el reporte."
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


@dataclass(kw_only=True)
class QueryState:
    """Private state for the retrieve_documents node in the researcher graph."""

    query: str


class IfReportSectionsAreApproved(BaseModel):
    """If report sections are approved by the user."""

    if_sections_are_approved: Literal["approved", "not_approved"] = Field(
        description="Si el usuario aprueba las secciones propuestas, devuelve 'approved', si no, devuelve 'not_approved'"
    )
    user_feedback: str = Field(
        description="Una breve explicación sobre cual es el requerimiento del usuario/humano."
    )


# %%
