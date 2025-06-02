"""Report generation agent."""

# %%
import logging
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.types import Command, interrupt

from agentic_workflow.schemas import (
    IfReportIsNeeded,
    PathwayGraphState,
    ReportGenerator,
)
from agentic_workflow.utils import get_llm


logger = logging.getLogger(__name__)


load_dotenv(override=True)
with Path("agentic_workflow/prompts/system_prompts.yaml").open("r") as f:
    prompts = yaml.safe_load(f)


def report_generator(
    state: PathwayGraphState,
) -> Command[Literal["planner", "ask_if_plot_is_needed"]]:
    """Report generator.

    This function generates a report using the chain_for_report_generator.
    """
    chain_for_if_report_is_needed = get_llm(
        model="gpt-4.1-nano"
    ).with_structured_output(IfReportIsNeeded)

    if_report_is_needed = interrupt(
        "¿Necesitas un reporte detallado basado en la información encontrada?",
    )
    response = chain_for_if_report_is_needed.invoke(if_report_is_needed)
    if response.if_report_is_needed:
        report_generator_prompt = prompts["report_generator"].format(
            question=state["user_question"],
            documents=state["documents"],
            web_search_results=state["web_search_results"],
            tables_results=state.get("tables_results", None),
            scratchpad=state.get("scratchpad", None),
        )

        report = (
            get_llm(provider="azure", model="gpt-4.1-mini")
            .with_structured_output(ReportGenerator)
            .invoke(
                [
                    SystemMessage(content=report_generator_prompt),
                    HumanMessage(content="Sigue las instrucciones."),
                ]
            )
        )

        next_node = "ask_if_plot_is_needed"
        return Command(
            goto=next_node,
            update={
                "report": report.report,
                "reasoning": [
                    report.reasoning,
                    f"proximo agente: {next_node.replace('_', ' ').title()}",
                ],
                "current_agent": "report_generator",
                "next_node": next_node,
                "llm_model": "deepseek-r1",  # Uses get_llm() for main report generation
            },
        )
    else:
        next_node = "planner"
        return Command(
            goto=next_node,
            update={
                # Propagate the user's response so we can continue the flow without asking again
                "messages": [
                    AIMessage(
                        content="¿Necesitas un reporte detallado basado en la información encontrada?"
                    ),
                    HumanMessage(
                        content="Continua con la conversación:" + if_report_is_needed
                    ),
                ],
                # We store an empty report placeholder to signal downstream steps without leaking full content to the chat stream
                "report": "",
                "reasoning": [
                    "No se hará un reporte.",
                    f"proximo agente: {next_node.replace('_', ' ').title()}",
                ],
                "current_agent": "report_generator",
                "next_node": next_node,
                "llm_model": "gpt-4.1-nano",  # Uses nano model for decision logic
            },
        )
