"""llm_chains.py.

This module contains the LLM chains for the agentic workflow.
"""

# %%
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from langchain.agents.agent_types import AgentType
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.agents.agent_toolkits import (
    create_pandas_dataframe_agent,
)

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

from tpp_agentic_graph_v4.fewshot.retriever import retriever  # noqa: E402
from tpp_agentic_graph_v4.schemas import (  # noqa: E402
    ForecastInput,
    IfAnotherForecastIsNeeded,
    IfReportIsNeeded,
    IfReportSectionsAreApproved,
    MultiQueryResponse,
    PlanRespond,
    QueriesToWebsearch,
    ReportGenerator,
    ReportSections,
    ResponseAfterPlan,
    TemporalSeriesChecker,
)
from tpp_agentic_graph_v4.utils import get_llm  # noqa: E402

with Path("prompts/system_prompts.yaml").open("r", encoding="utf-8") as f:
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
        | get_llm(
            provider="google", model="gemini-2.5-flash-preview-05-20"
        ).with_structured_output(PlanRespond)
    )

    return planning_chain


class StructuredPandasAgent:
    """Wrapper class for pandas agent that handles structured parameters."""

    def __init__(self, dataframes_list):
        """Initialize the structured pandas agent.

        Args:
            dataframes_list: List of tuples containing (name, dataframe)
        """
        self.list_of_dataframes = dataframes_list
        self.dataframes = [df for _, df in dataframes_list]

        # Create the base pandas agent
        self.pandas_agent = create_pandas_dataframe_agent(
            get_llm(provider="azure", model="gpt-4.1"),
            self.dataframes,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            allow_dangerous_code=True,
            max_iterations=100,
            max_execution_time=600.0,
            agent_executor_kwargs={"handle_parsing_errors": True},
        )

        # Define the extraction prompt template
        self.extraction_prompt_template = prompts["extraction_from_tables_prompt"]

    def invoke(
        self,
        input_text,
        nombre_de_la_tabla=None,
        nombre_de_la_serie_temporal=None,
        total_points=None,
    ):
        """Invoke the agent with structured parameters.

        Args:
            input_text: Required input text/query for the agent
            nombre_de_la_tabla: Optional table name (default: None)
            nombre_de_la_serie_temporal: Optional temporal series name (default: None)
            total_points: Optional number of points to return (default: None)

        Returns:
            Agent response
        """
        # Build the prompt parts conditionally
        prompt_parts = []

        if nombre_de_la_tabla is not None:
            prompt_parts.append(f"Nombre de la tabla: {nombre_de_la_tabla}")

        if nombre_de_la_serie_temporal is not None:
            prompt_parts.append(
                f"Nombre de la serie temporal: {nombre_de_la_serie_temporal}"
            )

        if total_points is not None:
            prompt_parts.append(
                f"IMPORTANTE: Debes entregar/responder al usuario los {total_points} últimos registros de la serie temporal."
            )

        # Build the final prompt
        if prompt_parts:
            formatted_prompt = (
                "Extraer la serie temporal de la tabla.\n"
                + "\n".join(prompt_parts)
                + f"\n\nInstrucción específica: {input_text}"
            )
        else:
            formatted_prompt = input_text

        # Invoke the pandas agent with the formatted prompt
        return self.pandas_agent.invoke({"input": formatted_prompt})


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


# Format the extracted data into forecast input structure
chain_for_forecast_input_formatter = get_llm(
    provider="groq", model="deepseek-r1-distill-llama-70b"
).with_structured_output(ForecastInput)


chain_for_if_another_forecast_is_needed = get_llm(
    provider="azure", model="gpt-4.1-mini"
).with_structured_output(IfAnotherForecastIsNeeded)

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompts["multi_query_generator"]),
        ("human", "{question}"),
    ]
)
chain_for_multi_query_retieval = route_prompt | get_llm(
    provider="azure", model="gpt-4.1-mini"
).with_structured_output(MultiQueryResponse)

ask_what_to_plot_prompt = ChatPromptTemplate.from_messages(
    [("system", prompts["ask_what_to_plot_prompt"]), ("human", "{input}")]
)

chain_for_ask_what_to_plot = ask_what_to_plot_prompt | get_llm(
    provider="google", model="gemini-2.5-flash-preview-05-20"
)


chain_for_plan_response = get_llm(
    provider="google", model="gemini-2.5-flash-preview-05-20"
).with_structured_output(ResponseAfterPlan)

chain_for_queries_to_websearch = get_llm().with_structured_output(QueriesToWebsearch)


chain_for_if_report_is_needed = get_llm(model="gpt-4.1-nano").with_structured_output(
    IfReportIsNeeded
)


sections_maker_prompt = ChatPromptTemplate.from_messages(
    [("system", prompts["report_sections_making_prompt"]), ("human", "{input}")]
)

chain_for_report_sections_maker = sections_maker_prompt | get_llm(
    provider="azure", model="gpt-4.1-nano"
).with_structured_output(ReportSections)


report_generator_prompt = ChatPromptTemplate.from_messages(
    [("system", prompts["report_generator"]), ("human", "{input}")]
)

chain_for_report_generator = report_generator_prompt | get_llm(
    provider="azure", model="gpt-4.1-mini"
).with_structured_output(ReportGenerator)

sections_approval_prompt = ChatPromptTemplate.from_messages(
    [("system", prompts["if_sections_are_approved_prompt"]), ("human", "{input}")]
)

chain_for_if_sections_are_approved = sections_approval_prompt | get_llm(
    provider="google", model="gemini-2.5-flash-preview-05-20"
).with_structured_output(IfReportSectionsAreApproved)

replan_next_step_prompt = ChatPromptTemplate.from_messages(
    [("system", prompts["replan_next_step_prompt"]), ("human", "{input}")]
)

chain_for_replan_next_steps = replan_next_step_prompt | get_llm(
    provider="azure", model="gpt-4.1-mini"
).with_structured_output(PlanRespond)

reasoner_prompt = ChatPromptTemplate.from_messages(
    [("system", prompts["reasoner_prompt"]), ("human", "{input}")]
)

chain_for_reasoner = reasoner_prompt | get_llm(provider="azure", model="gpt-4.1-mini")


if __name__ == "__main__":
    state = {
        "messages": [
            AIMessage(content="Hola, ¿cómo estás?"),
            HumanMessage(content="Estoy bien, gracias"),
        ]
    }
    # result = chain_for_ask_what_to_plot.invoke(
    #     {
    #         "input": "\n".join(
    #             [
    #                 f"{m.type.capitalize()}: {m.content}"
    #                 for m in state["messages"]
    #                 if m.type != "tool"
    #             ]
    #         )
    #     }
    # )
    # print(result.content)
    # result = chain_for_report_generator.invoke(
    #     {
    #         "input": "sigue las instrucciones",
    #         "question": "¿Cuál es el promedio de ventas de los últimos 12 meses? y como se ven las ventas en el ultimo mes?",
    #         "documents": "El ultimo mes tuvimos un incendio en la empresa",
    #         "web_search_results": "el ultimo mes de ventas fue de 7",
    #         "tables_results": "ultimos 3 meses: 100, 101, 102",
    #         "scratchpad": "el nombre de la empresa es Cerveceria del Norte",
    #     }
    # )
    # print(result.report)

    # result = chain_for_report_sections_maker.invoke(
    #     {
    #         "input": "genera las secciones del informe/reporte",
    #         "question": "¿Cuál es el promedio de ventas de los últimos 12 meses? y como se ven las ventas en el ultimo mes?",
    #         "documents": "El ultimo mes tuvimos un incendio en la empresa",
    #         "web_search_results": "el ultimo mes de ventas fue de 7",
    #         "tables_results": "ultimos 3 meses: 100, 101, 102",
    #         "scratchpad": "el nombre de la empresa es Cerveceria del Norte",
    #         "report_sections": "ninguna_seccion",
    #         "sections_user_feedback": None,
    #     }
    # )
    # print(f"type: {type(result)}")
    # print(f"sections: {result.sections}")
    # print(f"sections_approval_request: {result.sections_approval_request}")
    print(chain_for_planning.invoke({"input": [HumanMessage(content="hola")]}))

# %%
