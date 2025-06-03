"""Graph definition for the multiturn conversational agent."""

# %%
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, add_messages
from langgraph.types import Command, interrupt

from agentic_workflow.llm_chains import (
    chain_for_planning,
)
from agentic_workflow.schemas import PathwayGraphState, ResponseAfterPlan
from agentic_workflow.utils import get_llm


load_dotenv(override=True)


def multi_turn_planner(
    state: PathwayGraphState,
) -> Command[Literal["approve_plan"]]:
    """Entrypoint for the multiturn conversational graph."""
    while True:
        response = chain_for_planning.invoke(
            {
                "input": state["messages"],
            }
        )
        if response.what_to_do != "plan":
            user_input = interrupt(value=response.response)
            state["messages"] = add_messages(state["messages"], [user_input])
            continue
        elif response.what_to_do == "plan":
            next_node = "approve_plan"
            return Command(
                goto=next_node,
                update={
                    "plan_respond": response if next_node == "approve_plan" else None,
                    "steps": response.steps if next_node == "approve_plan" else None,
                    "messages": [response.response],
                    "user_question": state["messages"][-1].content,
                    "reasoning": [
                        response.reasoning,
                        *(
                            [f"proximo agente: {next_node.replace('_', ' ').title()}"]
                            if isinstance(next_node, str) and next_node != END
                            else []
                        ),
                    ],
                    "current_agent": "planner",
                    "next_node": next_node,
                    "llm_model": "gpt-4.1-mini",
                },
            )
        else:
            raise ValueError(
                f"multi_turn_planner: Expected valid response.what_to_do, got "
                f"'{response.what_to_do=}'"
            )


def approve_plan(
    state: PathwayGraphState,
) -> Command[Literal["conduct_plan", "planner"]]:
    """Approve plan.

    This function approves the plan.
    """
    steps = state["steps"]
    # Generate step list before the Command
    step_list = [f"- {s.step}" for idx, s in enumerate(steps)]
    message_for_interrupt = (
        "Voy a ejecutar el siguiente plan:\n"
        + "\n".join(step_list)
        + "\n\n\n\n¿Continuamos? (si/no)"
    )

    if_conduct_plan = interrupt(message_for_interrupt)
    affirmative_responses = [
        "si",
        "sí",
        "yes",
        "y",
        "ok",
        "okay",
    ]
    print(f"-------------------if_conduct_plan: {if_conduct_plan}")
    if if_conduct_plan.strip().lower() in affirmative_responses:
        next_node = "conduct_plan"
    else:
        next_node = "planner"
    return Command(
        goto=next_node,
    )


def conduct_plan(
    state: PathwayGraphState,
) -> Command[
    Literal[
        "rag",
        "websearch",
        "tables",
        "plot_generator",
        "forecast_information_checker",
        END,
    ]
]:
    """Conduct plan.

    This function conducts a plan using the chain_for_planning.
    """
    current_step = state["steps"][0]
    # Map agent types to node names
    if current_step.agent == "rag":
        next_node = "rag"
    elif current_step.agent == "websearch":
        next_node = "websearch"
    elif current_step.agent == "tables":
        next_node = "tables"
    elif current_step.agent == "plot_generator":
        next_node = "plot_generator"
    elif current_step.agent == "forecast_information_checker":
        next_node = "forecast_information_checker"
    else:
        raise ValueError(f"conduct_plan: Expected agent, got '{current_step.agent}'")

    return Command(
        goto=next_node,
        update={
            "messages": [f"Ejecutando: {current_step.step}"],
            "steps": state["steps"][1:],
            "current_step": current_step,
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


def check_if_plan_is_done(
    state: PathwayGraphState,
) -> Command[Literal["response_after_plan", "conduct_plan"]]:
    """Check if plan is done.

    This function checks if the plan is done.
    """
    next_node = "response_after_plan" if state["steps"] == [] else "conduct_plan"
    return Command(
        goto=next_node,
        update={
            "messages": [
                (
                    f"Quedan {len(state['steps'])} pasos por ejecutar."
                    if len(state["steps"]) > 0
                    else "Plan completado!"
                )
            ],
            "reasoning": [
                f"Quedan {len(state['steps'])} pasos por ejecutar."
                if len(state["steps"]) > 0
                else "Plan completado",
                *(
                    [f"proximo agente: {next_node.replace('_', ' ').title()}"]
                    if isinstance(next_node, str) and next_node != END
                    else []
                ),
            ],
            "current_agent": "check_if_plan_is_done",
            "next_node": next_node,
            "llm_model": "logic",  # Pure logic, no LLM used
        },
    )


with Path("agentic_workflow/prompts/system_prompts.yaml").open("r") as f:
    prompts = yaml.safe_load(f)


def response_after_plan(
    state: PathwayGraphState,
) -> Command[Literal["report_generator"]]:
    """Response after plan.

    This function generates a response after plan using the chain_for_plan_response.
    """
    plan_response_prompt = prompts["plan_response_generator"].format(
        question=state["user_question"],
        documents=state["documents"],
        web_search_results=state["web_search_results"],
        tables_results=state.get("tables_results", None),
        scratchpad=state["scratchpad"],
    )
    chain_for_plan_response = get_llm(
        provider="azure", model="gpt-4.1-mini"
    ).with_structured_output(ResponseAfterPlan)
    response = chain_for_plan_response.invoke(
        [
            SystemMessage(content=plan_response_prompt),
            HumanMessage(content="Sigue las instrucciones."),
        ]
    )
    return Command(
        goto="report_generator",
        update={
            "messages": [response.response],
            "reasoning": [
                response.reasoning,
                f"proximo agente: "
                f"{'ask_if_report_is_needed'.replace('_', ' ').title()}",
            ],
            "current_agent": "response_after_plan",
            "next_node": "ask_if_report_is_needed",
            "llm_model": "llama-3.3-70b",  # Uses get_llm() default model
        },
    )
