"""Graph definition for the multiturn conversational agent."""

# %%

import os
import sys
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, add_messages
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
    chain_for_plan_response,
    chain_for_planning,
    chain_for_replan_next_steps,
)
from tpp_agentic_graph_v4.schemas import PathwayGraphState  # noqa: E402

with Path("prompts/system_prompts.yaml").open("r", encoding="utf-8") as f:
    prompts = yaml.safe_load(f)


def info_print(message: str, width: int = 80, blank_lines: int = 3) -> None:
    """Print a message with blank-line padding and centered formatting.

    Args:
        message: The text to output.
        width: The total width used when centering the message.
        blank_lines: The number of blank lines to insert before and after.
    """
    print("\n" * blank_lines + str(message).center(width) + "\n" * blank_lines, end="")


def multi_turn_planner(
    state: PathwayGraphState,
) -> Command[Literal["approve_plan"]]:
    """Entrypoint for the multiturn conversational graph."""
    info_print(
        f"PASOS EJECUTADOS-------------------state['executed_steps']: {state.get('executed_steps', None)}"
    )
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
                    "llm_model": "gemini-2.5-flash",
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
    info_print(f"-------------------if_conduct_plan: {if_conduct_plan}")
    if if_conduct_plan.strip().lower() in affirmative_responses:
        next_node = "conduct_plan"
        return Command(
            goto=next_node,
            update={
                "llm_model": "logic",
                "current_agent": "approve_plan",
                "reasoning": [
                    "El plan ha sido generado y se presentó al usuario para su aprobación.",
                ],
            },
        )
    else:
        next_node = "planner"
        input_plan = interrupt(
            "¿Cómo deseas continuar? Por favor, especifica el plan que deseas ejecutar."
        )
        return Command(
            goto=next_node,
            update={
                "messages": [
                    ToolMessage(
                        content=f"Sigue estas últimas instrucciones para generar un nuevo plan: {input_plan}"
                    ),
                ],
                "llm_model": "logic",
                "current_agent": "approve_plan",
                "reasoning": [
                    "El plan ha sido generado y se presentó al usuario para su aprobación.",
                ],
            },
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

    updated_steps = None
    info_print(
        f"conduct_plan-------------------state['executed_steps']: {state.get('executed_steps', None)}"
    )

    if state.get("executed_steps", None):
        info_print("Replan next steps")
        replan_next_steps = chain_for_replan_next_steps.invoke(
            {
                "input": state["user_question"],
                "question": state["user_question"],
                "executed_steps": state["executed_steps"],
                "scratchpad": state["scratchpad"],
                "remaining_steps": state["steps"][1:],
            }
        )
        info_print(f"Pasos iniciales: {state['steps']}")
        info_print(f"Pasos actualizados: {replan_next_steps.steps}")
        updated_steps = replan_next_steps.steps

    if current_step.agent == "forecast_information_checker":
        next_node = "forecast_information_checker"
    else:
        raise ValueError(f"conduct_plan: Expected agent, got '{current_step.agent}'")

    # ------------------------------------------------------------------
    # Build scratchpad updates *only* if there is new content to append.
    # This prevents wiping the existing scratchpad with an empty list.
    # ------------------------------------------------------------------

    _new_scratchpad_msgs: list[AIMessage] = []
    if state.get("forecast_generated_response"):
        _new_scratchpad_msgs.append(
            AIMessage(content=state["forecast_generated_response"])
        )

    # NEW: Print the scratchpad messages that will be appended so we can track growth.
    info_print(
        f"_new_scratchpad_msgs (len={len(_new_scratchpad_msgs)}): {_new_scratchpad_msgs}"
    )

    # Base update payload
    _update_payload = {
        "messages": [f"Ejecutando: {current_step.step}"],
        "steps": updated_steps
        if state.get("executed_steps", None)
        else state["steps"][1:],
        "executed_steps": [
            *(
                [
                    s
                    for s in state.get("executed_steps", [])
                    if getattr(s, "step", None) != current_step.step
                ]
                if state.get("executed_steps")
                else []
            ),
            current_step,
        ],
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
    }

    # Add scratchpad only if we have new content to avoid overwriting.
    if _new_scratchpad_msgs:
        _update_payload["scratchpad"] = _new_scratchpad_msgs
    info_print(
        f"Scratchpad: {state.get('scratchpad', 'Scratchpad está vacía!!!!!!!!!!!')}"
    )
    return Command(
        goto=next_node,
        update=_update_payload,
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
                    else "Tareas completadas!"
                )
            ],
            "reasoning": [
                f"Quedan {len(state['steps'])} pasos por ejecutar."
                if len(state["steps"]) > 0
                else "Tareas completadas",
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


def response_after_plan(
    state: PathwayGraphState,
) -> Command[Literal[END]]:
    """Response after plan.

    This function generates a response after plan using the chain_for_plan_response.
    """

    info_print(
        f"response_after_plan-------------------state['scratchpad']: {state['scratchpad']}",
        blank_lines=5,
    )
    info_print(
        f"response_after_plan-------------------state['executed_steps']: {state['executed_steps']}",
        blank_lines=4,
    )
    plan_response_prompt = prompts["plan_response_generator"].format(
        question=state["user_question"],
        executed_steps=state["executed_steps"],
        scratchpad=state["scratchpad"],
    )

    response = chain_for_plan_response.invoke(
        [
            SystemMessage(content=plan_response_prompt),
            HumanMessage(content="Sigue las instrucciones."),
        ]
    )
    next_node = END  # "ask_if_report_is_needed"
    return Command(
        goto=next_node,
        update={
            "messages": [response.response, "¿En que más puedo ayudarte?"],
            "forecast_generated_response": None,
            "executed_steps": "delete",
            "reasoning": [
                response.reasoning,
                *(
                    [f"proximo agente: {next_node.replace('_', ' ').title()}"]
                    if isinstance(next_node, str) and next_node != END
                    else []
                ),
            ],
            "current_agent": "response_after_plan",
            "next_node": next_node,
            "llm_model": "gpt-4.1-mini",  #
        },
    )
