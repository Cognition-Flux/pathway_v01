"""reflect_over_info.
Este agente se encarga de razonar, pensar, tomar decisiones y generar una respuesta en
base a la pregunta del usuario y la información disponible.
"""

# %%
import logging
import os
import sys
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Command

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


load_dotenv(override=True)

os.chdir(package_root)
from tpp_agentic_graph_v4.llm_chains import (  # noqa: E402
    chain_for_reasoner,
)
from tpp_agentic_graph_v4.schemas import PathwayGraphState  # noqa: E402

logger = logging.getLogger(__name__)


def reasoner(
    state: PathwayGraphState,
) -> Command[Literal["check_if_plan_is_done"]]:
    """Reasoner
    Se encarga de razonar, pensar, tomar decisiones y generar una respuesta en base a la pregunta del usuario y la información disponible.
    """
    response = chain_for_reasoner.invoke(
        {
            "input": state["current_step"],
            "question": state["current_step"],
            "scratchpad": state["scratchpad"],
        }
    )
    next_node = "check_if_plan_is_done"
    # ✨ Agrega el nuevo razonamiento al scratchpad sin perder historial
    existing_pad = state.get("scratchpad", []) or []
    new_pad = [*existing_pad, AIMessage(content=response.content)]

    return Command(
        goto=next_node,
        update={
            "scratchpad": new_pad,
            "next_node": next_node,
        },
    )
