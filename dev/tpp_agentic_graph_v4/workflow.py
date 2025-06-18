"""Defines the main LangGraph workflow for the agentic system."""

# %%
import os
import sqlite3
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.types import Command

# pylint: disable=import-error,wrong-import-position
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


from tpp_agentic_graph_v4.agents.agent_for_forecast.forecasting import (  # noqa: E402
    ask_if_another_forecast_is_needed,
    extract_temporal_series_from_tables,
    forecast_information_checker,
    make_forecast,
)
from tpp_agentic_graph_v4.agents.agent_for_reasoning.reflect_over_info import (  # noqa: E402
    reasoner,
)
from tpp_agentic_graph_v4.agents.agent_multi_turn_planner.planner import (  # noqa: E402
    approve_plan,
    check_if_plan_is_done,
    conduct_plan,
    multi_turn_planner,
    response_after_plan,
)
from tpp_agentic_graph_v4.schemas import PathwayGraphState  # noqa: E402

load_dotenv(override=True)

builder = StateGraph(PathwayGraphState)

##### Agent for plan #####
builder.add_node("planner", multi_turn_planner)
builder.add_node("approve_plan", approve_plan)
builder.add_node("conduct_plan", conduct_plan)
builder.add_node("check_if_plan_is_done", check_if_plan_is_done)
builder.add_node("response_after_plan", response_after_plan)

##### Agent for forecast
builder.add_node("forecast_information_checker", forecast_information_checker)
builder.add_node(
    "extract_temporal_series_from_tables", extract_temporal_series_from_tables
)
builder.add_node("make_forecast", make_forecast)
builder.add_node("ask_if_another_forecast_is_needed", ask_if_another_forecast_is_needed)


##### Agent for reasoning
builder.add_node("reasoner", reasoner)

builder.add_edge(START, "planner")


conn = sqlite3.connect(":memory:", check_same_thread=False)
memory = SqliteSaver(conn)
graph = builder.compile(memory)


if __name__ == "__main__":
    thread_config = {"configurable": {"thread_id": uuid.uuid4()}}

    MESSAGE_FROM_USER = (
        # "necesito un forecast de la variable biomarcador de la tabla "
        # "patient_time_series, con contexto de 20 puntos y predicción de 10 puntos"
        # "busca en internet que es la metformina"
        #  "busca en los papersque es el envejecimiento"
        # "HOLA"
        # "si"
        # "no"
        "necesito hacer una proyeccion"
    )
    for chunk in graph.stream(
        (
            {"messages": MESSAGE_FROM_USER}
            if not graph.get_state(thread_config).next
            else Command(resume=MESSAGE_FROM_USER)
        ),
        config=thread_config,
        stream_mode="updates",
        subgraphs=True,
    ):
        print(chunk)
