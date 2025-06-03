"""Defines the main LangGraph workflow for the agentic system."""

# %%
import sqlite3
import uuid

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.types import Command

from agentic_workflow.agents.agent_for_forecast.forecasting import (
    ask_if_another_forecast_is_needed,
    extract_temporal_series_from_tables,
    forecast_information_checker,
    make_forecast,
)
from agentic_workflow.agents.agent_for_rag.get_documents import rag
from agentic_workflow.agents.agent_for_tables.tables_analysis import tables
from agentic_workflow.agents.agent_for_visualizations.visualizer import (
    ask_for_plot_edits,
    ask_if_plot_is_needed,
    plot_generator,
)
from agentic_workflow.agents.agent_for_websearch.search import websearch
from agentic_workflow.agents.agent_multi_turn_planner.planner import (
    approve_plan,
    check_if_plan_is_done,
    conduct_plan,
    multi_turn_planner,
    response_after_plan,
)
from agentic_workflow.agents.agente_for_reporting.report_generation import (
    report_generator,
)
from agentic_workflow.schemas import PathwayGraphState


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

##### Agent for report
builder.add_node("report_generator", report_generator)

##### Agent for visualization
builder.add_node("ask_if_plot_is_needed", ask_if_plot_is_needed)
builder.add_node("plot_generator", plot_generator)
builder.add_node("ask_for_plot_edits", ask_for_plot_edits)

##### Agent for websearch
builder.add_node("websearch", websearch)

##### Agent for rag
builder.add_node("rag", rag)

##### Agent for tables
builder.add_node("tables", tables)

builder.add_edge(START, "planner")


conn = sqlite3.connect(":memory:", check_same_thread=False)
memory = SqliteSaver(conn)
graph = builder.compile(memory)


if __name__ == "__main__":
    thread_config = {"configurable": {"thread_id": uuid.uuid4()}}

    mensaje_del_usuario = (
        # "necesito un forecast de la variable biomarcador de la tabla "
        # "patient_time_series, con contexto de 20 puntos y predicción de 10 puntos"
        # "busca en internet que es la metformina"
        # "busca en los papersque es el envejecimiento"
        "busca en las tablas si es que hays series temporales"
        # "si"
        # "no"
    )
    for chunk in graph.stream(
        (
            {"messages": mensaje_del_usuario}
            if not graph.get_state(thread_config).next
            else Command(resume=mensaje_del_usuario)
        ),
        config=thread_config,
        stream_mode="updates",
        subgraphs=True,
    ):
        print(chunk)
