"""Reducers are a way to modify the state of a graph.

Reducers are functions that take a state and an event, and return a new state.
"""

# %%

import os
from typing import Annotated, Literal

import aiosqlite
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import AzureChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from pydantic import BaseModel

load_dotenv(override=True)


def get_llm(model: str = "gpt-4.1-nano"):
    """Get a LLM."""
    return AzureChatOpenAI(
        azure_deployment=model,
        api_version=os.getenv("AZURE_API_VERSION"),
        temperature=0 if model != "o3-mini" else None,
        max_tokens=None,
        timeout=1200,
        max_retries=5,
        streaming=True,
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )


class HumanLastQuestion(BaseModel):
    """Human question."""

    last_question: str


class LastLLMResponse(BaseModel):
    """Last LLM response."""

    last_response: str


class State(MessagesState):
    """State of the graph."""

    human_messages: Annotated[list[HumanMessage], add_messages]
    ai_messages: Annotated[list[AIMessage], add_messages]
    tool_messages: Annotated[list[ToolMessage], add_messages]
    human_last_question: HumanLastQuestion
    ai_last_response: LastLLMResponse
    # list_messages: Annotated[list[str], add]
    # overwritten_list: list[str]
    # schema_messages: Annotated[list[AnyMessage], add_messages]
    # overwritten_schema: Annotated[list[AnyMessage], add_messages]
    # documents: Annotated[list[Document], add]
    # overwritten_documents: Annotated[list[Document], add]
    # tables: Annotated[list[Table], add]
    # overwritten_tables: Annotated[list[Table], add]
    # llm_summary: Annotated[str, add]


tool_web_search = TavilySearch(max_results=10)
tools = [tool_web_search]
websearch_agent = create_react_agent(
    model=get_llm(),
    tools=tools,
    name="websearch_agent",
    prompt="Responde en español",
)


async def node_1(state: State) -> Command[Literal[END]]:
    """Node that adds a new message to the state."""
    llm_response = await websearch_agent.ainvoke(
        {"messages": [HumanMessage(content=state["human_messages"][-1].content)]}
    )
    human_messages = []
    ai_messages = []
    tool_messages = []

    for message in llm_response["messages"]:
        if isinstance(message, AIMessage):
            ai_messages.append(message)
        elif isinstance(message, ToolMessage):
            tool_messages.append(message)
        elif isinstance(message, HumanMessage):
            human_messages.append(message)
        else:
            raise ValueError(f"Unknown message type: {type(message)}")

    return Command(
        goto=END,
        update={
            "human_last_question": HumanLastQuestion(
                last_question=state["human_messages"][-1].content
            ),
            "ai_last_response": LastLLMResponse(
                last_response=llm_response["messages"][-1].content
            ),
            "ai_messages": ai_messages,
            "tool_messages": tool_messages,
        },
    )


# Build graph
builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_edge(START, "node_1")


def get_memory():
    """Get a memory."""
    conn = aiosqlite.connect(":memory:")
    return AsyncSqliteSaver(conn=conn)


def get_graph():
    """Get a graph."""
    memory = get_memory()
    return builder.compile(checkpointer=memory, debug=True)


async def aget_next_state(compiled_graph: CompiledStateGraph, config: dict) -> State:
    """Get the next state of the graph."""
    latest_checkpoint = await compiled_graph.aget_state(config=config)
    return latest_checkpoint.next


if __name__ == "__main__":
    thread_config = {"configurable": {"thread_id": "123"}}
    graph = get_graph()

    next_state = aget_next_state(graph, thread_config)
    print(f"graph state: {next_state}")
    async for chunk in graph.astream(
        {
            "human_messages": [
                HumanMessage(content="busca en internet que es el metabolismo?")
            ]
        },
        config=thread_config,
        stream_mode="updates",
        subgraphs=True,
    ):
        print(chunk)
