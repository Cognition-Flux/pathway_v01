"""Reducers are a way to modify the state of a graph.

Reducers are functions that take a state and an event, and return a new state.
"""

# %%

import hashlib
import os
import uuid
from typing import Annotated, Any, Literal, Optional, Union

import aiosqlite
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import AzureChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from pydantic import BaseModel, Field


def _generate_uuid(page_content: str) -> str:
    """Generate a UUID for a document based on page content."""
    md5_hash = hashlib.md5(page_content.encode()).hexdigest()
    return str(uuid.UUID(md5_hash))


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


def reduce_docs(
    existing: Optional[list[Document]],
    new: Union[
        list[Document],
        list[dict[str, Any]],
        list[str],
        str,
        Literal["delete"],
    ],
) -> list[Document]:
    """Reduce and process documents based on the input type.

    This function handles various input types and converts them into a sequence of Document objects.
    It can delete existing documents, create new ones from strings or dictionaries, or return the existing documents.
    It also combines existing documents with the new one based on the document ID.

    Args:
        existing (Optional[Sequence[Document]]): The existing docs in the state, if any.
        new (Union[Sequence[Document], Sequence[dict[str, Any]], Sequence[str], str, Literal["delete"]]):
            The new input to process. Can be a sequence of Documents, dictionaries, strings, a single string,
            or the literal "delete".
    """
    if new == "delete":
        return []

    existing_list = list(existing) if existing else []
    if isinstance(new, str):
        return existing_list + [
            Document(page_content=new, metadata={"uuid": _generate_uuid(new)})
        ]

    new_list = []
    if isinstance(new, list):
        existing_ids = set(doc.metadata.get("uuid") for doc in existing_list)
        for item in new:
            if isinstance(item, str):
                item_id = _generate_uuid(item)
                new_list.append(Document(page_content=item, metadata={"uuid": item_id}))
                existing_ids.add(item_id)

            elif isinstance(item, dict):
                metadata = item.get("metadata", {})
                item_id = metadata.get("uuid") or _generate_uuid(
                    item.get("page_content", "")
                )

                if item_id not in existing_ids:
                    new_list.append(
                        Document(**{**item, "metadata": {**metadata, "uuid": item_id}})
                    )
                    existing_ids.add(item_id)

            elif isinstance(item, Document):
                item_id = item.metadata.get("uuid", "")
                if not item_id:
                    item_id = _generate_uuid(item.page_content)
                    new_item = item.copy(deep=True)
                    new_item.metadata["uuid"] = item_id
                else:
                    new_item = item

                if item_id not in existing_ids:
                    new_list.append(new_item)
                    existing_ids.add(item_id)

    return existing_list + new_list


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
    documents: Annotated[list[Document], reduce_docs] = Field(default_factory=list)
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
    pass
    # thread_config = {"configurable": {"thread_id": "123"}}
    # graph = get_graph()

    # next_state = aget_next_state(graph, thread_config)
    # print(f"graph state: {next_state}")
    # async for chunk in graph.astream(
    #     {
    #         "human_messages": [
    #             HumanMessage(content="busca en internet que es el metabolismo?")
    #         ]
    #     },
    #     config=thread_config,
    #     stream_mode="updates",
    #     subgraphs=True,
    # ):
    #     print(chunk)
