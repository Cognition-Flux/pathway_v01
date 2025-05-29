"""Graph definition for the multiturn conversational agent."""

# %%
import uuid

from dotenv import load_dotenv
from langchain_core.messages.tool import ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.func import entrypoint, task
from langgraph.graph import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command, interrupt

from agentic_workflow.utils import get_llm


load_dotenv(override=True)


@tool
def string_to_uuid(input_string: str) -> str:
    """Convert a string to a UUID5."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, input_string))


@tool(return_direct=True)
def transfer_to_hotel_advisor():
    """Ask hotel advisor agent for help."""
    return "Successfully transferred to hotel advisor"


# define an agent
travel_advisor_tools = [transfer_to_hotel_advisor]
travel_advisor = create_react_agent(get_llm(), travel_advisor_tools)


# define a task that calls an agent
@task
def call_travel_advisor(messages: list) -> list:
    """Call the travel advisor agent with the given messages."""
    response = travel_advisor.invoke({"messages": messages})
    return response["messages"]


checkpointer = MemorySaver()


@entrypoint(checkpointer)
def multi_turn_graph(messages: list, previous: list | None = None) -> list:
    """Entrypoint for the multiturn conversational graph."""
    previous = previous or []
    messages = add_messages(previous, messages)
    if len(messages) > 20:
        return interrupt("Too many messages")  # Interrupt if too many messages
    call_active_agent = call_travel_advisor
    while True:
        agent_messages = call_active_agent(messages).result()
        messages = add_messages(messages, agent_messages)

        if not isinstance(agent_messages[-1], ToolMessage):
            print(agent_messages[-1].pretty_print())
            user_input = interrupt(value="Ready for user input.")
            human_message = {
                "role": "user",
                "content": user_input,
                "id": string_to_uuid(user_input),
            }
            messages = add_messages(messages, [human_message])
            continue
        tool_call = agent_messages[-1]
        print(tool_call.pretty_print())
        if tool_call.name == "transfer_to_hotel_advisor":
            call_active_agent = call_travel_advisor
        else:
            raise ValueError(f"Expected transfer tool, got '{tool_call.name}'")
    return entrypoint.final(value=agent_messages[-1], save=messages)


thread_config = {"configurable": {"thread_id": uuid.uuid4()}}

inputs = [
    {
        "role": "user",
        "content": "transfer_to_hotel_advisor",
        "id": str(uuid.uuid4()),
    },
    Command(resume="cual fue mi ultima pregunta?"),
]

for idx, user_input in enumerate(inputs):
    print()
    print(f"--- Conversation Turn {idx + 1} ---")
    print()
    print(f"User: {user_input}")
    print()
    for update in multi_turn_graph.stream(
        user_input,
        config=thread_config,
        stream_mode="updates",
    ):
        for node_id, value in update.items():
            print(f"\n-- Node {node_id}")
            print(f"-- Value: {value}")
