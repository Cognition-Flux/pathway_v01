"""streamer.py.

This helper module acts as the **bridge between the multi-agent back-end and the
Reflex front-end**.

The :func:`stream_updates_including_subgraphs` generator consumes the LangGraph
`planner` graph and *yields incremental updates* that the UI can render in
real-time:

* **Natural-language chunks** - Partial assistant replies that can be appended
  to the chat as they arrive.
* **Reasoning traces** - Short strings giving visibility into the agentic
  thought process.  These are displayed in the right-hand sidebar.
* **Plotly visualisations** - If a node called ``plot_generator`` is executed
  the graph is returned as a **Plotly JSON string** (produced with
  ``fig.to_json()`` on the back-end).  We forward that JSON unchanged so the
  front-end can recreate the figure on the fly.

Keeping the generator purely text / JSON means we don't need to pickle complex
objects across the websocket boundary - everything is serialisable.
"""

# %%
import os
import sys
from collections.abc import Generator
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
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

# os.chdir(package_root)
import re  # placed after other stdlib imports (pylint: disable=wrong-import-position)

from tpp_agentic_graph_v4.workflow import graph  # noqa: E402


def format_reasoning_messages(
    messages: list[str | None] | str | None,
    node_name: str | None = None,
    current_agent: str | None = None,
    llm_model: str | None = None,
) -> str | None:
    """Format a list of reasoning messages into a well-formatted string.

    This function properly formats the reasoning messages, handling special
    cases like "proximo agente" messages. It only returns the latest reasoning
    update to avoid duplication in the UI.

    Args:
        messages: List of reasoning messages, a single message, or None.
        node_name: Name of the node that generated the reasoning.
        current_agent: Current agent from state if available.
        llm_model: LLM model used for the reasoning.

    Returns:
        A properly formatted string with the latest reasoning information, or None.
    """
    if not messages and not current_agent:
        return None

    # Ensure messages is a list for consistent processing
    if messages is None:
        message_list = []
    elif isinstance(messages, str):
        message_list = [messages]
    else:  # It's already a list
        message_list = messages

    next_agent = None
    latest_reasoning_content = []  # Renamed for clarity

    for msg_content in reversed(message_list):  # Iterate over content
        if msg_content is None or (
            isinstance(msg_content, str) and msg_content.strip().lower() == "none"
        ):
            continue

        if isinstance(msg_content, str) and "proximo agente:" in msg_content.lower():
            if not next_agent:
                next_agent = msg_content.split("proximo agente:")[-1].strip()
        else:
            cleaned_msg = str(msg_content)  # Ensure it's a string
            if node_name:
                pretty_name = node_name.replace("_", " ").title()
                header_to_check = f"**{pretty_name}**"
                # Check if the message starts with the header
                # (potentially with newlines before content)
                if header_to_check in cleaned_msg:
                    # Split by the known separator, take the part after it.
                    # This is more robust if there's other text before the
                    # actual reasoning.
                    parts = cleaned_msg.split(f"{header_to_check}\n\n---\n", 1)
                    if len(parts) > 1:
                        cleaned_msg = parts[1].strip()
                    else:  # If separator not found, try splitting by header only
                        parts_alt = cleaned_msg.split(header_to_check, 1)
                        if len(parts_alt) > 1:
                            cleaned_msg = (
                                parts_alt[1].strip().lstrip("-").strip()
                            )  # Remove potential leading --- and spaces

            latest_reasoning_content.append(cleaned_msg)
            break

    display_name = None
    if node_name:
        display_name = node_name.replace("_", " ").title()
    elif current_agent:
        # Use current_agent directly if it's a string, otherwise try to make it
        # presentable.
        display_name = (
            str(current_agent).replace("_", " ").title()
            if isinstance(current_agent, str)
            else "Agente Desconocido"
        )

    if display_name and llm_model:
        display_name = f"{display_name} ({llm_model})"

    result_parts = []
    if display_name:
        result_parts.append(f"#### {display_name} →")

    if latest_reasoning_content:
        reasoning_text = "\n\n".join(latest_reasoning_content)  # Join list of strings
        result_parts.append(reasoning_text)

    if next_agent and next_agent.lower() == "end":
        result_parts.append("#### END")

    if not result_parts:
        return None

    return "\n\n###SPLIT###\n\n".join(result_parts)


def _parse_forecast_output(raw_text: str) -> str:
    """Convert the plain-text forecast output into nicely formatted Markdown.

    The function expects the exact structure produced by
    ``get_forecast_moving_avg``.  It detects the *forecast table*, the list of
    *simulation sections* (one per date) and the *office configuration* block.

    Args:
        raw_text: Full text as produced by the tool.

    Returns:
        A Markdown-formatted string ready to display to the end user.
    """

    # --- Detect the header line ---
    header_match = re.match(r"^(Pronóstico[^\n]+)\n", raw_text)
    if not header_match:
        # Fallback: return wrapped in a code block if unexpected format
        return f"```text\n{raw_text}\n```"

    header_line = header_match.group(1)

    remaining = raw_text[len(header_line) + 1 :]

    # Split forecast table vs. simulation results
    split_token = "Resultados de la simulación a partir del forecast:"
    if split_token in remaining:
        forecast_table_txt, sim_part_full = remaining.split(split_token, 1)
    else:
        # Unexpected – just wrap entire remaining as block
        forecast_table_txt, sim_part_full = remaining, ""

    # Extract office configuration block (if present) before parsing simulation sections
    config_block: str | None = None
    if sim_part_full:
        cfg_match = re.search(
            r"configuración de oficina[\s\S]+$", sim_part_full, re.IGNORECASE
        )
        if cfg_match:
            config_block = cfg_match.group(0).strip()
            sim_part = sim_part_full[: cfg_match.start()]
        else:
            sim_part = sim_part_full
    else:
        sim_part = ""

    # Clean forecast table → keep as code block (preformatted)
    forecast_block = f"```text\n{forecast_table_txt.strip()}\n```"

    md_parts = [f"### {header_line}", forecast_block]

    # Parse simulation sections (delimited by === lines)
    if sim_part.strip():
        sections = re.split(r"={5,}[^\n]*={5,}", sim_part)
        headers = re.findall(r"={5,}\s*([^=]+?)\s*=+", sim_part)

        sim_md_parts = []
        for hdr, body in zip(
            headers, sections[1:]
        ):  # sections[0] is before first delimiter
            hdr_clean = hdr.strip().strip("=").strip()
            body_clean = body.strip()
            # Body may already contain markdown tables, keep as-is
            sim_md_parts.append(f"#### {hdr_clean}\n\n{body_clean}")

        if sim_md_parts:
            md_parts.append("### Resultados de la simulación")
            md_parts.extend(sim_md_parts)

    # Append office configuration block once (if found)
    if config_block:
        md_parts.append("### Configuración de la oficina usada en la simulación")
        md_parts.append(f"```text\n{config_block}\n```")

    return "\n\n".join(md_parts)


def stream_updates_including_subgraphs(
    mensaje_del_usuario: str,
    config: dict | None = None,
) -> Generator[tuple[str | None, str | None, str | None], None, None]:
    """Stream updates including subgraphs.

    This function streams updates from the planner graph, including messages and
    reasoning from all nodes.

    Args:
        mensaje_del_usuario: The user message to process.
        config: The configuration for the graph.

    Yields:
        A tuple of (message, reasoning, plot).
    """
    final_config = (
        config if config is not None else {"configurable": {"thread_id": "1"}}
    )
    final_config["recursion_limit"] = 200

    seen_reasoning = set()
    last_agent_context = {"current_agent": None, "llm_model": None}

    initial_graph_input: dict | Command
    if not graph.get_state(final_config).next:
        initial_graph_input = {
            "messages": mensaje_del_usuario,
            "scratchpad": [],
            "user_parameters_for_forecast": [],
        }
    else:
        initial_graph_input = Command(resume=mensaje_del_usuario)

    for chunk in graph.stream(
        initial_graph_input,
        final_config,
        stream_mode="updates",
        subgraphs=True,
    ):
        state, node_info = chunk
        print(f"------------node: {node_info}")
        print(f"------------state: {state}")

        reasoning_content: str | list[str] | None = None
        reasoning_source_node: str | None = None

        current_agent_from_node: str | None = None
        llm_model_from_node: str | None = None

        for _key, value in node_info.items():
            if isinstance(value, dict):
                current_agent_from_node = value.get("current_agent")
                llm_model_from_node = value.get("llm_model")

        if "__interrupt__" not in node_info and (
            current_agent_from_node or llm_model_from_node
        ):
            if current_agent_from_node:
                last_agent_context["current_agent"] = current_agent_from_node
            if llm_model_from_node:
                last_agent_context["llm_model"] = llm_model_from_node

        reasoning_data_source_key = next(
            (
                k
                for k in node_info
                if isinstance(node_info[k], dict) and "reasoning" in node_info[k]
            ),
            None,
        )
        if reasoning_data_source_key:
            reasoning_content = node_info[reasoning_data_source_key].get("reasoning")
            reasoning_source_node = reasoning_data_source_key

        plot_data: str | None = None

        def yield_node_updates(
            _node_key: str,
            messages: list[str | None] | str | None,
            default_msg: str | None = None,
            is_plot_node: bool = False,
            current_reasoning_content: str | list[str] | None = reasoning_content,
            current_reasoning_source_node: str | None = reasoning_source_node,
            current_agent: str | None = current_agent_from_node,
            current_llm_model: str | None = llm_model_from_node,
            current_plot_data_outer: str | None = plot_data,
        ):
            nonlocal plot_data
            node_reasoning_str = format_reasoning_messages(
                current_reasoning_content,
                current_reasoning_source_node,
                current_agent,
                current_llm_model,
            )

            reasoning_to_send_once = None
            if node_reasoning_str and node_reasoning_str not in seen_reasoning:
                seen_reasoning.add(node_reasoning_str)
                reasoning_to_send_once = node_reasoning_str

            actual_messages = []
            if messages:
                actual_messages = [messages] if isinstance(messages, str) else messages
            elif default_msg:
                actual_messages = [default_msg]

            if (
                not actual_messages
                and not reasoning_to_send_once
                and not (is_plot_node and current_plot_data_outer)
            ):
                return

            if not actual_messages and (
                reasoning_to_send_once or (is_plot_node and current_plot_data_outer)
            ):
                yield (
                    None,
                    reasoning_to_send_once,
                    current_plot_data_outer if is_plot_node else None,
                )
                return

            for idx, msg_text in enumerate(actual_messages):
                plot_to_yield_for_this_message = None
                if is_plot_node and idx == 0:
                    plot_to_yield_for_this_message = current_plot_data_outer

                reasoning_for_this_message = (
                    reasoning_to_send_once if idx == 0 else None
                )
                yield (
                    str(msg_text).strip() if msg_text else None,
                    reasoning_for_this_message,
                    plot_to_yield_for_this_message,
                )

        processed_node = False
        for node_name_key in node_info:
            if node_name_key == "__interrupt__":
                interrupt_msg_value = node_info[node_name_key][0].value
                interrupt_agent = last_agent_context.get("current_agent")
                interrupt_model = last_agent_context.get("llm_model")
                node_reasoning_str = format_reasoning_messages(
                    reasoning_content, interrupt_agent, interrupt_agent, interrupt_model
                )
                reasoning_to_send_interrupt = None
                if node_reasoning_str and node_reasoning_str not in seen_reasoning:
                    seen_reasoning.add(node_reasoning_str)
                    reasoning_to_send_interrupt = node_reasoning_str
                yield (interrupt_msg_value, reasoning_to_send_interrupt, None)
                processed_node = True
                break

            node_data = node_info[node_name_key]
            if isinstance(node_data, dict):
                messages_from_node = node_data.get("messages")

                default_message_for_node: str | None = None
                if node_name_key == "retrieve_documents" and not messages_from_node:
                    default_message_for_node = "Leyendo y recopilando documentos..."
                elif node_name_key == "rag" and not messages_from_node:
                    doc_count = len(node_data.get("documents", []))
                    default_message_for_node = (
                        f"Se recopilaron {doc_count} documentos..."
                    )
                elif node_name_key == "report_generator":
                    # Check for report content first
                    report_content = node_data.get("report")
                    regular_messages = node_data.get("messages")

                    # If there's a report, use that as the main message
                    if report_content:
                        messages_from_node = report_content
                    # If there's no report but there are regular messages, use those
                    elif regular_messages:
                        messages_from_node = regular_messages
                    else:
                        messages_from_node = None
                elif node_name_key == "plot_generator":
                    plot_data = node_data.get("plot")
                    default_message_for_node = "gráfico generado"
                elif node_name_key == "make_forecast":
                    plot_data = node_data.get("plot")
                    default_message_for_node = "📈 Forecast generado con visualización"

                    # Detect and prettify forecast text if present
                    fcast_resp = node_data.get("forecast_generated_response")
                    if fcast_resp:
                        pretty_fcast = _parse_forecast_output(fcast_resp)
                        messages_from_node = [pretty_fcast]

                if isinstance(messages_from_node, str):
                    # Mensaje simple en forma de cadena — se envía tal cual.
                    messages_list_for_yield = [messages_from_node]
                elif isinstance(messages_from_node, list):
                    # Lista de mensajes — filtramos cualquier ``ToolMessage``.
                    messages_list_for_yield = [
                        m for m in messages_from_node if not isinstance(m, ToolMessage)
                    ]
                elif isinstance(messages_from_node, ToolMessage):
                    # Mensaje único de tipo ``ToolMessage`` — lo descartamos.
                    messages_list_for_yield = []
                else:
                    messages_list_for_yield = []

                is_plot_type_node = node_name_key in ["plot_generator", "make_forecast"]
                yield from yield_node_updates(
                    node_name_key,
                    messages_list_for_yield,
                    default_message_for_node,
                    is_plot_type_node,
                    current_reasoning_content=reasoning_content,
                    current_reasoning_source_node=reasoning_source_node,
                    current_agent=current_agent_from_node,
                    current_llm_model=llm_model_from_node,
                    current_plot_data_outer=plot_data,
                )
                processed_node = True
                break

        if not processed_node and reasoning_content and not any(node_info.values()):
            node_reasoning_str = format_reasoning_messages(
                reasoning_content,
                reasoning_source_node,
                current_agent_from_node,
                llm_model_from_node,
            )
            if node_reasoning_str and node_reasoning_str not in seen_reasoning:
                seen_reasoning.add(node_reasoning_str)
                yield (None, node_reasoning_str, None)


if __name__ == "__main__":
    MESSAGE_FROM_USER = (
        "que puedes hacer para mi"
        # "necesito un forecast de la variable biomarcador de la tabla"
        # "patient_time_series"
        # ",con contexto de 20 puntos y predicción de 10 puntos"
        # "si"
        # "no"
    )

    for mensaje in stream_updates_including_subgraphs(MESSAGE_FROM_USER, config=None):
        print(mensaje)
