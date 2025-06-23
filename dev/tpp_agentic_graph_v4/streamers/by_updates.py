"""streamer.py.

Este módulo auxiliar actúa como el **puente entre el back-end de múltiples
agentes y el front-end de Reflex**. Su función principal es facilitar la
comunicación en tiempo real desde el sistema de agentes hacia la interfaz de
usuario.

El generador :func:`stream_updates_including_subgraphs` consume el grafo
`planner` de LangGraph y *produce actualizaciones incrementales* que la
interfaz de usuario (UI) puede renderizar en tiempo real. Estas actualizaciones
incluyen:

* **Fragmentos de lenguaje natural**: Respuestas parciales del asistente que se
  pueden agregar al chat a medida que llegan, creando una experiencia de
  streaming fluida.
* **Trazas de razonamiento**: Cadenas de texto cortas que ofrecen visibilidad
  sobre el proceso de pensamiento del agente. Estas se muestran en la barra
  lateral derecha de la UI para depuración y transparencia.
* **Visualizaciones de Plotly**: Si un nodo llamado ``plot_generator`` se
  ejecuta, el gráfico se devuelve como una **cadena JSON de Plotly** (producida
  con ``fig.to_json()`` en el back-end). Este JSON se reenvía sin cambios para
  que el front-end pueda recrear la figura dinámicamente.

Mantener el generador basado puramente en texto y JSON simplifica la
arquitectura, ya que no es necesario serializar (usando `pickle`) objetos
complejos a través del websocket. Todo es fácilmente serializable y
deserializable.
"""

# %%
# -----------------------------------------------------------------------------
# Importaciones y Configuración Inicial
# -----------------------------------------------------------------------------
import os
import sys
from collections.abc import Generator
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
from langgraph.types import Command

# pylint: disable=import-error,wrong-import-position

# --- Configuración dinámica del PYTHONPATH ---
# Esta sección ajusta el `sys.path` para asegurar que los módulos del proyecto
# se puedan importar correctamente, sin importar desde dónde se ejecute el
# script. Es una técnica robusta para manejar proyectos con estructuras de
# directorios complejas.

# Obtiene el nombre del módulo principal desde las variables de entorno.
MODULE_NAME = os.getenv("AGENTS_MODULE")
ALSO_MODULE = True
if ALSO_MODULE:
    # Resuelve la ruta del archivo actual.
    current_file = Path(__file__).resolve()
    # Navega hacia arriba en el árbol de directorios hasta encontrar la raíz del
    # paquete.
    package_root = current_file.parent
    while package_root.name != MODULE_NAME:
        package_root = package_root.parent

    # Agrega el directorio padre del paquete raíz al `sys.path` si aún no está
    # allí. Esto permite importaciones absolutas desde la raíz del proyecto.
    if str(package_root.parent) not in sys.path:
        sys.path.insert(0, str(package_root.parent))

# Carga las variables de entorno desde un archivo `.env`.
# `override=True` significa que las variables del archivo `.env` sobrescribirán
# las variables de entorno existentes en el sistema.
load_dotenv(override=True)


# Se importa `re` después de otras importaciones de la biblioteca estándar para
# cumplir con las convenciones de linting.
import re  # pylint: disable=wrong-import-position

# Importa el grafo principal de la aplicación. `noqa: E402` suprime una advertencia
# de linting sobre una importación que no está al principio del archivo.
from tpp_agentic_graph_v4.workflow import graph  # noqa: E402


# -----------------------------------------------------------------------------
# Funciones Auxiliares de Formateo
# -----------------------------------------------------------------------------
def format_reasoning_messages(
    messages: list[str | None] | str | None,
    node_name: str | None = None,
    current_agent: str | None = None,
    llm_model: str | None = None,
) -> str | None:
    """Formatea una lista de mensajes de razonamiento en una cadena bien formateada.

    Esta función procesa los mensajes de razonamiento generados por los agentes
    para que se muestren de forma clara en la interfaz de usuario. Se encarga de
    casos especiales, como los mensajes que indican el "próximo agente", y solo
    devuelve la última actualización de razonamiento para evitar duplicados en
    la UI.

    Args:
        messages: Una lista de mensajes de razonamiento, un solo mensaje o None.
        node_name: El nombre del nodo del grafo que generó el razonamiento.
        current_agent: El nombre del agente actual, extraído del estado del grafo.
        llm_model: El modelo de lenguaje (LLM) utilizado para generar el
                   razonamiento.

    Returns:
        Una cadena de texto formateada con la información de razonamiento más
        reciente, lista para ser mostrada en la UI, o None si no hay nada que
        mostrar.
    """
    # Si no hay mensajes ni un agente actual, no hay nada que formatear.
    if not messages and not current_agent:
        return None

    # Se asegura de que `messages` sea una lista para un procesamiento uniforme.
    if messages is None:
        message_list = []
    elif isinstance(messages, str):
        message_list = [messages]
    else:  # Ya es una lista
        message_list = messages

    next_agent = None
    latest_reasoning_content = []  # Almacena el contenido de razonamiento más reciente.

    # Itera sobre los mensajes en orden inverso para encontrar el más reciente.
    for msg_content in reversed(message_list):
        # Ignora mensajes vacíos o que son explícitamente "None".
        if msg_content is None or (
            isinstance(msg_content, str) and msg_content.strip().lower() == "none"
        ):
            continue

        # Comprueba si el mensaje indica el próximo agente a ejecutar.
        if isinstance(msg_content, str) and "proximo agente:" in msg_content.lower():
            if not next_agent:  # Solo toma el primero que encuentre.
                next_agent = msg_content.split("proximo agente:")[-1].strip()
        else:
            # Procesa el contenido del razonamiento.
            cleaned_msg = str(msg_content)
            if node_name:
                # Formatea el nombre del nodo para una mejor visualización.
                pretty_name = node_name.replace("_", " ").title()
                header_to_check = f"**{pretty_name}**"

                # Intenta limpiar el mensaje eliminando el encabezado autogenerado.
                # Esto es para evitar mostrar información redundante en la UI.
                if header_to_check in cleaned_msg:
                    parts = cleaned_msg.split(f"{header_to_check}\n\n---\n", 1)
                    if len(parts) > 1:
                        cleaned_msg = parts[1].strip()
                    else:
                        parts_alt = cleaned_msg.split(header_to_check, 1)
                        if len(parts_alt) > 1:
                            cleaned_msg = parts_alt[1].strip().lstrip("-").strip()

            latest_reasoning_content.append(cleaned_msg)
            # Una vez que se encuentra un contenido de razonamiento válido, se detiene.
            break

    # Determina el nombre a mostrar para el agente o nodo.
    display_name = None
    if node_name:
        display_name = node_name.replace("_", " ").title()
    elif current_agent:
        display_name = (
            str(current_agent).replace("_", " ").title()
            if isinstance(current_agent, str)
            else "Agente Desconocido"
        )

    # Agrega el nombre del modelo LLM si está disponible.
    if display_name and llm_model:
        display_name = f"{display_name} ({llm_model})"

    # Construye el resultado final en formato Markdown.
    result_parts = []
    if display_name:
        result_parts.append(f"#### {display_name} →")

    if latest_reasoning_content:
        reasoning_text = "\n\n".join(latest_reasoning_content)
        result_parts.append(reasoning_text)

    # Si el próximo agente es "END", indica el final del flujo.
    if next_agent and next_agent.lower() == "end":
        result_parts.append("#### END")

    # Si no hay partes para mostrar, devuelve None.
    if not result_parts:
        return None

    # Une todas las partes con un separador especial que la UI puede usar.
    return "\n\n###SPLIT###\n\n".join(result_parts)


def _parse_forecast_output(raw_text: str) -> str:
    """Convierte la salida de texto plano del forecast en un Markdown bien formateado.

    Esta función está diseñada para interpretar la estructura de texto específica
    producida por la herramienta `get_forecast_moving_avg`. Detecta la tabla de
    pronóstico, las secciones de simulación y el bloque de configuración de
    oficina, y los formatea para una visualización clara en la UI.

    Args:
        raw_text: El texto completo tal como lo produce la herramienta de forecast.

    Returns:
        Una cadena de texto en formato Markdown, lista para ser mostrada al usuario.
    """
    # --- Detección del encabezado ---
    # Busca la línea de encabezado que comienza con "Pronóstico".
    header_match = re.match(r"^(Pronóstico[^\n]+)\n", raw_text)
    if not header_match:
        # Si no se encuentra el formato esperado, devuelve el texto original
        # dentro de un bloque de código para mantener el formato.
        return f"```text\n{raw_text}\n```"

    header_line = header_match.group(1)
    # El resto del texto después del encabezado.
    remaining = raw_text[len(header_line) + 1 :]

    # --- Separación de la tabla de forecast y los resultados de simulación ---
    split_token = "Resultados de la simulación a partir del forecast:"
    if split_token in remaining:
        forecast_table_txt, sim_part_full = remaining.split(split_token, 1)
    else:
        # Si no se encuentran resultados de simulación, se asume que todo es
        # parte de la tabla de forecast.
        forecast_table_txt, sim_part_full = remaining, ""

    # --- Extracción del bloque de configuración de oficina ---
    config_block: str | None = None
    if sim_part_full:
        # Busca un bloque de texto que comience con "configuración de oficina".
        cfg_match = re.search(
            r"configuración de oficina[\s\S]+$", sim_part_full, re.IGNORECASE
        )
        if cfg_match:
            config_block = cfg_match.group(0).strip()
            # Elimina el bloque de configuración de la parte de simulación.
            sim_part = sim_part_full[: cfg_match.start()]
        else:
            sim_part = sim_part_full
    else:
        sim_part = ""

    # --- Formateo de las partes ---
    # La tabla de forecast se mantiene como texto preformateado en un bloque de código.
    forecast_block = f"```text\n{forecast_table_txt.strip()}\n```"

    # Lista para ensamblar las partes del Markdown final.
    md_parts = [f"### {header_line}", forecast_block]

    # --- Parseo de las secciones de simulación ---
    if sim_part.strip():
        # Las secciones están delimitadas por líneas de "===".
        sections = re.split(r"={5,}[^\n]*={5,}", sim_part)
        headers = re.findall(r"={5,}\s*([^=]+?)\s*=+", sim_part)

        sim_md_parts = []
        # Combina cada encabezado con su cuerpo correspondiente.
        for hdr, body in zip(headers, sections[1:]):
            hdr_clean = hdr.strip().strip("=").strip()
            body_clean = body.strip()
            # El cuerpo puede contener tablas Markdown, por lo que se mantiene tal cual.
            sim_md_parts.append(f"#### {hdr_clean}\n\n{body_clean}")

        if sim_md_parts:
            md_parts.append("### Resultados de la simulación")
            md_parts.extend(sim_md_parts)

    # --- Añadir el bloque de configuración de oficina al final ---
    if config_block:
        md_parts.append("### Configuración de la oficina usada en la simulación")
        md_parts.append(f"```text\n{config_block}\n```")

    # Une todas las partes en una sola cadena de texto Markdown.
    return "\n\n".join(md_parts)


# -----------------------------------------------------------------------------
# Generador Principal de Streaming
# -----------------------------------------------------------------------------
def stream_updates_including_subgraphs(
    mensaje_del_usuario: str,
    config: dict | None = None,
) -> Generator[tuple[str | None, str | None, str | None], None, None]:
    """Transmite actualizaciones del grafo, incluyendo mensajes de subgrafos.

    Esta función es un generador que se conecta al `stream` del grafo de LangGraph
    y produce una serie de actualizaciones que el front-end puede consumir. Se
    encarga de procesar los eventos del grafo, extraer información relevante
    (mensajes, razonamientos, gráficos) y cederla en un formato estandarizado.

    Args:
        mensaje_del_usuario: El mensaje inicial del usuario que desencadena el
                             proceso.
        config: Una configuración de LangGraph que incluye, por ejemplo, el ID
                del hilo de conversación (`thread_id`).

    Yields:
        Una tupla `(message, reasoning, plot)` donde:
        - `message` (str | None): Un fragmento de texto para mostrar en el chat.
        - `reasoning` (str | None): Una cadena formateada sobre el
          pensamiento del agente.
        - `plot` (str | None): Una cadena JSON que representa una figura de
          Plotly.
    """
    # --- Configuración de la ejecución del grafo ---
    # Asegura que haya una configuración válida, creando una por defecto si es
    # necesario.
    final_config = (
        config if config is not None else {"configurable": {"thread_id": "1"}}
    )
    # Aumenta el límite de recursión para permitir flujos de trabajo largos y
    # complejos.
    final_config["recursion_limit"] = 200

    # --- Variables de estado para el streaming ---
    # `seen_reasoning` evita enviar la misma traza de razonamiento varias veces.
    seen_reasoning = set()
    # `last_agent_context` almacena el último agente y modelo vistos para
    # atribuir correctamente los mensajes de interrupción.
    last_agent_context = {"current_agent": None, "llm_model": None}

    # --- Preparación de la entrada inicial para el grafo ---
    initial_graph_input: dict | Command
    # Si el estado del grafo está vacío, es una nueva conversación.
    if not graph.get_state(final_config).next:
        initial_graph_input = {
            "messages": mensaje_del_usuario,
            "scratchpad": [],
            "user_parameters_for_forecast": [],
        }
    else:
        # Si ya hay un estado, se reanuda la conversación con el nuevo mensaje.
        initial_graph_input = Command(resume=mensaje_del_usuario)

    # --- Bucle principal de streaming ---
    # Itera sobre las actualizaciones ("updates") del grafo. `subgraphs=True`
    # asegura que también se reciban eventos de los grafos anidados.
    for chunk in graph.stream(
        initial_graph_input,
        final_config,
        stream_mode="updates",
        subgraphs=True,
    ):
        state, node_info = chunk
        # Imprime información de depuración en la consola del servidor.
        print(f"------------node: {node_info}")
        print(f"------------state: {state}")

        # --- Extracción de datos del `chunk` de actualización ---
        reasoning_content: str | list[str] | None = None
        reasoning_source_node: str | None = None
        current_agent_from_node: str | None = None
        llm_model_from_node: str | None = None

        # Busca en `node_info` el agente actual y el modelo LLM.
        for _key, value in node_info.items():
            if isinstance(value, dict):
                current_agent_from_node = value.get("current_agent")
                llm_model_from_node = value.get("llm_model")

        # Actualiza el contexto del último agente visto si no es una interrupción.
        if "__interrupt__" not in node_info and (
            current_agent_from_node or llm_model_from_node
        ):
            if current_agent_from_node:
                last_agent_context["current_agent"] = current_agent_from_node
            if llm_model_from_node:
                last_agent_context["llm_model"] = llm_model_from_node

        # Busca el contenido de razonamiento en el `chunk`.
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

        # --- Función anidada para generar las actualizaciones ---
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
            """Función auxiliar para formatear y ceder una actualización.

            Esta función se encarga de ensamblar una tupla de actualización
            `(message, reasoning, plot)` y cederla. Maneja la lógica para no
            enviar razonamientos duplicados y asociar el gráfico con el primer
            mensaje de un nodo.
            """
            nonlocal plot_data
            # Formatea el razonamiento para su visualización.
            node_reasoning_str = format_reasoning_messages(
                current_reasoning_content,
                current_reasoning_source_node,
                current_agent,
                current_llm_model,
            )

            # Envía el razonamiento solo si es nuevo.
            reasoning_to_send_once = None
            if node_reasoning_str and node_reasoning_str not in seen_reasoning:
                seen_reasoning.add(node_reasoning_str)
                reasoning_to_send_once = node_reasoning_str

            # Prepara la lista de mensajes a enviar.
            actual_messages = []
            if messages:
                actual_messages = [messages] if isinstance(messages, str) else messages
            elif default_msg:
                actual_messages = [default_msg]

            # Si no hay nada que enviar (ni mensaje, ni razonamiento, ni gráfico),
            # simplemente retorna.
            if (
                not actual_messages
                and not reasoning_to_send_once
                and not (is_plot_node and current_plot_data_outer)
            ):
                return

            # Si solo hay razonamiento o un gráfico, envíalo sin mensaje.
            if not actual_messages and (
                reasoning_to_send_once or (is_plot_node and current_plot_data_outer)
            ):
                yield (
                    None,
                    reasoning_to_send_once,
                    current_plot_data_outer if is_plot_node else None,
                )
                return

            # Itera sobre los mensajes y los cede uno por uno.
            for idx, msg_text in enumerate(actual_messages):
                plot_to_yield_for_this_message = None
                # Asocia el gráfico solo con el primer mensaje del nodo.
                if is_plot_node and idx == 0:
                    plot_to_yield_for_this_message = current_plot_data_outer

                # Asocia el razonamiento solo con el primer mensaje.
                reasoning_for_this_message = (
                    reasoning_to_send_once if idx == 0 else None
                )

                # Cede la tupla de actualización.
                yield (
                    str(msg_text).strip() if msg_text else None,
                    reasoning_for_this_message,
                    plot_to_yield_for_this_message,
                )

        # --- Lógica de procesamiento de nodos ---
        processed_node = False
        for node_name_key in node_info:
            # Caso 1: El grafo está en un estado de interrupción (esperando entrada).
            if node_name_key == "__interrupt__":
                interrupt_msg_value = node_info[node_name_key][0].value
                interrupt_agent = last_agent_context.get("current_agent")
                interrupt_model = last_agent_context.get("llm_model")
                # Formatea cualquier razonamiento pendiente.
                node_reasoning_str = format_reasoning_messages(
                    reasoning_content, interrupt_agent, interrupt_agent, interrupt_model
                )
                reasoning_to_send_interrupt = None
                if node_reasoning_str and node_reasoning_str not in seen_reasoning:
                    seen_reasoning.add(node_reasoning_str)
                    reasoning_to_send_interrupt = node_reasoning_str
                # Cede el mensaje de interrupción y el razonamiento.
                yield (interrupt_msg_value, reasoning_to_send_interrupt, None)
                processed_node = True
                break

            # Caso 2: Procesamiento de un nodo estándar.
            node_data = node_info[node_name_key]
            if isinstance(node_data, dict):
                messages_from_node = node_data.get("messages")

                # --- Lógica para mensajes por defecto según el nodo ---
                # Si un nodo no genera un mensaje explícito, podemos proporcionar
                # uno por defecto para mejorar la experiencia del usuario.
                default_message_for_node: str | None = None
                if node_name_key == "retrieve_documents" and not messages_from_node:
                    default_message_for_node = "Leyendo y recopilando documentos..."
                elif node_name_key == "rag" and not messages_from_node:
                    doc_count = len(node_data.get("documents", []))
                    default_message_for_node = (
                        f"Se recopilaron {doc_count} documentos..."
                    )
                elif node_name_key == "report_generator":
                    # El generador de informes puede tener contenido en `report` o `messages`.
                    report_content = node_data.get("report")
                    regular_messages = node_data.get("messages")
                    if report_content:
                        messages_from_node = report_content
                    elif regular_messages:
                        messages_from_node = regular_messages
                    else:
                        messages_from_node = None
                elif node_name_key == "plot_generator":
                    plot_data = node_data.get("plot")  # Extrae el JSON del gráfico.
                    default_message_for_node = "gráfico generado"
                elif node_name_key == "make_forecast":
                    plot_data = node_data.get("plot")
                    default_message_for_node = "📈 Forecast generado con visualización"
                    # Si hay una respuesta de forecast en texto, la formatea.
                    fcast_resp = node_data.get("forecast_generated_response")
                    if fcast_resp:
                        pretty_fcast = _parse_forecast_output(fcast_resp)
                        messages_from_node = [pretty_fcast]

                # --- Normalización de mensajes ---
                # Se asegura de que `messages_list_for_yield` sea siempre una lista.
                if isinstance(messages_from_node, str):
                    messages_list_for_yield = [messages_from_node]
                elif isinstance(messages_from_node, list):
                    # Filtra `ToolMessage`, que no son para el usuario final.
                    messages_list_for_yield = [
                        m for m in messages_from_node if not isinstance(m, ToolMessage)
                    ]
                elif isinstance(messages_from_node, ToolMessage):
                    messages_list_for_yield = []
                else:
                    messages_list_for_yield = []

                # Llama a la función anidada para ceder las actualizaciones.
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
                break  # Sale del bucle una vez que se ha procesado un nodo.

        # Caso 3: Solo hay razonamiento, sin un nodo de mensaje activo.
        # Esto puede ocurrir si un agente "piensa" pero no produce una salida
        # de mensaje inmediata.
        if not processed_node and reasoning_content and not any(node_info.values()):
            node_reasoning_str = format_reasoning_messages(
                reasoning_content,
                reasoning_source_node,
                current_agent_from_node,
                llm_model_from_node,
            )
            # Si el razonamiento es nuevo, lo cede.
            if node_reasoning_str and node_reasoning_str not in seen_reasoning:
                seen_reasoning.add(node_reasoning_str)
                yield (None, node_reasoning_str, None)


# -----------------------------------------------------------------------------
# Bloque de Ejecución para Pruebas
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Este bloque de código solo se ejecuta cuando el script se corre
    # directamente (por ejemplo, `python streamer.py`). Es útil para probar y
    # depurar la lógica del streamer de forma aislada.

    # Define un mensaje de usuario de ejemplo para la prueba.
    # Puedes descomentar otras líneas para probar diferentes escenarios.
    MESSAGE_FROM_USER = (
        "que puedes hacer para mi"
        # "necesito un forecast de la variable biomarcador de la tabla"
        # "patient_time_series"
        # ",con contexto de 20 puntos y predicción de 10 puntos"
        # "si"
        # "no"
    )

    # Llama al generador con el mensaje de prueba y recorre las actualizaciones.
    for mensaje in stream_updates_including_subgraphs(MESSAGE_FROM_USER, config=None):
        # Imprime cada tupla de actualización en la consola.
        print(mensaje)
