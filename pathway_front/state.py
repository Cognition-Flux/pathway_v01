# %%
# state.py
"""Estado de la aplicación Reflex (front-end).

Esta clase se encarga de **recibir** los mensajes emitidos por
``stream_updates_including_subgraphs`` y transformarlos en una estructura que
la UI pueda mostrar.  El flujo completo para los gráficos es el siguiente::

    Back-end (LangGraph)          ->  stream_updates_including_subgraphs
    Plotly ``Figure``            ->  ``fig.to_json()``  (plain str)
    └─ ``plot`` field (JSON str) ->  Estado.frontend.answer
                                    └─ pio.from_json(plot) → Figure
                                       chat_history.append((None, Figure))
    Front-end (chat component)   ->  detecta ``Figure`` y llama a
                                    rx.plotly(data=figure)

Al mantener la figura como objeto ``go.Figure`` en el *state*, Reflex puede
serializarla correctamente al JSON que espera el componente `rx.plotly`.

# ruff: noqa: RUF012
"""

import json
import time
from collections.abc import AsyncGenerator
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import reflex as rx
import yaml
from dotenv import load_dotenv

from agentic_workflow.streamers.by_updates import stream_updates_including_subgraphs


load_dotenv(override=True)


class State(rx.State):
    """Estado principal de la aplicación."""

    # Estado para controlar la visualización del spinner de carga
    is_loading: bool = False
    # The current question being asked.
    question: str

    # Keep track of the chat history as a list of (question, answer) tuples.
    # Answer can be a string or a Plotly figure.
    chat_history: list[tuple[str | None, str | go.Figure | None]] = []

    # Keep track of reasoning messages
    reasoning_history: list[str] = []

    # Lista de PDFs indexados (extraídos del YAML)
    files_list: list[str] = []

    # Lista de tablas cargadas (extraídas de otro YAML)
    tables_list: list[str] = []

    # Pestaña seleccionada en la barra lateral izquierda ("documentos" | "faq")
    sidebar_tab: str = "faq"

    # Límite de refresco independiente para tablas (en segundos)
    last_update_tables: float = 0.0

    # Tiempo de la última actualización
    last_update: float = 0

    # Estado para animación del botón de actualización
    is_refreshing: bool = False

    def on_mount(self) -> None:
        """Se ejecuta cuando el componente se monta."""
        self.update_docs_files()
        self.update_tables()

    def update_docs_files(self) -> None:
        """Lee ``indexed_pdfs.yaml`` y extrae la lista de PDFs.

        El archivo YAML mantiene la *fuente de verdad* sobre los documentos
        que han sido procesados e indexados por la capa de vectorstore.  La
        estructura esperada es::

            indexed_pdfs:
              - name: some_file.pdf
                chunks: 123  # <- opcional
              - name: other.pdf
            total_pdfs: 2
            total_chunks: 456

        Sólo necesitamos los valores del campo ``name`` para mostrarlos en la
        barra lateral; cualquier otro metadato se ignora.
        """
        current_time = time.time()

        # Evitar refrescos excesivos (≥ 2 s entre llamadas)
        if current_time - self.last_update <= 2:
            return

        yaml_path = Path("pathway/vectorstore/indexed_pdfs.yaml")

        if not yaml_path.exists():
            # Si el archivo no existe, limpiamos la lista y salimos.
            self.files_list = []
            self.last_update = current_time
            return

        try:
            data: dict | None = yaml.safe_load(yaml_path.read_text())
        except yaml.YAMLError as exc:  # pragma: no cover – logging sencillo
            print(f"[State] Error parsing YAML: {exc}")
            self.files_list = []
            self.last_update = current_time
            return

        # Extraer nombres, asegurando que la estructura sea la prevista.
        indexed = data.get("indexed_pdfs", []) if isinstance(data, dict) else []
        self.files_list = [
            entry["name"]
            for entry in indexed
            if isinstance(entry, dict) and "name" in entry
        ]

        self.last_update = current_time

    def update_tables(self) -> None:
        """Lee ``loaded_tables.yaml`` y actualiza :pyattr:`tables_list`.

        El archivo se genera en la fase de carga de datos tabulares y su
        formato esperado es::

            loaded_tables:
              - name: my_table
                columns: 10
                rows: 123
        """
        current_time = time.time()

        if current_time - self.last_update_tables <= 2:
            return

        yaml_path = Path("pathway/agent_for_tables/loaded_tables.yaml")

        if not yaml_path.exists():
            self.tables_list = []
            self.last_update_tables = current_time
            return

        try:
            data: dict | None = yaml.safe_load(yaml_path.read_text())
        except yaml.YAMLError as exc:  # pragma: no cover
            print(f"[State] Error parsing tables YAML: {exc}")
            self.tables_list = []
            self.last_update_tables = current_time
            return

        loaded = data.get("loaded_tables", []) if isinstance(data, dict) else []
        self.tables_list = [
            entry["name"]
            for entry in loaded
            if isinstance(entry, dict) and "name" in entry
        ]

        self.last_update_tables = current_time

    def refresh_documents(self) -> None:
        """Fuerza la recarga de ``indexed_pdfs.yaml``.

        Se utiliza al pulsar el botón *refresh* de la barra lateral; marca
        temporalmente ``is_refreshing`` para que la UI muestre la animación.
        """
        # Marcar como refrescando para la animación
        self.is_refreshing = True
        # Forzar actualización cambiando last_update a 0
        self.last_update = 0
        self.update_docs_files()
        # Refrescar también las tablas para mantener coherencia visual
        self.last_update_tables = 0
        self.update_tables()
        # Finalizar la animación después de la actualización
        self.is_refreshing = False

    async def refresh_documents_async(self) -> None:
        """Versión *async* de :meth:`refresh_documents` con breve retardo.

        Permite que la animación de giro del icono sea perceptible antes de
        restaurar el estado por lo que se intercala un ``yield`` entre el
        comienzo y la finalización.
        """
        # Marcar como refrescando para la animación
        self.is_refreshing = True
        # Forzar actualización cambiando last_update a 0
        self.last_update = 0
        self.update_docs_files()
        self.last_update_tables = 0
        self.update_tables()
        # Esperar un pequeño periodo para que la animación sea visible
        yield
        # Finalizar la animación después de la actualización
        self.is_refreshing = False

    @rx.var
    def get_docs_files(self) -> list[str]:
        """Devuelve la lista de PDFs indexados (lazy refresh)."""
        # Actualizar la lista de archivos antes de devolverla
        self.update_docs_files()
        return self.files_list

    @rx.var
    def get_tables(self) -> list[str]:
        """Devuelve la lista de tablas cargadas (lazy refresh)."""
        self.update_tables()
        return self.tables_list

    def handle_key_press(self, key: str) -> None:
        """Maneja los eventos de teclado.

        Envía la pregunta actual cuando el usuario presiona **Enter** y el
        campo de texto no está vacío.
        """
        if key == "Enter" and self.question.strip():
            # Devolver la referencia al EventHandler para encadenarlo correctamente
            # en lugar de **invocarlo** (lo cual rompería el flujo de Reflex).
            return State.answer

    async def answer(self) -> AsyncGenerator:
        """Procesa la pregunta del usuario y devuelve respuestas generadas."""
        # Almacenar la pregunta del usuario
        input_question = self.question

        # Añadir la pregunta como nuevo mensaje del usuario
        self.chat_history.append((input_question, None))

        # Limpiar el campo de entrada y mostrar spinner
        self.question = ""
        self.is_loading = True
        yield rx.scroll_to("chat-bottom-anchor")

        # Variable para controlar si ya hemos recibido algún chunk
        chunk_received = False

        try:
            # Procesar cada chunk como un mensaje separado
            for chunk, reasoning, plot in stream_updates_including_subgraphs(
                input_question
            ):
                # Solo procesar chunks no vacíos
                if chunk and chunk.strip():
                    # Si es el primer chunk, lo ponemos como respuesta
                    # a la pregunta del usuario
                    if not chunk_received:
                        chunk_received = True
                        # Ensure chat history is not empty before accessing index -1
                        if self.chat_history:
                            self.chat_history[-1] = (
                                self.chat_history[-1][0],
                                chunk,
                            )
                        else:
                            # Handle case where chat_history might be empty
                            # unexpectedly
                            self.chat_history.append(("", chunk))
                    else:
                        # Para chunks posteriores, crear nuevos mensajes
                        # del asistente
                        self.chat_history.append(("", chunk))
                    # 1) Actualizar UI
                    yield
                    # 2) Desplazar al final una vez renderizado
                    yield rx.scroll_to("chat-bottom-anchor")

                # Procesar y almacenar el gráfico si existe
                if plot:
                    try:
                        # Deserialize the JSON string into a Plotly Figure object
                        # ``plot`` llega como *str* para poder cruzar el límite de
                        # serialización.  Utilizamos ``plotly.io.from_json`` para
                        # reconstruir la instancia ``go.Figure`` que entiende el
                        # componente `rx.plotly`.
                        fig = pio.from_json(plot)
                        # Append the figure to the chat history
                        self.chat_history.append((None, fig))
                        # 1) Actualizar UI con la figura
                        yield
                        # 2) Scroll tras render
                        yield rx.scroll_to("chat-bottom-anchor")
                    except (json.JSONDecodeError, ValueError) as e:
                        # Handle potential errors during JSON parsing or figure creation
                        print(f"Error processing plot data: {e}")
                        # Podríamos añadir un mensaje de error al chat.
                        # Decidimos omitirlo para no interrumpir la experiencia.
                        pass

                # Almacenar el razonamiento si existe y no es duplicado
                if reasoning:
                    raw_reasoning_content = reasoning.strip()
                    if (
                        raw_reasoning_content
                    ):  # Asegurar que el razonamiento no sea solo espacios en blanco
                        # Dividir el razonamiento en partes separadas si contiene el separador
                        if "###SPLIT###" in raw_reasoning_content:
                            parts = raw_reasoning_content.split("###SPLIT###")
                            for part in parts:
                                clean_part = part.strip()
                                if clean_part and (
                                    not self.reasoning_history
                                    or self.reasoning_history[-1] != clean_part
                                ):
                                    self.reasoning_history.append(clean_part)
                                    yield
                                    yield rx.scroll_to("reasoning-bottom-anchor")
                        else:
                            # Es una única pieza de razonamiento (ya formateada por el streamer)
                            if (
                                not self.reasoning_history
                                or self.reasoning_history[-1] != raw_reasoning_content
                            ):
                                self.reasoning_history.append(raw_reasoning_content)
                                yield
                                yield rx.scroll_to("reasoning-bottom-anchor")

        except Exception as exc:  # pragma: no cover – catch-all to avoid WS crash
            # Registrar el error en consola para depuración.
            print(f"[State.answer] Error inesperado: {exc}")

            # Mostrar mensaje de error amigable al usuario.
            self.chat_history.append(
                (None, "⚠️ Ocurrió un error procesando la petición. Inténtalo de nuevo.")
            )
            # Garantizar que la UI se actualice.
            yield rx.scroll_to("chat-bottom-anchor")

        # Si no recibimos ningún chunk, eliminamos la entrada que quedó con respuesta vacía o None
        if (
            not chunk_received
            and self.chat_history
            and (self.chat_history[-1][1] == "" or self.chat_history[-1][1] is None)
        ):
            self.chat_history.pop()

        self.is_loading = False
        # Asegurar desplazamiento al final después de ocultar el spinner
        yield rx.scroll_to("chat-bottom-anchor")


# %%
