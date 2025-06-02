# %%
"""**Capa de presentación de Pathway**.

Este módulo reúne **TODOS** los componentes de Reflex que conforman la interfaz
de usuario.  Está organizado en bloques lógicos claramente delimitados para que
cualquier desarrollador pueda entender la relación entre *estado → componente →
evento* de un vistazo.

Estructura a alto nivel
----------------------
1. **Pequeños _helpers_**
   • `sidebar_message` – formatea una cadena para la traza de razonamiento.
   • `qa`              – renderiza un par *pregunta ↔︎ respuesta*.
   • `plot_display`    – envuelve un `go.Figure` en `rx.plotly`.

2. **Sidebars**
   • `sidebar()`            – trazas de razonamiento (derecha).
   • `documents_sidebar()`  – PDFs y tablas indexadas (izquierda).

3. **Cuerpo principal**
   • `chat()`         – historial completo de mensajes.
   • `action_bar()`   – input + botón de envío.
   • `loading_spinner()` – indicador cuando `State.is_loading`.

4. **Página `index()`**
   Orquesta *sidebars + chat + barra de acción* y
   registra dos bloques `<script>` con lógica **pura-navegador** para permitir
   el *resizable* de los sidebars.  Estos scripts NO interactúan con Python
   directamente —únicamente manipulan el DOM—, por lo que se incluyen aquí
   como cadenas multilinea para mantener el proyecto auto-contenible.

Convenciones
-----------
* Todos los valores de estilo se centralizan en `style.py` (token design-system).
* Las `id` de nodos que se usan como anclas de scroll o referencias JS están
  declaradas en *snake_case* para diferenciarlas de las `class` CSS.
* El ancho de ambos sidebars se controla mediante CSS custom properties:
  `--sidebar-width` y `--left-sidebar-width`.  Esto permite transiciones fluidas
  y evita re-render de React al hacer `drag`.

Importante: **no** hay estado local de componente; todo vive en `pathway_front.state`.
De esta forma mantenemos un único _source of truth_ que sincroniza la interfaz
con las emisiones del *back-end* (LangGraph streamer).
"""

import plotly.graph_objects as go
import reflex as rx

from pathway_front import style
from pathway_front.state import State


# %%
def sidebar_message(message: str | rx.Var[str]) -> rx.Component:
    """Renderiza un mensaje en la barra lateral.

    Utiliza `rx.cond` para distinguir entre encabezados (líneas que comienzan
    con `####`) y mensajes normales, evitando evaluar un `Var` en un contexto
    booleano del lado de Python.
    """

    def _header(msg):
        """Renderiza un encabezado con flechas de transición entre agentes."""
        # Los encabezados tienen formato: #### Agent Name → o #### → Agent Name
        content = msg

        return rx.box(
            rx.markdown(
                content,
                style=style.sidebar_message_style
                | {
                    "padding": "0.1em 0.2em",
                    "margin_y": "0.3em",
                    "font_size": "0.8rem",  # Reducido un 20% del tamaño estándar
                    "line_height": "1.3",
                    "letter_spacing": "0.3px",
                    "background": "transparent",
                    "border_left": "none",
                    "color": "rgba(120, 180, 245, 0.95)",
                    "font_weight": "600",
                    "text_align": "center",
                },
            ),
            width="100%",
            padding_y="0.1em",
            margin_top="0.25em",
            margin_bottom="0.05em",
            cursor="default",  # Sin cursor pointer
            style={
                "pointer_events": "none",  # Sin eventos de click
                "transition": "none",  # Sin transiciones
            },
        )

    def _bubble(msg):
        """Renderiza una burbuja de mensaje estándar usando los estilos del chat principal."""
        return rx.box(
            rx.markdown(
                msg,
                style=style.sidebar_message_style
                | {
                    "font_size": "0.68rem",
                    "margin": "0",
                    "padding": "0.4em 0.6em",
                    "max_width": "100%",
                    "text_align": "justify",
                },
            ),
            width="100%",
            margin_bottom="0.15em",
            cursor="pointer",
            on_click=lambda m=msg: [
                rx.set_clipboard(m),
                rx.toast.success("Copiado", duration=1500),
            ],
            style={
                "transition": "transform 0.1s ease, opacity 0.1s ease",
                "_active": {"transform": "scale(0.96)", "opacity": "0.9"},
            },
        )

    # Verificar si es un encabezado (comienza con ####)
    is_header = False
    if isinstance(message, rx.Var):
        condition = message.startswith("####")
    else:
        msg_str = str(message).strip()
        is_header = msg_str.startswith("####")

    return rx.cond(
        condition if isinstance(message, rx.Var) else is_header,
        _header(message),
        _bubble(message),
    )


def sidebar() -> rx.Component:
    """Renderiza la barra lateral con respuestas del agente."""
    return rx.box(
        # Estilo para diferenciar entre nombres de agentes y texto de razonamiento
        rx.script(
            r"""
            (function() {
                // Aplicar estilos simplificados que mantienen la diferenciación
                const style = document.createElement('style');
                style.textContent = `
                    /* Estilo para encabezados (nombres de agentes) - SIN hover ni efectos */
                    #right-sidebar h4:not(#sidebar-title-heading),
                    #right-sidebar h1:not(#sidebar-title-heading),
                    #right-sidebar h2:not(#sidebar-title-heading),
                    #right-sidebar h3:not(#sidebar-title-heading),
                    #right-sidebar h5:not(#sidebar-title-heading),
                    #right-sidebar h6:not(#sidebar-title-heading) {
                        font-size: 0.7rem !important;
                        color: rgba(120, 180, 245, 0.95) !important;
                        padding: 0.2em 0 0.05em 0 !important;
                        margin: 0.3em 0 0.1em 0 !important;
                        font-weight: 600 !important;
                        background: transparent !important;
                        border: none !important;
                        line-height: 1.1 !important;
                        cursor: default !important;
                        pointer-events: none !important;
                        transition: none !important;
                    }

                    /* Flecha → para encabezados */
                    #right-sidebar h4:not(#sidebar-title-heading)::after,
                    #right-sidebar h1:not(#sidebar-title-heading)::after,
                    #right-sidebar h2:not(#sidebar-title-heading)::after,
                    #right-sidebar h3:not(#sidebar-title-heading)::after,
                    #right-sidebar h5:not(#sidebar-title-heading)::after,
                    #right-sidebar h6:not(#sidebar-title-heading)::after {
                        content: "";
                        display: inline-block;
                        margin-left: 0.25em;
                        width: 0;
                        height: 0;
                        border-left: 8px solid rgba(120, 180, 245, 0.95);
                        border-top: 5px solid transparent;
                        border-bottom: 5px solid transparent;
                        vertical-align: middle;
                        transform: translateY(-1px);
                        filter: drop-shadow(0 0 2px rgba(120, 180, 245, 0.5));
                    }

                    /* Estilo para el texto del modelo LLM - más claro y visible */
                    .llm-model-text {
                        font-size: 0.5rem !important;
                        font-family: 'SF Mono', 'Fira Code', 'Monaco', 'Inconsolata', 'Roboto Mono', 'Courier New', monospace !important;
                        font-weight: 500 !important;
                        opacity: 0.95 !important;
                        color: rgba(200, 220, 255, 0.9) !important;
                        letter-spacing: 0.3px !important;
                        margin-left: 0.15em !important;
                        text-shadow: 0 0 6px rgba(200, 220, 255, 0.4) !important;
                        border-radius: 4px !important;
                        padding: 0.1em 0.25em !important;
                        background: rgba(40, 55, 85, 0.6) !important;
                        backdrop-filter: blur(3px) !important;
                        border: 1px solid rgba(200, 220, 255, 0.3) !important;
                    }

                    /* Eliminar TODOS los hover de encabezados */
                    .header-container,
                    .header-container * {
                        pointer-events: none !important;
                        cursor: default !important;
                    }

                    .header-container:hover,
                    .header-container:hover * {
                        background: transparent !important;
                        transform: none !important;
                        box-shadow: none !important;
                        transition: none !important;
                    }
                `;
                document.head.appendChild(style);

                // Función para procesar los encabezados y estilizar el modelo LLM
                function processAgentHeaders() {
                    const headers = document.querySelectorAll('#right-sidebar h4:not(#sidebar-title-heading), #right-sidebar h1:not(#sidebar-title-heading), #right-sidebar h2:not(#sidebar-title-heading), #right-sidebar h3:not(#sidebar-title-heading), #right-sidebar h5:not(#sidebar-title-heading), #right-sidebar h6:not(#sidebar-title-heading)');

                    headers.forEach(header => {
                        let text = header.textContent || header.innerText;

                        // Quitar la flecha → existente si está presente
                        text = text.replace(/\s*→\s*$/, '');

                        // Buscar texto entre paréntesis
                        const regex = /^(.+?)\s*\(([^)]+)\)\s*$/;
                        const match = text.match(regex);

                        if (match) {
                            const agentName = match[1].trim();
                            const modelName = match[2].trim();

                            // Crear el nuevo HTML con el modelo estilizado SIN paréntesis
                            header.innerHTML = agentName + ' <span class="llm-model-text">' + modelName + '</span>';
                        }

                        // Marcar el contenedor padre como header-container para desactivar efectos
                        let container = header.closest('#right-sidebar > div > div > div');
                        if (container) {
                            container.classList.add('header-container');
                        }
                    });
                }

                // Ejecutar inmediatamente
                processAgentHeaders();

                // Observar cambios en el sidebar para procesar nuevos encabezados
                const observer = new MutationObserver(function(mutations) {
                    let shouldProcess = false;
                    mutations.forEach(function(mutation) {
                        if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                            shouldProcess = true;
                        }
                    });
                    if (shouldProcess) {
                        setTimeout(processAgentHeaders, 100); // Pequeño delay para asegurar que el DOM esté actualizado
                    }
                });

                const sidebar = document.getElementById('right-sidebar');
                if (sidebar) {
                    observer.observe(sidebar, {
                        childList: true,
                        subtree: true
                    });
                }
            })();
            """
        ),
        # Encabezado con el mismo estilo que el principal
        rx.box(
            rx.hstack(
                rx.icon(
                    "brain-circuit",
                    size=20,
                    style={
                        "color": "rgba(160, 174, 245, 0.85)",
                        "filter": "drop-shadow(0 0 6px rgba(99, 102, 241, 0.4))",
                        "margin_right": "8px",
                        "transform": "translateY(1px)",
                        "transition": "all 0.3s ease",
                    },
                ),
                rx.heading(
                    "Razonamiento y acciones",
                    size="4",
                    id="sidebar-title-heading",
                    style={
                        "background": "linear-gradient(135deg, rgba(235, 240, 255, 0.9) 0%, rgba(160, 174, 245, 0.8) 100%)",
                        "background_clip": "text",
                        "webkit_background_clip": "text",
                        "color": "transparent",
                        "font_weight": "500",
                        "letter_spacing": "1px",
                        "text_shadow": "0 1px 2px rgba(0, 0, 0, 0.2)",
                        "transition": "all 0.3s ease",
                    },
                ),
                rx.spacer(),
                width="100%",
                justify="between",
                align="center",
                padding_y="0.1em",
                spacing="0",
            ),
            width="100%",
            padding="0.3em",
            margin_bottom="0.5em",
            # Fondo más transparente
            background="linear-gradient(135deg, rgba(30, 41, 59, 0.08) 0%, rgba(51, 65, 85, 0.08) 100%)",
            backdrop_filter="blur(10px)",
            # Esquinas curvas
            border_radius="12px",
            box_shadow="none",  # Sin sombra
        ),
        # Resizer handle en el borde izquierdo (para arrastrar y cambiar ancho).
        rx.box(
            id="sidebar-resizer",
            width="6px",
            height="100%",
            position="absolute",
            left="-3px",
            top="0",
            cursor="ew-resize",
            background="transparent",
            z_index="10",
        ),
        # Botón toggle en el centro del borde izquierdo
        rx.box(
            "<",
            id="right-sidebar-toggle",
            role="button",
            aria_label="Toggle sidebar",
            position="absolute",
            left="0px",
            top="50%",
            transform="translateY(-50%)",
            # Mantener esquinas internas curvas usando estilo directamente
            # (Reflex no soporta valores múltiples en el prop border_radius)
            # Eliminamos el parámetro border_radius y lo definimos en style.
            background="linear-gradient(to bottom, rgba(25,110,185,0.45) 0%, rgba(55,140,205,0.40) 50%, rgba(95,175,230,0.35) 100%)",
            width="10px",
            height="96px",
            box_shadow="0 0 8px rgba(99, 102, 241, 0.6)",
            z_index="15",
            style={
                "color": "#ffffff",
                "display": "flex",
                "align_items": "center",
                "justify_content": "center",
                "font_size": "13px",
                "line_height": "1",
                "font_weight": "700",
                "border_top_left_radius": "96px",
                "border_bottom_left_radius": "96px",
                "transition": "box-shadow 0.2s, transform 0.25s",
                "_hover": {
                    "background": "linear-gradient(to bottom, rgba(25,110,185,0.65) 0%, rgba(55,140,205,0.60) 50%, rgba(95,175,230,0.55) 100%)",
                    "box_shadow": "0 0 12px rgba(95, 175, 230, 0.8)",
                },
            },
        ),
        # Contenedor para los mensajes
        rx.box(
            # Mostrar los mensajes de razonamiento del agente
            rx.foreach(
                State.reasoning_history,
                lambda reasoning: sidebar_message(reasoning),
            ),
            # Ancla para auto-scroll lateral.
            rx.box(id="reasoning-bottom-anchor", height="1px"),
            width="100%",
            height="calc(100vh - 70px)",  # Ajustado para compensar el título más pequeño
            overflow_y="auto",
            padding_y="0.5em",
            padding_x="0.3em",
            # Scroll suave
            style={
                "scrollBehavior": "smooth",
                "::-webkit-scrollbar": {
                    "width": "8px",
                },
                "::-webkit-scrollbar-track": {
                    "background": "rgba(30, 41, 59, 0.05)",
                    "borderRadius": "8px",
                },
                "::-webkit-scrollbar-thumb": {
                    "background": "linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(79, 70, 229, 0.15) 100%)",
                    "borderRadius": "8px",
                    "border": "2px solid transparent",
                    "backgroundClip": "padding-box",
                },
                "::-webkit-scrollbar-thumb:hover": {
                    "background": "linear-gradient(135deg, rgba(99, 102, 241, 0.4) 0%, rgba(79, 70, 229, 0.3) 100%)",
                    "boxShadow": "0 0 10px rgba(99, 102, 241, 0.5)",
                    "border": "1px solid rgba(255, 255, 255, 0.2)",
                },
            },
            background="rgba(15, 23, 42, 0.3)",
            backdrop_filter="blur(8px)",
            border_radius="20px",
        ),
        style=style.sidebar_style,
        id="right-sidebar",
    )


def documents_sidebar() -> rx.Component:
    """Renderiza la barra lateral izquierda para documentos."""
    return rx.box(
        # Contenedor con pestañas
        rx.vstack(
            # Selector de pestañas
            rx.cond(
                State.sidebar_tab == "documentos",
                rx.vstack(
                    rx.button(
                        "Preguntas frecuentes",
                        variant="ghost",
                        size="1",
                        on_click=State.set_sidebar_tab("faq"),
                        style={
                            "color": rx.cond(
                                State.sidebar_tab == "faq",
                                "rgba(255, 255, 255, 0.97)",
                                "rgba(180, 190, 255, 0.5)",
                            ),
                            "font_weight": rx.cond(
                                State.sidebar_tab == "faq", "700", "500"
                            ),
                            "padding": "0.35em 0.8em",
                            "border_radius": "8px",
                            "background": rx.cond(
                                State.sidebar_tab == "faq",
                                "linear-gradient(135deg, rgba(25,110,185,0.45) 0%, rgba(55,140,205,0.40) 50%, rgba(95,175,230,0.35) 100%)",
                                "linear-gradient(135deg, rgba(25,110,185,0.12) 0%, rgba(55,140,205,0.10) 50%, rgba(95,175,230,0.08) 100%)",
                            ),
                            "box_shadow": rx.cond(
                                State.sidebar_tab == "faq",
                                "0 4px 12px rgba(135, 206, 250, 0.35)",
                                "none",
                            ),
                            "backdrop_filter": rx.cond(
                                State.sidebar_tab == "faq", "blur(4px)", "none"
                            ),
                            "transform": rx.cond(
                                State.sidebar_tab == "faq", "translateY(-2px)", "none"
                            ),
                            "_hover": {
                                "background": "linear-gradient(135deg, rgba(25,110,185,0.25) 0%, rgba(55,140,205,0.20) 50%, rgba(95,175,230,0.15) 100%)",
                                "box_shadow": "0 2px 6px rgba(135, 206, 250, 0.2)",
                                "backdrop_filter": "blur(4px)",
                                "transform": "translateY(-2px)",
                            },
                            "_active": {
                                "transform": "scale(0.96)",
                                "opacity": "0.9",
                            },
                            "transition": "all 0.2s ease",
                            "animation": "0.3s cubic-bezier(0.25, 0.1, 0.25, 1) 0s 1 slideIn",
                        },
                    ),
                    rx.button(
                        "Base de conocimientos",
                        variant="ghost",
                        size="1",
                        on_click=State.set_sidebar_tab("documentos"),
                        style={
                            "color": "rgba(255, 255, 255, 0.97)",
                            "font_weight": "700",
                            "padding": "0.35em 0.8em",
                            "border_radius": "8px",
                            "background": "linear-gradient(135deg, rgba(25,110,185,0.45) 0%, rgba(55,140,205,0.40) 50%, rgba(95,175,230,0.35) 100%)",
                            "box_shadow": "0 4px 12px rgba(135, 206, 250, 0.35)",
                            "backdrop_filter": "blur(4px)",
                            "transform": "translateY(-2px)",
                            "animation": "0.3s cubic-bezier(0.25, 0.1, 0.25, 1) 0s 1 slideIn",
                        },
                    ),
                    spacing="4",
                    align="start",
                ),
                rx.vstack(
                    rx.button(
                        "Base de conocimientos",
                        variant="ghost",
                        size="1",
                        on_click=State.set_sidebar_tab("documentos"),
                        style={
                            "color": rx.cond(
                                State.sidebar_tab == "documentos",
                                "rgba(255, 255, 255, 0.97)",
                                "rgba(180, 190, 255, 0.5)",
                            ),
                            "font_weight": rx.cond(
                                State.sidebar_tab == "documentos", "700", "500"
                            ),
                            "padding": "0.35em 0.8em",
                            "border_radius": "8px",
                            "background": rx.cond(
                                State.sidebar_tab == "documentos",
                                "linear-gradient(135deg, rgba(25,110,185,0.45) 0%, rgba(55,140,205,0.40) 50%, rgba(95,175,230,0.35) 100%)",
                                "linear-gradient(135deg, rgba(25,110,185,0.12) 0%, rgba(55,140,205,0.10) 50%, rgba(95,175,230,0.08) 100%)",
                            ),
                            "box_shadow": rx.cond(
                                State.sidebar_tab == "documentos",
                                "0 4px 12px rgba(135, 206, 250, 0.35)",
                                "none",
                            ),
                            "backdrop_filter": rx.cond(
                                State.sidebar_tab == "documentos", "blur(4px)", "none"
                            ),
                            "transform": rx.cond(
                                State.sidebar_tab == "documentos",
                                "translateY(-2px)",
                                "none",
                            ),
                            "_hover": {
                                "background": "linear-gradient(135deg, rgba(25,110,185,0.25) 0%, rgba(55,140,205,0.20) 50%, rgba(95,175,230,0.15) 100%)",
                                "box_shadow": "0 2px 6px rgba(135, 206, 250, 0.2)",
                                "backdrop_filter": "blur(4px)",
                                "transform": "translateY(-2px)",
                            },
                            "_active": {
                                "transform": "scale(0.96)",
                                "opacity": "0.9",
                            },
                            "transition": "all 0.2s ease",
                            "animation": "0.3s cubic-bezier(0.25, 0.1, 0.25, 1) 0s 1 slideIn",
                        },
                    ),
                    rx.button(
                        "Preguntas frecuentes",
                        variant="ghost",
                        size="1",
                        on_click=State.set_sidebar_tab("faq"),
                        style={
                            "color": "rgba(255, 255, 255, 0.97)",
                            "font_weight": "700",
                            "padding": "0.35em 0.8em",
                            "border_radius": "8px",
                            "background": "linear-gradient(135deg, rgba(25,110,185,0.45) 0%, rgba(55,140,205,0.40) 50%, rgba(95,175,230,0.35) 100%)",
                            "box_shadow": "0 4px 12px rgba(135, 206, 250, 0.35)",
                            "backdrop_filter": "blur(4px)",
                            "transform": "translateY(-2px)",
                            "animation": "0.3s cubic-bezier(0.25, 0.1, 0.25, 1) 0s 1 slideIn",
                        },
                    ),
                    spacing="4",
                    align="start",
                ),
            ),
            rx.cond(
                State.sidebar_tab == "documentos",
                # Contenido Documentos
                rx.box(
                    # Subencabezado Documentos
                    rx.hstack(
                        rx.icon(
                            "file-text",
                            size=20,
                            style={
                                "color": "rgba(160, 174, 245, 0.85)",
                                "margin_right": "8px",
                                "transform": "translateY(1px)",
                                "transition": "all 0.3s ease",
                            },
                        ),
                        rx.heading(
                            "Documentos",
                            size="4",
                            style=style.sidebar_title_style,
                        ),
                        spacing="0",
                        width="100%",
                        justify="start",
                        align="center",
                    ),
                    # Lista de PDFs
                    rx.foreach(
                        State.get_docs_files,
                        lambda filename: rx.box(
                            rx.hstack(
                                rx.icon(
                                    "file-text",
                                    size=16,
                                    style={
                                        "color": "rgba(220, 220, 255, 0.9)",
                                        "margin_right": "8px",
                                        "width": "16px",
                                        "height": "16px",
                                        "flex": "0 0 16px",
                                        "display": "flex",
                                        "align_items": "center",
                                        "justify_content": "center",
                                    },
                                ),
                                rx.text(
                                    filename,
                                    size="2",
                                    style={
                                        "color": "rgba(220, 220, 255, 0.9)",
                                        "font_size": "0.8em",
                                        "white_space": "normal",
                                        "overflow": "visible",
                                        "text_overflow": "unset",
                                        "max_width": "calc(var(--left-sidebar-width, 200px) - 40px)",
                                        "line_height": "1",
                                    },
                                ),
                                width="100%",
                                justify="start",
                                align="center",
                                spacing="1",
                            ),
                            padding="0.25em 0.5em",
                            margin_bottom="0.05em",
                            border_radius="8px",
                            background="rgba(30, 41, 59, 0.2)",
                            cursor="pointer",
                            on_click=lambda f=filename: [
                                rx.set_clipboard(f),
                                rx.toast.success("Copiado", duration=1500),
                            ],
                            _hover={
                                "background": "linear-gradient(to bottom, rgba(25, 110, 185, 0.45) 0%, rgba(55, 140, 205, 0.40) 50%, rgba(95, 175, 230, 0.35) 100%)",
                                "box_shadow": "0 4px 12px rgba(135, 206, 250, 0.35)",
                                "backdrop_filter": "blur(4px)",
                                "border": "none",
                                "transform": "translateY(-2px)",
                                "transition": "all 0.25s ease",
                            },
                            animation="glow 3s ease-in-out infinite alternate",
                            animationPlayState="paused",
                        ),
                    ),
                    # Tabla
                    rx.box(
                        rx.hstack(
                            rx.icon(
                                "table",
                                size=20,
                                style={
                                    "color": "rgba(160, 174, 245, 0.85)",
                                    "margin_right": "8px",
                                    "transform": "translateY(1px)",
                                    "transition": "all 0.3s ease",
                                },
                            ),
                            rx.heading(
                                "Tablas",
                                size="4",
                                style=style.sidebar_title_style,
                            ),
                            spacing="0",
                            width="100%",
                            justify="start",
                            align="center",
                        ),
                        width="100%",
                        padding="0.1em",
                        margin_top="0.5em",
                        margin_bottom="0.05em",
                        background="linear-gradient(135deg, rgba(30, 41, 59, 0.08) 0%, rgba(51, 65, 85, 0.08) 100%)",
                        backdrop_filter="blur(10px)",
                        border_radius="12px",
                        box_shadow="none",
                    ),
                    rx.foreach(
                        State.get_tables,
                        lambda table_name: rx.box(
                            rx.hstack(
                                rx.icon(
                                    "database",
                                    size=16,
                                    style={
                                        "color": "rgba(220, 220, 255, 0.9)",
                                        "margin_right": "8px",
                                        "width": "16px",
                                        "height": "16px",
                                        "flex": "0 0 16px",
                                        "display": "flex",
                                        "align_items": "center",
                                        "justify_content": "center",
                                    },
                                ),
                                rx.text(
                                    table_name,
                                    size="2",
                                    style={
                                        "color": "rgba(220, 220, 255, 0.9)",
                                        "font_size": "0.8em",
                                        "white_space": "normal",
                                        "overflow": "visible",
                                        "text_overflow": "unset",
                                        "max_width": "calc(var(--left-sidebar-width, 200px) - 40px)",
                                        "line_height": "1",
                                    },
                                ),
                                width="100%",
                                justify="start",
                                align="center",
                                spacing="1",
                            ),
                            padding="0.25em 0.5em",
                            margin_bottom="0.05em",
                            border_radius="8px",
                            background="rgba(30, 41, 59, 0.2)",
                            cursor="pointer",
                            on_click=lambda t=table_name: [
                                rx.set_clipboard(t),
                                rx.toast.success("Copiado", duration=1500),
                            ],
                            _hover={
                                "background": "linear-gradient(to bottom, rgba(25, 110, 185, 0.45) 0%, rgba(55, 140, 205, 0.40) 50%, rgba(95, 175, 230, 0.35) 100%)",
                                "box_shadow": "0 4px 12px rgba(135, 206, 250, 0.35)",
                                "backdrop_filter": "blur(4px)",
                                "border": "none",
                                "transform": "translateY(-2px)",
                                "transition": "all 0.25s ease",
                            },
                            animation="glow 3s ease-in-out infinite alternate",
                            animationPlayState="paused",
                        ),
                    ),
                    width="100%",
                    height="calc(100vh - 100px)",
                    overflow_y="auto",
                    padding_top="0.1em",
                    padding_bottom="0em",
                    padding_x="0.2em",
                    style={
                        "scrollBehavior": "smooth",
                        "::-webkit-scrollbar": {"width": "8px"},
                        "::-webkit-scrollbar-track": {
                            "background": "rgba(30, 41, 59, 0.05)",
                            "borderRadius": "8px",
                        },
                        "::-webkit-scrollbar-thumb": {
                            "background": "linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(79, 70, 229, 0.15) 100%)",
                            "borderRadius": "8px",
                            "border": "2px solid transparent",
                            "backgroundClip": "padding-box",
                        },
                        "::-webkit-scrollbar-thumb:hover": {
                            "background": "linear-gradient(135deg, rgba(99, 102, 241, 0.4) 0%, rgba(79, 70, 229, 0.3) 100%)",
                            "boxShadow": "0 0 10px rgba(99, 102, 241, 0.5)",
                            "border": "1px solid rgba(255, 255, 255, 0.2)",
                        },
                    },
                    background="rgba(15, 23, 42, 0.3)",
                    backdrop_filter="blur(8px)",
                    border_radius="20px",
                ),
                # Contenido FAQ
                rx.box(
                    rx.vstack(
                        *[
                            rx.box(
                                q,
                                on_click=lambda q=q: [
                                    State.set_question(q),
                                    State.answer,
                                ],
                                cursor="pointer",
                                style={
                                    "padding": "0.4em 0.6em",
                                    "border_radius": "8px",
                                    "margin_y": "0.05em",
                                    "background": "rgba(30, 41, 59, 0.2)",
                                    "color": "#F8FAFC",
                                    "font_size": "0.85em",
                                    "box_shadow": "rgba(0, 0, 0, 0.35) 0px 4px 12px",
                                    "transition": "all 0.25s ease",
                                    "_hover": {
                                        "background": "linear-gradient(to bottom, rgba(25, 110, 185, 0.45) 0%, rgba(55, 140, 205, 0.40) 50%, rgba(95, 175, 230, 0.35) 100%)",
                                        "box_shadow": "0 4px 12px rgba(135, 206, 250, 0.35)",
                                        "backdrop_filter": "blur(4px)",
                                        "border": "none",
                                        "transform": "translateY(-2px)",
                                    },
                                    "_active": {
                                        "transform": "scale(0.96)",
                                        "opacity": "0.9",
                                    },
                                    "animation": "0.3s cubic-bezier(0.25, 0.1, 0.25, 1) 0s 1 slideIn",
                                },
                            )
                            for q in [
                                "Busca datos de biomarcadores en las tablas y genera un gráfico de serie temporal",
                                "Genera una proyección de los biomarcadores",
                                "Impacto de la comida sobre el envejecimiento",
                                "¿El ejercicio físico puede ralentizar el envejecimiento?",
                                "¿Dormir bien influye en el envejecimiento celular?",
                                "¿El estrés acelera el envejecimiento?",
                                "¿Fumar envejece más rápido el cuerpo?",
                                "¿Dónde viven las personas más longevas del mundo?",
                                "¿La soledad afecta el envejecimiento?",
                                "¿Hay diferencia entre envejecer en la ciudad o en el campo?",
                                "¿Qué rol tiene la comunidad en un envejecimiento saludable?",
                                "¿La contaminación ambiental envejece más rápido?",
                            ]
                        ],
                        spacing="2",
                        width="100%",
                    ),
                    width="100%",
                    height="calc(100vh - 100px)",
                    padding_top="0.3em",
                    padding_bottom="0em",
                    overflow_y="auto",
                    background="rgba(15, 23, 42, 0.3)",
                    backdrop_filter="blur(8px)",
                    border_radius="20px",
                ),
            ),
            spacing="1",
            width="100%",
        ),
        # Resizer handle en el borde derecho para arrastrar y cambiar el ancho.
        rx.box(
            id="left-sidebar-resizer",
            width="6px",
            height="100%",
            position="absolute",
            right="-3px",
            top="0",
            cursor="ew-resize",
            background="transparent",
            z_index="10",
        ),
        rx.box(
            ">",
            id="left-sidebar-toggle",
            role="button",
            aria_label="Toggle sidebar",
            position="absolute",
            right="0px",
            top="50%",
            transform="translateY(-50%)",
            background="linear-gradient(to bottom, rgba(25,110,185,0.45) 0%, rgba(55,140,205,0.40) 50%, rgba(95,175,230,0.35) 100%)",
            width="10px",
            height="96px",
            box_shadow="0 0 8px rgba(99, 102, 241, 0.6)",
            z_index="15",
            style={
                "color": "#ffffff",
                "display": "flex",
                "align_items": "center",
                "justify_content": "center",
                "font_size": "13px",
                "line_height": "1",
                "font_weight": "700",
                "border_top_right_radius": "96px",
                "border_bottom_right_radius": "96px",
                "cursor": "pointer",
                "transition": "box-shadow 0.2s, transform 0.25s",
                "_hover": {
                    "background": "linear-gradient(to bottom, rgba(25,110,185,0.65) 0%, rgba(55,140,205,0.60) 50%, rgba(95,175,230,0.55) 100%)",
                    "box_shadow": "0 0 12px rgba(95, 175, 230, 0.8)",
                },
            },
        ),
        style=style.left_sidebar_style,
        id="left-sidebar",
    )


def qa(question: str | None, answer: str | None) -> rx.Component:
    """Renderiza un par pregunta-respuesta en el chat."""
    children: list[rx.Component] = []

    if question is not None:
        children.append(
            rx.box(
                rx.text(
                    question,
                    style=style.question_style,
                ),
                on_click=lambda: [
                    rx.set_clipboard(question),
                    rx.toast.success("Copiado", duration=1500),
                ],
                cursor="pointer",
                text_align="right",
                width="100%",
                padding_x="0.8em",  # Reducido de 1em
                style={
                    "transition": "transform 0.1s ease, opacity 0.1s ease",
                    "_active": {
                        "transform": "scale(0.96)",
                        "opacity": "0.9",
                    },
                },
            )
        )

    if answer is not None:
        children.append(
            rx.box(
                rx.markdown(
                    answer,
                    style=style.answer_style,
                ),
                on_click=lambda: [
                    rx.set_clipboard(answer),
                    rx.toast.success("Copiado", duration=1500),
                ],
                cursor="pointer",
                text_align="left",
                width="100%",
                padding_x="0.8em",  # Reducido de 1em
                style={
                    "transition": "transform 0.1s ease, opacity 0.1s ease",
                    "_active": {
                        "transform": "scale(0.96)",
                        "opacity": "0.9",
                    },
                },
            )
        )

    return rx.box(
        *children,
        margin_y="0.15em",  # Reducido de 0.5em
        width="100%",
    )


def plot_display(figure: go.Figure | rx.Var) -> rx.Component:
    """Renderiza un gráfico Plotly en el chat."""
    # El componente ``rx.plotly`` exige que ``data`` sea un ``go.Figure`` o
    # un ``Var`` que *se resuelva* a ``go.Figure``.  Cuando la figura proviene
    # directamente del estado llega como ``Var``.  Aquí la convertimos para
    # satisfacer el *type checker* de Reflex sin perder la reactividad.
    data_prop = (
        figure.to(go.Figure)  # type: ignore[attr-defined]
        if hasattr(figure, "to") and isinstance(figure, rx.Var)
        else figure
    )

    # Envolver la figura en un contenedor con *glass-effect* y sombra suave
    # para que combine con las burbujas de chat sin parecer una burbuja.
    return rx.box(
        rx.plotly(
            data=data_prop,
            layout={
                "autosize": True,
                "responsive": True,
            },  # Enhanced responsive configuration
            config={
                "responsive": True,
                "displayModeBar": True,
                "displaylogo": False,
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": "forecast_chart",
                    "height": 600,
                    "width": 1000,
                    "scale": 2,
                },
            },
            use_resize_handler=True,  # Ajusta al redimensionar contenedor
            style={
                "position": "absolute",
                "inset": "0",
                "width": "100%",
                "height": "100%",
            },
        ),
        # Estilo del contenedor externo que integra el plot con el chat
        style={
            "width": "57%",  # Reduced by 1/3: 85% * (2/3) = 57%
            "aspect_ratio": "2 / 1",  # Height = half of width (2:1 ratio)
            "min_width": "213px",  # Reduced by 1/3: 320px * (2/3) = 213px
            "max_width": "800px",  # Reduced by 1/3: 1200px * (2/3) = 800px
            "margin": "0.4em auto",  # Centrado horizontal y margen superior/inferior
            "background": style.answer_style[
                "background"
            ],  # Mismo gradiente que respuestas
            **{
                k: v for k, v in style.glass_effect.items() if k != "background"
            },  # conserva blur y borde
            "border_radius": "20px",  # Curvatura más pronunciada
            "border_left_color": "rgba(16, 185, 129, 0.7)",  # Mantiene verde en hover
            "border_left_width": "6px",
            "border": "2px solid rgba(255, 255, 255, 0.12)",  # Contorno translúcido efecto vidrio
            "overflow": "hidden",
            "box_shadow": f"{style.shadow}, 0 0 6px rgba(16, 185, 129, 0.15)",  # Sombra + glow verde sutil
            "transition": "all 0.3s ease",  # Transición suave sin animación
            "position": "relative",  # Ensure proper positioning for absolute child
            "_hover": {
                "transform": "translateY(-2px)",  # Movimiento sutil hacia arriba
                "box_shadow": "0 8px 20px rgba(16, 185, 129, 0.25), 0 0 12px rgba(16, 185, 129, 0.3)",  # Glow sutil
                "backdrop_filter": "blur(16px)",
                "border_color": "rgba(255, 255, 255, 0.15)",  # Leve intensificación del borde
                "border_left_color": "rgba(16, 185, 129, 0.8)",  # Leve intensificación del borde verde
                "border_left_width": "6px",
                "filter": "brightness(1.02)",  # Brillo muy sutil
                "transition": "all 0.3s ease",
            },
            # Responsive behavior for smaller screens
            "@media (max-width: 768px)": {
                "width": "63%",  # Reduced by 1/3: 95% * (2/3) = 63%
                "min_width": "187px",  # Reduced by 1/3: 280px * (2/3) = 187px
                "aspect_ratio": "2 / 1",  # Keep 2:1 ratio on tablets
            },
            "@media (max-width: 480px)": {
                "width": "65%",  # Reduced by 1/3: 98% * (2/3) = 65%
                "min_width": "173px",  # Reduced by 1/3: 260px * (2/3) = 173px
                "aspect_ratio": "2 / 1",  # Keep 2:1 ratio on mobile
                "margin": "0.2em auto",
            },
        },
    )


def render_message(
    message_tuple: tuple[str | None, str | go.Figure | None],
) -> rx.Component:
    """Renderiza un mensaje de chat, que puede ser texto o un gráfico."""
    question = message_tuple[0]
    answer_or_plot = message_tuple[1]

    return rx.vstack(
        # ------------------------------------------------------------------
        # 1️⃣  Renderizar la pregunta (si existe).  Se hace primero para que la
        #     figura aparezca debajo de la pregunta correspondiente.
        rx.cond(
            question.is_not_none()
            if hasattr(question, "is_not_none")
            else (question is not None),
            qa(question=question, answer=None),
            rx.fragment(),
        ),
        # ------------------------------------------------------------------
        # 2️⃣  Renderizar la *respuesta* o el *gráfico*.
        #     Utilizamos ``js_type()`` para distinguir strings de figuras en
        #     tiempo de compilación – imprescindible cuando ``answer_or_plot``
        #     es un ``Var``.
        rx.cond(
            answer_or_plot.is_not_none()
            if hasattr(answer_or_plot, "is_not_none")
            else (answer_or_plot is not None),
            rx.cond(
                (answer_or_plot.js_type() == "string")
                if hasattr(answer_or_plot, "js_type")
                else isinstance(answer_or_plot, str),
                qa(question=None, answer=answer_or_plot),
                plot_display(figure=answer_or_plot),
            ),
            rx.fragment(),
        ),
        width="100%",
        align_items="stretch",
        spacing="1",
    )


def chat() -> rx.Component:
    """Renderiza el historial de chat."""
    return rx.box(
        rx.foreach(
            State.chat_history,
            render_message,  # Use the new rendering function
        ),
        loading_spinner(),
        # Ancla invisible para auto-scroll.
        rx.box(id="chat-bottom-anchor", height="1px"),
        width="100%",
        padding_x="0",
        flex="1",
        overflow_y="auto",
        padding_bottom="1em",  # Reducido de 1.5em
        padding_top="0.2em",  # Añadido para dar un poco de espacio arriba
        gap="0.1em",  # Añadido para reducir espacio entre mensajes
        style={
            "scrollBehavior": "auto",
            "::-webkit-scrollbar": {
                "width": "12px",
            },
            "::-webkit-scrollbar-track": {
                "background": "rgba(30, 41, 59, 0.05)",
                "borderRadius": "10px",
                "backdropFilter": "blur(12px)",
            },
            "::-webkit-scrollbar-thumb": {
                "background": "linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(79, 70, 229, 0.15) 100%)",
                "borderRadius": "10px",
                "border": "2px solid transparent",
                "backgroundClip": "padding-box",
                "backdropFilter": "blur(8px)",
                "boxShadow": "inset 0 0 6px rgba(99, 102, 241, 0.1)",
            },
            "::-webkit-scrollbar-thumb:hover": {
                "background": "linear-gradient(135deg, rgba(99, 102, 241, 0.5) 0%, rgba(79, 70, 229, 0.4) 100%)",
                "boxShadow": "0 0 10px rgba(99, 102, 241, 0.5)",
                "border": "1px solid rgba(255, 255, 255, 0.2)",
                "backdropFilter": "blur(12px)",
            },
        },
    )


def loading_spinner() -> rx.Component:
    """Renderiza un spinner de carga cuando se está esperando una respuesta."""
    return rx.cond(
        State.is_loading,
        rx.center(
            rx.vstack(
                rx.spinner(
                    color="rgb(99, 102, 241)",
                    size="3",
                    thickness="5px",
                    speed="0.5s",
                    empty_color="rgba(255, 255, 255, 0.15)",
                ),
                rx.text(
                    "Trabajando...",
                    color="rgb(220, 220, 255)",
                    font_weight="500",
                    margin_top="0.5em",
                    font_size="0.95em",
                    letter_spacing="0.5px",
                ),
                spacing="0",
                align="center",
                justify="center",
            ),
            padding="1.5em",
            width="100%",
            background="rgba(30, 41, 59, 0.6)",
            border_radius="12px",
            margin_y="1em",
            animation="glow 1.2s infinite alternate, fadeIn 0.5s",
            overflow="hidden",
            display="flex",
            align_items="center",
            justify_content="center",
            box_shadow="0 0 15px rgba(99, 102, 241, 0.2)",
            transition="all 0.3s ease",
        ),
        rx.fragment(),
    )


def action_bar() -> rx.Component:
    """Renderiza la barra de acción para enviar mensajes."""
    return rx.box(
        rx.hstack(
            rx.input(
                value=State.question,
                placeholder="Escribe tu pregunta...",
                on_change=State.set_question,
                on_key_down=State.handle_key_press,
                is_disabled=State.is_loading,
                style={
                    **style.input_style,
                    "height": "30px",
                    "padding": "0 16px",
                    "font_size": "14px",
                    "border_width": "1px",
                    "border_color": "rgba(255, 255, 255, 0.2)",
                    # Hover especializado para el input principal: tonos más oscuros y el doble de transparencia.
                    "_hover": {
                        "background": "linear-gradient(to bottom, rgba(15, 40, 75, 0.22) 0%, rgba(25, 60, 100, 0.20) 50%, rgba(35, 80, 120, 0.17) 100%)",
                        "box_shadow": "0 4px 12px rgba(50, 80, 120, 0.4)",
                        "backdrop_filter": "blur(4px)",
                        "border": "none",
                        "transform": "translateY(-2px)",
                        "transition": "all 0.25s ease",
                    },
                },
                auto_focus=True,
                size="2",
            ),
            rx.button(
                rx.hstack(
                    rx.text("Enviar"),
                    spacing="0",
                    align="center",
                    justify="center",
                ),
                on_click=State.answer,
                is_disabled=State.is_loading,
                style={
                    **style.button_style,
                    "height": "30px",
                    "padding": "0 16px",
                    "display": "flex",
                    "align_items": "center",
                    "justify_content": "center",
                    "background": "linear-gradient(135deg, rgba(30, 58, 138, 0.5) 0%, rgba(56, 79, 157, 0.4) 100%)",
                    "backdrop_filter": "blur(8px)",
                    "border": "none",
                    "color": "rgba(235, 240, 255, 0.95)",
                    "font_weight": "500",
                    "letter_spacing": "0.5px",
                    "box_shadow": "none",
                    "transition": "all 0.25s cubic-bezier(0.4, 0, 0.2, 1)",
                    "cursor": "pointer",
                    "border_radius": "8px",
                    "transform_origin": "center",
                    "position": "relative",
                    ":hover": {
                        # Hover unificado con el de los mensajes.
                        "background": "linear-gradient(to bottom, rgba(25, 110, 185, 0.45) 0%, rgba(55, 140, 205, 0.40) 50%, rgba(95, 175, 230, 0.35) 100%)",
                        "box_shadow": "0 4px 12px rgba(135, 206, 250, 0.35)",
                        "backdrop_filter": "blur(4px)",
                        "border": "none",
                        "transform": "translateY(-2px) scale(1.02)",
                        "color": "rgba(255, 255, 255, 1)",
                        "transition": "all 0.25s ease",
                    },
                    "active": {
                        "transform": "translateY(1px) scale(0.97)",
                        "background": "linear-gradient(135deg, rgba(30, 58, 138, 0.7) 0%, rgba(56, 79, 157, 0.6) 100%)",
                        "box_shadow": "inset 0 2px 4px rgba(0, 0, 0, 0.1)",
                        "transition": "all 0.05s cubic-bezier(0.4, 0, 0.2, 1)",
                        "border": "none",
                        "animation": "buttonGlow 0.6s ease-in-out",
                    },
                    "focus": {
                        "outline": "none",
                        "box_shadow": "0 0 0 2px rgba(99, 102, 241, 0.4)",
                    },
                    "@media (max-width: 768px)": {
                        "padding": "0 12px",
                        "height": "28px",
                        "font_size": "0.9rem",
                    },
                    "@media (max-width: 480px)": {
                        "padding": "0 10px",
                        "height": "26px",
                        "letter_spacing": "0.3px",
                    },
                    "_after": {
                        "content": "''",
                        "position": "absolute",
                        "top": "-2px",
                        "left": "-2px",
                        "right": "-2px",
                        "bottom": "-2px",
                        "border_radius": "10px",
                        "opacity": "0",
                        "z_index": "-1",
                        "transition": "opacity 0.3s ease",
                    },
                    "_active:after": {
                        "opacity": "1",
                        "border": "2px solid transparent",
                        "background": "linear-gradient(135deg, #4F46E5, #6366F1, #818CF8, #4F46E5)",
                        "background_size": "400% 400%",
                        "animation": "borderGlow 1.5s ease infinite, gradientShift 3s ease infinite",
                    },
                },
                size="2",
            ),
            width="100%",
            justify="center",
            padding="0.4em",
            spacing="3",
        ),
        width="100%",
        padding_x="0",
        padding_y="0.5em",
        position="sticky",
        bottom="0",
        background="linear-gradient(135deg, rgba(30, 41, 59, 0.15) 0%, rgba(51, 65, 85, 0.15) 100%)",
        backdrop_filter="blur(16px)",
        # border_top eliminado para suprimir línea divisoria
        box_shadow="0 -5px 20px rgba(0, 0, 0, 0.1)",
        z_index="10",
        # Bordes redondeados superiores para apariencia curvada
        border_radius="12px 12px 0 0",
    )


def index() -> rx.Component:
    """Renderiza la página principal de la aplicación."""
    return rx.box(
        rx.toast.provider(
            position="top-center",  # Centrar horizontalmente
            offset="8vh",  # Más arriba en el área de chat
            toast_options=rx.toast.options(
                duration=800,
                style={
                    "transition": "opacity 0.2s ease, transform 0.2s ease",
                    # Mismo gradiente que el hover (armonía visual)
                    "background": "linear-gradient(to bottom, rgba(25, 110, 185, 0.85) 0%, rgba(55, 140, 205, 0.80) 50%, rgba(95, 175, 230, 0.75) 100%)",
                    "color": "rgba(255, 255, 255, 0.95)",
                    "backdropFilter": "blur(4px)",
                    "boxShadow": "0 0 12px rgba(55, 140, 205, 0.6), 0 0 24px rgba(55, 140, 205, 0.4)",
                    "border": "none",
                    "borderRadius": "8px",
                    "padding": "0.4em 0.9em",
                    "width": "fit-content",
                    "maxWidth": "90%",
                },
            ),
        ),
        rx.hstack(
            # Añadir el sidebar de documentos en el extremo izquierdo
            documents_sidebar(),
            rx.box(
                rx.vstack(
                    rx.box(
                        rx.hstack(
                            rx.icon(
                                "route",
                                size=26,
                                style={
                                    "color": "rgba(160, 174, 245, 0.85)",
                                    "filter": "drop-shadow(0 0 6px rgba(99, 102, 241, 0.4))",
                                    "margin_right": "7px",
                                    "transform": "translateY(1px)",
                                    "transition": "all 0.3s ease",
                                },
                            ),
                            rx.heading(
                                "Pathway™ v0.0 (Proof-of-Concept)",
                                size="4",
                                style={
                                    "background": "linear-gradient(135deg, rgba(235, 240, 255, 0.9) 0%, rgba(160, 174, 245, 0.8) 100%)",
                                    "background_clip": "text",
                                    "webkit_background_clip": "text",
                                    "color": "transparent",
                                    "font_weight": "500",
                                    "letter_spacing": "1px",
                                    "text_shadow": "0 1px 2px rgba(0, 0, 0, 0.2)",
                                    "transition": "all 0.3s ease",
                                },
                            ),
                            width="100%",
                            justify="start",
                            align="center",
                        ),
                        padding="0.6em",
                        # Bordes redondeados inferiores para apariencia curvada
                        border_radius="0 0 12px 12px",
                        # border_bottom eliminado para suprimir línea divisoria
                        background="linear-gradient(135deg, rgba(30, 41, 59, 0.15) 0%, rgba(51, 65, 85, 0.15) 100%)",
                        backdrop_filter="blur(16px)",
                        position="sticky",
                        top="0",
                        z_index="10",
                        width="100%",
                        box_shadow="0 5px 20px rgba(0, 0, 0, 0.1)",
                    ),
                    chat(),
                    action_bar(),
                    width="100%",
                    max_width="100vw",
                    height="100vh",
                    align="stretch",
                    spacing="0",
                    position="relative",
                    overflow_x="hidden",
                    style={
                        **style.container_style,
                        "backgroundImage": "url('/svg/chat_background.svg')",
                        "backgroundSize": "contain",
                        "backgroundAttachment": "fixed",
                        "backgroundPosition": "center",
                        "backgroundRepeat": "repeat",
                        "backgroundColor": "rgba(10, 10, 20, 0.1)",
                    },
                ),
                style=style.main_container_with_sidebar,
            ),
            sidebar(),
            # Script robusto para redimensionar el sidebar con soporte pointer/touch.
            rx.script(
                """
                (() => {
                  const sidebar = document.getElementById('right-sidebar');
                  const resizer = document.getElementById('sidebar-resizer');
                  if (!sidebar || !resizer) return;

                  // Asegurar que existan los custom properties por defecto.
                  const root = document.documentElement;
                  if (!root.style.getPropertyValue('--sidebar-width')) {
                    root.style.setProperty('--sidebar-width', sidebar.offsetWidth + 'px');
                  }

                  const MIN = 220;
                  const MAX = 600;

                  resizer.addEventListener('pointerdown', (ev) => {
                    ev.preventDefault();
                    const startX = ev.clientX;
                    const startWidth = sidebar.getBoundingClientRect().width;

                    // Evitar selección de texto mientras se arrastra.
                    const prevSelect = document.body.style.userSelect;
                    document.body.style.userSelect = 'none';

                    const onPointerMove = (e) => {
                      const dx = e.clientX - startX;
                      const newWidth = Math.min(Math.max(startWidth - dx, MIN), MAX);
                      root.style.setProperty('--sidebar-width', newWidth + 'px');
                      window.dispatchEvent(new Event('resize'));
                    };

                    const onPointerUp = () => {
                      document.removeEventListener('pointermove', onPointerMove);
                      document.removeEventListener('pointerup', onPointerUp);
                      document.body.style.userSelect = prevSelect;
                      const finalW = sidebar.getBoundingClientRect().width;
                      const toggleBtn = document.getElementById('right-sidebar-toggle');
                      if (toggleBtn) {
                        toggleBtn.style.transform = finalW > (MIN + MAX) / 2
                          ? 'translateY(-50%) rotate(180deg)'
                          : 'translateY(-50%) rotate(0deg)';
                      }
                    };

                    document.addEventListener('pointermove', onPointerMove);
                    document.addEventListener('pointerup', onPointerUp);
                  });

                  // Toggle button para plegar/desplegar
                  const toggleBtn = document.getElementById('right-sidebar-toggle');
                  if (toggleBtn) {
                    toggleBtn.addEventListener('click', () => {
                      const currentW = sidebar.getBoundingClientRect().width;
                      const target = currentW > (MIN + MAX) / 2 ? MIN : MAX;
                      root.style.setProperty('--sidebar-width', target + 'px');
                      const finalW = sidebar.getBoundingClientRect().width;
                      toggleBtn.style.transform = finalW > (MIN + MAX) / 2
                        ? 'translateY(-50%) rotate(180deg)'
                        : 'translateY(-50%) rotate(0deg)';
                      window.dispatchEvent(new Event('resize'));
                    });
                  }
                })();
                """,
            ),
            # Script para redimensionar el sidebar izquierdo.
            rx.script(
                """
                (() => {
                  const sidebar = document.getElementById('left-sidebar');
                  const resizer = document.getElementById('left-sidebar-resizer');
                  if (!sidebar || !resizer) return;

                  const root = document.documentElement;
                  if (!root.style.getPropertyValue('--left-sidebar-width')) {
                    // Valor inicial por defecto incrementado ~266 px
                    root.style.setProperty('--left-sidebar-width', '266px');
                  }

                  const MIN = 200;
                  const MAX = 600;

                  resizer.addEventListener('pointerdown', (ev) => {
                    ev.preventDefault();
                    const startX = ev.clientX;
                    const startWidth = sidebar.getBoundingClientRect().width;

                    const prevSelect = document.body.style.userSelect;
                    document.body.style.userSelect = 'none';

                    const onPointerMove = (e) => {
                      const dx = e.clientX - startX;
                      const newWidth = Math.min(Math.max(startWidth + dx, MIN), MAX);
                      root.style.setProperty('--left-sidebar-width', newWidth + 'px');
                      window.dispatchEvent(new Event('resize'));
                    };

                    const onPointerUp = () => {
                      document.removeEventListener('pointermove', onPointerMove);
                      document.removeEventListener('pointerup', onPointerUp);
                      document.body.style.userSelect = prevSelect;
                      const finalWL = sidebar.getBoundingClientRect().width;
                      const toggleBtnL = document.getElementById('left-sidebar-toggle');
                      if (toggleBtnL) {
                        toggleBtnL.style.transform = finalWL > (MIN + MAX) / 2
                          ? 'translateY(-50%) rotate(180deg)'
                          : 'translateY(-50%) rotate(0deg)';
                      }
                    };

                    document.addEventListener('pointermove', onPointerMove);
                    document.addEventListener('pointerup', onPointerUp);
                  });

                  // Toggle button
                  const toggleBtnL = document.getElementById('left-sidebar-toggle');
                  if (toggleBtnL) {
                    toggleBtnL.addEventListener('click', () => {
                      const currentWL = sidebar.getBoundingClientRect().width;
                      const target = currentWL > (MIN + MAX) / 2 ? MIN : MAX;
                      root.style.setProperty('--left-sidebar-width', target + 'px');
                      const finalWL = sidebar.getBoundingClientRect().width;
                      toggleBtnL.style.transform = finalWL > (MIN + MAX) / 2
                        ? 'translateY(-50%) rotate(180deg)'
                        : 'translateY(-50%) rotate(0deg)';
                      window.dispatchEvent(new Event('resize'));
                    });
                  }
                })();
                """,
            ),
            # Silent clipboard writeText errors when document is not focused
            rx.script(
                """
                (() => {
                  // Detect if clipboard API is available
                  if (navigator.clipboard && navigator.clipboard.writeText) {
                    // Store the original function
                    const originalWriteText = navigator.clipboard.writeText.bind(navigator.clipboard);

                    // Replace with wrapped version that catches errors
                    navigator.clipboard.writeText = async (text) => {
                      try {
                        // Try the original function
                        return await originalWriteText(text);
                      } catch (err) {
                        // Handle the error
                        console.warn('Clipboard operation failed:', err.message);

                        // Create a fallback using a hidden textarea (works when not focused)
                        try {
                          const textArea = document.createElement('textarea');
                          textArea.value = text;
                          textArea.style.position = 'fixed';
                          textArea.style.top = '-999999px';
                          textArea.style.left = '-999999px';
                          document.body.appendChild(textArea);
                          textArea.focus();
                          textArea.select();

                          const success = document.execCommand('copy');
                          document.body.removeChild(textArea);

                          if (success) {
                            console.log('Used fallback clipboard method');
                            return Promise.resolve();
                          } else {
                            return Promise.reject(new Error('Fallback clipboard failed'));
                          }
                        } catch (fallbackErr) {
                          console.error('All clipboard methods failed', fallbackErr);
                          return Promise.resolve(); // Resolve anyway to prevent UI errors
                        }
                      }
                    };
                  }
                })();
                """,
            ),
            width="100vw",
            height="100vh",
            spacing="0",
            align="start",
            overflow="hidden",
        ),
        width="100vw",
        height="100vh",
        padding="0",
        margin="0",
        overflow="hidden",
    )


app = rx.App(
    toaster=None,
    style={
        "body": {
            "background": "#030712",
            "min_height": "100vh",
            "margin": "0",
            "font_family": '"Inter", system-ui, sans-serif',
            "color": "white",
        },
        **style.animations,
        "@keyframes buttonGlow": {
            "0%": {"box-shadow": "0 0 5px rgba(99, 102, 241, 0.4)"},
            "50%": {
                "box-shadow": "0 0 20px rgba(99, 102, 241, 0.8), 0 0 30px rgba(99, 102, 241, 0.6)"
            },
            "100%": {"box-shadow": "0 0 5px rgba(99, 102, 241, 0.4)"},
        },
        "@keyframes borderGlow": {
            "0%": {"box-shadow": "0 0 5px rgba(99, 102, 241, 0.4)"},
            "50%": {
                "box-shadow": "0 0 15px rgba(99, 102, 241, 0.8), 0 0 20px rgba(79, 70, 229, 0.6)"
            },
            "100%": {"box-shadow": "0 0 5px rgba(99, 102, 241, 0.4)"},
        },
        "@keyframes gradientShift": {
            "0%": {"background-position": "0% 50%"},
            "50%": {"background-position": "100% 50%"},
            "100%": {"background-position": "0% 50%"},
        },
    },
)
app.add_page(index)
