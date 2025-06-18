# style.py
"""Guía de estilos centralizada para Pathway.

Este módulo contiene **únicamente** constantes (dictionaries) que agrupan
propiedades CSS para los distintos elementos de la interfaz.  La decisión de
externalizar los estilos responde a varias razones::

1. Reutilización: un mismo *token* de estilo (p. ej. ``message_style``) puede
   aplicarse a múltiples componentes sin duplicar código.
2. Cohesión: se mantiene la lógica de presentación separada de la lógica de
   interacción.
3. Autonomía del front-end: Reflex serializa directamente los diccionarios a
   objetos JS, evitando la dependencia de hojas de estilo externas.

Los nombres de las variables siguen la convención *snake_case* y describen con
precisión su propósito.  No existe código ejecutable – sólo datos – por lo que
modificar este archivo **no** impacta en la lógica de la aplicación.
"""

# ---------------------------------------------------------------------------
# TOKENS BÁSICOS
# ---------------------------------------------------------------------------
# Sombras estándar aplicadas a múltiples elementos (botones, mensajes, etc.).
# Mantener un único *token* evita que surjan inconsistencias en caso de
# cambiar valores de diseño globales.
shadow = "rgba(0, 0, 0, 0.35) 0px 8px 20px"

# Efecto *glassmorphism* reutilizado en fondos translúcidos.
# Agrupar las propiedades relacionadas con blur, fondo y borde facilita su
# reutilización y ajustes finos (mantener coherencia visual).

glass_effect = {
    "backdrop_filter": "blur(16px)",
    "background": "rgba(20, 20, 30, 0.2)",
    "border": "1px solid rgba(255, 255, 255, 0.15)",
    "box_shadow": "0 4px 15px rgba(0, 0, 0, 0.2)",
}

# ---------------------------------------------------------------------------
# EFECTO DE REFLEJO PARA VIDRIO
# ---------------------------------------------------------------------------
# Versión más sutil (baja opacidades) del reflejo de vidrio.
glass_reflection = (
    "linear-gradient(140deg, "
    "rgba(255,255,255,0.12) 0%, "  # Destello inicial tenue
    "rgba(255,255,255,0.06) 20%, "  # Transición suave
    "rgba(255,255,255,0.015) 50%, "  # Zona difusa casi imperceptible
    "rgba(255,255,255,0.04) 80%, "  # Segundo reflejo muy leve
    "rgba(255,255,255,0.01) 100%)"  # Desvanecido final
)

# ---------------------------------------------------------------------------
# ESTILOS PARA MENSAJES DE CHAT
# ---------------------------------------------------------------------------
# "message_style" funciona como base común para preguntas y respuestas.  No
# incluye propiedades específicas de alineación o color, que se definen en los
# diccionarios derivados ``question_style`` y ``answer_style``.
message_style = {
    "padding": "clamp(0.35em, 1.5vw, 0.6em) clamp(0.5em, 2vw, 0.8em)",  # Escala con viewport
    "border_radius": "8px",
    "margin_y": "0.3em",
    "box_shadow": shadow,
    "max_width": "70%",
    "display": "inline-block",
    "transform_origin": "center",
    "animation": "0.5s cubic-bezier(0.25, 0.1, 0.25, 1) 0s 1 slideIn",
    "backdrop_filter": "blur(16px)",
    "line_height": "1.3",
    "position": "relative",
    "_after": {
        "content": "'copiar'",
        "position": "absolute",
        "bottom": "-1.3em",
        "right": "0.5em",
        "font_size": "0.8em",
        "font_weight": "600",
        "color": "rgba(235, 240, 255, 0.7)",
        "opacity": "0",
        "transition": "opacity 0.2s",
        "pointer_events": "none",
    },
    "_hover": {
        "_after": {"opacity": "1"},
    },
}

# Definición de keyframes para animaciones
animations = {
    "@keyframes slideIn": {
        "from": {"opacity": "0", "transform": "translateY(15px)"},
        "to": {"opacity": "1", "transform": "translateY(0)"},
    },
    "@keyframes fadeIn": {
        "from": {"opacity": "0"},
        "to": {"opacity": "1"},
    },
    "@keyframes pulse": {
        "0%": {"transform": "scale(0.95)"},
        "50%": {"transform": "scale(1.05)"},
        "100%": {"transform": "scale(0.95)"},
    },
    "@keyframes glow": {
        "0%": {"box-shadow": "0 0 5px rgba(120, 180, 255, 0.5)"},
        "50%": {"box-shadow": "0 0 20px rgba(120, 180, 255, 0.8)"},
        "100%": {"box-shadow": "0 0 5px rgba(120, 180, 255, 0.5)"},
    },
    "@keyframes spin": {
        "0%": {"transform": "rotate(0deg)"},
        "100%": {"transform": "rotate(360deg)"},
    },
}

# Add markdown-specific styles to make content more compact
markdown_style = {
    "p": {
        "margin_top": "0.3em",
        "margin_bottom": "0.3em",
    },
    "ul, ol": {
        "padding_left": "1em",
        "margin_top": "0.3em",
        "margin_bottom": "0.3em",
    },
    "li": {
        "margin_bottom": "0.1em",
    },
    "h1, h2, h3, h4, h5, h6": {
        "margin_top": "0.4em",
        "margin_bottom": "0.3em",
    },
    "code": {
        "padding": "0.1em 0.2em",
    },
    "pre": {
        "margin": "0.3em 0",
        "padding": "0.4em",
    },
    "blockquote": {
        "margin": "0.3em 0",
        "padding_left": "0.6em",
    },
}

# Set specific styles for questions and answers.
question_style = message_style | {
    "margin_left": "auto",
    # Capa de reflejo + capa base ligeramente oscura
    "background": f"{glass_reflection}, linear-gradient(135deg, rgba(42, 48, 66, 0.015) 0%, rgba(62, 74, 102, 0.015) 100%)",
    "color": "#F8FAFC",
    "transform_origin": "right",
    # Barra vertical tipo vidrio anaranjado oscuro (transparente)
    "border_left": "4px solid rgba(220, 110, 40, 0.25)",
    "letter_spacing": "0.3px",
    "_hover": {
        # 👇 Los valores alfa (0.25, 0.20, 0.15) controlan la TRANSPARENCIA del gradiente.
        "background": "linear-gradient(to bottom, rgba(25, 110, 185, 0.12) 0%, rgba(55, 140, 205, 0.07) 50%, rgba(95, 175, 230, 0.04) 100%)",
        "box_shadow": "0 4px 12px rgba(135, 206, 250, 0.35)",
        "backdrop_filter": "blur(4px)",
        "border": "none",
        "transform": "translateY(-2px)",
        "transition": "all 0.25s ease",
        "_after": {"opacity": "1"},
    },
    **markdown_style,
}
answer_style = message_style | {
    "margin_right": "auto",
    "background": f"{glass_reflection}, linear-gradient(135deg, rgba(30, 41, 59, 0.015) 0%, rgba(51, 65, 85, 0.015) 100%)",
    "color": "#F8FAFC",
    "transform_origin": "left",
    # Barra vertical tipo vidrio anaranjado oscuro (transparente)
    "border_left": "4px solid rgba(220, 110, 40, 0.25)",
    "letter_spacing": "0.3px",
    "_hover": {
        # 👇 Los valores alfa (0.25, 0.20, 0.15) controlan la TRANSPARENCIA del gradiente.
        "background": "linear-gradient(to bottom, rgba(25, 110, 185, 0.12) 0%, rgba(55, 140, 205, 0.07) 50%, rgba(95, 175, 230, 0.04) 100%)",
        "box_shadow": "0 4px 12px rgba(135, 206, 250, 0.35)",
        "backdrop_filter": "blur(4px)",
        "border": "none",
        "transform": "translateY(-2px)",
        "transition": "all 0.25s ease",
        "_after": {"opacity": "1"},
    },
    **markdown_style,
}

# Estilos para encabezados con gradiente
heading_gradient_style = {
    "background": "linear-gradient(135deg, rgba(235, 240, 255, 0.9) 0%, rgba(160, 174, 245, 0.8) 100%)",
    "background_clip": "text",
    "webkit_background_clip": "text",
    "color": "transparent",
    "font_weight": "500",
    "font_size": "clamp(0.95rem, 1vw + 0.6rem, 1.4rem)",  # Escala fluida
    "letter_spacing": "1px",
    "text_shadow": "0 1px 2px rgba(0, 0, 0, 0.2)",
    "transition": "all 0.3s ease",
}

# Sidebar styles
sidebar_style = {
    "width": "var(--sidebar-width, 300px)",
    "height": "100vh",
    "position": "fixed",
    "top": "0",
    "right": "0",
    "background": "radial-gradient(circle at top right, #111827 0%, #030712 100%)",
    "background_image": "url('/svg/chat_background.svg')",
    "background_size": "contain",
    "background_attachment": "fixed",
    "background_position": "center",
    "background_repeat": "repeat",
    "backdrop_filter": "blur(16px)",
    "box_shadow": "-5px 0 20px rgba(0, 0, 0, 0.3)",
    "display": "flex",
    "flex_direction": "column",
    "z_index": "5",
    "padding": "0.8em",
    # "border_left" eliminado para suprimir línea divisoria
    "overflow_y": "auto",
    "animation": "fadeIn 0.5s ease",
    "resize": "horizontal",
    "overflow": "hidden",
    "min_width": "220px",
    "max_width": "600px",
}

# Left sidebar style - similar to sidebar_style but positioned on the left
left_sidebar_style = {
    "width": "var(--left-sidebar-width, 266px)",
    "height": "100vh",
    "position": "fixed",
    "top": "0",
    "left": "0",
    "background": "radial-gradient(circle at top left, #111827 0%, #030712 100%)",
    "background_image": "url('/svg/chat_background.svg')",
    "background_size": "contain",
    "background_attachment": "fixed",
    "background_position": "center",
    "background_repeat": "repeat",
    "backdrop_filter": "blur(16px)",
    "box_shadow": "5px 0 20px rgba(0, 0, 0, 0.3)",
    "display": "flex",
    "flex_direction": "column",
    "z_index": "5",
    "padding": "0.8em",
    # "border_right" eliminado para suprimir línea divisoria
    "overflow_y": "auto",
    "animation": "fadeIn 0.5s ease",
    "min_width": "200px",
    # Bordes redondeados para simular una pestaña de navegador
    "border_top_right_radius": "16px",
    "border_bottom_right_radius": "16px",
}

# Estilo para el título del sidebar que coincide con Multi-Agentic Workflow
sidebar_title_style = {
    "margin": "0",
    "padding": "0.1em 0.3em",
    "font_size": "1rem",  # Tamaño ligeramente menor
    "background": "linear-gradient(135deg, rgba(235, 240, 255, 0.9) 0%, rgba(160, 174, 245, 0.8) 100%)",
    "background_clip": "text",
    "webkit_background_clip": "text",
    "color": "transparent",
    "font_weight": "500",
    "letter_spacing": "0.8px",  # Espaciado de letras ligeramente menor
    "text_shadow": "0 1px 2px rgba(0, 0, 0, 0.2)",
    "display": "inline-block",
    "line_height": "1.1",  # Altura de línea más compacta
}

# Estilo unificado para mensajes en la barra lateral utilizando la misma
# paleta y bordes que las burbujas del chat principal.
# Se parte de ``answer_style`` para heredar el mismo "look & feel" y se
# ajustan únicamente propiedades de tamaño/márgenes que son específicas del
# contenedor lateral.
sidebar_message_style = answer_style | {
    # Ajustes propios del sidebar (no afectan colores ni efecto vidrio)
    "margin": "0.5em 0",
    "width": "95%",
    "max_width": "100%",
    # Borde verde oscuro para el sidebar de Razonamiento y acciones
    "border_left": "4px solid rgba(20, 83, 45, 0.55)",
    "font_size": "0.8rem",  # Reducido un 20% del tamaño estándar
    "animation": "0.5s cubic-bezier(0.25, 0.1, 0.25, 1) 0s 1 slideIn",
    "backdrop_filter": "blur(10px)",
    "box_shadow": "0 2px 10px rgba(0, 0, 0, 0.15)",
}

# Estilo específico para ítems de "Preguntas frecuentes" con borde morado
faq_message_style = sidebar_message_style | {
    "border_left": "4px solid rgba(68, 28, 135, 0.6)",  # Morado oscuro
    "_after": {
        **message_style["_after"],  # Hereda posición y estilos base
        "content": "'Preguntar'",
    },
}

# Estilos responsivos para diferentes dispositivos
responsive_styles = {
    "@media (max-width: 768px)": {
        "message_style": {
            "max_width": "85%",
            "padding": "clamp(0.4em, 1.8vw, 0.7em)",
        },
        "input_container": {
            "width": "95%",
        },
    },
    "@media (max-width: 480px)": {
        "message_style": {
            "max_width": "95%",
            "padding": "clamp(0.35em, 2vw, 0.6em)",
        },
    },
    "@media (max-width: 600px)": {
        "#right-sidebar": {"display": "none"},
        "#left-sidebar": {"display": "none"},
        ":root": {
            "--sidebar-width": "0px",
            "--left-sidebar-width": "0px",
        },
        "main_container_with_sidebar": {
            "margin_left": "0",
            "margin_right": "0",
            "width": "100%",
        },
    },
}

# Estilos para el spinner de carga
spinner_style = {
    "animation": "2s linear infinite spin, 3s infinite glow",
    "transform_origin": "center",
}

# Styles for the action bar.
input_style = {
    "border_width": "1px",
    "padding": "0.8em 1.2em",
    "box_shadow": shadow,
    "width": "80%",
    "border_radius": "48px",
    "border_color": "rgba(99, 102, 241, 0.3)",
    "background": "rgba(30, 41, 59, 0.2)",
    "backdrop_filter": "blur(16px)",
    "color": "#F1F5F9",
    "font_size": "1rem",
    "letter_spacing": "0.3px",
    "transition": "box-shadow 0.2s ease",
    "_hover": {
        "box_shadow": "0 0 6px rgba(135, 206, 250, 0.35)",
    },
    "_focus": {
        "outline": "none",
        "box_shadow": "none",
        "border_color": "rgba(255, 255, 255, 0.2)",
    },
}
button_style = {
    "background": "linear-gradient(135deg, #4F46E5 0%, #6366F1 100%)",
    "box_shadow": shadow,
    "color": "white",
    "font_weight": "600",
    "border_radius": "48px",
    "padding": "0.8em 1.5em",
    "border": "none",
    "transition": "all 0.3s cubic-bezier(0.25, 0.1, 0.25, 1)",
    "hover": {
        "box_shadow": "0 4px 12px rgba(135, 206, 250, 0.35)",
        "background": "linear-gradient(to bottom, rgba(25, 110, 185, 0.45) 0%, rgba(55, 140, 205, 0.40) 50%, rgba(95, 175, 230, 0.35) 100%)",
        "backdrop_filter": "blur(4px)",
        "border": "1px solid rgba(255, 255, 255, 0.18)",
        "transform": "translateY(-2px) scale(1.02)",
    },
    "active": {
        "transform": "translateY(0) scale(0.98)",
    },
}

# Estilos para el contenedor principal
container_style = {
    "width": "100%",
    "max_width": "1200px",
    "margin": "0 auto",
    "padding": "0",
    "height": "100dvh",
    "display": "flex",
    "flex_direction": "column",
    "background": "radial-gradient(circle at top right, #111827 0%, #030712 100%)",
}

# Estilo para el contenedor principal con sidebar
main_container_with_sidebar = {
    "margin_right": "var(--sidebar-width, 300px)",
    "margin_left": "var(--left-sidebar-width, 266px)",
    "width": "calc(100% - var(--sidebar-width, 300px) - var(--left-sidebar-width, 266px))",
    "transition": "none",
}

# ---------------------------------------------------------------------------
# BARRAS DE ENCABEZADO (FAQ y Razonamiento)
# ---------------------------------------------------------------------------
# Para que los encabezados de "Preguntas frecuentes" y "Razonamiento y
# acciones" compartan el mismo estilo base de las burbujas, creamos un token
# reutilizable que solo modifica aspectos de tamaño y layout respecto a
# ``answer_style``.

header_bar_style = answer_style | {
    "width": "100%",
    "max_width": "100%",  # Garantiza que el encabezado use todo el ancho disponible
    "overflow": "visible",  # Evita que iconos o texto sean recortados
    "border_radius": "12px",
    "padding": "0.2em 0.3em",
    "margin_bottom": "0.4em",
    # Un header no debería tener el efecto de copia ni hover tan marcado.
    "_after": {"content": "''"},
    "_hover": answer_style["_hover"] | {"transform": "none"},
}

# ---------------------------------------------------------------------------
# ESTILO GLOBAL PARA SCROLLBARS (consistencia en toda la app)
# ---------------------------------------------------------------------------
global_scrollbar_style = {
    "html": {
        # Para Firefox – color del pulgar y del track
        "scrollbarColor": "rgba(99, 102, 241, 0.4) transparent",
        "scrollbarWidth": "thin",
    },
    "::-webkit-scrollbar": {
        "width": "8px",
        "height": "8px",
    },
    "::-webkit-scrollbar-track": {
        "background": "rgba(30, 41, 59, 0.05)",
        "borderRadius": "8px",
        "backdropFilter": "blur(12px)",
    },
    "::-webkit-scrollbar-thumb": {
        "background": "linear-gradient(135deg, rgba(99, 102, 241, 0.3) 0%, rgba(79, 70, 229, 0.25) 100%)",
        "borderRadius": "8px",
        "border": "2px solid transparent",
        "backgroundClip": "padding-box",
    },
    "::-webkit-scrollbar-thumb:hover": {
        "background": "linear-gradient(135deg, rgba(99, 102, 241, 0.5) 0%, rgba(79, 70, 229, 0.4) 100%)",
        "boxShadow": "0 0 10px rgba(99, 102, 241, 0.5)",
        "border": "1px solid rgba(255, 255, 255, 0.2)",
    },
}
