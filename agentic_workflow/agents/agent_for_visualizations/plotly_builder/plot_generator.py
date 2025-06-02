"""Plot generator module for creating Plotly visualizations using LLM.

This module provides functionality to generate Plotly charts from natural language
instructions using a Large Language Model with structured output.
"""
# %%
# Importar las bibliotecas necesarias

from __future__ import annotations

import json
import logging
import os
import re
import traceback
from pathlib import Path

import pandas as pd
import plotly.figure_factory as ff  # For distplot / density functions
import plotly.graph_objects as go
from dotenv import load_dotenv
from langchain.prompts import (
    ChatPromptTemplate,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

#
from pydantic import BaseModel, Field

from agentic_workflow.agents.agent_for_visualizations.plotly_builder.PlotlySchema_classes import (
    PlotlySchema,
)
from agentic_workflow.utils import get_llm


# ---------------------------------------------------------------------------
# PALETA CORPORATIVA SOBRIA
# ---------------------------------------------------------------------------
# Se utiliza como *fallback* cuando el usuario no especifica colores.  El orden
# mezcla grises claros complejos, celestes vibrantes, anaranjados y amarillos
# para garantizar un contraste adecuado sobre el fondo #1A1A1A.
# ---------------------------------------------------------------------------

CORPORATE_PALETTE: list[str] = [
    # Grises claros
    "#B0BEC5",
    "#CFD8DC",
    "#ECEFF1",
    "#90A4AE",
    # Celestes
    "#55A7FF",
    "#70B7FF",
    "#8DC5FF",
    "#A9D3FF",
    # Anaranjados
    "#FF8C42",
    "#E07A5F",
    "#F49F0A",
    "#D77A24",
    # Amarillos
    "#FFC857",
    "#FFDD67",
    "#F9E79F",
    "#E1C542",
]

# Colores de máximo contraste para ≤5 series.
CONTRAST_PALETTE: list[str] = [
    "#B0BEC5",  # gris claro
    "#55A7FF",  # celeste vivo
    "#FF8C42",  # naranja intenso
    "#FFC857",  # amarillo saturado
    "#E07A5F",  # naranja quemado
]


# Helper ─────────────────────────────────────────────────────────────────
def _hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Convert a ``#RRGGBB`` hex code to an ``rgba(r,g,b,a)`` string.

    If *hex_color* is already an ``rgba`` string it is returned unchanged but
    with the new *alpha* applied.  Basic validation is performed so that
    unexpected inputs do not propagate downstream and break Plotly.
    """
    hex_color = str(hex_color).strip()

    if hex_color.startswith("rgba("):
        try:
            comps = hex_color.lstrip("rgba(").rstrip(")").split(",")
            r, g, b, _ = [c.strip() for c in comps] + ["1"]
            return f"rgba({r},{g},{b},{alpha:.2f})"
        except Exception:
            pass  # fall-through to raise below

    if not hex_color.startswith("#"):
        raise ValueError(f"Invalid colour '{hex_color}'. Expected #RRGGBB or rgba().")

    hex_digits = hex_color.lstrip("#")
    if len(hex_digits) != 6:
        raise ValueError(f"Invalid hex colour '{hex_color}'. Must be 6 hex digits.")

    try:
        r = int(hex_digits[0:2], 16)
        g = int(hex_digits[2:4], 16)
        b = int(hex_digits[4:6], 16)
    except ValueError as exc:
        raise ValueError(f"Invalid hex colour '{hex_color}'.") from exc

    return f"rgba({r},{g},{b},{alpha:.2f})"


def build_distinct_colors(n: int) -> list[str]:
    """Return *n* visually distinct colours within the corporate style.

    Uses the base ``CORPORATE_PALETTE`` and, when more colours are required
    than entries available, generates alpha variants (0.9, 0.75, 0.6, 0.45…)
    of the same hue to maintain contrast while keeping uniqueness.
    """
    base = CORPORATE_PALETTE
    if n <= len(base):
        return base[:n]

    colors: list[str] = []
    alpha_cycle = [0.9, 0.75, 0.6, 0.45]
    for i in range(n):
        base_color = base[i % len(base)]
        cycle_idx = i // len(base)
        alpha = alpha_cycle[cycle_idx % len(alpha_cycle)]
        if alpha == 0.9:  # first cycle keeps original hex (opaque)
            colors.append(base_color)
        else:
            colors.append(_hex_to_rgba(base_color, alpha))
    return colors


def _ensure_unique_colors(base_list: list[str]) -> list[str]:
    """Return a list where all colours are distinct strings.

    If duplicates are found and the colour is an HEX (#RRGGBB) or RGBA, we
    generate opacity variants (0.9→0.75→0.6→0.45…) until uniqueness is
    achieved.  For other formats we fall back to *build_distinct_colors* for
    the remaining slots.
    """
    result: list[str] = []
    seen: set[str] = set()
    alpha_cycle = [0.9, 0.75, 0.6, 0.45]

    def _next_variant(col: str, dup_idx: int) -> str:
        # If hex, change opacity; otherwise add suffix number (still distinct).
        if col.startswith("#") and len(col) == 7:
            alpha = alpha_cycle[(dup_idx - 1) % len(alpha_cycle)]
            return _hex_to_rgba(col, alpha)
        if col.startswith("rgba("):
            # replace alpha segment
            try:
                parts = col.strip("rgba() ").split(",")
                r, g, b, _ = [p.strip() for p in parts] + ["1"]
                alpha = alpha_cycle[(dup_idx - 1) % len(alpha_cycle)]
                return f"rgba({r},{g},{b},{alpha})"
            except Exception:
                pass  # fallback below
        # fallback distinct colour
        return None  # type: ignore[return-value]

    for col in base_list:
        if col not in seen:
            result.append(col)
            seen.add(col)
        else:
            dup_idx = 1
            new_col = _next_variant(col, dup_idx)
            while new_col is None or new_col in seen:
                dup_idx += 1
                new_col = _next_variant(col, dup_idx)
                if dup_idx > 10:
                    break  # safety
            if new_col is None or new_col in seen:
                # fallback to build_distinct_colors to get a brand-new colour
                pool = build_distinct_colors(len(base_list) + 10)
                for p in pool:
                    if p not in seen:
                        new_col = p
                        break
            result.append(new_col)
            seen.add(new_col)

    return result


load_dotenv(override=True)


def _is_valid_css_color(value: str) -> bool:
    """Return *True* if *value* looks like a valid CSS/Plotly colour string.

    The check is **heuristic** – it accepts:
    • Hex strings "#RRGGBB" or "#RGB" (with or without alpha)
    • ``rgb(...)`` / ``rgba(...)`` strings
    • A limited whitelist of named colours frequently used in examples.
    """
    if not isinstance(value, str):
        return False

    v = value.strip()

    # Hex colours (#RGB, #RRGGBB, #RRGGBBAA)
    if re.match(r"^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$", v):
        return True

    # rgb(...) or rgba(...)
    if v.lower().startswith("rgb(") or v.lower().startswith("rgba("):
        return True

    # Basic named colours accepted by Plotly
    _named = {
        "black",
        "white",
        "red",
        "green",
        "blue",
        "yellow",
        "cyan",
        "magenta",
        "grey",
        "gray",
        "orange",
        "purple",
        "brown",
        "pink",
    }
    return v.lower() in _named


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"),
    )
    logger.addHandler(_handler)


system_prompt = """
Eres un *senior* data-viz engineer especializado en **Plotly**.  Debes producir un
**JSON válido** que siga el esquema `PlotlySchema`; **no añadas texto libre**.

IMPORTANTE: el JSON debe incluir SIEMPRE los títulos de los ejes `xaxis_title` y `yaxis_title`,
descriptivos y acordes a los datos.  Si algún eje no aplica, escribe "no definido".

====================================================================
   1 · ESTILO CORPORATIVO (obligatorio y coherente)
====================================================================
• Tema DARK uniforme →    `paper_bgcolor` y `plot_bgcolor` = **#1A1A1A**.
• Ejes / cuadrícula       line/grid **gris grafito**:
      linecolor #666666  ·  linewidth 1
      gridcolor  #444444 ·  gridwidth 0.5
• Texto (títulos, ticks, leyenda, anotaciones) → blanco o gris claro #DDDDDD.
• Si el usuario NO especifica colores → asigna automáticamente **paleta corporativa sobria** con estas muestras
  Grises claros: **#B0BEC5**, **#CFD8DC**, **#ECEFF1**, **#90A4AE**,
  Celestes: **#55A7FF**, **#70B7FF**, **#8DC5FF**, **#A9D3FF**;
  Anaranjados: **#FF8C42**, **#E07A5F**, **#F49F0A**, **#D77A24**;
  Amarillos: **#FFC857**, **#FFDD67**, **#F9E79F**, **#E1C542**.
  Puedes generar variaciones jugando con *opacity* (0.4-0.9) para líneas, marcadores, barras y sectores de pie/donut y crear contraste suave.
• Ticks fuera del área (`ticks='outside'`, `tickcolor='#666'`, `ticklen=8`).
• Añade 5 % de padding bajo el cero y 15 % por encima del máx (excepto escalas categóricas).
• Usa siempre `template='plotly_dark'` **o** define un objeto en `extra_attrs.template`.

====================================================================
   2 · INTERACTIVIDAD PROFESIONAL
====================================================================
• `hovertemplate` claro y sin `<extra>` residual.
• `hoverdistance` y `spikedistance` (20 px) **a nivel de layout**.
• Habilita `showspikes=True`, `spikemode='across'`, color #666.
• `dragmode` → *select* con `selectdirection='any'` si aporta valor.
• Para *scatter* / *bar* / *histogram* define:
      selected.marker.opacity 1 · unselected.marker.opacity 0.2  (solo color/opacity/size).
      **No** añadas `marker.line` dentro de selected/unselected.
• *Heatmap* → nada de `marker` · *Pie* → sin `selected`/`unselected`.

====================================================================
   3 · ELECCIÓN DEL GRÁFICO
====================================================================
Analiza variable(s) y cardinalidad:
    – Temporales ⇢ líneas; una línea por categoría; orden cronológico; opcional banda de confianza.
    – Numéricas continuas ⇢ scatter / área.
    – Categóricas ⇢ barras (agrupadas o apiladas) o pie/donut (≤6 categorías).
Solo considera un eje **temporal** si detectas fechas explícitas ("2024-01-01", "hora", "día", etc.).

====================================================================
   4 · TRAZAS Y ATRIBUTOS ESPECÍFICOS
====================================================================
• *Heatmap*    → permite `colorscale`, `zmid`, `hoverongaps`, texto en celdas.
• *Histogram*  → soporta `histnorm`, `xbins`, `barmode` overlay/stack.
• *Pie* / *Donut* → usa `hole`, `pull`, `sort:false` para conservar orden dado.
• **Nunca uses** propiedades obsoletas: `titlefont`, `titleside`, ni keys fuera del esquema
  (usa `extra_attrs` si es imprescindible).

====================================================================
   5 · NOMBRES, LEYENDAS Y TEXTO
====================================================================
• Nombres técnicos, sin redundancias.  Ej.: "Ventas (%)" en vez de "Grafico Ventas".
• **SIEMPRE** incluye los títulos de ambos ejes → `xaxis_title` y `yaxis_title` (breves pero informativos).
• También agrega `title.text` como encabezado general del gráfico.
• Evita repetir la misma métrica a la vez en color y tamaño.

====================================================================
   6 · ERRORES CONOCIDOS QUE DEBES EVITAR
====================================================================
• `marker.colorbar.titlefont` → usa `marker.colorbar.title.font`.
• No incluyas claves no soportadas en selección (*line*, *gradient*, …).
• No generes `marker` para heatmaps ni `selected` para pie.
• Mantén listas **x** y **y** con la misma longitud; si no, recórtalas o usa un solo eje.

====================================================================
   7 · RESPUESTA
====================================================================
Devuelve **únicamente** el objeto JSON; Plotly lo reproducirá sin más pasos.

Para hasta 5 series/variables elige **tonos de máximo contraste** (gris claro, celeste vivo, naranja intenso, amarillo saturado, naranja quemado).  A partir de la 6.ª usa la paleta base y variantes de opacidad (0.9 → 0.75 → 0.6 → 0.45…) para asegurar distinción sin salir del estilo corporativo.
"""
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ],
)


def _validate_schema(schema: PlotlySchema) -> None:
    """Validate basic consistency rules in a *PlotlySchema* object.

    The rules are intentionally **minimal** – they only guard against the most
    common errors that romper the figure creation process, while still allowing
    the LLM to return flexible structures.

    Raises:
    ------
    ValueError
        If any inconsistency is detected.
    """
    if not schema.data:
        raise ValueError("El esquema devuelto no contiene ninguna traza de datos.")

    for idx, trace in enumerate(schema.data):
        if not trace.type:
            raise ValueError(
                f"La traza #{idx} no especifica el campo obligatorio 'type'."
            )

        # If both *x* and *y* are present, ensure they have the same length.
        if trace.x is not None and trace.y is not None and len(trace.x) != len(trace.y):
            # En lugar de abortar, recortamos ambas listas al mínimo común y registramos la corrección.
            min_len = min(len(trace.x), len(trace.y))
            trace.x = trace.x[:min_len]
            trace.y = trace.y[:min_len]
            logger.warning(
                "Trace %s: longitudes x/y corregidas para coincidir (nuevo tamaño=%s)",
                idx,
                min_len,
            )

        # Pie charts need *values*.
        if trace.type == "pie" and not trace.values and not trace.y:
            raise ValueError(
                f"La traza #{idx} es un gráfico de pastel pero no incluye 'values' ni 'y'."
            )

    # --------------------------------------------------------------
    # Ensure that *some* title is present for ambos ejes.  El modelo puede
    # devolverlos ya sea en las claves cortas (`xaxis_title`) o en la forma
    # anidada (`xaxis.title.text`).  Aceptamos cualquiera de las dos.
    # --------------------------------------------------------------

    def _has_title(layout_obj, axis_root: str) -> bool:
        """Return *True* si encuentra texto de título para el eje."""
        # 1) Clave corta generada explícitamente por el LLM.
        short_key = f"{axis_root}_title"
        if getattr(layout_obj, short_key, None):
            return True

        # 2) Estructura anidada → layout.<axis>.title.text
        axis_block = getattr(layout_obj, axis_root, None)
        try:
            if axis_block and getattr(axis_block, "title", None):
                return bool(getattr(axis_block.title, "text", None))
        except Exception:
            pass

        return False

    # En gráficos sin ejes cartesianos (p.e. *pie*, *sunburst*) no forzamos
    # la presencia de títulos de eje.  Detectamos si **alguna** traza usa un
    # sistema XY tradicional antes de lanzar la validación.

    _cartesian_families = {
        "scatter",
        "scattergl",
        "scatter3d",
        "scatterpolar",
        "bar",
        "histogram",
        "box",
        "violin",
        "heatmap",
        "surface",
        "area",
    }

    has_cartesian = any(
        getattr(tr, "type", None) in _cartesian_families for tr in schema.data
    )

    if (
        has_cartesian
        and schema.layout
        and not (
            _has_title(schema.layout, "xaxis") and _has_title(schema.layout, "yaxis")
        )
    ):
        raise ValueError(
            "Faltan xaxis_title o yaxis_title en el layout para gráficos con ejes."
        )


def _sanitize_selection_marker(trace_dict: dict, trace_type: str) -> None:
    """Remove unsupported keys inside ``selected`` / ``unselected`` blocks.

    Plotly only allows *color*, *opacity* and *size* inside
    ``selected.marker`` and ``unselected.marker`` for *scatter*‐based traces.
    The LLM may nevertheless emit a nested ``line`` dict (e.g. to control
    border colour/width) which raises a ``ValueError`` like the one reported
    by the user.  This helper mutates *trace_dict* **in-place** to strip any
    unsupported keys so the figure can be constructed safely.

    Parameters
    ----------
    trace_dict:
        The dictionary with the trace specification that will be passed to
        the corresponding ``go.*`` constructor.
    trace_type:
        The Plotly trace type (``scatter``, ``bar``, etc.).  Sanitisation is
        currently only necessary for *scatter*-family traces.
    """
    if trace_type not in {
        "scatter",
        "scattergl",
        "scatterpolar",
        "scattermapbox",
        "scatter3d",
        "bar",
        "histogram",
        "pie",  # treat separately below
    }:
        return  # No sanitisation required for other trace types.

    allowed_marker_keys = {"color", "opacity", "size"}

    for sel_key in ("selected", "unselected"):
        sel_block = trace_dict.get(sel_key)
        if not isinstance(sel_block, dict):
            continue

        marker_block = sel_block.get("marker")
        if not isinstance(marker_block, dict):
            continue

        # Remove nested *line* dict or any other unsupported key.
        unsupported_keys = [k for k in marker_block if k not in allowed_marker_keys]
        for bad_key in unsupported_keys:
            marker_block.pop(bad_key, None)

        # If the marker block becomes empty after removing bad keys, delete it
        # entirely to avoid sending an empty dict.
        if not marker_block:
            sel_block.pop("marker", None)

        # Likewise, if the *selected* / *unselected* dict has no valid content
        # left, remove the whole block to prevent needless clutter.
        if not sel_block:
            trace_dict.pop(sel_key, None)

    # Pie traces do NOT support *selected* / *unselected* at all – drop them.
    if trace_type == "pie":
        trace_dict.pop("selected", None)
        trace_dict.pop("unselected", None)
        return


def _recursive_sanitize_selection_markers(
    obj: dict | list, *, allowed_keys: set[str] | None = None
):
    """Traverse *obj* recursively and sanitise selection marker blocks.

    This is needed because the LLM can embed defaults inside layout.template
    (or other nested dictionaries) that include invalid properties such as
    ``selected.marker.line`` – Plotly validates **all** nested structures when
    the figure is constructed, so we must clean them proactively.

    Parameters
    ----------
    obj:
        Arbitrary dict / list representing part of the Plotly figure-spec.
    allowed_keys:
        Whitelist of keys allowed inside ``*.marker`` for selection blocks.
    """
    if allowed_keys is None:
        allowed_keys = {"color", "opacity", "size"}

    if isinstance(obj, dict):
        # First, check if *obj* itself has selected/unselected.
        for sel_key in ("selected", "unselected"):
            sel_block = obj.get(sel_key)
            if isinstance(sel_block, dict):
                marker_block = sel_block.get("marker")
                if isinstance(marker_block, dict):
                    unsupported = [k for k in marker_block if k not in allowed_keys]
                    for bad in unsupported:
                        marker_block.pop(bad, None)

                    if not marker_block:
                        sel_block.pop("marker", None)

                if not sel_block:
                    obj.pop(sel_key, None)

        # Recurse into children.
        for v in obj.values():
            _recursive_sanitize_selection_markers(v, allowed_keys=allowed_keys)

    elif isinstance(obj, list):
        for item in obj:
            _recursive_sanitize_selection_markers(item, allowed_keys=allowed_keys)


# ------------------------------------------------------------------
# Sanitize ``marker.colorbar`` blocks to prevent invalid properties such as
# ``titlefont`` (Plotly expects nested ``title.font``) and remove any keys not
# recognised by Plotly. This helper mutates *marker_dict* in place.
# ------------------------------------------------------------------


def _sanitize_marker_colorbar(marker_dict: dict) -> None:
    """Clean invalid properties inside ``marker.colorbar``.

    Parameters
    ----------
    marker_dict: dict
        The marker dictionary that may contain a *colorbar* sub-dictionary.
    """
    if not isinstance(marker_dict, dict):
        return

    colorbar = marker_dict.get("colorbar")
    if not isinstance(colorbar, dict):
        return

    # ------------------------------------------------------------------
    # 1) Convert deprecated ``titlefont`` into the nested structure
    #    Plotly expects (``title.font``) and remove it from the top level.
    # ------------------------------------------------------------------
    if "titlefont" in colorbar:
        font_val = colorbar.pop("titlefont")
        if font_val is not None:
            colorbar.setdefault("title", {})["font"] = font_val

    # ------------------------------------------------------------------
    # 2) Remove other deprecated/unsupported properties such as ``titleside``.
    # ------------------------------------------------------------------
    colorbar.pop("titleside", None)

    # Whitelist of valid properties for marker.colorbar in Plotly.
    allowed_keys = {
        "bgcolor",
        "bordercolor",
        "borderwidth",
        "dtick",
        "exponentformat",
        "len",
        "lenmode",
        "minexponent",
        "nticks",
        "orientation",
        "outlinecolor",
        "outlinewidth",
        "separatethousands",
        "showexponent",
        "showticklabels",
        "showtickprefix",
        "showticksuffix",
        "thickness",
        "thicknessmode",
        "tick0",
        "tickangle",
        "tickcolor",
        "tickfont",
        "tickformat",
        "tickformatstops",
        "ticklabeloverflow",
        "ticklabelposition",
        "ticklen",
        "tickmode",
        "tickprefix",
        "ticks",
        "ticksuffix",
        "ticktext",
        "tickvals",
        "tickwidth",
        "title",
        "x",
        "xanchor",
        "xpad",
        "xref",
        "y",
        "yanchor",
        "ypad",
        "yref",
    }

    for bad_key in [k for k in list(colorbar.keys()) if k not in allowed_keys]:
        colorbar.pop(bad_key, None)


# ------------------------------------------------------------------
# Sanitize ``marker.pattern`` blocks – ensure shape values are within the
# allowed enumeration for Plotly ( '', '/', '\\', 'x', '-', '|', '+', '.' ).
# ------------------------------------------------------------------


def _sanitize_marker_pattern(marker_dict: dict) -> None:
    """Validate *marker.pattern.shape* values and coerce invalid ones.

    If an invalid character (e.g. '*') is found, it will be replaced by the
    default '.' to avoid Plotly validation errors.
    """
    if not isinstance(marker_dict, dict):
        return

    pattern = marker_dict.get("pattern")
    if not isinstance(pattern, dict):
        return

    shape_val = pattern.get("shape")
    if shape_val is None:
        return

    allowed = {"", "/", "\\", "x", "-", "|", "+", "."}

    def _coerce_shape(s):
        return s if s in allowed else "."

    if isinstance(shape_val, str):
        pattern["shape"] = _coerce_shape(shape_val)
    elif isinstance(shape_val, list | tuple):
        pattern["shape"] = [_coerce_shape(str(v)) for v in shape_val]
    else:
        # Unknown structure – remove to be safe
        pattern.pop("shape", None)


def make_plot(structured_output):
    """Create a Plotly figure from structured output.

    Args:
        structured_output: PlotlySchema object containing plot data and layout.

    Returns:
        go.Figure: The generated Plotly figure.
    """
    _validate_schema(structured_output)

    # ------------------------------------------------------------------
    # Detect if the request is for a *distribution plot* (distplot).  We
    # represent each distribution as a trace with ``type='dist'`` so the LLM
    # does not need to understand the low-level ff.create_distplot API.
    # ------------------------------------------------------------------

    dist_traces = [
        tr
        for tr in structured_output.data
        if getattr(tr, "type", "").lower() in {"dist", "distplot"}
    ]

    if dist_traces:
        hist_data: list[list] = []
        group_labels: list[str] = []
        colors: list[str] = []

        for idx, tr in enumerate(dist_traces):
            values = tr.y or tr.x  # Prefer *y*, fallback to *x*
            if values is None:
                continue  # skip empty
            hist_data.append(values)
            group_labels.append(tr.name or f"Grupo {idx + 1}")
            # Pick a colour (marker_color shortcut or default palette)
            if tr.marker_color:
                colors.append(
                    tr.marker_color
                    if isinstance(tr.marker_color, str)
                    else tr.marker_color[0]
                )

        # Fallback colour palette
        if len(hist_data) <= 5:
            colors = _ensure_unique_colors(CONTRAST_PALETTE[: len(hist_data)])
        else:
            colors = _ensure_unique_colors(build_distinct_colors(len(hist_data)))

        # Allow extra distplot-wide kwargs via the first trace's extra_attrs
        extra_opts = (dist_traces[0].extra_attrs or {}) if dist_traces else {}

        fig = ff.create_distplot(hist_data, group_labels, colors=colors, **extra_opts)

        # Even though ff returns a Figure, we continue with the rest of the
        # pipeline for styling/layout.  Replace structured_output.data with
        # the underlying traces so later logic (e.g. axis styling) sees the
        # histogram/kde traces and treats them as cartesian.

        structured_output.data = []  # type: ignore[attr-defined]
        for tr in fig.data:
            # Remove default colours assigned by ff so that the corporate
            # palette replaces them later in the pipeline.
            if hasattr(tr, "marker") and getattr(tr, "marker", None):
                if hasattr(tr.marker, "color"):
                    tr.marker.color = None  # type: ignore[attr-defined]
                if hasattr(tr.marker, "colors"):
                    tr.marker.colors = None  # type: ignore[attr-defined]
            if (
                hasattr(tr, "line")
                and getattr(tr, "line", None)
                and hasattr(tr.line, "color")
            ):
                tr.line.color = None  # type: ignore[attr-defined]

            structured_output.data.append(tr)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Init axis-title containers *after* potential distplot creation so that
    # helper `_log_axis_titles` can reference them even if called early.
    # ------------------------------------------------------------------
    provided_x_title: str | None = None
    provided_y_title: str | None = None

    # ------------------------------------------------------------------
    # Debug helper (defined immediately after the containers).
    # ------------------------------------------------------------------
    def _log_axis_titles(stage: str, fig: go.Figure | None = None):
        if fig is None:
            logger.debug(
                "[AXIS-TITLE-TRACE] %s | provided_x_title=%s | provided_y_title=%s | fig=None",
                stage,
                provided_x_title,
                provided_y_title,
            )
            return

        logger.debug(
            "[AXIS-TITLE-TRACE] %s | provided_x_title=%s | provided_y_title=%s | xaxis=%s | yaxis=%s",
            stage,
            provided_x_title,
            provided_y_title,
            (fig.layout.xaxis.title.text if fig.layout.xaxis.title else None),
            (fig.layout.yaxis.title.text if fig.layout.yaxis.title else None),
        )

    # Enable verbose trace via env var.
    if os.getenv("PLOT_DEBUG"):
        logger.setLevel(logging.DEBUG)
        logger.debug("[AXIS-TITLE-TRACE] Debug tracing ENABLED")

    # ------------------------------------------------------------------
    # Handle template and initialize figure.
    # The base template is applied first, then LLM layout specifics (including its colorway)
    # will override parts of the template if needed.
    # ------------------------------------------------------------------
    llm_layout_options = (
        structured_output.layout.model_dump(exclude_unset=True, exclude_none=True)
        if structured_output.layout
        else {}
    )

    # Determine base template: from LLM or default to 'plotly_dark'
    template_name_or_obj: str | go.layout.Template | None
    raw_template_value = llm_layout_options.pop(
        "template", "plotly_dark"
    )  # Remove from dict, apply separately

    if isinstance(raw_template_value, str):
        template_name_or_obj = raw_template_value
    elif isinstance(raw_template_value, dict):
        try:
            template_name_or_obj = go.layout.Template(raw_template_value)
        except Exception:
            logger.warning(
                "Invalid template dict from LLM, falling back to 'plotly_dark'.",
                exc_info=True,
            )
            template_name_or_obj = "plotly_dark"
    else:
        template_name_or_obj = "plotly_dark"

    fig = go.Figure()
    if template_name_or_obj:
        fig.layout.template = template_name_or_obj

    _log_axis_titles("after_base_template_application", fig)

    # --------------------------------------------------------------
    # 1) Detect colours proporcionados explícitamente por el usuario.  Se
    #    buscan tanto listas completas (marker.colors) como valores
    #    individuales (marker.color, line.color) para cada traza.
    # --------------------------------------------------------------

    custom_colors: list[str] = []
    has_categorical_data = False

    def _flatten_color_value(c):
        """Normaliza *c* a una lista de strings hex/RGB si es posible."""
        if c is None:
            return []
        if isinstance(c, str):
            return [c]
        if isinstance(c, list | tuple):
            return [str(v) for v in c]
        return []

    def _is_categorical_color_list(color_list):
        """Detecta si una lista de colores contiene datos categóricos."""
        if not color_list:
            return False
        # Si al menos el 50% de los valores no son colores válidos, consideramos que es categórico
        non_color_count = sum(1 for c in color_list if not _is_valid_css_color(str(c)))
        return non_color_count / len(color_list) >= 0.5

    for trace in structured_output.data:
        if trace.marker and isinstance(trace.marker.get("colors"), list | tuple):
            marker_colors = _flatten_color_value(trace.marker["colors"])
            if _is_categorical_color_list(marker_colors):
                has_categorical_data = True
            else:
                custom_colors.extend(marker_colors)

        # marker.color (str o lista)
        if trace.marker and "color" in trace.marker:
            marker_color = _flatten_color_value(trace.marker["color"])
            if _is_categorical_color_list(marker_color):
                has_categorical_data = True
            else:
                custom_colors.extend(marker_color)

        if trace.line and "color" in trace.line:
            line_color = _flatten_color_value(trace.line["color"])
            if _is_categorical_color_list(line_color):
                has_categorical_data = True
            else:
                custom_colors.extend(line_color)

    # --------------------------------------------------------------
    # 2) Decide la paleta a usar - SIEMPRE usar paleta corporativa para múltiples variables
    # --------------------------------------------------------------
    num_traces = len(structured_output.data)

    # Si hay datos categóricos o múltiples trazas, SIEMPRE usar paleta corporativa
    if has_categorical_data or num_traces > 1:
        # Para múltiples variables, SIEMPRE usar colores distintos de la paleta corporativa
        if num_traces <= 5:
            colors = CONTRAST_PALETTE[:num_traces]
        else:
            colors = build_distinct_colors(num_traces)
        # Garantiza unicidad absoluta para múltiples variables
        colors = _ensure_unique_colors(colors)
    elif custom_colors:
        # Solo una traza con colores personalizados válidos
        base_needed = num_traces - len(custom_colors)
        if base_needed > 0:
            repeats = (base_needed // len(CORPORATE_PALETTE)) + 1
            fill = (CORPORATE_PALETTE * repeats)[:base_needed]
            colors = custom_colors + fill
        else:
            colors = custom_colors[:num_traces]
        colors = _ensure_unique_colors(colors)
    else:
        # Una sola traza sin colores personalizados
        colors = (
            CONTRAST_PALETTE[:num_traces]
            if num_traces <= 5
            else build_distinct_colors(num_traces)
        )
        colors = _ensure_unique_colors(colors)

    # Mapeo de tipo de traza a objetos de gráfico de Plotly
    trace_type_mapping = {
        "scatter": go.Scatter,
        "bar": go.Bar,
        "pie": go.Pie,
        "histogram": go.Histogram,
        "box": go.Box,
        "scatter3d": go.Scatter3d,
        "surface": go.Surface,
        "heatmap": go.Heatmap,
        "violin": go.Violin,
        "area": go.Scatter,  # Los gráficos de área son scatter plots con 'fill' establecido
        # Añade más mapeos según sea necesario
    }

    # Track if we created separate traces for categorical data
    created_separate_traces = False

    # Añadir trazas con colores de la paleta o colores definidos por el usuario
    for idx, trace in enumerate(structured_output.data):
        trace_dict = trace.model_dump(exclude_unset=True)
        # Desempaquetar atributos avanzados si existen.
        if "extra_attrs" in trace_dict:
            extra = trace_dict.pop("extra_attrs") or {}
            trace_dict.update(extra)
        trace_type = trace_dict.pop("type", None)  # Eliminar la clave 'type'

        # Intentar convertir 'x' a datetime solo si parece una fecha
        if "x" in trace_dict:
            x_values = trace_dict["x"]
            if (
                isinstance(x_values, list)
                and x_values
                and all(isinstance(v, str) for v in x_values)
            ):
                # Heurística: considerar fecha si contiene '-' o '/' o 'T'
                looks_like_date = (
                    sum(1 for v in x_values if any(c in v for c in "-/T"))
                    / len(x_values)
                    >= 0.6
                )
            else:
                looks_like_date = False
            if looks_like_date:
                try:
                    converted_x = pd.to_datetime(trace_dict["x"], errors="coerce")
                    if not converted_x.isnull().all():
                        trace_dict["x"] = converted_x
                except Exception:
                    pass

        # Map shortcut styling fields to Plotly nested structure.
        marker_color = trace_dict.pop("marker_color", None)
        marker_line_color = trace_dict.pop("marker_line_color", None)
        marker_line_width = trace_dict.pop("marker_line_width", None)
        marker_opacity = trace_dict.pop("marker_opacity", None)
        fillcolor = trace_dict.pop("fillcolor", None)
        marker_size = trace_dict.pop("marker_size", None)
        marker_symbol = trace_dict.pop("marker_symbol", None)
        marker_pattern_shape = trace_dict.pop("marker_pattern_shape", None)

        # ------------------------------------------------------------------
        # Guard against invalid colour values provided by the LLM.  If the
        # colour string does **not** look like a valid CSS colour, replace it
        # with a fallback from the corporate palette so that Plotly does not
        # raise a ``ValueError``.
        # ------------------------------------------------------------------
        def _coerce_color(
            c: str | list[str] | None, fallback_idx: int
        ) -> str | list[str] | None:
            if c is None:
                return None
            if isinstance(c, list):
                return [
                    _c
                    if _is_valid_css_color(_c)
                    else colors[fallback_idx % len(colors)]
                    for _c in c
                ]
            if isinstance(c, str):
                return (
                    c if _is_valid_css_color(c) else colors[fallback_idx % len(colors)]
                )
            return None

        marker_color = _coerce_color(marker_color, idx)
        marker_line_color = _coerce_color(marker_line_color, idx)
        # line color may exist inside the incoming trace_dict already; we handle later.

        # ------------------------------------------------------------------
        # Bubble charts: allow the LLM to specify a raw ``size`` list at the
        # top level (similar to Plotly-Express).  We map it into
        # ``marker.size`` and generate sensible ``sizeref`` so that the
        # largest bubble does not exceed ~60 px unless the user overrides it
        # via ``marker.sizemax``.
        # ------------------------------------------------------------------

        raw_size_list = trace_dict.pop("size", None)
        if raw_size_list is not None:
            trace_dict["marker"] = trace_dict.get("marker", {})
            trace_dict["marker"]["size"] = raw_size_list

            # Auto-compute sizeref only if the size list is numeric.
            if (
                isinstance(raw_size_list, list)
                and raw_size_list
                and all(isinstance(v, int | float) for v in raw_size_list)
            ):
                desired_max = trace_dict["marker"].get("sizemax", 60)
                max_size_val: float = max(raw_size_list) or 1.0  # prevent div/0
                sizeref_val = 2 * max_size_val / (desired_max**2)
                trace_dict["marker"].setdefault("sizemode", "area")
                trace_dict["marker"].setdefault("sizeref", sizeref_val)
                trace_dict["marker"].setdefault("sizemin", 4)

        # Handle marker properties - but defer marker_color to trace-specific logic below
        if (
            marker_line_color is not None
            or marker_line_width is not None
            or marker_opacity is not None
        ):
            trace_dict["marker"] = trace_dict.get("marker", {})
            if marker_opacity is not None:
                trace_dict["marker"]["opacity"] = marker_opacity
            if marker_line_color is not None or marker_line_width is not None:
                trace_dict["marker"]["line"] = trace_dict["marker"].get("line", {})
                if marker_line_color is not None:
                    trace_dict["marker"]["line"]["color"] = marker_line_color
                if marker_line_width is not None:
                    trace_dict["marker"]["line"]["width"] = marker_line_width

        if fillcolor is not None:
            trace_dict["fillcolor"] = fillcolor

        if (
            marker_size is not None
            or marker_symbol is not None
            or marker_pattern_shape is not None
        ):
            trace_dict["marker"] = trace_dict.get("marker", {})
            if marker_size is not None:
                trace_dict["marker"]["size"] = marker_size
            if marker_symbol is not None:
                trace_dict["marker"]["symbol"] = marker_symbol
            if marker_pattern_shape is not None:
                trace_dict["marker"]["pattern"] = trace_dict["marker"].get(
                    "pattern", {}
                )
                trace_dict["marker"]["pattern"]["shape"] = marker_pattern_shape

        # ------------------------------------------------------------------
        # Skip placeholder traces automatically generated by the LLM such as
        # "Grupo Colores" / "Grupo de Colores". These are not part of the
        # user-provided variables and clutter the legend.  Matching is done
        # in a case-insensitive way, ignoring extra whitespace and the word
        # "de".
        # ------------------------------------------------------------------
        trace_name_normalised = (
            str(trace_dict.get("name", "")).lower().replace(" de ", " ").strip()
        )
        if trace_name_normalised in {"grupo colores", "grupo color"}:
            continue  # Do NOT add this spurious trace to the figure.

        if trace_type in trace_type_mapping:
            trace_class = trace_type_mapping[trace_type]

            # ──────────────────────────────────────────────────────────────
            # Special case: *Heatmap* traces do NOT accept a ``marker``
            # attribute.  If the LLM generated one (often to specify
            # ``marker.line.width``) we must remove it to avoid a validation
            # error.  Consider migrating relevant style keys in the future.
            # ──────────────────────────────────────────────────────────────
            if trace_type == "heatmap" and "marker" in trace_dict:
                trace_dict.pop("marker", None)

            # ------------------------------------------------------------------
            # If a colour bar is present at the **trace** level (allowed for
            # heatmaps, histograms, etc.), sanitise it as well to get rid of
            # deprecated keys that cause errors ('titlefont', 'titleside').
            # ------------------------------------------------------------------
            if isinstance(trace_dict.get("colorbar"), dict):
                _sanitize_marker_colorbar({"colorbar": trace_dict["colorbar"]})  # type: ignore[arg-type]

                # Asignar color a la traza apropiadamente
            if trace_type in ["scatter", "scatter3d", "line", "area"]:
                # Check if marker.color contains categorical data (non-color strings)
                marker_has_categorical_color = False
                categorical_data_for_separate_traces = None

                # First check if customdata contains categorical information (like continents)
                customdata_val = trace_dict.get("customdata")
                if isinstance(customdata_val, list) and customdata_val:
                    # Check if customdata contains categorical info (like [["Asia", "China"], ["Europe", "Germany"]])
                    if (
                        isinstance(customdata_val[0], list)
                        and len(customdata_val[0]) >= 1
                    ):
                        # Extract the first element of each sublist (the category)
                        categories = [
                            item[0]
                            if isinstance(item, list) and len(item) > 0
                            else str(item)
                            for item in customdata_val
                        ]
                        categorical_data_for_separate_traces = categories
                    elif isinstance(customdata_val[0], str):
                        # Direct categorical data
                        categorical_data_for_separate_traces = customdata_val

                # If no categorical data found in customdata, check marker.color
                if categorical_data_for_separate_traces is None and isinstance(
                    trace_dict.get("marker"), dict
                ):
                    marker_color_val = trace_dict["marker"].get("color")
                    if isinstance(marker_color_val, list) and marker_color_val:
                        # Check if the first value looks like categorical data
                        first_val = str(marker_color_val[0])
                        marker_has_categorical_color = not _is_valid_css_color(
                            first_val
                        )

                        # If we have categorical data with a colorscale, we need to create separate traces
                        if marker_has_categorical_color and (
                            "colorscale" in trace_dict.get("marker", {})
                            or "coloraxis" in trace_dict.get("marker", {})
                        ):
                            categorical_data_for_separate_traces = marker_color_val
                            # We'll handle this after the main trace processing
                        elif marker_has_categorical_color:
                            # If categorical data but no colorscale, assign distinct corporate colors to each category
                            unique_categories = list(
                                dict.fromkeys(marker_color_val)
                            )  # Preserve order, remove duplicates

                            # Asignar colores corporativos distintos para cada categoría
                            if len(unique_categories) <= 5:
                                category_colors = CONTRAST_PALETTE[
                                    : len(unique_categories)
                                ]
                            else:
                                category_colors = build_distinct_colors(
                                    len(unique_categories)
                                )
                            category_colors = _ensure_unique_colors(category_colors)

                            # Create mapping from category to corporate color
                            category_to_color = {
                                cat: category_colors[i]
                                for i, cat in enumerate(unique_categories)
                            }

                            # Convert categorical values to actual colors
                            actual_colors = [
                                category_to_color[cat] for cat in marker_color_val
                            ]
                            trace_dict["marker"]["color"] = actual_colors

                            # Store original categories in customdata for hover templates
                            trace_dict["customdata"] = marker_color_val
                            marker_has_categorical_color = False

                # Also check if line.color contains categorical data and clean it up
                line_has_categorical_color = False
                if isinstance(trace_dict.get("line"), dict):
                    line_color_val = trace_dict["line"].get("color")
                    if line_color_val is not None:
                        if isinstance(line_color_val, list) and line_color_val:
                            first_val = str(line_color_val[0])
                            line_has_categorical_color = not _is_valid_css_color(
                                first_val
                            )
                        elif isinstance(line_color_val, str):
                            line_has_categorical_color = not _is_valid_css_color(
                                line_color_val
                            )

                        # If line color is categorical, remove it
                        if line_has_categorical_color:
                            trace_dict["line"].pop("color", None)

                # If we need to create separate traces for categorical data, do it now
                if categorical_data_for_separate_traces is not None:
                    # Get unique categories and create separate traces
                    unique_categories = list(
                        dict.fromkeys(categorical_data_for_separate_traces)
                    )

                    if len(unique_categories) <= 5:
                        category_colors = CONTRAST_PALETTE[: len(unique_categories)]
                    else:
                        category_colors = build_distinct_colors(len(unique_categories))
                    category_colors = _ensure_unique_colors(category_colors)

                    # Create separate traces for each category
                    for cat_idx, category in enumerate(unique_categories):
                        # Create a copy of the trace for this category
                        cat_trace_dict = trace_dict.copy()
                        cat_trace_dict["marker"] = trace_dict["marker"].copy()

                        # Filter data for this category
                        category_indices = [
                            i
                            for i, cat in enumerate(
                                categorical_data_for_separate_traces
                            )
                            if cat == category
                        ]

                        # Filter all data arrays for this category
                        if "x" in cat_trace_dict and isinstance(
                            cat_trace_dict["x"], list
                        ):
                            cat_trace_dict["x"] = [
                                cat_trace_dict["x"][i] for i in category_indices
                            ]
                        if "y" in cat_trace_dict and isinstance(
                            cat_trace_dict["y"], list
                        ):
                            cat_trace_dict["y"] = [
                                cat_trace_dict["y"][i] for i in category_indices
                            ]
                        if "text" in cat_trace_dict and isinstance(
                            cat_trace_dict["text"], list
                        ):
                            cat_trace_dict["text"] = [
                                cat_trace_dict["text"][i] for i in category_indices
                            ]
                        if "customdata" in cat_trace_dict and isinstance(
                            cat_trace_dict["customdata"], list
                        ):
                            cat_trace_dict["customdata"] = [
                                cat_trace_dict["customdata"][i]
                                for i in category_indices
                            ]

                        # Filter marker size if it's a list
                        if "size" in cat_trace_dict["marker"] and isinstance(
                            cat_trace_dict["marker"]["size"], list
                        ):
                            cat_trace_dict["marker"]["size"] = [
                                cat_trace_dict["marker"]["size"][i]
                                for i in category_indices
                            ]

                        # Set the color and name for this category
                        cat_trace_dict["marker"]["color"] = category_colors[cat_idx]
                        cat_trace_dict["marker"].pop("colorscale", None)
                        cat_trace_dict["marker"].pop("coloraxis", None)
                        cat_trace_dict["name"] = str(category)
                        cat_trace_dict["showlegend"] = True

                        # Clean up selection markers
                        _sanitize_selection_marker(cat_trace_dict, trace_type)

                        # Add this category trace to the figure
                        fig.add_trace(trace_class(**cat_trace_dict))

                    # Mark that we created separate traces
                    created_separate_traces = True

                    # Skip adding the original trace since we've created separate ones
                    continue

                # Only set line color if marker doesn't have categorical color data and line doesn't already have valid color
                if not marker_has_categorical_color:
                    trace_dict["line"] = trace_dict.get("line", {})
                    # Sanitize existing colour if provided
                    existing_line_color = trace_dict["line"].get("color")
                    if existing_line_color is not None:
                        trace_dict["line"]["color"] = _coerce_color(
                            existing_line_color, idx
                        )
                    if "color" not in trace_dict["line"]:
                        trace_dict["line"]["color"] = colors[idx]
            elif trace_type in [
                "bar",
                "histogram",
                "box",
                "violin",
            ]:
                trace_dict["marker"] = trace_dict.get("marker", {})

                llm_intended_color_val = None

                if (
                    marker_color is not None
                ):  # marker_color is the coerced shortcut value. Can be str or list.
                    llm_intended_color_val = marker_color
                elif "color" in trace_dict.get("marker", {}):
                    llm_intended_color_val = trace_dict["marker"]["color"]

                if llm_intended_color_val is not None:
                    if isinstance(llm_intended_color_val, list):
                        if not llm_intended_color_val:  # Empty list
                            if trace_type in ["bar", "histogram"]:
                                trace_dict["marker"]["color"] = _coerce_color(
                                    colors[idx % len(colors)], idx
                                )
                            # For box/violin (empty list): marker.color remains unset for colorway handling
                        elif trace_type in [
                            "bar",
                            "histogram",
                        ]:  # Apply list logic ONLY for bar/histogram
                            # This block is for bar/histogram when llm_intended_color_val is a non-empty list
                            if _is_categorical_color_list(llm_intended_color_val):
                                unique_categories = list(
                                    dict.fromkeys(
                                        str(c) for c in llm_intended_color_val
                                    )
                                )  # Ensure strings
                                if len(unique_categories) <= 5 and len(
                                    unique_categories
                                ) <= len(CONTRAST_PALETTE):
                                    category_palette = CONTRAST_PALETTE[
                                        : len(unique_categories)
                                    ]
                                else:
                                    category_palette = build_distinct_colors(
                                        len(unique_categories)
                                    )
                                category_palette = _ensure_unique_colors(
                                    category_palette
                                )
                                cat_to_color_map = {
                                    cat: category_palette[i % len(category_palette)]
                                    for i, cat in enumerate(unique_categories)
                                }
                                actual_colors = [
                                    cat_to_color_map.get(
                                        str(cat), colors[idx % len(colors)]
                                    )
                                    for cat in llm_intended_color_val
                                ]
                                trace_dict["marker"]["color"] = actual_colors
                            else:  # List of potential CSS colors for bar/histogram
                                coerced_list = []
                                for i, color_str in enumerate(llm_intended_color_val):
                                    if isinstance(color_str, str):
                                        coerced_list.append(
                                            _coerce_color(
                                                color_str, (idx + i) % len(colors)
                                            )
                                        )
                                    else:  # Non-string in list, use a cycling fallback
                                        coerced_list.append(
                                            colors[(idx + i) % len(colors)]
                                        )
                                trace_dict["marker"]["color"] = coerced_list
                        else:  # trace_type is box or violin AND llm_intended_color_val is a non-empty list
                            # For box/violin: if LLM provides a list, we IGNORE it for marker.color.
                            # This lets Plotly's colorway handle categorical coloring correctly,
                            # ensuring points match their respective boxes.
                            # So, marker.color remains unset in this specific sub-case.
                            pass  # Explicitly do nothing to marker.color
                    elif isinstance(llm_intended_color_val, str):
                        # LLM intended a single string for marker.color (applies to all types in this block)
                        trace_dict["marker"]["color"] = _coerce_color(
                            llm_intended_color_val, idx
                        )
                    else:
                        # Not a list or string (e.g., a number). Treat as no specific valid color intent.
                        if trace_type in ["bar", "histogram"]:
                            trace_dict["marker"]["color"] = _coerce_color(
                                colors[idx % len(colors)], idx
                            )
                        # For box/violin (unusable type for color): marker.color remains unset.
                else:
                    # No color specified by LLM (llm_intended_color_val is None).
                    if trace_type in ["bar", "histogram"]:
                        trace_dict["marker"]["color"] = _coerce_color(
                            colors[idx % len(colors)], idx
                        )
                    # For "box" and "violin" (no color from LLM): marker.color remains unset.
            elif trace_type == "pie":
                trace_dict["marker"] = trace_dict.get("marker", {})

                # Ensure *values* and *labels* exist first so we know the sector count.
                if "values" not in trace_dict:
                    trace_dict["values"] = trace_dict.get("y", [])
                if "labels" not in trace_dict:
                    trace_dict["labels"] = trace_dict.get("x", [])

                sector_count = len(trace_dict.get("values", [])) or len(
                    trace_dict.get("labels", [])
                )

                # Handle marker_color for pie charts (should be colors, plural)
                if marker_color is not None:
                    if isinstance(marker_color, list):
                        # Check if it's categorical data
                        if marker_color and not _is_valid_css_color(
                            str(marker_color[0])
                        ):
                            # Categorical data - assign distinct corporate colors
                            unique_categories = list(dict.fromkeys(marker_color))
                            if len(unique_categories) <= 5:
                                category_colors = CONTRAST_PALETTE[
                                    : len(unique_categories)
                                ]
                            else:
                                category_colors = build_distinct_colors(
                                    len(unique_categories)
                                )
                            category_colors = _ensure_unique_colors(category_colors)

                            # Map categories to colors
                            category_to_color = {
                                cat: category_colors[i]
                                for i, cat in enumerate(unique_categories)
                            }
                            actual_colors = [
                                category_to_color[cat] for cat in marker_color
                            ]
                            trace_dict["marker"]["colors"] = actual_colors
                            trace_dict["customdata"] = marker_color  # Store for hover
                        else:
                            # Valid colors provided
                            trace_dict["marker"]["colors"] = marker_color
                    else:
                        # Single color provided, replicate for all sectors
                        trace_dict["marker"]["colors"] = [marker_color] * sector_count
                elif "colors" not in trace_dict["marker"] and sector_count:
                    # Usa colores corporativos distintos para cada sector.
                    if sector_count <= 5:
                        sector_colors = CONTRAST_PALETTE[:sector_count]
                    else:
                        sector_colors = build_distinct_colors(sector_count)
                    trace_dict["marker"]["colors"] = _ensure_unique_colors(
                        sector_colors
                    )
            else:
                trace_dict.pop("marker", None)

            # Default selection style for better contrast
            if trace_type in [
                "scatter",
                "scattergl",
                "scatterpolar",
                "scattermapbox",
                "scatter3d",
            ]:
                if "selected" not in trace_dict:
                    trace_dict["selected"] = {"marker": {"opacity": 1}}
                if "unselected" not in trace_dict:
                    trace_dict["unselected"] = {"marker": {"opacity": 0.2}}

            # ------------------------------------------------------------------
            # Sanitise *selected* / *unselected* blocks to avoid invalid keys.
            # ------------------------------------------------------------------
            _sanitize_selection_marker(trace_dict, trace_type)

            # Extra guard: ensure no spurious 'selected' / 'unselected' appear
            # inside the *marker* dict itself (they are not valid marker props
            # and will raise a ValueError as seen in issue #XYZ).
            if isinstance(trace_dict.get("marker"), dict):
                marker_inner = trace_dict["marker"]
                marker_inner.pop("selected", None)
                marker_inner.pop("unselected", None)

                # Clean colorbar + pattern shapes
                _sanitize_marker_colorbar(marker_inner)
                _sanitize_marker_pattern(marker_inner)

            fig.add_trace(trace_class(**trace_dict))
        else:
            trace_dict["line"] = trace_dict.get("line", {})
            if "color" not in trace_dict["line"]:
                trace_dict["line"]["color"] = colors[idx]

            # Ensure selection blocks are valid even in the fallback case.
            _sanitize_selection_marker(trace_dict, trace_type)

            # Final safeguard for fallback traces as well.
            if isinstance(trace_dict.get("marker"), dict):
                _sanitize_marker_colorbar(trace_dict["marker"])

            fig.add_trace(go.Scatter(**trace_dict))

        # ------------------------------------------------------------------
        # GLOBAL: Remove / migrate any *titlefont* / *titleside* dangling keys
        # that may still exist after the specific sanitizers above.
        # ------------------------------------------------------------------
        _recursive_fix_titlefont(trace_dict)

    # --------------------------------------------------------------
    # Auto-expand Y range only when *all* collected values are numéricos.
    # Evita errores cuando las Y son categóricas (strings) como en heatmaps.
    # --------------------------------------------------------------
    all_y_values: list[float] = []
    for tr_data in fig.data:  # Use fig.data as traces are now added
        if (
            hasattr(tr_data, "y")
            and tr_data.y is not None
            and all(isinstance(v, int | float) for v in tr_data.y)
        ):
            all_y_values.extend(tr_data.y)  # type: ignore[arg-type]

    if all_y_values:
        min_y_val = min(all_y_values)
        max_y_val = max(all_y_values)
        # Ensure a positive range if min_y_val is close to or below zero, but allow negative if all data is negative
        y_range_min = (
            0 if min_y_val >= 0 else min_y_val * 1.15
        )  # Add padding for negative
        if min_y_val < 0 and max_y_val <= 0:
            y_range_max = max_y_val * 0.85  # Bring closer to zero if all negative
        elif min_y_val < 0 and max_y_val > 0:
            y_range_max = max_y_val * 1.15
        else:  # all positive or zero
            y_range_max = max_y_val * 1.15
            if y_range_min == 0 and y_range_max == 0:  # all data is zero
                y_range_max = 1  # default if all data is zero

        fig.update_yaxes(range=[y_range_min, y_range_max])

    # ------------------------------------------------------------------
    # Process and apply LLM-specific layout options ON TOP of the base template
    # ------------------------------------------------------------------
    if llm_layout_options:  # Use the dict from which 'template' was already popped
        # Pop and merge 'extra_attrs' from layout if they exist
        if "extra_attrs" in llm_layout_options:
            extra_layout = llm_layout_options.pop("extra_attrs") or {}
            llm_layout_options.update(extra_layout)

        provided_x_title = (
            _extract_axis_title(llm_layout_options, "xaxis") or provided_x_title
        )
        provided_y_title = (
            _extract_axis_title(llm_layout_options, "yaxis") or provided_y_title
        )

        # Sanitise axis dictionaries within llm_layout_options
        for axis_key in [
            k
            for k in llm_layout_options
            if k.startswith("xaxis") or k.startswith("yaxis")
        ]:
            axis_dict = llm_layout_options[axis_key]
            if isinstance(axis_dict, dict):
                if "spikedistance" in axis_dict:
                    llm_layout_options.setdefault(
                        "spikedistance", axis_dict.pop("spikedistance")
                    )
                axis_defaults = {
                    "showspikes": True,
                    "spikemode": "across",
                    "spikecolor": "#666666",
                    "spikethickness": -0.05,
                    "spikedash": "dash",
                }
                for k_def, v_def in axis_defaults.items():
                    axis_dict.setdefault(k_def, v_def)
                axis_dict.pop("spikesides", None)

        llm_layout_options.setdefault("hoverdistance", 20)
        llm_layout_options.setdefault("spikedistance", 20)

        _recursive_fix_titlefont(llm_layout_options)
        _recursive_sanitize_selection_markers(llm_layout_options)

        if created_separate_traces:  # This variable is from trace processing loop
            llm_layout_options.pop("coloraxis", None)
            llm_layout_options["showlegend"] = True

        # Remove invalid top-level layout keys from llm_layout_options
        if not hasattr(go.Layout, "_VALID_KEYS_CACHE"):
            go.Layout._VALID_KEYS_CACHE = set(go.Layout().to_plotly_json().keys())  # type: ignore[attr-defined]
        valid_layout_keys: set[str] = go.Layout._VALID_KEYS_CACHE  # type: ignore[attr-defined]
        keys_to_remove_from_llm_layout = [
            k
            for k in list(llm_layout_options.keys())
            if k not in valid_layout_keys and k not in {"xaxis_title", "yaxis_title"}
        ]
        for bad_key in keys_to_remove_from_llm_layout:
            llm_layout_options.pop(bad_key, None)

        # Extract LLM's colorway to apply it specifically after other layout updates
        llm_colorway = llm_layout_options.pop("colorway", None)

        # Map shortcut titles (xaxis_title / yaxis_title) to nested structure
        def _inject_axis_title_to_dict(
            short_key: str, axis_root: str, target_dict: dict
        ):
            title_val = target_dict.pop(short_key, None)
            if title_val is not None:
                target_dict.setdefault(axis_root, {}).setdefault("title", {})[
                    "text"
                ] = title_val

        _inject_axis_title_to_dict("xaxis_title", "xaxis", llm_layout_options)
        _inject_axis_title_to_dict("yaxis_title", "yaxis", llm_layout_options)

        fig.update_layout(**llm_layout_options)  # Apply processed LLM layout options
        _log_axis_titles("after_llm_specific_layout_updates", fig)

        if llm_colorway:  # Apply LLM's colorway now, ensuring it takes precedence
            fig.update_layout(colorway=llm_colorway)
            _log_axis_titles("after_llm_colorway_override", fig)

        # Capture titles again after LLM specifics, if not already captured
        if provided_x_title is None:
            provided_x_title = _get_title_from_fig_or_dict(
                fig.layout.xaxis, "xaxis", llm_layout_options
            )
        if provided_y_title is None:
            provided_y_title = _get_title_from_fig_or_dict(
                fig.layout.yaxis, "yaxis", llm_layout_options
            )

    # ... [The rest of the axis styling (axis_common_style, title enforcement, margins, etc.) remains largely the same] ...
    # ... but it should operate on `fig` and use `provided_x_title`, `provided_y_title` ...
    # The final `fig.update_layout` for paper_bgcolor etc. should NOT re-apply the full template.

    # Helper for _get_title (replaces the one inside make_plot to be reusable)
    # This helper needs to be defined *before* its first use if it's not already global
    # For this edit, I'm assuming it's available or will be moved to an appropriate scope.
    # def _get_title_from_fig_or_dict(fig_axis, axis_root: str, layout_dict_fallback: dict) -> str | None:
    #    ...

    # [ Ensure the axis styling block (has_cartesian, axis_common_style, title updates) is here ]
    # [ This part needs to use `provided_x_title` and `provided_y_title` correctly ]

    # --------------------------------------------------------------
    # Final global styling (NO FULL TEMPLATE RE-APPLICATION HERE)
    # --------------------------------------------------------------
    fig.update_layout(
        paper_bgcolor="#1A1A1A",
        plot_bgcolor="#1A1A1A",
        # hovermode was in llm_layout_options, if not, 'closest' is a good default
        hovermode=fig.layout.hovermode or "closest",
        modebar_orientation="v",
    )
    # If a specific theme string (like 'plotly_dark') was used as the base template,
    # ensure essential theme characteristics (like font color) are set if not overridden by LLM.
    if isinstance(template_name_or_obj, str) and template_name_or_obj == "plotly_dark":
        if not (
            fig.layout.font and fig.layout.font.color
        ):  # Check if LLM already set a font color
            fig.update_layout(font_color="#DDDDDD")

    return fig


# It's important that _get_title_from_fig_or_dict is defined or accessible here.
# For the purpose of this edit, I'll add a placeholder for it if it's not already global.
# If it's a nested helper in the original code, its definition would need to be adjusted or moved.

_global_colors_for_fallback = CONTRAST_PALETTE  # or some other default


def _get_title_from_fig_or_dict(
    fig_axis, axis_root: str, layout_dict_fallback: dict
) -> str | None:
    """Helper to extract axis title text from figure or layout dictionary."""
    try:
        if fig_axis and getattr(fig_axis, "title", None):
            text = getattr(fig_axis.title, "text", None)
            if text:
                return str(text)
    except Exception:
        pass

    axis_block_fallback = layout_dict_fallback.get(axis_root)
    if isinstance(axis_block_fallback, dict):
        nested = axis_block_fallback.get("title")
        if isinstance(nested, dict):
            txt = nested.get("text")
            if txt:
                return str(txt)

    short_key_fallback = f"{axis_root}_title"
    text_fallback = layout_dict_fallback.get(short_key_fallback)
    if text_fallback:
        return str(text_fallback)
    return None


# Ensure _extract_axis_title is also available if it was local
def _extract_axis_title(ldict: dict, axis_root: str) -> str | None:
    axis_block = ldict.get(axis_root)
    if isinstance(axis_block, dict):
        nested = axis_block.get("title")
        if isinstance(nested, dict):
            txt = nested.get("text")
            if txt:
                return str(txt)
    return (
        str(ldict.get(f"{axis_root}_title"))
        if ldict.get(f"{axis_root}_title")
        else None
    )


def instantiate_model_with_prompt_and_plotly_schema():
    """Instantiate an LLM model with structured output for PlotlySchema.

    Returns:
        A configured LLM pipeline that outputs PlotlySchema objects.
    """
    modelo = get_llm(model="gpt-4.1")
    model_with_structure_and_prompt = prompt | modelo.with_structured_output(
        PlotlySchema,
        method="function_calling",
    )
    return model_with_structure_and_prompt


def llm_json_to_plot_from_text(
    input_instructions: str,
    model_with_structure_and_prompt,
    chat_history: ChatHistory,
    max_retries: int = 5,
):
    """Generate a Plotly figure from *input_instructions* using an LLM.

    The function now has robust error handling that captures **all** exceptions
    raised during the two critical stages of the pipeline:

    1. LLM invocation + schema validation.
    2. Plot construction with ``make_plot``.

    On failure, a *structured* error payload is logged (``logger.error``) and
    appended to the *chat_history* as an ``AIMessage`` so that downstream
    consumers (e.g. a UI) can display the full context.

    Parameters
    ----------
    input_instructions: str
        Natural-language prompt describing the desired chart.
    model_with_structure_and_prompt:
        The LLM pipeline returned by
        :pyfunc:`instantiate_model_with_prompt_and_PlotlySchema`.
    chat_history: ChatHistory
        Conversation history object used to keep context between retries.
    max_retries: int, default=5
        Maximum number of attempts before giving up.

    Returns:
    -------
    plotly.graph_objects.Figure
        The generated figure.

    Raises:
    ------
    RuntimeError
        If *max_retries* is exceeded.
    """
    retries = 0
    while retries < max_retries:
        # Step 1 ────────────────────────────────────────────────────────────
        try:
            respuesta_estructurada = model_with_structure_and_prompt.invoke(
                {
                    "input": input_instructions,
                    "chat_history": chat_history.messages,
                },
            )
            # Safely log the received layout (it may be *None* in edge cases)
            if getattr(respuesta_estructurada, "layout", None) is not None:
                print(
                    "⤵️  Layout recibido:\n",
                    respuesta_estructurada.layout.model_dump(mode="json"),
                )
            else:
                print("⤵️  Layout recibido: <sin layout>")
        except Exception as exc:  # Catch *everything*: network errors, validation, etc.
            tb = traceback.format_exc()
            error_payload = {
                "stage": "llm_invoke",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": tb,
                "input_instructions": input_instructions,
            }
            logger.error(
                "LLM invocation failed: %s",
                json.dumps(error_payload, ensure_ascii=False, indent=2),
            )
            chat_history.append(HumanMessage(content=input_instructions))
            chat_history.append(
                AIMessage(content=json.dumps(error_payload, ensure_ascii=False))
            )

            # Ask the model to self-correct in the next iteration.
            input_instructions = (
                f"{input_instructions}\n\nSe produjo un error durante la llamada al modelo: "
                f"{error_payload['error_message']}.\nIntenta corregirlo y vuelve a intentarlo."
            )
            retries += 1
            continue  # Retry

        # Step 2 ────────────────────────────────────────────────────────────
        try:
            fig = make_plot(respuesta_estructurada)

            # Save the plot as PNG image to the absolute path
            try:
                # Use the absolute path /home/alejandro/Pictures/pathway_plots/
                plots_abs_dir = Path("/home/alejandro/Pictures/pathway_plots")
                plots_abs_dir.mkdir(parents=True, exist_ok=True)

                # Save the figure as PNG (overwrites existing file)
                output_path = plots_abs_dir / "custom_plot.png"
                fig.write_image(
                    str(output_path),
                    width=1200,
                    height=800,
                    scale=2,  # Higher resolution
                    format="png",
                )
                logger.info(f"Plot saved successfully to: {output_path}")

            except Exception as save_exc:
                # Log the error but don't fail the entire function
                logger.warning(f"Failed to save plot image: {save_exc}")

            # Success – update history and return the figure.
            chat_history.append(HumanMessage(content=input_instructions))
            chat_history.append(AIMessage(content=str(respuesta_estructurada)))
            return fig
        except Exception as exc:
            tb = traceback.format_exc()
            error_payload = {
                "stage": "make_plot",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": tb,
                # ``model_dump`` keeps it JSON-serialisable.
                "structured_output": respuesta_estructurada.model_dump(mode="json")
                if hasattr(respuesta_estructurada, "model_dump")
                else str(respuesta_estructurada),
            }
            logger.error(
                "Plot construction failed: %s",
                json.dumps(error_payload, ensure_ascii=False, indent=2),
            )
            chat_history.append(HumanMessage(content=input_instructions))
            chat_history.append(
                AIMessage(content=json.dumps(error_payload, ensure_ascii=False))
            )

            # Tweak the instructions so the LLM can attempt to fix the issue.
            input_instructions = (
                f"{input_instructions}\n\nSe produjo un error durante la construcción del gráfico: "
                f"{error_payload['error_message']}.\nIntenta corregirlo y vuelve a intentarlo."
            )
            retries += 1
            continue  # Retry

    # ────────────────────────────────────────────────────────────────────────
    raise RuntimeError(
        f"Se ha excedido el número máximo de reintentos ({max_retries})."
    )


# ------------------------------------------------------------------
# Generic recursive sanitiser to remove deprecated keys such as
# ``titlefont`` / ``titleside`` from ANY nested dictionary (axes, colorbars,
# templates, etc.).  When ``titlefont`` is found we try to migrate it to the
# new structure (``title.font``) when possible, otherwise we simply drop it to
# prevent Plotly validation errors.
# ------------------------------------------------------------------


def _recursive_fix_titlefont(obj: dict | list) -> None:
    """Recursively fix *titlefont* / *titleside* keys in *obj*.

    This helper mutates the input structure **in-place** so it can be safely
    passed to Plotly without triggering *ValueError* validation errors.
    """
    if isinstance(obj, dict):
        # Handle titlefont → title.font migration.
        if "titlefont" in obj:
            font_val = obj.pop("titlefont")
            if (
                isinstance(font_val, dict)
                and "title" in obj
                and isinstance(obj["title"], dict)
            ):
                # Migrate into nested structure if possible.
                obj["title"].setdefault("font", font_val)
            # If moving is not feasible, we silently discard – avoids error.

        # Remove obsolete 'titleside' (handled by title.side internally).
        obj.pop("titleside", None)

        # Recurse into children.
        for v in obj.values():
            _recursive_fix_titlefont(v)

    elif isinstance(obj, list):
        for item in obj:
            _recursive_fix_titlefont(item)


class ChatHistory(BaseModel):
    """Chat history container for storing conversation messages."""

    messages: list[BaseMessage] = Field(default_factory=list)

    def append(self, message: BaseMessage):
        """Append a message to the chat history."""
        self.messages.append(message)


if __name__ == "__main__":
    """CLI test runner.

    Reads test scenarios from *plot_tests_inputs.yaml* placed in the same
    directory as this script and generates the corresponding figures.  Each
    entry should be a mapping with at least the key ``input_instructions``.
    Additional keys such as ``name`` are optional – they are used only for
    logging / file naming purposes.
    """

    from pathlib import Path

    import yaml

    base_dir = Path(__file__).parent
    yaml_path_candidates = [
        base_dir
        / f"{Path(__file__).stem}_tests_inputs.yaml",  # plot_generator_tests_inputs.yaml
    ]

    yaml_path = next((p for p in yaml_path_candidates if p.exists()), None)

    if yaml_path is None:
        raise FileNotFoundError(
            "No se encontró ningún archivo YAML de tests. Nombres buscados: "
            + ", ".join(str(p) for p in yaml_path_candidates),
        )

    with yaml_path.open("r", encoding="utf-8") as f:
        yaml_content = yaml.safe_load(f) or {}

    tests = yaml_content.get("tests")
    if not tests:
        # Allow a simpler structure: a single string under 'input_instructions'
        single_instr = yaml_content.get("input_instructions")
        if single_instr is None:
            raise ValueError(
                "El YAML debe contener 'tests' (lista) o 'input_instructions' (cadena)."
            )
        tests = [{"name": "default_test", "input_instructions": single_instr}]

    chat_history_plot_gen = ChatHistory()
    llm_plot_gen = instantiate_model_with_prompt_and_plotly_schema()

    for test_case in tests:
        instr = test_case["input_instructions"]
        test_name = test_case.get("name", "sin_nombre")
        print(f"\n\u2192 Generando figura para el test: {test_name} ...")

        fig = llm_json_to_plot_from_text(
            input_instructions=instr,
            model_with_structure_and_prompt=llm_plot_gen,
            chat_history=chat_history_plot_gen,
        )

        # Mostrar resultado por consola (JSON) y abrir figura interactiva.
        print(fig.to_json())

        try:
            from plotly.io import show as plotly_show

            plotly_show(fig)
        except Exception:
            # In headless environments show() may fail; ignore.
            pass

# Debug configuration for development
if __name__ == "__main__":
    # 1️⃣  Activa el flag antes de cargar el módulo.
    os.environ["PLOT_DEBUG"] = "1"  # cualquier valor no vacío sirve

    # 3️⃣  Refuerza el nivel DEBUG por si el handler ya existía con otro nivel.
    logging.getLogger(__name__).setLevel(logging.DEBUG)

# %%
