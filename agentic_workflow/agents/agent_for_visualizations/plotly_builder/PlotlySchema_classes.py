# %%
from typing import Any, Optional

import pandas as pd
import plotly.express as px  # Para paletas de colores y temas
import plotly.graph_objects as go
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator


load_dotenv(override=True)


class Trace(BaseModel):
    type: str = Field(
        ...,
        description="Tipo de traza (por ejemplo, 'scatter', 'bar', 'pie', 'histogram', etc.)",
    )
    x: list[Any] | None = Field(
        None,
        description="Puntos de datos del eje X (pueden ser fechas (dates) para series temporales)",
    )
    y: list[Any] | None = Field(None, description="Puntos de datos del eje Y")
    mode: str | None = Field(
        None,
        description="Modo de dibujo para trazas scatter (por ejemplo, 'lines', 'markers', 'lines+markers')",
    )
    name: str | None = Field(
        None,
        description="Nombre de la traza para la leyenda",
    )
    text: list[Any] | None = Field(
        None,
        description="Elementos de texto asociados a cada punto de datos",
    )
    hovertext: list[str] | None = Field(
        None,
        description="Elementos de texto al pasar el ratón",
    )
    marker: dict | None = Field(
        None,
        description="Opciones de estilo del marcador (por ejemplo, 'size', 'color', 'symbol', 'line').",
    )
    line: dict | None = Field(
        None,
        description="Opciones de estilo de línea (por ejemplo, 'width', 'color', 'dash').",
    )
    fill: str | None = Field(
        None,
        description="Opciones de relleno de área (por ejemplo, 'tozeroy', 'tonexty')",
    )
    opacity: float | None = Field(
        None,
        description="Opacidad de la traza entre 0 y 1",
    )
    orientation: str | None = Field(
        None,
        description="Orientación para gráficos de barras ('v' o 'h')",
    )
    labels: list[Any] | None = Field(
        None,
        description="Etiquetas para sectores del gráfico de pastel",
    )
    values: list[Any] | None = Field(
        None,
        description="Valores para sectores del gráfico de pastel",
    )
    textposition: str | None = Field(
        None,
        description="Posición de las etiquetas de texto (por ejemplo, 'inside', 'outside')",
    )
    hole: float | None = Field(
        None,
        description="Tamaño del agujero en un gráfico de dona (0 a 1)",
    )
    direction: str | None = Field(
        None,
        description="Dirección de los sectores del gráfico de pastel ('clockwise', 'counterclockwise')",
    )
    sort: bool | None = Field(
        None,
        description="Si ordenar los sectores del gráfico de pastel",
    )
    customdata: list[Any] | None = Field(
        None,
        description="Datos adicionales para eventos de pasar el ratón y clic",
    )
    error_x: dict | None = Field(
        None,
        description="Barras de error para el eje x",
    )
    error_y: dict | None = Field(
        None,
        description="Barras de error para el eje y",
    )
    visible: bool | None = Field(None, description="Visibilidad de la traza")
    showlegend: bool | None = Field(
        None,
        description="Si mostrar la traza en la leyenda",
    )
    legendgroup: str | None = Field(
        None,
        description="Nombre del grupo para la leyenda",
    )
    offsetgroup: str | None = Field(
        None,
        description="Nombre del grupo para barras agrupadas",
    )
    hoverinfo: str | None = Field(
        None,
        description="Información que se muestra al pasar el ratón",
    )
    texttemplate: str | None = Field(
        None,
        description="Plantilla para el texto mostrado en las etiquetas",
    )
    width: float | None = Field(
        None,
        description="Ancho de las barras en gráficos de barras",
    )
    # Permite atributos arbitrarios avanzados.
    extra_attrs: dict[str, Any] | None = Field(
        None,
        description="Otros atributos avanzados de Plotly no listados explícitamente (se incluirán tal cual).",
    )
    # Añade más campos según sea necesario para diferentes tipos de traza

    marker_color: str | list[Any] | None = Field(
        None,
        description="Color o lista de colores para los marcadores (equivalente a marker.color).",
    )
    marker_line_color: str | None = Field(
        None,
        description="Color del borde de los marcadores (marker.line.color).",
    )
    marker_line_width: float | None = Field(
        None,
        description="Ancho del borde de los marcadores (marker.line.width).",
    )
    marker_opacity: float | None = Field(
        None,
        description="Opacidad individual de los marcadores (marker.opacity).",
    )
    line_width: float | None = Field(
        None,
        description="Ancho de la línea para trazas lineales (line.width).",
    )
    line_dash: str | None = Field(
        None,
        description="Estilo de línea (solid, dash, dot, etc.) (line.dash).",
    )
    fillcolor: str | None = Field(
        None,
        description="Color de relleno cuando se usa 'fill' (fillcolor).",
    )

    marker_size: int | list[Any] | None = Field(
        None,
        description="Tamaño o lista de tamaño de marcadores (equivalente a marker.size).",
    )
    marker_symbol: str | list[Any] | None = Field(
        None,
        description="Símbolo o lista de símbolos para marcadores (marker.symbol).",
    )

    # Bubble charts – tamaño de la burbuja (equivalente a marker.size)
    size: list[Any] | None = Field(
        None,
        description="Lista numérica que indica el tamaño relativo de cada punto en gráficos tipo bubble.",
    )

    # Propiedades específicas de gráficos de pastel / dona.
    pull: float | list[Any] | None = Field(
        None,
        description="Desplazamiento radial de sectores en un pie chart (float o lista).",
    )

    @field_validator("text")
    def convert_text_to_string(cls, v):
        if v is not None:
            return [str(item) for item in v]
        return v

    # --- Interactividad y hover ---
    hovertemplate: str | None = Field(
        None,
        description="Plantilla personalizada de hover (hovertemplate).",
    )
    hoverlabel: dict | None = Field(
        None,
        description="Configuración de etiquetas de hover (hoverlabel).",
    )
    hoveron: str | None = Field(
        None,
        description="Determina cuáles elementos de la traza disparan hover ('points', 'fills', etc.).",
    )

    class Config:
        # Permite campos extra no declarados para que el LLM pueda añadir
        # propiedades avanzadas de Plotly sin restricciones.  De este modo,
        # **cualquier** atributo adicional se preservará y se enviará a la fase
        # de construcción del gráfico.
        extra = "allow"


class Layout(BaseModel):
    """Schema for the Plotly *layout* object.

    The LLM occasionally returns the *title* or *template* fields as full
    dictionaries (e.g. ``{"text": "Mi gráfico"}`` or an entire Plotly template
    definition) instead of simple strings.  To avoid a hard validation error
    when that happens we accept both ``str`` and ``dict`` for these two keys.
    """

    # Accept both ``str`` and ``dict`` so that a complete Plotly *Title* or
    # *Template* specification returned by the model does not raise a
    # ``ValidationError``.
    title: str | dict | None = Field(
        None,
        description="Título del gráfico o definición completa del título.",
    )
    xaxis: Optional["_Axis"] = Field(
        None,
        description="Configuración detallada del eje X (incluye title.text).",
    )
    yaxis: Optional["_Axis"] = Field(
        None,
        description="Configuración detallada del eje Y (incluye title.text).",
    )
    legend: dict | None = Field(
        None,
        description="Configuración de la leyenda",
    )
    template: str | dict | None = Field(
        None,
        description=(
            "Plantilla de tema del gráfico.  Puede ser una cadena que haga "
            "referencia a un *built-in* de Plotly (e.g. 'plotly_dark') o un "
            "diccionario/objeto con la definición completa de la plantilla."
        ),
    )
    margin: dict | None = Field(None, description="Márgenes del gráfico")
    annotations: list[dict] | None = Field(
        None,
        description="Lista de anotaciones para agregar al gráfico",
    )
    shapes: list[dict] | None = Field(
        None,
        description="Lista de formas para agregar al gráfico",
    )
    bargap: float | None = Field(
        None,
        description="Espacio entre barras en gráficos de barras",
    )
    bargroupgap: float | None = Field(
        None,
        description="Espacio entre grupos de barras en gráficos de barras",
    )
    barmode: str | None = Field(
        None,
        description="Modo de gráfico de barras ('stack', 'group', 'overlay', 'relative')",
    )
    hovermode: str | None = Field(
        None,
        description="Modo de interacción al pasar el ratón",
    )
    polar: dict | None = Field(
        None,
        description="Configuración para gráficos polares",
    )
    radialaxis: dict | None = Field(
        None,
        description="Configuración del eje radial en gráficos polares",
    )
    angularaxis: dict | None = Field(
        None,
        description="Configuración del eje angular en gráficos polares",
    )
    showlegend: bool | None = Field(None, description="Si mostrar la leyenda")
    font: dict | None = Field(
        None,
        description="Configuración de fuente para el texto del gráfico",
    )
    paper_bgcolor: str | None = Field(
        None,
        description="Color de fondo del papel",
    )
    plot_bgcolor: str | None = Field(
        None,
        description="Color de fondo del área del gráfico",
    )
    width: int | None = Field(None, description="Ancho del gráfico en píxeles")
    height: int | None = Field(
        None,
        description="Altura del gráfico en píxeles",
    )
    autosize: bool | None = Field(
        None,
        description="Si el gráfico debe ajustarse automáticamente al tamaño del contenedor",
    )
    title_font: dict | None = Field(
        None,
        description="Configuración de fuente para el título",
    )
    xaxis_title_font: dict | None = Field(
        None,
        description="Configuración de fuente para el título del eje X",
    )
    yaxis_title_font: dict | None = Field(
        None,
        description="Configuración de fuente para el título del eje Y",
    )
    xaxis_tickfont: dict | None = Field(
        None,
        description="Configuración de fuente para las etiquetas del eje X",
    )
    yaxis_tickfont: dict | None = Field(
        None,
        description="Configuración de fuente para las etiquetas del eje Y",
    )
    legend_font: dict | None = Field(
        None,
        description="Configuración de fuente para la leyenda",
    )
    # Permite atributos arbitrarios avanzados.
    extra_attrs: dict[str, Any] | None = Field(
        None,
        description="Otros atributos avanzados de Plotly para el layout que no estén listados explícitamente.",
    )
    # Añade más campos según sea necesario

    uniformtext: dict | None = Field(
        None,
        description="Configuración global del texto (ej. mode, minsize) para el gráfico.",
    )
    grid: dict | None = Field(
        None,
        description="Configuración de la cuadrícula de subplots (layout.grid). Permite controlar filas, columnas, patrones, espaciado, etc.",
    )
    xaxis2: dict | None = Field(
        None,
        description="Configuración adicional para un segundo eje X (layout.xaxis2)",
    )
    yaxis2: dict | None = Field(
        None,
        description="Configuración adicional para un segundo eje Y (layout.yaxis2)",
    )

    # Control fino de etiquetas y ticks de los ejes
    xaxis_title: str | None = Field(
        None,
        description="OBLIGATORIO: Título del eje X (alternativa corta a xaxis.title.text).",
    )
    yaxis_title: str | None = Field(
        None,
        description="OBLIGATORIO: Título del eje Y (alternativa corta a yaxis.title.text).",
    )
    xaxis_tickformat: str | None = Field(
        None,
        description="Formato de ticks del eje X (ej. '%Y-%m').",
    )
    yaxis_tickformat: str | None = Field(
        None,
        description="Formato de ticks del eje Y.",
    )
    xaxis_tickangle: int | None = Field(
        None,
        description="Ángulo de rotación de etiquetas de ticks en X.",
    )
    yaxis_tickangle: int | None = Field(
        None,
        description="Ángulo de rotación de etiquetas de ticks en Y.",
    )

    colorway: list[str] | None = Field(
        None,
        description="Secuencia de colores por defecto para trazas (layout.colorway).",
    )
    coloraxis: dict | None = Field(
        None,
        description="Mapa de color compartido entre trazas (layout.coloraxis).",
    )

    hoverdistance: int | None = Field(
        None,
        description="Distancia en píxeles para activar eventos de hover (layout.hoverdistance).",
    )
    spikedistance: int | None = Field(
        None,
        description="Distancia en píxeles para spike lines (layout.spikedistance).",
    )
    dragmode: str | None = Field(
        None,
        description="Modo de arrastre/interacción (zoom, pan, select, lasso, drawline, drawrect, etc.).",
    )
    selectdirection: str | None = Field(
        None,
        description="Dirección de selección (h, v, d) cuando dragmode es 'select'.",
    )
    newshape: dict | None = Field(
        None,
        description="Opciones por defecto para nuevas formas dibujadas en modo 'draw*'.",
    )

    # ─────────────────────────────────────────────────────────────
    # Histograma & distribución
    # ─────────────────────────────────────────────────────────────
    nbinsx: int | None = Field(
        None,
        description="Número de bins en X para histogramas.",
    )
    nbinsy: int | None = Field(
        None,
        description="Número de bins en Y para histogramas 2D.",
    )
    histnorm: str | None = Field(
        None,
        description="Normalización del histograma (e.g. 'probability', 'probability density').",
    )
    histfunc: str | None = Field(
        None,
        description="Función aplicada a los datos de histograma (e.g. 'count', 'sum', 'avg').",
    )

    # ─────────────────────────────────────────────────────────────
    # Heatmap
    # ─────────────────────────────────────────────────────────────
    z: list[list[Any]] | None = Field(
        None,
        description="Matriz Z para heatmaps o surface charts.",
    )
    zmid: float | None = Field(
        None,
        description="Valor central para escalas divergentes en heatmap.",
    )
    colorscale: str | list[Any] | None = Field(
        None,
        description="Escala de colores (nombre o lista de pares).",
    )
    xgap: float | None = Field(
        None,
        description="Separación en píxeles entre celdas (heatmap.xgap).",
    )
    ygap: float | None = Field(
        None,
        description="Separación en píxeles entre filas (heatmap.ygap).",
    )

    # ─────────────────────────────────────────────────────────────
    # Patrón de barras (marker.pattern) y otras opciones avanzadas
    # ─────────────────────────────────────────────────────────────
    marker_pattern: dict | None = Field(
        None,
        description="Configuración de patrón para barras (marker.pattern.*).",
    )

    class Config:
        # Permite campos extra no declarados para que el LLM pueda añadir
        # propiedades avanzadas de Plotly sin restricciones.  De este modo,
        # **cualquier** atributo adicional se preservará y se enviará a la fase
        # de construcción del gráfico.
        extra = "allow"


class PlotlySchema(BaseModel):
    data: list[Trace] = Field(..., description="Lista de trazas de datos")
    layout: Layout | None = Field(None, description="Configuración del diseño")

    class Config:
        # Permite campos extra no declarados para que el LLM pueda añadir
        # propiedades avanzadas de Plotly sin restricciones.  De este modo,
        # **cualquier** atributo adicional se preservará y se enviará a la fase
        # de construcción del gráfico.
        extra = "allow"


# Inicializar el modelo de lenguaje
def generar_grafico(model, prompt):
    # Vincular el esquema al modelo

    instrucciones = f"""
    Recuerda poner nombres de las variables y título.

    Los datos pueden ser no-temporales o temporales.
    A continuación, solo y exclusivamente si los datos son NO-TEMPORALES:
        - Si es una sola/unica oficina, haga una barra para cada variable con diferentes colores y NO MUESTRE leyenda y no muestre (esconda) eje Y (axis-y).
        - Cuando son varias oficinas, utilizar barras agrupadas (barmode='group') y no muestre (esconda) eje Y (axis-y).
        - Siempre: Mostrar los valores de las variables.
        - Siempre: Las etiquetas (valores de las variables) posicionarlas afuera, 'outside'.

    A continuación, solo y exclusivamente si los datos son series temporales (time series):
        Utiliza los datos proporcionados para generar un gráfico de líneas que muestre las atenciones diarias a lo largo del tiempo para cada oficina. Asegúrate de:

        - Usar la fecha como eje X (asegúrate de que las fechas estén en el formato adecuado).
        - Diferenciar cada oficina con colores distintos.
        - Si hay más de una oficina, incluir una leyenda para identificar cada línea.
        - Asegurarte de que el eje X esté ordenado cronológicamente.
    Ahora aquí tienes los datos:
        {prompt}
    """

    model_with_structure = model.with_structured_output(PlotlySchema)
    structured_output = model_with_structure.invoke(instrucciones)

    # Aplicar el tema especificado por el usuario o por defecto 'plotly_dark'
    theme = (
        structured_output.layout.template
        if structured_output.layout and structured_output.layout.template
        else "plotly_dark"
    )

    fig = go.Figure()

    # Definir colores personalizados si se proporcionan
    custom_colors = []
    for trace in structured_output.data:
        if trace.marker and "colors" in trace.marker:
            custom_colors.extend(trace.marker["colors"])

    # Usar colores personalizados o paleta predeterminada
    if custom_colors:
        colors = custom_colors
    else:
        color_palette = px.colors.qualitative.Plotly
        num_traces = len(structured_output.data)
        if num_traces > len(color_palette):
            colors = color_palette * ((num_traces // len(color_palette)) + 1)
        else:
            colors = color_palette

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

    # Añadir trazas con colores de la paleta o colores definidos por el usuario
    for idx, trace in enumerate(structured_output.data):
        trace_dict = trace.model_dump(exclude_unset=True)
        trace_type = trace_dict.pop("type", None)  # Eliminar la clave 'type'

        # Intentar convertir 'x' a datetime si es apropiado
        if "x" in trace_dict:
            try:
                converted_x = pd.to_datetime(trace_dict["x"])
                # Comprobar si la conversión tuvo éxito sin generar valores NaT
                if not converted_x.isnull().all():
                    trace_dict["x"] = converted_x
            except (ValueError, TypeError):
                # Si falla la conversión, dejar 'x' como está
                pass

        if trace_type in trace_type_mapping:
            trace_class = trace_type_mapping[trace_type]

            # Asignar color a la traza apropiadamente
            if trace_type in ["scatter", "scatter3d", "line", "area"]:
                trace_dict["line"] = trace_dict.get("line", {})
                if "color" not in trace_dict["line"]:
                    trace_dict["line"]["color"] = colors[idx]
            elif trace_type in [
                "bar",
                "histogram",
                "box",
                "violin",
                "heatmap",
            ]:
                trace_dict["marker"] = trace_dict.get("marker", {})
                if "color" not in trace_dict["marker"]:
                    trace_dict["marker"]["color"] = colors[idx]
            elif trace_type == "pie":
                trace_dict["marker"] = trace_dict.get("marker", {})
                if "colors" not in trace_dict["marker"]:
                    trace_dict["marker"]["colors"] = colors[
                        : len(trace_dict.get("values", []))
                    ]
                if "values" not in trace_dict:
                    trace_dict["values"] = trace_dict.get("y", [])
                if "labels" not in trace_dict:
                    trace_dict["labels"] = trace_dict.get("x", [])
            else:
                pass

            fig.add_trace(trace_class(**trace_dict))
        else:
            trace_dict["line"] = trace_dict.get("line", {})
            if "color" not in trace_dict["line"]:
                trace_dict["line"]["color"] = colors[idx]
            fig.add_trace(go.Scatter(**trace_dict))

    # Ajustar el rango del eje Y si es necesario
    all_y_values = []
    for trace in structured_output.data:
        if trace.y:
            all_y_values.extend(trace.y)

    if all_y_values:
        max_y = max(all_y_values)
        adjusted_max_y = max_y * 1.15
        fig.update_yaxes(range=[0, adjusted_max_y])

    # Actualizar el diseño con cualquier opción de personalización adicional
    if structured_output.layout:
        fig.update_layout(
            **structured_output.layout.model_dump(exclude_unset=True),
        )

    # Asegurar que las líneas de cuadrícula se muestren por defecto y aplicar el tema
    fig.update_layout(
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        template=theme,
    )
    return fig


# def get_env_variable(name):
#     """Obtiene una variable de entorno o detiene la aplicación si no se encuentra."""
#     value = os.environ.get(name)
#     if not value:
#         st.error(f"La variable de entorno '{name}' no está definida.")
#         st.stop()
#     return value


# # Obtener las variables de entorno necesarias
# AZURE_API_KEY = get_env_variable("AZURE_API_KEY")
# AZURE_ENDPOINT = get_env_variable("AZURE_ENDPOINT")
# AZURE_API_VERSION = get_env_variable("AZURE_API_VERSION")


# def initialize_model():
#     """Inicializa el modelo AzureChatOpenAI."""
#     try:
#         model_instance = AzureChatOpenAI(
#             azure_deployment="gpt-4o",
#             api_version=AZURE_API_VERSION,
#             temperature=0,
#             max_tokens=None,
#             timeout=None,
#             max_retries=5,
#             api_key=AZURE_API_KEY,
#             azure_endpoint=AZURE_ENDPOINT,
#             streaming=True,
#         )
#         return model_instance
#     except Exception as e:
#         st.error(f"Error al inicializar el modelo AzureChatOpenAI: {e}")
#         logging.error(f"Error al inicializar el modelo AzureChatOpenAI: {e}")
#         st.stop()


# generador_de_gráfico = initialize_model()

# # visualizador = initialize_model()
# # generador_de_gráfico = initialize_model()

# llm_chat = """
# Aquí tienes el nivel de servicio (SLA) para las oficinas '005 - Los Leones', '064 - Tobalaba' y '145 - La Florida' con un tiempo de espera máximo de 900 segundos (15 minutos):

# | Oficina          | Nivel de Servicio (%) | Tiempo de Espera Promedio (minutos) |
# |------------------|-----------------------|-------------------------------------|
# | 005 - Los Leones | 95.54                 | 5.67                                |
# | 064 - Tobalaba   | 75.26                 | 11.56                               |


# # Si necesitas más detalles o información adicional, por favor házmelo saber.
# # """

# llm_chat = """
# Para la oficina '005 - Los Leones' el día 30/09/2024:
# Nivel de Servicio (SLA): 95.54% (clientes atendidos en menos de 15 minutos).
# Tiempo Medio de Espera Global: 5.67 minutos.
# Total Atenciones: 202.
# Total Abandonos: 2 (0.99% de abandono).
# Si necesitas más detalles específicos, por favor indícalo.
# """


# llm_chat = """
# Aquí tienes el resumen de las atenciones diarias del último mes para las oficinas '005 - Los Leones' y '145 - La Florida':

# | Oficina          | Fecha       | Atenciones Totales |
# |------------------|-------------|--------------------|
# | 005 - Los Leones | 16/09/2024  | 142                |
# | 005 - Los Leones | 17/09/2024  | 127                |
# | 005 - Los Leones | 23/09/2024  | 130                |
# | 005 - Los Leones | 24/09/2024  | 148                |
# | 005 - Los Leones | 25/09/2024  | 99                 |
# | 005 - Los Leones | 26/09/2024  | 133                |
# | 005 - Los Leones | 27/09/2024  | 161                |
# | 005 - Los Leones | 30/09/2024  | 202                |
# | 005 - Los Leones | 01/10/2024  | 191                |
# | 005 - Los Leones | 02/10/2024  | 178                |
# | 005 - Los Leones | 03/10/2024  | 150                |
# | 005 - Los Leones | 04/10/2024  | 182                |
# | 005 - Los Leones | 07/10/2024  | 171                |
# | 005 - Los Leones | 08/10/2024  | 142                |
# | 005 - Los Leones | 09/10/2024  | 135                |
# | 005 - Los Leones | 10/10/2024  | 140                |
# | 005 - Los Leones | 11/10/2024  | 136                |
# | 005 - Los Leones | 14/10/2024  | 151                |
# | 145 - La Florida | 16/09/2024  | 435                |
# | 145 - La Florida | 17/09/2024  | 340                |
# | 145 - La Florida | 23/09/2024  | 379                |
# | 145 - La Florida | 24/09/2024  | 320                |
# | 145 - La Florida | 25/09/2024  | 296                |
# | 145 - La Florida | 26/09/2024  | 263                |
# | 145 - La Florida | 27/09/2024  | 342                |
# | 145 - La Florida | 30/09/2024  | 365                |
# | 145 - La Florida | 01/10/2024  | 299                |
# | 145 - La Florida | 02/10/2024  | 268                |
# | 145 - La Florida | 03/10/2024  | 280                |
# | 145 - La Florida | 04/10/2024  | 293                |
# | 145 - La Florida | 07/10/2024  | 291                |
# | 145 - La Florida | 08/10/2024  | 244                |
# | 145 - La Florida | 09/10/2024  | 245                |
# | 145 - La Florida | 10/10/2024  | 231                |
# | 145 - La Florida | 11/10/2024  | 279                |
# | 145 - La Florida | 14/10/2024  | 297                |

# En el último mes, la oficina '005 - Los Leones' tuvo un promedio de 151 atenciones diarias, mientras que la oficina '145 - La Florida' tuvo un promedio de 303.72 atenciones diarias.
# """


# fig = generar_grafico(model=generador_de_gráfico, prompt=llm_chat)
# fig.show()
# %%


# %%
# def main():
#     st.title("Generador de Gráficos Interactivos")

#     # Obtener el prompt del usuario
#     prompt = st.text_area("Introduce el prompt para generar el gráfico:")

#     if st.button("Generar Gráfico"):
#         if prompt:
#             model = AzureChatOpenAI(
#                 azure_deployment="gpt-4o",
#                 api_version="2024-09-01-preview",
#                 temperature=0,
#                 max_tokens=None,
#                 timeout=None,
#                 max_retries=5,
#                 api_key=os.environ.get("AZURE_API_KEY"),
#                 azure_endpoint=os.environ.get("AZURE_ENDPOINT"),
#                 streaming=True,
#             )
#             fig = generar_grafico(model=model, prompt=prompt)

#             # Mostrar el gráfico en la aplicación Streamlit
#             st.plotly_chart(fig)
#         else:
#             st.warning(
#                 "Por favor, introduce un prompt para generar el gráfico.",
#             )


# if __name__ == "__main__":
#     main()


# %%
# model = AzureChatOpenAI(
#     azure_deployment="gpt-4o",
#     api_version="2024-09-01-preview",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=5,
#     api_key=os.environ.get("AZURE_API_KEY"),
#     azure_endpoint=os.environ.get("AZURE_ENDPOINT"),
#     streaming=True,
# )
# prompt = """
# Genera un gráfico interactivo que represente los siguientes datos:

# - SLA para Santiago: 85%
# - SLA para Rancagua: 35%

# Requisitos:
# - Utiliza un gráfico de pastel (pie chart).
# - Usa colores azul claro y verde para las secciones.
# - Coloca las etiquetas de porcentaje fuera de las secciones.
# - Añade un título al gráfico: 'SLA por Región'.
# - Establece el tamaño del agujero en 0.4 para crear un gráfico de dona (donut chart).
# """
# generar_grafico(model=model, prompt=prompt)

# """
# Impletement this code as a streamlit app, were the prompt should be input by the user, and the
# generated fig should be displayed
# might be something like
# # Display the chart in the Streamlit app
# st.plotly_chart(fig)

# """

# ─────────────────────────────────────────────────────────────
#  Modelos anidados para título y ejes — aseguran que
#  ``layout.xaxis.title.text`` y ``layout.yaxis.title.text``
#  estén claramente definidos y validados.
# ─────────────────────────────────────────────────────────────


class _AxisTitle(BaseModel):
    """Título de un eje con propiedades básicas.

    Se define como modelo aparte para garantizar que el texto del título
    exista y sea accesible vía la ruta estándar ``axis.title.text``.
    """

    text: str = Field(
        ...,
        description="Texto del título del eje (obligatorio).",
    )
    font: dict | None = Field(
        None,
        description="Configuración de la fuente del título (opcional).",
    )
    standoff: int | None = Field(
        None,
        description="Separación en píxeles entre el título y las etiquetas de tick.",
    )

    class Config:
        extra = "allow"  # Permite keys avanzadas (color, side, etc.)


class _Axis(BaseModel):
    """Representa un eje Plotly minimamente tipado con *title* obligatorio.

    Solo se exige el sub-objeto *title* para asegurar la presencia del texto;
    cualquier otra propiedad del eje se acepta tal cual mediante ``extra=allow``.
    """

    title: _AxisTitle = Field(
        ...,
        description="Objeto con el título y su estilo.",
    )

    class Config:
        extra = "allow"
