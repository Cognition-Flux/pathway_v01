import reflex as rx


# ---------------------------------------------------------------------------
# rxconfig.py
# ---------------------------------------------------------------------------
# Este fichero define la configuración *global* de Reflex para la aplicación.
# Al mantenerse en la raíz del proyecto, **Reflex** lo detecta automáticamente
# al arrancar y utiliza los parámetros aquí declarados —principalmente
# `app_name`— para:
#
# 1. Nombrar la instancia de la aplicación (se refleja en los atributos
#    HTML/JS generados y en los mensajes de consola).
# 2. Crear la estructura de carpetas de compilación (`.web/`) y los artefactos
#    resultantes del build front-end.
# 3. Facilitar la importación relativa de componentes (p. ej. `from app import
#    state`).
#
# En proyectos más complejos, este archivo suele incluir ajustes de logging,
# rutas de static/assets o *feature flags*.  En nuestro caso basta con indicar
# el nombre de la app ya que todos los demás parámetros utilizan los valores
# por defecto suministrados por Reflex.
# ---------------------------------------------------------------------------

config = rx.Config(
    app_name="pathway_front",
    tailwind=None,  # Explicitly disable Tailwind CSS since this project doesn't use it
)
