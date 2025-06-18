"""
Este fichero define la configuración *global* de Reflex para la aplicación.
Al mantenerse en la raíz del proyecto, **Reflex** lo detecta automáticamente
al arrancar y utiliza los parámetros aquí declarados —principalmente
`app_name`— para:

1. Nombrar la instancia de la aplicación (se refleja en los atributos
"""

import os

import reflex as rx

app_name = os.getenv("REFLEX_APP_NAME", "pathway_front")

config = rx.Config(
    app_name=app_name,
    frontend_port=3000,
    tailwind=None,  # Explicitly disable Tailwind CSS since this project doesn't use it
)
