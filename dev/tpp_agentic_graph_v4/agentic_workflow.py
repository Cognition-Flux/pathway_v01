"""Orquestador de alto nivel para el *TPP agentic graph*.

Este módulo actúa como punto de entrada cuando se ejecutan experimentos
localmente.  Configura el entorno para que los imports relativos
funcionen tanto si se importa el paquete como si se lanza este archivo
con `python`.  Además, expone de forma explícita los símbolos públicos
que vive en el paquete.
"""

# %%
from __future__ import annotations

import sys
from pathlib import Path

# pylint: disable=import-error,wrong-import-position

MODULE_NAME = "tpp_agentic_graph_v4"
if __package__ in (None, ""):
    current_file = Path(__file__).resolve()
    # Buscamos el directorio raíz del paquete (`tpp_agentic_graph_v4`) para
    # añadir su directorio contenedor (`dev`) a `sys.path`.
    package_root = current_file.parent
    while package_root.name != MODULE_NAME:
        package_root = package_root.parent

    # Añadimos el directorio `dev` a `sys.path`.
    sys.path.insert(0, str(package_root.parent))

from tpp_agentic_graph_v4.chains import HOLA  # noqa: E402

if __name__ == "__main__":
    print(HOLA)

# %%
