# %%
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# pylint: disable=import-error,wrong-import-position

MODULE_NAME = "tpp_agentic_graph_v4"
ALSO_NOTEBOOK = True
if ALSO_NOTEBOOK:
    current_file = Path(__file__).resolve()
    package_root = current_file.parent
    while package_root.name != MODULE_NAME:
        package_root = package_root.parent

    if str(package_root.parent) not in sys.path:
        sys.path.insert(0, str(package_root.parent))

from tpp_agentic_graph_v4.chains import AA, HOLA  # noqa: E402

load_dotenv(override=True)


if __name__ == "__main__":
    print(os.getcwd())
    print(HOLA)
    print(AA)

# %%
