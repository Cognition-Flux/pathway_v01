"""Agentic graph playground package.

This package gathers experimental agents, tools and prompts used while
building the *TPP agentic graph* prototype.  The goal is to be able to
import any function or class that lives anywhere under this directory by
simply doing::

    from dev.tpp_agentic_graph_v4 import MyFunction, MyClass

All sub-packages are automatically discovered on import so that their
public symbols are re-exported from this package's namespace.  Public
symbols are defined by the ``__all__`` variable of every sub-module.  If
``__all__`` is not present, **all** top-level names that do not start
with an underscore ("_*") are considered public.

The traversal happens only once, at import time, and is lightweight
because it relies on ``pkgutil.walk_packages`` which inspects the
filesystem without executing the byte-code of each module.  Actual
modules are imported lazily on first access via ``importlib.import_module``.
"""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType


__all__: list[str] = []  # aggregated public names exposed by this package


def _expose_public_attributes(module: ModuleType) -> None:
    """Add public attributes from *module* to this package's globals.

    A *public attribute* is defined as any attribute listed in the module's
    ``__all__``.  When ``__all__`` is not defined we fallback to the names
    that do **not** start with an underscore.
    """
    public_names = getattr(module, "__all__", None)
    if public_names is None:
        public_names = [name for name in dir(module) if not name.startswith("_")]

    globals().update({name: getattr(module, name) for name in public_names})
    __all__.extend(public_names)


def _discover_and_import_submodules() -> None:
    """Walk the package tree and import every sub-module once."""
    # ``__path__`` is defined by the import machinery for packages.
    package_name = __name__
    for module_info in pkgutil.walk_packages(__path__, prefix=f"{package_name}."):
        # Avoid importing the current package again.
        if module_info.name == package_name:
            continue

        try:
            module = importlib.import_module(module_info.name)
        except Exception as exc:  # pragma: no cover – best effort import
            # Swallow import errors silently but make them accessible for debugging.
            # This design choice allows the rest of the package to remain usable even
            # if a single experimental module is broken.
            globals()[module_info.name.rsplit(".", 1)[-1]] = exc  # type: ignore[assignment]
            continue

        _expose_public_attributes(module)


# Perform the discovery at import time so that the package behaves like a *flat* module.
_discover_and_import_submodules()
