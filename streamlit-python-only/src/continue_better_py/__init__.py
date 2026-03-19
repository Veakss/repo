"""Compatibility shim during the package rename.

The active codebase now imports ``streamlit_python_only`` directly. This module
remains only to avoid breaking any residual legacy imports while the rename is
fully absorbed across the repo and external tooling.
"""

from importlib import import_module

_new_pkg = import_module("streamlit_python_only")

__all__ = getattr(_new_pkg, "__all__", [])
__path__ = _new_pkg.__path__
