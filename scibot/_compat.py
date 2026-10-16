"""Compatibility shim for the ollama Python client.

ollama>=0.5 returns pydantic models from API calls; 0.4.x returns
plain dicts. This helper reads a field from either, so SciBot works
with whatever version the user has.
"""

from __future__ import annotations

from typing import Any


def field(resp: Any, name: str, default: Any = None) -> Any:
    if isinstance(resp, dict):
        return resp.get(name, default)
    return getattr(resp, name, default)
