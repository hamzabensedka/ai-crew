"""Web search tool interface."""

from __future__ import annotations


def web_search(query: str) -> str:
    """Placeholder search — returns guidance to use documentation."""
    return (
        f"Search query: {query}\n"
        "Use official documentation for the relevant library or framework. "
        "Integrate a real search API (e.g. Serper, Tavily) for production use."
    )
