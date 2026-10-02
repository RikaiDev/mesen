"""Vulture whitelist: names that look dead but are used dynamically.

Add a line per intentional case with a comment saying who consumes it, e.g.:

    # FastAPI route, mounted via decorator in serve/server.py
    health_check  # noqa: V104

Keep this file empty rather than lowering --min-confidence: the 60% tier is
pydantic fields, enum members and framework-called methods, i.e. noise.
"""
