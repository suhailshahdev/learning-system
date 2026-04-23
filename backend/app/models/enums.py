"""Shared enums used across models.

This file holds every enum that either maps to a database column
or gets used by more than one model. Keeping them together means
you can open one file to see every valid value for every enum
column in the schema.

All enums inherit from `StrEnum`, so each member is both an enum
and a plain string. That matters because the database stores the
string value (for example, `"language"`) and the JSON API sends
the same string back to the frontend. No extra conversion step,
no surprise object wrappers.
"""

from __future__ import annotations

from enum import StrEnum


class DomainKind(StrEnum):
    """Kind of top-level domain. Used by `domain.kind`."""

    LANGUAGE = "language"
    FRAMEWORK = "framework"
    LIBRARY = "library"
    CONCEPT = "concept"
    TOOL = "tool"
    PRACTICE = "practice"
    OTHER = "other"
