# Architecture

This document describes the architecture of the learning system at a high level. It is kept current as the project evolves.

**Last updated:** initial draft.

## Goals

- Capture every learning session into a structured, queryable local database.
- Keep Claude as the teaching engine, behind an abstraction so the transport layer can change without touching business logic.
- Maintain a single source of truth for what the user knows, across sessions and over time.
- Stay simple enough for one developer to hold in their head.
