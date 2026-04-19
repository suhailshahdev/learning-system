# Learning System

A local learning tool that uses Claude as an interactive tutor and keeps a persistent record of everything you study.

**Status:** pre-alpha, under construction. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the plan.

## What it is

A personal study tool. Claude runs live sessions, asking questions and explaining concepts across software topics (languages, frameworks, patterns, tools). The local app captures each session, tags it by domain and difficulty, and makes it reviewable later.

It exists because chatting with Claude directly has no memory across sessions and no structure. This tool adds both.

Features:

- Captures each session into a local database with domain and difficulty tags
- Supports multiple modes: flashcards, type-the-answer, code-with-explanation, Socratic, and others
- Tracks prerequisites so sessions warn before teaching something that depends on unlearned material
- Replayable offline, retestable as new sessions
- Optional resume and job description inputs pivot the curriculum toward interview prep
- Runs locally, no cloud

## License

TBD.