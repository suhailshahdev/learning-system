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

## Disclaimer

This is a personal learning project, built in public for educational
purposes. It includes browser automation against claude.ai, which is
against Anthropic's Terms of Service. The author accepts this risk
for personal, single-user use only.

If you fork or use this code, you do so at your own risk. The author
is not responsible for any consequences, including but not limited to
account termination, legal action, or any other impact on your
relationship with Anthropic or any other service this project
interacts with.

This project is not affiliated with, endorsed by, or otherwise
associated with Anthropic.

For any production or commercial use, use the official Anthropic API.

## License

TBD.
