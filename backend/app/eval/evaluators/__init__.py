"""Evaluators score a target's output against an eval item's expectation.

Deterministic evaluators (parser, retrieval) are pure comparisons. The
judge evaluator (teaching) calls an LLM. Each returns an ItemScore so the
runner collects them uniformly regardless of kind.
"""

from __future__ import annotations
