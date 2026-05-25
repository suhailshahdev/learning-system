"""Evaluation framework for the learning system.

Versioned eval sets scored against a target, with run records that the
next run diffs against. The gate that protects later AI behavior from
silent regression: build the gate before shipping the thing it guards.

Code lives here under app.eval. Eval-set data and run records live as
files under backend/eval/ (sets/ and runs/), separate from code the way
scripts/ and the database file already separate.
"""

from __future__ import annotations
