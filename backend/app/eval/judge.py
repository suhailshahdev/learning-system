"""LLM-as-judge for teaching-turn quality.

Given a rubric and a teaching turn, the judge scores how well the turn
meets the rubric. The judge is a separate concern from the teaching driver
and from the response parser: it has its own static intro and its own tiny
output format (a score and a rationale), parsed by its own parser. The
teaching discriminated union is not extended to carry a judge verdict,
which has nothing to do with teaching.

The judge runs on a transport whose model differs from the model that
produced the turn (enforced at JudgeTarget construction): a model grading
its own output scores it generously. The caller supplies the judge
transport, this module does not choose it.

Score is a float 0.0 to 1.0 in tenths. The intro constrains the judge to
eleven buckets so it cannot fabricate two-decimal precision the rubric
cannot support, which also tightens cross-run variance. An out-of-range or
unparseable score raises rather than clamping: a clamped score would hide
a judge that did not follow the format, and the evaluator turns the raise
into an ERROR outcome (could not measure) rather than a poor score.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from app.transport.base import TransportError

if TYPE_CHECKING:
    from app.eval.schemas import TeachingSetup
    from app.schemas.parsed_response import ParsedTurn
    from app.transport.base import LLMTransport

# Static judge intro. States field-presence explicitly: both fields always,
# score quantized, nothing else. An implied format spec is not enough; an
# LLM omits fields it judges self-evident unless told otherwise.
_JUDGE_INTRO = """\
You are evaluating the quality of a single teaching turn against a rubric.
You will be given the rubric, the topic and mode the turn was meant to use,
and the teaching turn itself. Score how well the turn meets the rubric.

Reply in exactly this format and nothing else. Emit both fields every time,
even if the rationale is brief:

---SCORE---
<a number from 0.0 to 1.0 in steps of 0.1: one of 0.0, 0.1, 0.2, 0.3, 0.4,
0.5, 0.6, 0.7, 0.8, 0.9, 1.0>
---RATIONALE---
<one or two sentences explaining the score against the rubric>
---END---

Do not add any text before or after this block. Do not use any other
delimiters. The score must be one of the eleven listed values.
"""

# Delimiter line, anchored to a whole line, same convention as the response
# parser (a stray ---SCORE--- inside rationale prose is not a delimiter).
_JUDGE_DELIMITER_RE = re.compile(r"^---([A-Z_]+)---$", re.MULTILINE)

# Valid quantized scores. The judge is instructed to emit one of these.
# A value outside the set is a format violation, not a score to clamp.
_VALID_SCORES = frozenset({i / 10 for i in range(11)})


class JudgeError(Exception):
    """The judge could not produce a usable score.

    Raised on transport failure, on a response that does not contain both
    required fields, or on a score outside the quantized 0.0 to 1.0 set.
    The evaluator turns this into an ERROR-outcome score: the item could
    not be judged, distinct from the teaching turn scoring poorly.
    """

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.cause = cause


async def judge_teaching_turn(
    transport: LLMTransport[Any],
    setup: TeachingSetup,
    turn: ParsedTurn,
    rubric: str,
) -> tuple[float, str]:
    """Score a teaching turn against a rubric. Returns (score, rationale).

    Opens a fresh chat on the judge transport with the static judge intro
    and a prompt carrying the rubric, the setup, and the rendered turn.
    Raises JudgeError on transport failure or an unparseable/out-of-range
    response.
    """
    prompt = _build_judge_prompt(setup, turn, rubric)

    try:
        _chat, response = await transport.start_new_chat(_JUDGE_INTRO, prompt)
    except TransportError as e:
        raise JudgeError(f"Judge transport failed: {e.message}", cause=e) from e

    return parse_judge_response(response.text)


def _build_judge_prompt(setup: TeachingSetup, turn: ParsedTurn, rubric: str) -> str:
    """Assemble the judge prompt from rubric, setup, and the turn to score."""
    return f"""\
RUBRIC:
{rubric}

INTENDED TOPIC: {setup.topic_path}
INTENDED MODE: {setup.mode.value}
INTENDED DIFFICULTY: {setup.difficulty.value}

TEACHING TURN TO SCORE:
TOPIC: {turn.topic_path}
DIFFICULTY: {turn.difficulty.value}
MODE: {turn.mode.value}
QUESTION: {turn.question}
EXPECTED_ANSWER: {turn.expected_answer or "OPEN"}
"""


def parse_judge_response(text: str) -> tuple[float, str]:
    """Extract (score, rationale) from a judge response.

    Tolerates leading prose before the first delimiter and parses
    the SCORE and RATIONALE blocks. Raises JudgeError if either
    field is missing or the score is not one of the eleven quantized values.
    """
    blocks = _split_judge_blocks(text)
    if "SCORE" not in blocks:
        raise JudgeError("Judge response missing SCORE field.")
    if "RATIONALE" not in blocks:
        raise JudgeError("Judge response missing RATIONALE field.")

    score = _parse_score(blocks["SCORE"])
    rationale = blocks["RATIONALE"].strip()
    if not rationale:
        raise JudgeError("Judge response has empty RATIONALE.")
    return score, rationale


def _split_judge_blocks(text: str) -> dict[str, str]:
    """Split a judge response into {marker: content}, ignoring the END marker.

    Walks the delimiter regex match by match, same as the response parser's
    block splitter. Content is the text between a delimiter and the next.
    Duplicate markers keep the last (consistent with the handover parser).
    """
    matches = list(_JUDGE_DELIMITER_RE.finditer(text))
    blocks: dict[str, str] = {}
    for i, match in enumerate(matches):
        marker = match.group(1)
        if marker == "END":
            continue
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        blocks[marker] = text[start:end].strip()
    return blocks


def _parse_score(raw: str) -> float:
    """Parse and validate a quantized score. Raises JudgeError if invalid."""
    stripped = raw.strip()
    try:
        value = float(stripped)
    except ValueError as e:
        raise JudgeError(f"Judge SCORE is not a number: {stripped!r}.", cause=e) from e

    # Round to one decimal before the set check so 0.30000000004 from a
    # float parse still matches 0.3. Values genuinely outside 0.0 to 1.0
    # (1.5, -0.2) do not round into the set and raise.
    rounded = round(value, 1)
    if rounded not in _VALID_SCORES:
        raise JudgeError(
            f"Judge SCORE {stripped!r} is not one of the quantized values 0.0 to 1.0.",
        )
    return rounded
