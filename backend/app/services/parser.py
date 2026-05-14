"""Parser for delimited LLM responses.

The transport layer hands back raw text. This module turns that
text into one of the structured shapes from app/schemas/parsed_response.py.

Three response kinds are recognized, dispatched on the first
delimiter line in the text:

  ---TOPIC---                  -> ParsedTurn
  ---SESSION_END_PROPOSAL---   -> ParsedSessionEnd
  ---HANDOVER---               -> ParsedHandover

Anything else is a ParseError. The parser is strict by design:
every required field must be present, every enum value must be
valid, every prerequisite pair must be well-formed. Loud failures
catch prompt format drift early. Lenient parsing would hide real
bugs in saved session data.
"""

from __future__ import annotations

import json
import re

from pydantic import TypeAdapter, ValidationError

from app.models.enums import Difficulty, GradingVerdict, LearningMode
from app.schemas.common import Prerequisite
from app.schemas.parsed_response import (
    CodeBlock,
    ParsedGrading,
    ParsedHandover,
    ParsedProposal,
    ParsedResponse,
    ParsedSessionEnd,
    ParsedToolCall,
    ParsedTurn,
)
from app.schemas.tools import ToolCall

# Matches a delimiter line and captures the name. Anchored to whole
# line so a stray `---FOO---` inside content (e.g., a comment in a
# code block) does not get mistaken for a delimiter.
DELIMITER_RE = re.compile(r"^---([A-Z_]+)---$", re.MULTILINE)

# Sentinels in the wire format that map to None / [] in the model.
SENTINEL_OPEN = "OPEN"
SENTINEL_NONE = "NONE"

# Required fields per kind. Kept here so the dispatcher and the
# field-extractor agree on what counts as a complete response.
TURN_FIELDS = (
    "TOPIC",
    "DIFFICULTY",
    "PREREQUISITES",
    "MODE",
    "QUESTION",
    "QUESTION_CODE",
    "EXPECTED_ANSWER",
    "REQUIREMENTS",
    "FOLLOWUP",
    "TAGS",
)

GRADING_FIELDS = (
    "GRADING",
    "GRADING_EXPLANATION",
    "GRADING_EXPLANATION_CODE",
)

HANDOVER_KEYS = (
    "DOMAIN_FOCUS",
    "COVERED",
    "LAST_QUESTION",
    "NEXT_PLANNED",
    "OPEN_THREADS",
    "USER_STATE",
)

PROPOSAL_KEYS = (
    "TOPIC_PATH",
    "REASONING",
)

# A single-block response (header + body) must be followed by an
# end marker. This is the minimum block count for SESSION_END_PROPOSAL
# and HANDOVER, both of which have one body block plus a closing marker.
MIN_BLOCKS_WITH_END_MARKER = 2

# A valid code block has a language tag on the first line and a code
# body on subsequent lines: minimum two lines after the field is split.
MIN_CODE_BLOCK_LINES = 2


# Module-level adapter so the discriminated-union validator is built
# once. TypeAdapter compilation is the expensive part so the same
# instance is reused to keep tool-call parsing cheap.
_TOOL_CALL_ADAPTER: TypeAdapter[ToolCall] = TypeAdapter(ToolCall)


class ParseError(Exception):
    """A response could not be parsed into a known shape.

    Carries the raw response so the session engine can log it to
    error_log for later inspection. The message describes what
    went wrong; the cause is the underlying validation error if
    one fired.
    """

    def __init__(
        self,
        message: str,
        raw_response: str,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.raw_response = raw_response
        self.cause = cause


def parse_response(text: str) -> ParsedResponse:
    """Parse a transport response into a ParsedResponse.

    Raises ParseError on any structural or semantic problem.
    """
    blocks = _split_blocks(text)
    if not blocks:
        raise ParseError("No delimiters found in response.", raw_response=text)

    first_marker, _ = blocks[0]
    if first_marker == "TOPIC":
        return _parse_turn(blocks, raw=text)
    if first_marker == "SESSION_END_PROPOSAL":
        return _parse_session_end(blocks, raw=text)
    if first_marker == "HANDOVER":
        return _parse_handover(blocks, raw=text)
    if first_marker == "TOOL_CALL":
        return _parse_tool_call(blocks, raw=text)
    if first_marker == "GRADING":
        return _parse_grading(blocks, raw=text)
    if first_marker == "PROPOSAL":
        return _parse_proposal(blocks, raw=text)

    raise ParseError(f"Unknown leading delimiter: {first_marker!r}.", raw_response=text)


def _split_blocks(text: str) -> list[tuple[str, str]]:
    """Split text into (marker, content) pairs.

    Walks the delimiter regex match-by-match. Content is everything
    between this delimiter and the next one. A final END or
    END_HANDOVER marker is included; its content (anything after it)
    is discarded.
    """
    matches = list(DELIMITER_RE.finditer(text))
    blocks: list[tuple[str, str]] = []
    for i, match in enumerate(matches):
        marker = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        blocks.append((marker, content))
    return blocks


def _parse_turn(blocks: list[tuple[str, str]], raw: str) -> ParsedTurn:
    """Build a ParsedTurn from a block list starting with TOPIC."""
    fields = _collect_fields(blocks, expected=TURN_FIELDS, end_marker="END", raw=raw)

    payload = {
        "topic_path": fields["TOPIC"],
        "difficulty": _parse_enum(fields["DIFFICULTY"], Difficulty, "DIFFICULTY", raw),
        "prerequisites": _parse_prerequisites(fields["PREREQUISITES"], raw),
        "mode": _parse_enum(fields["MODE"], LearningMode, "MODE", raw),
        "question": fields["QUESTION"],
        "question_code": _parse_code_block(fields["QUESTION_CODE"], "QUESTION_CODE", raw),
        "expected_answer": _none_if_sentinel(fields["EXPECTED_ANSWER"], SENTINEL_OPEN),
        "requirements": _none_if_sentinel(fields["REQUIREMENTS"], SENTINEL_NONE),
        "followup": _none_if_sentinel(fields["FOLLOWUP"], SENTINEL_NONE),
        "tags": _parse_tags(fields["TAGS"]),
    }

    try:
        return ParsedTurn.model_validate(payload)
    except ValidationError as e:
        raise ParseError("Turn failed schema validation.", raw_response=raw, cause=e) from e


def _parse_session_end(blocks: list[tuple[str, str]], raw: str) -> ParsedSessionEnd:
    """Build a ParsedSessionEnd from a block list starting with SESSION_END_PROPOSAL."""
    if len(blocks) < MIN_BLOCKS_WITH_END_MARKER or blocks[1][0] != "END":
        raise ParseError("SESSION_END_PROPOSAL must be followed by END.", raw_response=raw)

    summary = blocks[0][1]
    try:
        return ParsedSessionEnd.model_validate({"summary": summary})
    except ValidationError as e:
        raise ParseError("Session end failed schema validation.", raw_response=raw, cause=e) from e


def _parse_handover(blocks: list[tuple[str, str]], raw: str) -> ParsedHandover:
    """Build a ParsedHandover from a block list starting with HANDOVER.

    The handover format is one block of KEY: value lines, terminated
    by ---END_HANDOVER---. Different from the other two kinds, which
    use one delimiter per field.
    """
    if len(blocks) < MIN_BLOCKS_WITH_END_MARKER or blocks[1][0] != "END_HANDOVER":
        raise ParseError("HANDOVER must be followed by END_HANDOVER.", raw_response=raw)

    body = blocks[0][1]
    fields = _parse_handover_body(body, raw=raw)

    try:
        return ParsedHandover.model_validate(fields)
    except ValidationError as e:
        raise ParseError("Handover failed schema validation.", raw_response=raw, cause=e) from e


def _parse_proposal(blocks: list[tuple[str, str]], raw: str) -> ParsedProposal:
    """Build a ParsedProposal from a block list starting with PROPOSAL.

    The proposal format is one block of KEY: value lines, terminated
    by ---END_PROPOSAL---. Same shape as a handover block but with
    different keys (TOPIC_PATH, REASONING) and a smaller required
    set.

    Field-presence is strict: missing keys raise ParseError
    rather than defaulting silently. A diagnostic LLM that omits
    REASONING has misunderstood the format and a loud failure surfaces
    that early.
    """
    if len(blocks) < MIN_BLOCKS_WITH_END_MARKER or blocks[1][0] != "END_PROPOSAL":
        raise ParseError("PROPOSAL must be followed by END_PROPOSAL.", raw_response=raw)

    body = blocks[0][1]
    fields = _parse_proposal_body(body, raw=raw)

    try:
        return ParsedProposal.model_validate(fields)
    except ValidationError as e:
        raise ParseError("Proposal failed schema validation.", raw_response=raw, cause=e) from e


def _parse_proposal_body(body: str, raw: str) -> dict[str, str]:
    """Parse the KEY: value lines inside a proposal block.

    Mirrors _parse_handover_body. Each line is KEY: value, keys are
    validated against PROPOSAL_KEYS, missing keys raise. The block
    must contain every required key once. Duplicate keys overwrite
    silently (last wins) which is consistent with handover behavior.
    """
    fields: dict[str, str] = {}
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" not in line:
            raise ParseError(f"Malformed proposal line (no colon): {line!r}", raw_response=raw)
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if key not in PROPOSAL_KEYS:
            raise ParseError(f"Unknown proposal key: {key!r}", raw_response=raw)
        fields[_proposal_field_name(key)] = value

    missing = [k for k in PROPOSAL_KEYS if _proposal_field_name(k) not in fields]
    if missing:
        raise ParseError(f"Proposal missing fields: {missing}", raw_response=raw)
    return fields


def _proposal_field_name(key: str) -> str:
    """Convert wire-format KEY (uppercase) to model field name (lowercase)."""
    return key.lower()


def _parse_grading(blocks: list[tuple[str, str]], raw: str) -> ParsedGrading:
    """Build a ParsedGrading from a block list starting with GRADING.

    Standalone grading response shape: three required field blocks
    (GRADING, GRADING_EXPLANATION, GRADING_EXPLANATION_CODE) followed
    by END. Same structure as a ParsedTurn but smaller. The verdict
    must be a valid GradingVerdict value (not the NONE sentinel that
    embedded grading-on-turn accepted, since a standalone grading
    response by definition has a verdict).
    """
    fields = _collect_fields(blocks, expected=GRADING_FIELDS, end_marker="END", raw=raw)

    payload = {
        "verdict": _parse_enum(fields["GRADING"], GradingVerdict, "GRADING", raw),
        "explanation": fields["GRADING_EXPLANATION"],
        "explanation_code": _parse_code_block(
            fields["GRADING_EXPLANATION_CODE"], "GRADING_EXPLANATION_CODE", raw
        ),
    }

    try:
        return ParsedGrading.model_validate(payload)
    except ValidationError as e:
        raise ParseError("Grading failed schema validation.", raw_response=raw, cause=e) from e


def _parse_tool_call(blocks: list[tuple[str, str]], raw: str) -> ParsedToolCall:
    """Build a ParsedToolCall from a block list starting with TOOL_CALL.

    The block carries a single JSON payload between the TOOL_CALL
    delimiter and the END delimiter. Validates against the
    discriminated ToolCall union which dispatches on the `name`
    field to the matching call envelope.

    Returns a ParsedToolCall carrying both the validated call and
    the raw block text. The session-service loop logs raw_text to
    error_log if the handler later fails so the original LLM
    output is preserved for inspection.
    """
    if len(blocks) < MIN_BLOCKS_WITH_END_MARKER or blocks[1][0] != "END":
        raise ParseError("TOOL_CALL must be followed by END.", raw_response=raw)

    body = blocks[0][1]
    if not body:
        raise ParseError("TOOL_CALL block is empty.", raw_response=raw)

    try:
        data = json.loads(body)
    except json.JSONDecodeError as e:
        raise ParseError(
            f"TOOL_CALL body is not valid JSON: {e.msg}",
            raw_response=raw,
            cause=e,
        ) from e

    if not isinstance(data, dict):
        raise ParseError(
            f"TOOL_CALL body must be a JSON object, got {type(data).__name__}.",
            raw_response=raw,
        )

    try:
        call = _TOOL_CALL_ADAPTER.validate_python(data)
    except ValidationError as e:
        raise ParseError(
            "Tool call failed schema validation.",
            raw_response=raw,
            cause=e,
        ) from e

    try:
        return ParsedToolCall.model_validate({"call": call, "raw_text": body})
    except ValidationError as e:
        raise ParseError(
            "ParsedToolCall failed schema validation.",
            raw_response=raw,
            cause=e,
        ) from e


def _parse_handover_body(body: str, raw: str) -> dict[str, str]:
    """Parse the KEY: value lines inside a handover block."""
    fields: dict[str, str] = {}
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" not in line:
            raise ParseError(f"Malformed handover line (no colon): {line!r}", raw_response=raw)
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if key not in HANDOVER_KEYS:
            raise ParseError(f"Unknown handover key: {key!r}", raw_response=raw)
        fields[_handover_field_name(key)] = value

    missing = [k for k in HANDOVER_KEYS if _handover_field_name(k) not in fields]
    if missing:
        raise ParseError(f"Handover missing fields: {missing}", raw_response=raw)
    return fields


def _handover_field_name(key: str) -> str:
    """Convert wire-format KEY (uppercase) to model field name (lowercase)."""
    return key.lower()


def _collect_fields(
    blocks: list[tuple[str, str]],
    expected: tuple[str, ...],
    end_marker: str,
    raw: str,
) -> dict[str, str]:
    """Build a {marker: content} dict, enforcing the expected sequence."""
    if not blocks or blocks[-1][0] != end_marker:
        raise ParseError(f"Response must terminate with ---{end_marker}---.", raw_response=raw)

    field_blocks = blocks[:-1]
    if len(field_blocks) != len(expected):
        raise ParseError(
            f"Expected {len(expected)} fields, got {len(field_blocks)}: "
            f"{[m for m, _ in field_blocks]}",
            raw_response=raw,
        )

    out: dict[str, str] = {}
    for i, (marker, content) in enumerate(field_blocks):
        if marker != expected[i]:
            raise ParseError(
                f"Field {i} expected {expected[i]!r}, got {marker!r}.",
                raw_response=raw,
            )
        out[marker] = content
    return out


def _parse_enum[E: (Difficulty, LearningMode, GradingVerdict)](
    value: str, enum_cls: type[E], field_name: str, raw: str
) -> E:
    """Coerce a string to an enum member, raising ParseError on miss."""
    try:
        return enum_cls(value.strip().lower())
    except ValueError as e:
        valid = [m.value for m in enum_cls]
        raise ParseError(
            f"Invalid {field_name}: {value!r}. Valid values: {valid}.",
            raw_response=raw,
            cause=e,
        ) from e


def _parse_code_block(value: str, field_name: str, raw: str) -> CodeBlock | None:
    """Parse a code-block field. Returns None for the NONE sentinel.

    The wire format puts the language tag on the first line and the
    code body on the lines that follow. Both must be non-empty. A
    field that is just NONE means no code block. A language without
    a body or a body without a language is a parse error.
    """
    stripped = value.strip()
    if stripped == SENTINEL_NONE or not stripped:
        return None

    lines = value.split("\n", 1)
    if len(lines) < MIN_CODE_BLOCK_LINES:
        raise ParseError(
            f"{field_name} must have a language tag on the first line "
            f"and a code body on subsequent lines.",
            raw_response=raw,
        )

    language = lines[0].strip()
    body = lines[1].strip()
    if not language:
        raise ParseError(f"{field_name} missing language tag.", raw_response=raw)
    if not body:
        raise ParseError(f"{field_name} missing code body.", raw_response=raw)

    try:
        return CodeBlock(language=language, body=body)
    except ValidationError as e:
        raise ParseError(
            f"{field_name} failed schema validation.", raw_response=raw, cause=e
        ) from e


def _parse_prerequisites(value: str, raw: str) -> list[Prerequisite]:
    """Parse the PREREQUISITES content into a list of Prerequisite models."""
    stripped = value.strip()
    if stripped == SENTINEL_NONE or not stripped:
        return []

    out: list[Prerequisite] = []
    for raw_pair in stripped.split(","):
        pair = raw_pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise ParseError(f"Malformed prerequisite (no colon): {pair!r}", raw_response=raw)
        # rpartition so a topic path containing colons still parses;
        # difficulty (rightmost token) is unambiguous.
        path, _, diff = pair.rpartition(":")
        path = path.strip()
        diff = diff.strip().lower()
        if not path:
            raise ParseError(f"Prerequisite missing topic_path: {pair!r}", raw_response=raw)
        try:
            min_diff = Difficulty(diff)
        except ValueError as e:
            raise ParseError(
                f"Invalid prerequisite difficulty: {diff!r} in {pair!r}.",
                raw_response=raw,
                cause=e,
            ) from e
        out.append(Prerequisite(topic_path=path, min_difficulty=min_diff))
    return out


def _parse_tags(value: str) -> list[str]:
    """Split TAGS content on commas, strip each, drop empties."""
    stripped = value.strip()
    if not stripped or stripped == SENTINEL_NONE:
        return []
    return [t.strip() for t in stripped.split(",") if t.strip()]


def _none_if_sentinel(value: str, sentinel: str) -> str | None:
    """Return None if the field's content is the literal sentinel, else the stripped value."""
    stripped = value.strip()
    return None if stripped == sentinel else stripped
