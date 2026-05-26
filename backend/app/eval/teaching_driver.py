"""Drive one teaching turn from a setup, for the teaching-turn eval.

Produces a teaching turn the judge can score. Mirrors the retest-grading
driver's shape: a fresh chat, a static intro, one prompt, one parsed
response, no database, no tool surface. An eval run is not a learning
session, so this does not touch get_or_create_topic, prereq checks,
session rows, or build_intro (which reads live db state and would
make two runs of the same item non-reproducible).

The intro is static and declares the same delimited output format the
parser expects. It duplicates that format spec rather than importing
build_intro, for the same reason retest grading has its own intro:
build_intro is db-coupled and advertises tools and domain context
this driver must not include. Format drift is caught structurally: every
driver call parses its response through the real parser, so a drifted
format spec produces a parse failure in the driver's own tests.

No tools. The intro does not advertise a tool surface, so a well-behaved
LLM returns a teaching turn directly. A response that is anything other
than a teaching turn (a tool call, a session-end proposal) is a driver
error: the setup asked for a teaching turn and got something else.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.prompts.first_prompt import build_first_prompt
from app.schemas.parsed_response import ParsedTurn
from app.services.parser import ParseError, parse_response
from app.transport.base import TransportError

if TYPE_CHECKING:
    from app.eval.schemas import TeachingSetup
    from app.transport.base import LLMTransport

# Static intro for the teaching driver. Declares the delimited teaching-turn
# format and nothing else: no domain list, no knowledge summary, no tool
# surface. Duplicated from the teaching format spec rather than imported
# because build_intro is db-coupled.
_TEACHING_DRIVER_INTRO = """\
You are a teaching assistant. You produce one teaching turn at a time in a
strict delimited format. Reply with exactly one teaching turn and nothing
else: no tool calls, no session-end proposals, no conversational preamble.

The format is:

---TOPIC---
<Domain > Category > Subtopic>
---DIFFICULTY---
<beginner | intermediate | advanced>
---PREREQUISITES---
<comma-separated "path:difficulty" pairs, or NONE>
---MODE---
<one of: flashcard | type_the_answer | code_with_explanation | multiple_choice | explain_back | socratic>
---QUESTION---
<the question or teaching prompt>
---QUESTION_CODE---
<language tag on the first line then code, or NONE>
---EXPECTED_ANSWER---
<the canonical answer, or OPEN for free-form modes>
---REQUIREMENTS---
<setup the learner needs, or NONE>
---FOLLOWUP---
<an optional hint or follow-up, or NONE>
---TAGS---
<comma-separated tags, or NONE>
---END---
"""


class TeachingDriverError(Exception):
    """The teaching driver could not produce a teaching turn.

    Raised when the transport fails, the response does not parse, or the
    response parses to something other than a teaching turn. The evaluator
    turns this into an ERROR-outcome score: the eval item could not be
    measured, which is distinct from the teaching turn being judged poor.
    """

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.cause = cause


async def drive_teaching_turn(transport: LLMTransport[Any], setup: TeachingSetup) -> ParsedTurn:
    """Produce one teaching turn for the given setup.

    Opens a fresh chat with the static teaching intro and a first prompt
    built from the setup's topic, then parses the response. Returns the
    ParsedTurn. Raises TeachingDriverError if the transport fails, the
    response does not parse, or it parses to a non-teaching-turn kind.
    """
    first_prompt = build_first_prompt(setup.topic_path)

    try:
        _chat, response = await transport.start_new_chat(_TEACHING_DRIVER_INTRO, first_prompt)
    except TransportError as e:
        raise TeachingDriverError(
            f"Transport failed driving teaching turn: {e.message}", cause=e
        ) from e

    try:
        parsed = parse_response(response.text)
    except ParseError as e:
        raise TeachingDriverError("Teaching-turn response did not parse.", cause=e) from e

    if not isinstance(parsed, ParsedTurn):
        raise TeachingDriverError(
            f"Expected a teaching turn, got {parsed.kind!r}.",
        )

    return parsed
