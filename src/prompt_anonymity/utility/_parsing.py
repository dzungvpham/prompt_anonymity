"""Shared string wrangling for the utility metrics: reply cleanup and conversation rendering.

Every metric here has to do the same two unglamorous jobs -- turn a chat model's reply into
structured data despite markdown fences and reasoning preambles, and turn a stored conversation
cell into text a model can read. Those helpers live here so a new metric inherits them.

What is deliberately **not** shared: the rubrics, the actual verdict/score extraction, and the
result shapes. Those differ per metric by design, and collapsing them into a common abstraction
would couple metrics that should be free to disagree.
"""

from __future__ import annotations

import json
import re

from ..defenses._backends import split_turns

#: Opening / closing markdown code fences some models wrap JSON in.
_FENCE_OPEN = re.compile(r"^```[a-zA-Z0-9]*\s*")
_FENCE_CLOSE = re.compile(r"\s*```$")

#: Appended when :func:`render_turns` truncates, so the judge knows it is reading a prefix rather
#: than assuming the conversation simply ended there.
TRUNCATION_MARKER = "\n[...truncated]"


def strip_code_fence(raw: str) -> str:
    """Strip a surrounding ```` ```json ... ``` ```` fence and outer whitespace.

    Models routinely wrap JSON in a fence despite being told to return JSON only; stripping it is
    the difference between a parsed verdict and an unparsed row.
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        text = _FENCE_OPEN.sub("", text)
        text = _FENCE_CLOSE.sub("", text).strip()
    return text


def json_object(text: str) -> dict | None:
    """Parse ``text`` as a JSON object with lowercased keys, or ``None`` if it is not one.

    Keys are lowercased so a rubric asking for ``{"Score": ...}`` still parses when the model
    replies with ``{"score": ...}``. Returns ``None`` (rather than raising) for anything that is
    not a JSON object, so callers can fall through to their regex tier.
    """
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    return {str(k).lower(): v for k, v in obj.items()}


def render_turns(text, *, label: bool = False, drop_blank: bool = False,
                 max_chars: int | None = None) -> str:
    """Render one conversation cell as readable text for a model.

    On disk a cell is a user's turns joined by the literal ``\\n===\\n`` delimiter (see
    :data:`prompt_anonymity.defenses._backends.TURN_DELIM`); this splits on that and rejoins with
    real newlines so the model never sees the delimiter.

    Parameters
    ----------
    text : str
        The conversation cell.
    label : bool, default False
        Prefix each turn with ``[Turn i]`` (1-based). Needed when a whole conversation is shown at
        once: without labels the turns collapse into one wall of text, and a judge can neither say
        *which* turn broke nor notice that a defense dropped one. Left off when turns are being
        shown individually, where the label would be noise.
    drop_blank : bool, default False
        Drop whitespace-only turns. **Defaults off on purpose**: with both flags off this function
        is byte-for-byte identical to the plain ``"\\n".join(split_turns(text))`` that
        :mod:`.answer_judge` has always used, and that rendered text *is* the key of a cache full
        of paid completions -- filtering by default would silently miss every row whose defended
        turn came back whitespace-only. Turn on when rendering a whole conversation, where a blank
        turn is just a delimiter artifact that would skew the ``[Turn i]`` numbering.
    max_chars : int, optional
        Truncate the result to this many characters, appending :data:`TRUNCATION_MARKER`. Applied
        per rendered side by the caller, so a truncated comparison stays fair.
    """
    turns = split_turns(str(text))
    if drop_blank:
        turns = [turn for turn in turns if turn.strip()]
    if label:
        turns = [f"[Turn {i}] {turn}" for i, turn in enumerate(turns, start=1)]
    rendered = "\n".join(turns)
    if max_chars is not None and len(rendered) > max_chars:
        rendered = rendered[:max_chars] + TRUNCATION_MARKER
    return rendered
