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

from ...defenses._backends import split_turns

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
        Prefix each turn with ``[Turn i]`` (1-based). The conversation judge used this until
        2026-09-25 and now reads :func:`render_turn_pairs` instead, which pairs the two versions
        turn by turn rather than numbering each on its own.
    drop_blank : bool, default False
        Drop whitespace-only turns. **Defaults off on purpose**: with both flags off this function
        is byte-for-byte the plain ``"\\n".join(split_turns(text))``, which is what a caller
        rendering turns individually wants -- a blank turn there is a real (empty) rewrite worth
        seeing, not noise. Turn it on when rendering a whole conversation, where a blank turn is a
        delimiter artifact that would skew the ``[Turn i]`` numbering. Note the rendered text is
        the cache key for paid judge replies, so flipping either default silently orphans every
        entry already on disk.
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


def render_turn_pairs(original, modified, *, max_chars: int | None = None) -> str:
    """Render two versions of a conversation as **position-aligned turn pairs**, in XML.

    ::

        <conversation>
        <turn_1>
        <original>...</original>
        <modified>...</modified>
        </turn_1>
        ...
        </conversation>

    Pairing is done here, by position, rather than asked of the judge. The defenses rewrite a
    conversation one user turn at a time, so position *is* the correspondence; handing the judge
    two separate conversations made it re-derive that alignment, and made the numbering fragile.

    * **An absent side is an empty element**: ``<original></original>`` marks a turn that exists
      only in the modified version (an addition), ``<modified></modified>`` an original turn with
      no counterpart (a drop). A whitespace-only turn renders the same way -- for utility, a blank
      turn and a missing one both carry nothing.
    * **Blank turns keep their position.** :func:`render_turns` drops them *before* numbering,
      which renumbered the modified side whenever a defense blanked a turn and paired the wrong
      turns. Only a pair blank on **both** sides -- a delimiter artifact -- is skipped, and the
      remaining pairs are numbered consecutively.
    * **The text is not XML-escaped.** The corpora are full of literal placeholders (``<URL>``,
      ``<PATH>``) and code, and escaping would change what the judge reads. The structure is for the
      model, not a parser; a turn containing a literal ``</original>`` would be ambiguous, which is
      accepted.

    Parameters
    ----------
    original, modified : str
        The two conversation cells (turns joined by :data:`TURN_DELIM`).
    max_chars : int, optional
        Per-side character budget, spent turn by turn: once a side's turns exceed it, the turn that
        crosses it is cut and marked with :data:`TRUNCATION_MARKER`, and that side's later turns
        render empty. Applied to both sides alike, so a truncated comparison stays fair.
    """
    original_turns = _budget(split_turns(str(original)), max_chars)
    modified_turns = _budget(split_turns(str(modified)), max_chars)
    length = max(len(original_turns), len(modified_turns))
    original_turns += [""] * (length - len(original_turns))
    modified_turns += [""] * (length - len(modified_turns))

    blocks = ["<conversation>"]
    index = 0
    for original_turn, modified_turn in zip(original_turns, modified_turns):
        if not original_turn.strip() and not modified_turn.strip():
            continue
        index += 1
        blocks.append(
            f"<turn_{index}>\n"
            f"<original>{_block(original_turn)}</original>\n"
            f"<modified>{_block(modified_turn)}</modified>\n"
            f"</turn_{index}>"
        )
    blocks.append("</conversation>")
    return "\n".join(blocks)


def _block(text: str) -> str:
    """A turn's text on its own lines inside its element, or nothing at all when blank."""
    return f"\n{text}\n" if text.strip() else ""


def _budget(turns: list[str], max_chars: int | None) -> list[str]:
    """Spend a per-side character budget over ``turns`` in order (see :func:`render_turn_pairs`)."""
    if max_chars is None:
        return list(turns)
    kept, remaining = [], max_chars
    for turn in turns:
        if remaining <= 0:
            kept.append("")
        elif len(turn) > remaining:
            kept.append(turn[:remaining] + TRUNCATION_MARKER)
            remaining = 0
        else:
            kept.append(turn)
            remaining -= len(turn)
    return kept
