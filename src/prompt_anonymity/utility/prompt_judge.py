"""Conversation-level utility: how much of the original survives the defense, scored 1-5.

This is the *prompt-side* utility metric. Where :mod:`.answer_judge` asks a response model to
answer each turn and judges whether the two **answers** match, this shows a judge the two
**conversations** -- original and defended, whole, in one call -- and asks for a 1-5 score.

Two reasons that is worth having alongside the per-turn predicate:

* **Cross-turn structure.** A defense can preserve every turn in isolation and still break the
  thread: back-references stop resolving, or a turn vanishes. A judge scoring turns one at a time
  cannot see either failure, because neither is visible inside a single turn.
* **Resolution.** PASS/FAIL puts every defense that mostly works in the same bucket. A 1-5 scale
  separates a light paraphrase from a heavily degraded but still answerable rewrite, which is what
  ranking defenses actually needs.

It is also far cheaper: one call per conversation instead of two responses plus a verdict per
changed turn -- roughly 18x fewer calls on WildChat's ~6-turn conversations, paid for with longer
inputs.

**The rubric scores task utility, not textual similarity.** The judge is told explicitly that
stripping identifiers and rewriting style is what the defense is *for* and must not cost points.
Without that carve-out the metric would quietly re-measure the privacy axis -- a defense that
anonymizes well would score badly precisely *because* it worked -- and utility would stop being an
independent reading. Whether that instruction is actually landing is checkable, and worth checking:
read the ``reason`` column of the worst-scoring rows (see :meth:`ConversationUtilityResult.to_csv`).
If the reasons complain that names were removed or the voice changed, the carve-out is not holding
and the numbers should not be trusted yet.

Judging runs on OpenRouter (default ``anthropic/claude-sonnet-5``) through
:class:`~prompt_anonymity.utility._openrouter.OpenRouterChat`, and every verdict is cached by the
package's content-addressed :class:`~prompt_anonymity.caching.TransformCache` under
``<cache_dir>/utility/conversation_judge``. Conversations the defense left untouched short-circuit
to 5 with no API call, so scoring ``--defense none`` is free.

Caveats:

* **Prompts, not answers.** This measures semantic preservation of the request; it does not verify
  that a model still *answers* it well. A redaction the judge reads as minor could still sink the
  response. Read this and :mod:`.answer_judge` as complements, not substitutes.
* **Unbounded input.** Two whole conversations go into one request, so unlike the per-turn judge
  the input is not inherently bounded -- see ``max_chars``.
* **Absolute, not comparative.** Each conversation is scored on its own, so cross-defense
  comparisons inherit whatever bias the judge has about the scale. Check ``score_counts`` for
  compression before reading small differences as real.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from ._openrouter import OpenRouterChat
from ._parsing import json_object, render_turns, strip_code_fence
from .base import DEFAULT_SEED, UtilityMetric, UtilityResult

#: Conversation-level utility rubric. Scores 1-5 how much of the original's substance survives,
#: judging the conversation as a whole. The "style and identity changes are expected" carve-out is
#: stated twice -- once in the framing, once inside the scale -- because it is the instruction most
#: likely to be dropped, and dropping it silently turns this into a privacy metric.
CONVERSATION_JUDGE_SYSTEM_PROMPT = (
    "You are a utility evaluator. You will be shown two versions of the SAME conversation's user "
    "messages: the ORIGINAL, and a DEFENDED rewrite produced by a privacy defense whose goal is to "
    "hide the author's identity and writing style.\n\n"
    "Judge content, not style. The defense is SUPPOSED to change wording, tone, register, "
    "verbosity, and identifying details; none of that is a utility loss. A utility loss is a "
    "change in what the user is asking for, what they told you, or what they need -- in what an "
    "assistant reading only the defended version would do differently. Removing a specific detail "
    "IS a loss when that detail was needed to answer, and is NOT a loss when it was incidental or "
    "identifying.\n\n"
    "Weigh the conversation AS A WHOLE: whether the turns still follow from one another, whether "
    "references back to earlier turns still resolve, and whether the overall arc of what the user "
    "was trying to accomplish survives. Cross-turn coherence that was present in the original and "
    "is broken in the rewrite is a utility loss even when every turn reads fine on its own.\n\n"
    "Score on this scale:\n"
    "5 -- Fully faithful. Every request, constraint, fact, and nuance survives, and the turns "
    "still hang together. An assistant answering the defended version would give an answer just as "
    "useful to the original user. Only surface wording and identity changed.\n"
    "4 -- Substantially faithful. All main requests and constraints survive; a minor detail, "
    "hedge, or shade of nuance is blurred, but the answer would still fully serve the user.\n"
    "3 -- Partially faithful. Topic and primary request survive, but a meaningful requirement, "
    "constraint, or piece of context is lost, distorted, or made vague -- or a back-reference no "
    "longer resolves -- so the answer would be noticeably less useful, or would need a clarifying "
    "question first.\n"
    "2 -- Largely unfaithful. A core request is missing, garbled, or turned into a different "
    "question; or so much specific detail is gone that only a generic answer is possible. Some "
    "recognizable connection to the original remains.\n"
    "1 -- Unusable. Does not convey what the user wanted: incoherent, empty, off-topic, inventing "
    "requests the user never made, or redacted so heavily no useful answer is possible.\n\n"
    "Input contains two tagged sections, <original_conversation> and <defended_conversation>, each "
    "with the user's turns labeled [Turn 1], [Turn 2], and so on.\n\n"
    "Return JSON and nothing else: {\"Score\": 1-5, \"Reason\": YOUR_ONE_LINE_EXPLANATION}"
)

#: Default OpenRouter judge model -- the same current Sonnet the LLM-judge attacks use. EDIT ME (or
#: pass ``judge_model=``) to retarget; it is part of the cache key, so a swap re-caches cleanly.
DEFAULT_CONVERSATION_JUDGE_MODEL = "anthropic/claude-sonnet-5"

#: Output budget: a one-line JSON score plus a short reason.
DEFAULT_CONVERSATION_JUDGE_MAX_TOKENS = 512

#: Per-side character cap on the conversation shown to the judge, or ``None`` for no cap. Two whole
#: conversations plus the rubric can overrun a context window, and ``OpenRouterChat`` fails fast on
#: the resulting 4xx rather than retrying -- which aborts the batch. Set this if a split has very
#: long conversations.
DEFAULT_MAX_CHARS = None

#: Score at or above which a conversation counts as "defense preserved it" in ``usable_rate``.
#: Deliberately NOT part of the cache key: it only post-processes an already-cached score, so
#: re-sweeping the threshold costs nothing.
USABLE_SCORE_THRESHOLD = 4

#: Manual logic version for the conversation-judge cache; bump to force a full recompute. Separate
#: from :data:`prompt_anonymity.utility.answer_judge.ANSWER_UTILITY_VERSION` so bumping one metric never
#: throws away the other's paid results.
CONVERSATION_UTILITY_VERSION = "1"

#: Valid scores, and the labeled-field fallback for a reply whose JSON did not parse.
_VALID_SCORES = (1, 2, 3, 4, 5)
_SCORE_FIELD_RE = re.compile(r'"?score"?\s*:\s*"?\s*([1-5])', re.IGNORECASE)
_ANY_SCORE_RE = re.compile(r"\b([1-5])\b")


def _judge_input(original: str, defended: str) -> str:
    """The judge's user message: the two tagged conversations.

    Built by direct string join (not ``str.format``) so braces anywhere in the conversation text
    are never mangled -- the same rule :mod:`.answer_judge` follows.

    The original always comes first and the sides are never shuffled. The sibling LLM-judge attacks
    randomize presentation order to cancel position bias, but that does not apply here: the two
    sides play asymmetric roles (the original is the referent the defended version is measured
    against), so their order carries meaning and swapping them would change the question.
    """
    return (
        f"<original_conversation>\n{original}\n</original_conversation>\n"
        f"<defended_conversation>\n{defended}\n</defended_conversation>"
    )


def _parse_score(raw: str) -> tuple[int | None, str]:
    """Parse a judge reply into ``(score, reason)``; ``score`` is ``None`` when unparseable.

    Layered like :func:`prompt_anonymity.utility.answer_judge._parse_verdict`: strip a code fence,
    try JSON, then a labeled ``"Score": N`` field, then a bare digit.

    Two decisions worth stating:

    * A score outside 1-5 is **discarded, not clamped**. A reply of ``0`` or ``10`` means the judge
      ignored the rubric; clamping it to a valid value would invent a data point that no judge
      actually produced.
    * The bare-digit fallback takes the **first** ``[1-5]`` in the text, not the last. The rubric
      puts the score before the reason, and reasons routinely contain digits ("turn 3 lost the
      constraint"). This is the opposite of ``_parse_choice`` in the LLM-judge attacks, where the
      answer trails any reasoning -- same tier, different position, because the output shape differs.
    """
    text = strip_code_fence(raw)

    score: int | None = None
    reason = ""
    obj = json_object(text)
    if obj is not None:
        reason = str(obj.get("reason", "") or "")
        value = obj.get("score")
        if isinstance(value, bool):  # bool is an int subclass; a True here is not a score of 1
            value = None
        if isinstance(value, (int, float)):
            score = int(round(value))
        elif isinstance(value, str):
            match = _ANY_SCORE_RE.search(value)
            if match is not None:
                score = int(match.group(1))

    if score is None:  # JSON missing or garbled -> scan the raw text
        match = _SCORE_FIELD_RE.search(text) or _ANY_SCORE_RE.search(text)
        if match is not None:
            score = int(match.group(1))

    if score not in _VALID_SCORES:
        return None, f"UNPARSED: {text[:200]}"
    return score, reason


@dataclass
class ConversationUtilityResult(UtilityResult):
    """Conversation-level utility for one (original, defended) comparison.

    Attributes
    ----------
    mean_score : float
        Mean score over **parsed rows only**, in ``[1, 5]``. Rows the judge answered unparseably are
        excluded rather than scored 1: on a 1-5 scale, imputing the floor for what is usually a
        *formatting* failure moves the mean far more than it would move a pass rate, and makes "the
        judge misbehaved" indistinguishable from "the defense destroyed the conversation". Read it
        together with ``n_unparsed``; ``0.0`` when nothing parsed.
    usable_rate : float
        Fraction of parsed rows scoring at least :data:`USABLE_SCORE_THRESHOLD`.
    n : int
        Conversations scored (blank originals are skipped entirely and not counted).
    n_scored, n_unchanged, n_unparsed : int
        How many produced a usable score (including short-circuited 5s), how many were left
        untouched by the defense and short-circuited with no API call, and how many could not be
        parsed.
    score_counts : dict
        ``{1: count, ..., 5: count}``. Worth reading even when the mean looks fine: a bimodal 1/5
        split and a uniform 3 both average to 3 and say completely different things about a defense.
    sampled_from : int or None
        Full split size when ``limit`` was used, else ``None``.
    table : pandas.DataFrame
        Per-conversation detail (``conv_index, conv_id, original, defended, score, reason``).
    """

    mean_score: float
    usable_rate: float
    n: int
    n_scored: int
    n_unchanged: int
    n_unparsed: int
    score_counts: dict
    table: pd.DataFrame
    sampled_from: int | None = None

    #: Lower scores are worse, so a plain ascending sort already puts the worst rows first.
    sort_column: str = field(default="score", repr=False)

    def summary(self) -> str:
        if not self.n_scored:
            return (
                f"{self.sample_note()}Conversation utility  n={self.n}  "
                f"NO PARSED SCORES (unparsed={self.n_unparsed})"
            )
        dist = {s: self.score_counts.get(s, 0) for s in reversed(_VALID_SCORES)}
        return (
            f"{self.sample_note()}Conversation utility  n={self.n}  "
            f"mean={self.mean_score:.2f}  usable(>={USABLE_SCORE_THRESHOLD})={self.usable_rate:.4f}  "
            f"dist={dist}  (unchanged={self.n_unchanged}, unparsed={self.n_unparsed})"
        )


class ConversationUtility(UtilityMetric):
    """Score defense utility by showing a judge both whole conversations and asking for 1-5.

    The judge client is built lazily on first real need, so a fully cached run makes no API calls
    and needs no ``OPENROUTER_API_KEY``. The model, rubric, and truncation cap are part of the cache
    key, so changing any of them re-caches automatically.

    Parameters
    ----------
    judge_model : str
        OpenRouter model id for the judge.
    judge_system_prompt : str
        The rubric. Swap it to recalibrate; the cache follows.
    judge_max_tokens : int
        Output budget -- enough for the score plus a one-line reason.
    max_chars : int, optional
        Per-side character cap on the rendered conversation; see :data:`DEFAULT_MAX_CHARS`.
    """

    name = "conversation_judge"
    version = CONVERSATION_UTILITY_VERSION

    def __init__(self, *, judge_model: str = DEFAULT_CONVERSATION_JUDGE_MODEL,
                 judge_system_prompt: str = CONVERSATION_JUDGE_SYSTEM_PROMPT,
                 judge_max_tokens: int = DEFAULT_CONVERSATION_JUDGE_MAX_TOKENS,
                 max_chars: int | None = DEFAULT_MAX_CHARS):
        self.judge_model = judge_model
        self.judge_system_prompt = judge_system_prompt
        self.judge_max_tokens = judge_max_tokens
        self.max_chars = max_chars
        self._judge_client: OpenRouterChat | None = None

    def params(self) -> dict:
        # Everything here changes what the judge is sent. USABLE_SCORE_THRESHOLD is intentionally
        # absent: it only post-processes cached scores, so re-sweeping it should cost nothing.
        return {"judge_model": self.judge_model,
                "judge_system_prompt": self.judge_system_prompt,
                "max_chars": self.max_chars}

    def _judge(self, inputs: list[str]) -> list[str]:
        if self._judge_client is None:
            self._judge_client = OpenRouterChat(
                self.judge_model, self.judge_system_prompt, max_tokens=self.judge_max_tokens
            )
        return self._judge_client.complete_batch(inputs)

    def score(self, data, *, cache_dir, reference, side: str = "unknown",
              limit: int | None = None, seed: int = DEFAULT_SEED) -> ConversationUtilityResult:
        """Score the conversation utility of ``data`` (post-defense) against ``reference``.

        See :meth:`prompt_anonymity.utility.base.UtilityMetric.score` for the parameters. Scoring
        is per conversation: one judge call compares the whole original against the whole defended
        rewrite.
        """
        sides = self._load_sides(data, reference, side, limit=limit, seed=seed)
        ids = getattr(data, f"{side}_ids", None)

        # One unit per conversation. A blank original is skipped outright rather than scored --
        # there is nothing to preserve, so any score would be meaningless in either direction.
        units: list[tuple[int, object, str, str]] = []  # (conv_index, conv_id, original, defended)
        for position, row in enumerate(sides.indices):
            original, defended = sides.original[position], sides.defended[position]
            if not original.strip():
                continue
            conv_id = ids[row] if ids is not None else row
            units.append((row, conv_id, original, defended))
        n = len(units)

        # A conversation the defense left untouched trivially preserves everything, so short-circuit
        # it to 5 without an API call. This is what makes scoring `--defense none` free.
        scores: list[int | None] = [5] * n
        reasons: list[str] = ["unchanged conversation"] * n
        changed = [u for u, (_, _, original, defended) in enumerate(units) if defended != original]

        if changed:
            judge_inputs = [
                _judge_input(
                    render_turns(units[u][2], label=True, drop_blank=True, max_chars=self.max_chars),
                    render_turns(units[u][3], label=True, drop_blank=True, max_chars=self.max_chars),
                )
                for u in changed
            ]
            verdicts = self._cache(cache_dir).apply_batch(judge_inputs, self._judge)
            for k, u in enumerate(changed):
                scores[u], reasons[u] = _parse_score(verdicts[k])

        parsed = [s for s in scores if s is not None]
        n_unparsed = n - len(parsed)
        score_counts = {s: parsed.count(s) for s in _VALID_SCORES}
        mean_score = sum(parsed) / len(parsed) if parsed else 0.0
        usable = sum(1 for s in parsed if s >= USABLE_SCORE_THRESHOLD)
        usable_rate = usable / len(parsed) if parsed else 0.0

        table = pd.DataFrame({
            "conv_index": [u[0] for u in units],
            "conv_id": [u[1] for u in units],
            "original": [u[2] for u in units],
            "defended": [u[3] for u in units],
            "score": scores,          # None for unparsed -> sorts first in to_csv
            "reason": reasons,
        })
        return ConversationUtilityResult(
            mean_score=mean_score, usable_rate=usable_rate, n=n, n_scored=len(parsed),
            n_unchanged=n - len(changed), n_unparsed=n_unparsed, score_counts=score_counts,
            table=table, sampled_from=sides.sampled_from,
        )


def conversation_utility(data, *, cache_dir, reference, side: str = "unknown",
                          limit: int | None = None, seed: int = DEFAULT_SEED,
                          **kwargs) -> ConversationUtilityResult:
    """Convenience wrapper: score ``data`` vs. ``reference`` with a default
    :class:`ConversationUtility`.

    Mirrors :func:`prompt_anonymity.utility.answer_utility`. Any
    :class:`ConversationUtility` constructor argument (``judge_model``, ``judge_system_prompt``,
    ``judge_max_tokens``, ``max_chars``) may be passed through ``kwargs``.
    """
    return ConversationUtility(**kwargs).score(
        data, cache_dir=cache_dir, reference=reference, side=side, limit=limit, seed=seed
    )
