"""Conversation-level utility: how much of the original survives the defense, scored 1-5.

A linkage defense is only worth using if rewriting a prompt does not ruin what the user was trying
to get. This module measures that: a judge is shown the two **conversations** -- original and
defended, whole, in one call -- and asked for a 1-5 score. Utility is the mean over conversations,
with ``usable_rate`` (the share scoring at least :data:`USABLE_SCORE_THRESHOLD`) as the headline.

Judging the conversation as a unit, rather than turn by turn, is the point rather than a shortcut.
A defense can preserve every turn in isolation and still break the thread -- back-references that
stop resolving, a turn that vanishes -- and neither failure is visible inside a single turn. A 1-5
scale also separates a light paraphrase from a heavily degraded but still answerable rewrite, which
is what *ranking* defenses needs and what a PASS/FAIL predicate cannot give.

**The rubric lives in** ``conversation_judge.yaml`` **beside this module**, not in the Python: it is
the part of the metric most likely to need iterating, and tuning it means reading the judge's
worst-scoring reasons and rewording, which is a text edit. See that file's header for the three
invariants an edit has to preserve -- the short version is that the judge is never told what
produced the modification (a judge that knows a rewrite was meant to hide the author starts grading
*that*, which is the attacks' job), that changed wording and incidental specifics cost nothing, and
that anything the rewrite *adds* is outside the judgement entirely.

**Whether those instructions are landing is checkable, and worth checking**: sort
:attr:`ConversationUtilityResult.table` by ``score`` and read the ``reason`` column of the worst
rows. If the reasons complain that names were removed, that the voice changed, or that the rewrite
said more than it needed to, the rubric is not holding and the numbers should not be trusted yet.

Judging runs through :class:`~prompt_anonymity.evaluation.utility._deepseek.DeepSeekJudge` -- a concurrent
fan-out against a DeepSeek deployment over its OpenAI-compatible API.
Every verdict is cached by the package's content-addressed
:class:`~prompt_anonymity.caching.TransformCache` under ``<cache_dir>/utility/conversation_judge``,
and conversations the defense left untouched short-circuit to 5 with no API call, so scoring
``--defense none`` is free.

Caveats:

* **Prompts, not answers.** This measures semantic preservation of the request; it does not verify
  that a model still *answers* it well. A redaction the judge reads as minor could still sink the
  response.
* **Unbounded input.** Two whole conversations go into one request, so the input is not inherently
  bounded -- see ``max_chars``.
* **Absolute, not comparative.** Each conversation is scored on its own, so cross-defense
  comparisons inherit whatever bias the judge has about the scale. Check ``score_counts`` for
  compression before reading small differences as real.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from ._deepseek import (
    DEFAULT_JUDGE_MODEL as DEEPSEEK_DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DeepSeekJudge,
    JudgeUsage,
)
from ._parsing import json_object, render_turns, strip_code_fence
from .base import DEFAULT_SEED, UtilityMetric, UtilityResult

#: The rubric file shipped with the package. Read at import; override per run by passing
#: ``judge_system_prompt=`` (or ``eval_utility.py --judge-prompt <path>``).
JUDGE_PROMPT_FILE = Path(__file__).with_name("conversation_judge.yaml")


def load_judge_prompt(path=None) -> str:
    """The judge rubric from a YAML file with a ``system_prompt`` key.

    Failures are loud rather than defaulted: a missing file, unreadable YAML, or absent/blank key
    all raise. There is no sensible fallback -- a judge running without its rubric would return
    scores that look like measurements and are not, and every one of them would be paid for and
    then cached.
    """
    path = Path(path) if path is not None else JUDGE_PROMPT_FILE
    try:
        with open(path, encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
    except FileNotFoundError:
        raise FileNotFoundError(f"judge rubric not found at {path}.") from None
    except yaml.YAMLError as error:
        raise ValueError(f"judge rubric at {path} is not valid YAML: {error}") from None
    prompt = (document or {}).get("system_prompt") if isinstance(document, dict) else None
    if not (prompt or "").strip():
        raise ValueError(f"judge rubric at {path} has no non-empty 'system_prompt' key.")
    return str(prompt).strip()


#: Conversation-level utility rubric: scores 1-5 how much of the original's substance survives the
#: rewrite, judging the conversation as a whole. See :data:`JUDGE_PROMPT_FILE` for the text and the
#: invariants behind it.
CONVERSATION_JUDGE_SYSTEM_PROMPT = load_judge_prompt()

#: Default judge model / deployment name. EDIT ME (or pass ``judge_model=``) to retarget; it is part
#: of the cache key, so a swap re-caches cleanly instead of mixing two models' scores in one number.
DEFAULT_CONVERSATION_JUDGE_MODEL = DEEPSEEK_DEFAULT_MODEL

#: Reasoning budget for the judge, ``"low"`` since 2026-08-11 (on request, following DeepSeek's
#: thinking-mode guide). Part of the cache key. **On this deployment "low" turns reasoning on**,
#: because it was off without the field entirely -- see
#: :data:`~._deepseek.DEFAULT_REASONING_EFFORT` for the probe behind that and for why the level is
#: only a coarse dial here.
DEFAULT_JUDGE_REASONING_EFFORT = DEFAULT_REASONING_EFFORT

#: Sampling controls, both 1.0 since 2026-08-11 (on request). **Neither does anything while
#: reasoning is on** -- the API accepts them for compatibility and ignores them -- so 1.0 is the
#: neutral value rather than a claim about determinism. The previous 0.0 was chosen as a
#: reproducibility lever, and under thinking mode it was not one: **the response cache is now the
#: only thing that makes a re-run reproducible**, and two uncached runs of the same conversation
#: can disagree. Both stay in the cache key so a run under different settings does not silently
#: blend into one at the defaults.
DEFAULT_JUDGE_TEMPERATURE = DEFAULT_TEMPERATURE
DEFAULT_JUDGE_TOP_P = DEFAULT_TOP_P

#: Output budget. Larger than a one-line JSON verdict needs, because a model that reasons before
#: answering spends it too: an under-budgeted request comes back truncated (and unparseable) rather
#: than erroring, which would read as "the judge misbehaved" when it was a configuration slip.
DEFAULT_CONVERSATION_JUDGE_MAX_TOKENS = 8192

#: Conversations judged before results are written to the cache. This is a **checkpoint interval**,
#: not an API or concurrency limit -- concurrency is the client's ``max_workers``. A request that
#: fails after the SDK's retries aborts the chunk it is in, and everything already completed in
#: that chunk is lost with it (the cache is written per chunk, not per reply), so this bounds what
#: a failure can cost. Small because there is nothing to amortize: unlike a submitted batch, a
#: fan-out has no per-chunk overhead to pay for a larger one.
DEFAULT_JUDGE_CHUNK_SIZE = 50

#: Per-side character cap on the conversation shown to the judge, or ``None`` for no cap. Two whole
#: conversations plus the rubric can overrun a context window; with a 1M-token window that is
#: unlikely, so the default is off. Set it if a split has pathologically long conversations.
DEFAULT_MAX_CHARS = None

#: Score at or above which a conversation counts as "defense preserved it" in ``usable_rate``.
#: Deliberately NOT part of the cache key: it only post-processes an already cached score, so
#: re-sweeping the threshold costs nothing.
USABLE_SCORE_THRESHOLD = 4

#: Manual logic version for the conversation-judge cache; bump to force a full recompute. Since
#: :meth:`UtilityMetric.logic_classes` is empty, this is the *only* automatic invalidation lever
#: besides :meth:`params`. ``"3"`` marks the move to DeepSeek over the OpenAI-compatible API
#: (``"2"`` was the Anthropic Foundry judge, ``"1"`` OpenRouter).
CONVERSATION_UTILITY_VERSION = "3"

#: Valid scores, and the labeled-field fallback for a reply whose JSON did not parse.
_VALID_SCORES = (1, 2, 3, 4, 5)
_SCORE_FIELD_RE = re.compile(r'"?score"?\s*:\s*"?\s*([1-5])', re.IGNORECASE)
_ANY_SCORE_RE = re.compile(r"\b([1-5])\b")


def _judge_input(original: str, defended: str) -> str:
    """The judge's user message: the two tagged conversations.

    Built by direct string join (not ``str.format``) so braces anywhere in the conversation text
    are never mangled.

    **The defended side is tagged ``<modified_conversation>``, not "defended".** Nothing the judge
    reads names what produced the rewrite -- see the rubric file's first invariant. The parameter
    keeps the package's own vocabulary because that is what it is; only the wire format is neutral.

    The original always comes first and the sides are never shuffled. The sibling LLM-judge attacks
    randomize presentation order to cancel position bias, but that does not apply here: the two
    sides play asymmetric roles (the original is the referent the other version is measured
    against), so their order carries meaning and swapping them would change the question.
    """
    return (
        f"<original_conversation>\n{original}\n</original_conversation>\n"
        f"<modified_conversation>\n{defended}\n</modified_conversation>"
    )


def _parse_score(raw: str) -> tuple[int | None, str]:
    """Parse a judge reply into ``(score, reason)``; ``score`` is ``None`` when unparseable.

    Layered: strip a code fence, try JSON, then a labeled ``"Score": N`` field, then a bare digit.

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
        Per-conversation detail (``conv_id, score, reason, cost_usd``).
    usage : JudgeUsage
        What this run actually spent at the API, in total. Note it counts **requests made, not
        conversations scored**: a verdict served from the cache and a conversation the defense left
        unchanged both cost nothing, so a fully cached re-run reports zero requests -- which is the
        correct record of what *that* run spent. The same spend broken out per conversation is
        ``table``'s ``cost_usd``.
    judge_model : str
        Model the requests were billed against; kept so the token counts can be re-priced later
        without guessing which model produced them.
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
    usage: JudgeUsage = field(default_factory=JudgeUsage)
    judge_model: str = ""

    #: The judge's verdict, and what it cost to get it. Deliberately unannotated, so ``@dataclass``
    #: leaves it a class attribute rather than making it a constructor argument. ``reason`` stays
    #: off the score file: it is a paragraph per row, and the file is meant to hold numbers a
    #: defense can be compared on -- read reasons off ``table`` when calibrating the rubric.
    score_columns = {"judge_score": "score", "judge_cost_usd": "cost_usd"}

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
    and needs no credentials. The model, rubric, temperature, and truncation cap are all part of
    the cache key, so changing any of them re-caches automatically rather than mixing two
    configurations' scores into one mean.

    Parameters
    ----------
    judge_model : str
        Model / deployment name for the judge.
    judge_system_prompt : str
        The rubric. Swap it to recalibrate; the cache follows.
    judge_temperature, judge_top_p : float
        Sampling controls; see :data:`DEFAULT_JUDGE_TEMPERATURE` (both are ignored by the API
        while reasoning is on, and are kept in the cache key rather than dropped).
    judge_reasoning_effort : str or None
        How much the judge thinks before answering; see
        :data:`DEFAULT_JUDGE_REASONING_EFFORT`.
    judge_max_tokens : int
        Output budget per request, shared between the reasoning and the verdict.
    max_chars : int, optional
        Per-side character cap on the rendered conversation; see :data:`DEFAULT_MAX_CHARS`.
    chunk_size : int
        Conversations judged before results are written to the cache (see
        :data:`DEFAULT_JUDGE_CHUNK_SIZE`).
    """

    name = "conversation_judge"
    version = CONVERSATION_UTILITY_VERSION

    def __init__(self, *, judge_model: str = DEFAULT_CONVERSATION_JUDGE_MODEL,
                 judge_system_prompt: str = CONVERSATION_JUDGE_SYSTEM_PROMPT,
                 judge_temperature: float = DEFAULT_JUDGE_TEMPERATURE,
                 judge_top_p: float = DEFAULT_JUDGE_TOP_P,
                 judge_reasoning_effort: str | None = DEFAULT_JUDGE_REASONING_EFFORT,
                 judge_max_tokens: int = DEFAULT_CONVERSATION_JUDGE_MAX_TOKENS,
                 max_chars: int | None = DEFAULT_MAX_CHARS,
                 chunk_size: int = DEFAULT_JUDGE_CHUNK_SIZE):
        self.judge_model = judge_model
        self.judge_system_prompt = judge_system_prompt
        self.judge_temperature = judge_temperature
        self.judge_top_p = judge_top_p
        self.judge_reasoning_effort = judge_reasoning_effort
        self.judge_max_tokens = judge_max_tokens
        self.max_chars = max_chars
        self.chunk_size = chunk_size
        self._judge_client: DeepSeekJudge | None = None

    def params(self) -> dict:
        # Everything here changes what the judge is sent or how it samples. The sampling pair is
        # included even though thinking mode ignores it: a cache key describes the configuration a
        # verdict was bought under, and "the API ignored it" is a fact about today's deployment,
        # not a promise. Two knobs are intentionally absent: USABLE_SCORE_THRESHOLD
        # (post-processes a cached score, so re-sweeping it should cost nothing) and chunk_size (a
        # checkpoint interval -- it changes how the work is submitted, never what any one request
        # contains).
        return {"judge_model": self.judge_model,
                "judge_system_prompt": self.judge_system_prompt,
                "judge_temperature": self.judge_temperature,
                "judge_top_p": self.judge_top_p,
                "judge_reasoning_effort": self.judge_reasoning_effort,
                "max_chars": self.max_chars}

    def _judge(self, inputs: list[str]) -> list[str]:
        if self._judge_client is None:
            self._judge_client = DeepSeekJudge(
                self.judge_model, self.judge_system_prompt,
                max_tokens=self.judge_max_tokens, temperature=self.judge_temperature,
                top_p=self.judge_top_p, reasoning_effort=self.judge_reasoning_effort,
            )
        return self._judge_client.complete_batch(inputs)

    def score(self, data, *, cache_dir, reference, side: str = "unknown",
              limit: int | None = None, seed: int = DEFAULT_SEED) -> ConversationUtilityResult:
        """Score the conversation utility of ``data`` (post-defense) against ``reference``.

        See :meth:`prompt_anonymity.evaluation.utility.base.UtilityMetric.score` for the parameters. Scoring
        is per conversation: one judge request compares the whole original against the whole
        defended rewrite.
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
        # What each conversation cost *this run*. Unchanged ones cost nothing and never will;
        # judged ones are filled in below from the request they caused.
        costs: list[float] = [0.0] * n
        changed = [u for u, (_, _, original, defended) in enumerate(units) if defended != original]

        if changed:
            judge_inputs = [
                _judge_input(
                    render_turns(units[u][2], label=True, drop_blank=True, max_chars=self.max_chars),
                    render_turns(units[u][3], label=True, drop_blank=True, max_chars=self.max_chars),
                )
                for u in changed
            ]
            # Judge in chunks so each chunk's verdicts are cached before the next starts: a run
            # that dies partway keeps everything already paid for, and re-running resumes.
            cache = self._cache(cache_dir)
            verdicts: list[str] = []
            cached_count = 0
            step = max(1, self.chunk_size)
            for start in range(0, len(judge_inputs), step):
                chunk = judge_inputs[start:start + step]
                verdicts.extend(cache.apply_batch(chunk, self._judge))
                cached_count += cache.hits
            print(f"Utility: {len(judge_inputs):,} changed conversations "
                  f"({cached_count:,} served from cache, "
                  f"{len(judge_inputs) - cached_count:,} judged)")

            for k, u in enumerate(changed):
                scores[u], reasons[u] = _parse_score(verdicts[k])
                # Zero for a verdict this run read back from the cache: the request that paid for
                # it belongs to the run that made it, whose figure is already in the score file.
                if self._judge_client is not None:
                    costs[u] = self._judge_client.request_cost(judge_inputs[k])

        parsed = [s for s in scores if s is not None]
        n_unparsed = n - len(parsed)
        score_counts = {s: parsed.count(s) for s in _VALID_SCORES}
        mean_score = sum(parsed) / len(parsed) if parsed else 0.0
        usable = sum(1 for s in parsed if s >= USABLE_SCORE_THRESHOLD)
        usable_rate = usable / len(parsed) if parsed else 0.0

        # The conversations themselves are deliberately not carried here: they are the bulk of the
        # dataset, they are already on disk in the parquet pair this was loaded from, and nothing
        # downstream reads them back. `reason` stays because it is what a rubric is calibrated on.
        table = pd.DataFrame({
            "conv_id": [u[1] for u in units],
            "score": scores,          # None where the reply could not be parsed
            "reason": reasons,
            "cost_usd": costs,
        })
        # A fully cached run never builds a client, so there is nothing to read -- report the empty
        # tally rather than nothing, since "this run spent zero" is itself the record.
        usage = self._judge_client.usage if self._judge_client is not None else JudgeUsage()
        return ConversationUtilityResult(
            mean_score=mean_score, usable_rate=usable_rate, n=n, n_scored=len(parsed),
            n_unchanged=n - len(changed), n_unparsed=n_unparsed, score_counts=score_counts,
            table=table, sampled_from=sides.sampled_from,
            usage=usage, judge_model=self.judge_model,
        )


def conversation_utility(data, *, cache_dir, reference, side: str = "unknown",
                          limit: int | None = None, seed: int = DEFAULT_SEED,
                          **kwargs) -> ConversationUtilityResult:
    """Convenience wrapper: score ``data`` vs. ``reference`` with a default
    :class:`ConversationUtility`.

    Mirrors :func:`prompt_anonymity.defenses.apply_defense`. Any :class:`ConversationUtility`
    constructor argument (``judge_model``, ``judge_system_prompt``, ``judge_temperature``,
    ``judge_max_tokens``, ``max_chars``, ``chunk_size``) may be passed through ``kwargs``.
    """
    return ConversationUtility(**kwargs).score(
        data, cache_dir=cache_dir, reference=reference, side=side, limit=limit, seed=seed
    )
