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

Judging runs through one of two :data:`JUDGE_BACKENDS`, both a concurrent fan-out over an
OpenAI-compatible API: ``deepseek`` (the default -- hosted and billed) or ``local`` (a self-hosted
vLLM server, free). The backend changes where a request goes and what it costs; the rubric, input
format and parser are shared.
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

import json
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
    OpenAICompatibleJudge,
    TokenRates,
    unpack_reply,
)
from ._vllm_judge import (
    DEFAULT_LOCAL_REASONING_EFFORT,
    DEFAULT_LOCAL_TEMPERATURE,
    DEFAULT_LOCAL_TOP_P,
    VLLMJudge,
    local_base_url,
    served_model_name,
)
from ._parsing import json_object, render_turn_pairs, strip_code_fence
from ...defenses._backends import split_turns
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

#: Reasoning budget for the judge. Part of the cache key. **On this deployment "low" turns
#: reasoning on**, since it is off entirely without the field -- see
#: :data:`~._deepseek.DEFAULT_REASONING_EFFORT`.
DEFAULT_JUDGE_REASONING_EFFORT = DEFAULT_REASONING_EFFORT

#: Sampling controls. **Neither does anything while reasoning is on** -- the API accepts them for
#: compatibility and ignores them -- so these are the neutral values rather than a claim about
#: determinism. **The response cache is the only thing that makes a re-run reproducible**; two
#: uncached runs of the same conversation can disagree. Both stay in the cache key so a run under
#: different settings does not silently blend into one at the defaults.
DEFAULT_JUDGE_TEMPERATURE = DEFAULT_TEMPERATURE
DEFAULT_JUDGE_TOP_P = DEFAULT_TOP_P

#: Valid scores -- the rubric's 1-5 scale.
_VALID_SCORES = (1, 2, 3, 4, 5)

#: JSON schema for a verdict, for backends that can constrain decoding to one
#: (:attr:`JudgeBackend.structured_output`). It mirrors the rubric's own output line, field for
#: field, **in generation order**:
#:
#: * ``Turns`` -- one entry per turn that has an ORIGINAL, each ``{Turn, Chain_of_thought, Score}``:
#:   the judge rates the conversation **turn by turn** (the ``judge_turn_scores`` column), writing
#:   its analysis of a turn before that turn's score.
#: * ``Connections`` -- the cross-turn check (rubric Step 2).
#: * ``Score`` -- the overall verdict, written last so it is conditioned on everything above.
#: * ``Reason`` -- one line.
#:
#: The per-turn analyses and ``Connections`` together are the chain of thought kept in the
#: ``judge_reasoning`` column; they replace the model's hidden thinking (the local judge runs with
#: ``reasoning_effort="none"``).
#:
#: **Why a schema, and why thinking is off.** Without a schema, an unconstrained reply drifts to
#: prose on the wrong scale and becomes hard to parse reliably. With a schema but thinking left on,
#: the model's hidden reasoning can land on a different scale (e.g. reasoning "9/10") than the
#: constrained output enum allows, silently truncating it. Written reasoning next to a restated
#: scale puts the scale where the model is actually writing.
#:
#: **Constrained decoding does not show the schema to the model** -- it only masks tokens -- so a
#: ``description`` alone is never read. :data:`JUDGE_OUTPUT_INSTRUCTION` appends the schema to the
#: system prompt for that reason.
_SCALE_REMINDER = (
    "An integer on the 1-5 scale, NOT out of 10: 5 = fully faithful (only surface wording or "
    "incidental specifics changed), 4 = substantially faithful (a minor detail blurred), "
    "3 = partially faithful (a meaningful requirement or context lost), 2 = largely unfaithful "
    "(a core request missing or changed), 1 = unusable."
)
JUDGE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "utility_verdict",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "Turns": {
                    "type": "array",
                    "description": (
                        "One entry per turn that has an ORIGINAL (a non-empty <original>), in "
                        "order. ADDED turns (empty <original>) get no entry."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "Turn": {"type": "integer",
                                     "description": "k, from the <turn_k> tag."},
                            "Chain_of_thought": {
                                "type": "string",
                                "description": (
                                    "Written BEFORE this turn's score: what the <original> asked "
                                    "or conveyed, and whether that survives in the <modified>. "
                                    "Changed wording, tone, incidental specifics and anything "
                                    "added do not count against it."
                                ),
                            },
                            "Score": {"type": "integer", "enum": list(_VALID_SCORES),
                                      "description": "This turn alone. " + _SCALE_REMINDER},
                        },
                        "required": ["Turn", "Chain_of_thought", "Score"],
                        "additionalProperties": False,
                    },
                },
                "Connections": {
                    "type": "string",
                    "description": (
                        "Whether references and the user's arc across ORIGINAL turns still hold "
                        "in the MODIFIED version. ADDED turns do not count, whatever they say."
                    ),
                },
                "Score": {"type": "integer", "enum": list(_VALID_SCORES),
                          "description": "The whole conversation, from the turn scores and the "
                                         "connections. " + _SCALE_REMINDER},
                "Reason": {"type": "string", "description": "One-line explanation of the score."},
            },
            "required": ["Turns", "Connections", "Score", "Reason"],
            "additionalProperties": False,
        },
    },
}

#: Appended to the rubric for a backend with :attr:`JudgeBackend.structured_output`, so the model
#: actually reads the schema it is constrained to (see :data:`JUDGE_RESPONSE_FORMAT`), scale
#: reminders included.
JUDGE_OUTPUT_INSTRUCTION = (
    "OUTPUT FORMAT. Return one JSON object matching this JSON schema, fields in this order. Every "
    "Score is on the 1-5 scale defined above, never out of 10:\n"
    + json.dumps(JUDGE_RESPONSE_FORMAT["json_schema"]["schema"], indent=2)
)


@dataclass(frozen=True)
class JudgeBackend:
    """Where judge requests go, and the settings a judge there defaults to.

    ``default_model`` is ``None`` when the model is read off the server instead
    (:func:`~._vllm_judge.served_model_name`). The sampling defaults are per backend because what
    an endpoint does with them is: DeepSeek ignores ``temperature``/``top_p`` while reasoning,
    vLLM honours them. ``structured_output`` sends :data:`JUDGE_RESPONSE_FORMAT` and appends
    :data:`JUDGE_OUTPUT_INSTRUCTION` to the rubric. ``record_reasoning`` also keeps the model's
    hidden thinking trace, if it produced one, as the ``judge_reasoning`` column when the reply
    carries no ``Chain_of_thought`` field; off for DeepSeek only because turning it on would re-key
    every verdict already paid for.
    """

    client: type
    default_model: str | None
    temperature: float
    top_p: float
    reasoning_effort: str | None
    structured_output: bool = False
    record_reasoning: bool = False
    #: Skip conversations whose judge input exceeds the server's context (see
    #: :meth:`ConversationUtility.score`). Needs a client that can count exact prompt tokens --
    #: :class:`~._vllm_judge.VLLMJudge` asks its server; DeepSeek has no such endpoint.
    checks_context: bool = False


#: Judge backends, selectable with ``judge_backend=`` / ``eval_utility.py --judge-backend``.
#: ``deepseek`` is the default and **stays out of the cache key** (see
#: :meth:`ConversationUtility.params`), so every verdict bought before the second backend existed
#: is still found.
JUDGE_BACKENDS: dict[str, JudgeBackend] = {
    "deepseek": JudgeBackend(DeepSeekJudge, DEEPSEEK_DEFAULT_MODEL, DEFAULT_TEMPERATURE,
                             DEFAULT_TOP_P, DEFAULT_REASONING_EFFORT),
    "local": JudgeBackend(VLLMJudge, None, DEFAULT_LOCAL_TEMPERATURE, DEFAULT_LOCAL_TOP_P,
                          DEFAULT_LOCAL_REASONING_EFFORT, structured_output=True,
                          record_reasoning=True, checks_context=True),
}
DEFAULT_JUDGE_BACKEND = "deepseek"

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
#: besides :meth:`params`.
CONVERSATION_UTILITY_VERSION = "3"

#: The labeled-field fallback for a reply whose JSON did not parse. ``(?!\d)`` so a ``"Score": 10``
#: is not read as a 1.
_SCORE_FIELD_RE = re.compile(r'"?score"?\s*:\s*"?\s*([1-5])(?!\d)', re.IGNORECASE)
_ANY_SCORE_RE = re.compile(r"\b([1-5])\b")


def _judge_input(original: str, defended: str, max_chars: int | None = None) -> str:
    """The judge's user message: the two versions as aligned turn pairs
    (:func:`~._parsing.render_turn_pairs`), ``<turn_k><original>...</original>
    <modified>...</modified></turn_k>``.

    Paired in code rather than left for the judge to align, so the rubric's turn-by-turn procedure
    (invariant 4) can read each pair directly.

    **The defended side is tagged ``<modified>``, not "defended".** Nothing the judge reads names
    what produced the rewrite -- see the rubric file's first invariant. The parameter keeps the
    package's own vocabulary because that is what it is; only the wire format is neutral.

    Within a pair the original always comes first and the sides are never shuffled. The sibling
    LLM-judge attacks randomize presentation order to cancel position bias, but that does not apply
    here: the two sides play asymmetric roles (the original is the referent the other version is
    measured against), so their order carries meaning and swapping them would change the question.
    """
    return render_turn_pairs(original, defended, max_chars=max_chars)


def _original_turns_intact(original: str, defended: str) -> bool:
    """Whether every ORIGINAL turn survives **byte-for-byte**, in order, as the defended version's
    leading turns -- i.e. the defense changed nothing and at most appended turns after them.

    Such a conversation has lost nothing the original conveyed, so it scores 5 without asking the
    judge: added turns are outside the judgement (rubric invariant 3), and an added turn cannot
    reach back and change an earlier one (invariant 4). Decided in code rather than by the rubric,
    because a judge does not reliably hold that line against an adversarial addition that claims
    the earlier turns are void.
    """
    original_turns = split_turns(original)
    return split_turns(defended)[:len(original_turns)] == original_turns


@dataclass
class ParsedVerdict:
    """One judge reply, parsed. ``score`` is ``None`` when the overall score is unparseable."""

    score: int | None
    reason: str
    reasoning: str | None = None
    turn_scores: list[int | None] | None = None


def _turn_reasoning(turns: list, connections) -> str | None:
    """The per-turn analyses and the connections check as one readable block."""
    lines = []
    for entry in turns:
        if isinstance(entry, dict):
            entry = {str(k).lower(): v for k, v in entry.items()}
            lines.append(f"turn_{entry.get('turn', '?')} [score {entry.get('score', '?')}]: "
                         f"{entry.get('chain_of_thought', '') or ''}")
    if connections:
        lines.append(f"connections: {connections}")
    return "\n".join(lines) or None


def _turn_score(entry) -> int | None:
    """One ``Turns`` entry's score, or ``None`` when absent or outside 1-5."""
    if not isinstance(entry, dict):
        return None
    value = {str(k).lower(): v for k, v in entry.items()}.get("score")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = int(round(value))
    return value if value in _VALID_SCORES else None


def _parse_score(raw: str) -> ParsedVerdict:
    """Parse a judge reply (the rubric's per-turn JSON) into a :class:`ParsedVerdict`.

    Layered: strip a code fence and read the JSON; failing that, take the **last** labeled
    ``"Score": N`` in the text. Last, because the overall score follows the per-turn ones in the
    requested format, so the first match would be turn 1's.

    Decisions worth stating:

    * A score outside 1-5 is **discarded, not clamped**. A reply of ``0`` or ``10`` means the judge
      ignored the rubric; clamping it to a valid value would invent a data point that no judge
      actually produced.
    * **There is no bare-digit fallback.** With the input structured as ``<turn_k>`` and replies
      discussing turns by number, a free digit is far more often a turn index than a score. An
      unlabeled reply is reported as unparsed, which is visible, rather than guessed.
    """
    text = strip_code_fence(raw)

    score: int | None = None
    reason = ""
    reasoning: str | None = None
    turn_scores: list[int | None] | None = None
    obj = json_object(text)
    if obj is not None:
        reason = str(obj.get("reason", "") or "")
        turns = obj.get("turns")
        if isinstance(turns, list):
            turn_scores = [_turn_score(entry) for entry in turns]
            reasoning = _turn_reasoning(turns, obj.get("connections"))
        else:  # an older single-verdict reply
            reasoning = obj.get("chain_of_thought") or None
        value = obj.get("score")
        if isinstance(value, bool):  # bool is an int subclass; a True here is not a score of 1
            value = None
        if isinstance(value, (int, float)):
            score = int(round(value))
        elif isinstance(value, str):
            match = _ANY_SCORE_RE.search(value)
            if match is not None:
                score = int(match.group(1))

    if score is None:  # JSON missing or garbled -> the last labeled score in the raw text
        matches = _SCORE_FIELD_RE.findall(text)
        if matches:
            score = int(matches[-1])

    if score not in _VALID_SCORES:
        return ParsedVerdict(None, f"UNPARSED: {text[:200]}", reasoning, turn_scores)
    return ParsedVerdict(score, reason, reasoning, turn_scores)


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
        How many produced a usable score (including short-circuited 5s), how many had every
        original turn left untouched by the defense -- identical, or with turns only appended after
        them (:func:`_original_turns_intact`) -- and short-circuited with no API call, and how many
        could not be parsed.
    score_counts : dict
        ``{1: count, ..., 5: count}``. Worth reading even when the mean looks fine: a bimodal 1/5
        split and a uniform 3 both average to 3 and say completely different things about a defense.
    sampled_from : int or None
        Full split size when ``limit`` was used, else ``None``.
    table : pandas.DataFrame
        Per-conversation detail (``conv_id, score, reason, reasoning, turn_scores, cost_usd``).
        ``turn_scores`` is a JSON list of the judge's per-turn 1-5 ratings, one per ORIGINAL turn.
    usage : JudgeUsage
        What this run actually spent at the API, in total. Note it counts **requests made, not
        conversations scored**: a verdict served from the cache and a conversation the defense left
        unchanged both cost nothing, so a fully cached re-run reports zero requests -- which is the
        correct record of what *that* run spent. The same spend broken out per conversation is
        ``table``'s ``cost_usd``.
    judge_model : str
        Model the requests were billed against; kept so the token counts can be re-priced later
        without guessing which model produced them.
    n_skipped : int
        Conversations not judged because their input exceeded the judge's context (local backend
        only). Their ``score`` is ``None`` and they are excluded from every aggregate, like an
        unparsed row -- but counted apart, since a skip is a length fact, not a judge failure.
    judge_rates : TokenRates or None
        The per-token prices this run's spend was computed at -- zero for a self-hosted backend,
        ``None`` when unknown or when no client was built (a fully cached run).
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
    judge_rates: TokenRates | None = None
    n_skipped: int = 0

    #: The judge's verdict, what it cost, and why. Deliberately unannotated, so ``@dataclass``
    #: leaves it a class attribute rather than making it a constructor argument. The one-line
    #: ``reason`` and the chain of thought (``reasoning``; empty for a backend that does not record
    #: it, or for an unchanged conversation that was never judged) ride in the score file so a
    #: suspicious score can be audited in place.
    score_columns = {"judge_score": "score", "judge_cost_usd": "cost_usd",
                     "judge_turn_scores": "turn_scores",
                     "judge_reason": "reason", "judge_reasoning": "reasoning"}

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
            f"dist={dist}  (unchanged={self.n_unchanged}, unparsed={self.n_unparsed}, "
            f"skipped as too long={self.n_skipped})"
        )


class ConversationUtility(UtilityMetric):
    """Score defense utility by showing a judge both whole conversations and asking for 1-5.

    The judge client is built lazily on first real need, so a fully cached run makes no API calls
    and needs no credentials. The model, rubric, temperature, and truncation cap are all part of
    the cache key, so changing any of them re-caches automatically rather than mixing two
    configurations' scores into one mean.

    Parameters
    ----------
    judge_backend : str
        A key of :data:`JUDGE_BACKENDS`: ``"deepseek"`` (default, hosted, billed) or ``"local"``
        (a self-hosted vLLM server, free). The settings below that are left ``None`` take this
        backend's defaults.
    judge_model : str, optional
        Model / deployment name for the judge. For ``"local"``, ``None`` asks the server which
        model it serves -- which needs the server up even for a fully cached run.
    judge_base_url : str, optional
        ``"local"`` only: the server's ``/v1`` URL (default :func:`~._vllm_judge.local_base_url`).
        Not in the cache key -- the model name is what identifies the judge, not its address.
    judge_system_prompt : str
        The rubric. Swap it to recalibrate; the cache follows.
    judge_temperature, judge_top_p : float, optional
        Sampling controls; see :data:`DEFAULT_JUDGE_TEMPERATURE` (DeepSeek ignores both while
        reasoning is on; vLLM honours them). Kept in the cache key either way.
    judge_reasoning_effort : str or None, optional
        How much the judge thinks before answering; see :data:`DEFAULT_JUDGE_REASONING_EFFORT`
        and, for the local server, :mod:`._vllm_judge`.
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

    def __init__(self, *, judge_backend: str = DEFAULT_JUDGE_BACKEND,
                 judge_model: str | None = None,
                 judge_base_url: str | None = None,
                 judge_system_prompt: str = CONVERSATION_JUDGE_SYSTEM_PROMPT,
                 judge_temperature: float | None = None,
                 judge_top_p: float | None = None,
                 judge_reasoning_effort: str | None = None,
                 judge_max_tokens: int = DEFAULT_CONVERSATION_JUDGE_MAX_TOKENS,
                 max_chars: int | None = DEFAULT_MAX_CHARS,
                 chunk_size: int = DEFAULT_JUDGE_CHUNK_SIZE):
        if judge_backend not in JUDGE_BACKENDS:
            raise ValueError(f"unknown judge backend {judge_backend!r}; "
                             f"available: {sorted(JUDGE_BACKENDS)}")
        if judge_base_url is not None and judge_backend != "local":
            raise ValueError("judge_base_url only applies to judge_backend='local'; the DeepSeek "
                             "endpoint comes from DEEPSEEK_BASE_URL.")
        backend = JUDGE_BACKENDS[judge_backend]
        self.judge_backend = judge_backend
        self.judge_base_url = judge_base_url
        if judge_backend == "local":
            self.judge_base_url = judge_base_url or local_base_url()
        # The model is resolved *now*, not when the client is built, because it is in the cache
        # key: a local judge must name the model the server really runs before any lookup.
        self.judge_model = (judge_model or backend.default_model
                            or served_model_name(self.judge_base_url))
        self.judge_system_prompt = judge_system_prompt
        self.judge_temperature = (backend.temperature if judge_temperature is None
                                  else judge_temperature)
        self.judge_top_p = backend.top_p if judge_top_p is None else judge_top_p
        self.judge_reasoning_effort = (backend.reasoning_effort if judge_reasoning_effort is None
                                       else judge_reasoning_effort)
        self.judge_max_tokens = judge_max_tokens
        self.max_chars = max_chars
        self.chunk_size = chunk_size
        self._judge_client: OpenAICompatibleJudge | None = None

    def params(self) -> dict:
        # Everything here changes what the judge is sent or how it samples. The sampling pair is
        # included even though thinking mode ignores it: a cache key describes the configuration a
        # verdict was bought under, and "the API ignored it" is a fact about today's deployment,
        # not a promise. Two knobs are intentionally absent: USABLE_SCORE_THRESHOLD
        # (post-processes a cached score, so re-sweeping it should cost nothing) and chunk_size (a
        # checkpoint interval -- it changes how the work is submitted, never what any one request
        # contains).
        params = {"judge_model": self.judge_model,
                  "judge_system_prompt": self.judge_system_prompt,
                  "judge_temperature": self.judge_temperature,
                  "judge_top_p": self.judge_top_p,
                  "judge_reasoning_effort": self.judge_reasoning_effort,
                  "max_chars": self.max_chars}
        # The backend joins the key only when it is not the default, so the DeepSeek key -- and
        # with it every verdict already paid for -- is byte-identical to before backends existed.
        if self.judge_backend != DEFAULT_JUDGE_BACKEND:
            params["judge_backend"] = self.judge_backend
        if JUDGE_BACKENDS[self.judge_backend].structured_output:
            params["response_format"] = JUDGE_RESPONSE_FORMAT
        # Changes what is cached (an envelope, not the bare reply), so it must key the namespace.
        if JUDGE_BACKENDS[self.judge_backend].record_reasoning:
            params["record_reasoning"] = True
        return params

    def _system_prompt_sent(self) -> str:
        """The rubric, plus :data:`JUDGE_OUTPUT_INSTRUCTION` for a schema-constrained backend.

        Not a separate cache-key entry: it is a function of the rubric and the schema, which
        :meth:`params` already carries.
        """
        if JUDGE_BACKENDS[self.judge_backend].structured_output:
            return f"{self.judge_system_prompt}\n\n{JUDGE_OUTPUT_INSTRUCTION}"
        return self.judge_system_prompt

    def _judge(self, inputs: list[str]) -> list[str]:
        return self._client().complete_batch(inputs)

    def _client(self) -> OpenAICompatibleJudge:
        """The judge client, built on first use. A fully cached DeepSeek run never builds one; a
        local run always does, since the context check below asks the server to count tokens."""
        if self._judge_client is None:
            endpoint = {"base_url": self.judge_base_url} if self.judge_backend == "local" else {}
            backend = JUDGE_BACKENDS[self.judge_backend]
            if backend.structured_output:
                endpoint["response_format"] = JUDGE_RESPONSE_FORMAT
            endpoint["record_reasoning"] = backend.record_reasoning
            self._judge_client = JUDGE_BACKENDS[self.judge_backend].client(
                self.judge_model, self._system_prompt_sent(),
                max_tokens=self.judge_max_tokens, temperature=self.judge_temperature,
                top_p=self.judge_top_p, reasoning_effort=self.judge_reasoning_effort, **endpoint,
            )
        return self._judge_client

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

        # A conversation whose original turns the defense left untouched trivially preserves
        # everything, so short-circuit it to 5 without an API call -- whether it is identical (which
        # is what makes scoring `--defense none` free) or only had turns appended after them.
        scores: list[int | None] = [5] * n
        reasons: list[str] = [
            "unchanged conversation" if defended == original
            else "original turns unchanged; turns only appended"
            for (_, _, original, defended) in units
        ]
        # "" rather than None where there is nothing to record, so a fresh row always replaces the
        # score file's previous reasoning instead of letting a stale one survive the merge.
        reasonings: list[str] = [""] * n
        # One score per ORIGINAL turn, as a JSON list. A short-circuited conversation kept every
        # original turn, so each of its (non-blank) turns is a 5.
        turn_scores: list[str] = [
            json.dumps([5] * sum(1 for turn in split_turns(original) if turn.strip()))
            for (_, _, original, _) in units
        ]
        # What each conversation cost *this run*. Unchanged ones cost nothing and never will;
        # judged ones are filled in below from the request they caused.
        costs: list[float] = [0.0] * n
        changed = [u for u, (_, _, original, defended) in enumerate(units)
                   if not _original_turns_intact(original, defended)]

        # Conversations too long for the judge's context are skipped rather than judged: their
        # score stays None (reported as n_skipped, never folded into the mean) and nothing is
        # cached for them, so a server started with a longer context picks them up next run.
        # Checked before the cache on purpose -- a skip is a fact about today's server, not about
        # the conversation, and must not be stored as if it were a verdict.
        n_skipped = 0
        if changed and JUDGE_BACKENDS[self.judge_backend].checks_context:
            client = self._client()
            budget = client.input_budget()
            candidates = [_judge_input(units[u][2], units[u][3], max_chars=self.max_chars)
                          for u in changed]
            counts = client.input_token_counts(candidates)
            fits = []
            for u, tokens in zip(changed, counts):
                if tokens > budget:
                    scores[u] = None
                    reasons[u] = (f"SKIPPED: judge input is {tokens:,} tokens, over the "
                                  f"{budget:,}-token budget (context minus max_tokens)")
                    turn_scores[u] = ""
                    n_skipped += 1
                else:
                    fits.append(u)
            if n_skipped:
                print(f"Utility: skipped {n_skipped:,} of {len(changed):,} changed conversations "
                      f"as too long for the judge ({budget:,}-token input budget)")
            changed = fits

        if changed:
            judge_inputs = [
                _judge_input(units[u][2], units[u][3], max_chars=self.max_chars)
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
                answer, thinking = unpack_reply(verdicts[k])
                verdict = _parse_score(answer)
                scores[u], reasons[u] = verdict.score, verdict.reason
                # The schema's written-out analysis when there is one, else the hidden thinking.
                reasonings[u] = verdict.reasoning or thinking or ""
                turn_scores[u] = ("" if verdict.turn_scores is None
                                  else json.dumps(verdict.turn_scores))
                # Zero for a verdict this run read back from the cache: the request that paid for
                # it belongs to the run that made it, whose figure is already in the score file.
                if self._judge_client is not None:
                    costs[u] = self._judge_client.request_cost(judge_inputs[k])

        parsed = [s for s in scores if s is not None]
        n_unparsed = n - len(parsed) - n_skipped
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
            "reasoning": reasonings,  # "" where not recorded
            "turn_scores": turn_scores,  # JSON list, one per ORIGINAL turn; "" where not given
            "cost_usd": costs,
        })
        # A fully cached run never builds a client, so there is nothing to read -- report the empty
        # tally rather than nothing, since "this run spent zero" is itself the record.
        usage = self._judge_client.usage if self._judge_client is not None else JudgeUsage()
        return ConversationUtilityResult(
            mean_score=mean_score, usable_rate=usable_rate, n=n, n_scored=len(parsed),
            n_unchanged=n - len(changed) - n_skipped, n_unparsed=n_unparsed, n_skipped=n_skipped, score_counts=score_counts,
            table=table, sampled_from=sides.sampled_from,
            usage=usage, judge_model=self.judge_model,
            judge_rates=self._judge_client.rates if self._judge_client is not None else None,
        )


def conversation_utility(data, *, cache_dir, reference, side: str = "unknown",
                          limit: int | None = None, seed: int = DEFAULT_SEED,
                          **kwargs) -> ConversationUtilityResult:
    """Convenience wrapper: score ``data`` vs. ``reference`` with a default
    :class:`ConversationUtility`.

    Mirrors :func:`prompt_anonymity.defenses.apply_defense`. Any :class:`ConversationUtility`
    constructor argument (``judge_backend``, ``judge_model``, ``judge_base_url``,
    ``judge_system_prompt``, ``judge_temperature``,
    ``judge_max_tokens``, ``max_chars``, ``chunk_size``) may be passed through ``kwargs``.
    """
    return ConversationUtility(**kwargs).score(
        data, cache_dir=cache_dir, reference=reference, side=side, limit=limit, seed=seed
    )
