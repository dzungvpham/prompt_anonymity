"""Utility utility: the paper's LLM-as-a-judge PASS/FAIL predicate for a defense.

This is the *answer-side* utility metric: it judges the **answers** two prompts elicit. For the
prompt-side complement -- one 1-5 score comparing the two whole conversations directly, which sees
cross-turn breakage this cannot -- see :mod:`.prompt_judge`.

A defense is only useful if a prompt still gets an equally-good answer *after* it is rewritten.
This module scores that the way "Operationalizing Data Minimization for Privacy-Preserving LLM
Prompting" (ICLR 2026, App. E) defines **utility**: a response model ``F`` answers the original
prompt (reference response ``A = F(x)``) and the defended prompt (candidate response
``B = F(defended)``), then a judge model decides whether ``B`` still addresses every key point of
the user's message -- returning **PASS** or **FAIL**. The utility of a defense is the fraction of
prompts that PASS.

Scoring is **per user turn**: a conversation cell is a user's turns joined by the ``\\n===\\n``
delimiter, and the per-turn rewrite defenses defend each turn independently, so each
(original turn, defended turn) pair is judged on its own. A defense that does not preserve the turn
structure (e.g. one that rewrites the whole joined conversation) is scored as a single whole-cell
unit instead.

The response model and the judge both run on OpenRouter (default ``openai/gpt-4o``, the paper's
judge) via :class:`~prompt_anonymity.utility._openrouter.OpenRouterChat`. Both calls are cached
with the package's content-addressed :class:`~prompt_anonymity.caching.TransformCache` under
``<cache_dir>/utility``: responses are keyed by prompt text (so an identical prompt -- e.g. the
same original across defenses, or a defense that left a row unchanged -- is answered once) and
verdicts by the composed judge input, so re-runs and shared turns cost no API calls. A model or
prompt swap re-caches automatically (it is part of the cache key), and unchanged turns short-circuit
to PASS with no API call at all, mirroring how :func:`prompt_anonymity.features.apply_featurizer`
reuses unchanged rows.

Caveats (faithful-to-paper limits, bounded by what this package retains):

* **No placeholder restore-back.** The paper restores ``[PERSON_n]`` placeholders in ``B`` before
  judging; this package keeps no placeholder->original map (the OpenAnonymity defense keeps only the
  REDACT half), so ``B`` is judged as-is. Redaction/placeholder defenses therefore score lower **by
  design** -- read the number as a utility floor and compare against a paraphrase defense.
* **Open-ended, single decode.** WildChat prompts are open-ended, so the paper's ``k=5`` closed-
  ended verification path does not apply -- one decode per prompt.
* **Combined defenses** (e.g. StyleRemix + OpenAnonymity) are judged against the true original
  (the ``reference``), which is the intended reading of end-to-end utility.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from ..defenses._backends import split_turns
from ._openrouter import OpenRouterChat
from ._parsing import json_object, render_turns, strip_code_fence
from .base import DEFAULT_SEED, UtilityMetric, UtilityResult

#: Neutral persona for the response model F -- it just answers the user's prompt.
RESPONSE_SYSTEM_PROMPT = "You are a helpful assistant."

#: Utility judge rubric, verbatim from the paper (App. E). Returns a JSON PASS/FAIL verdict.
ANSWER_JUDGE_SYSTEM_PROMPT = (
    "You are a utility evaluator. Your task is to assess whether a second response adequately "
    "addresses a user's original message. You will be provided with: (i) the user message (what the "
    "user is asking or talking about), (ii) a reference response A (a standard answer that is known "
    "to be good), and (iii) a candidate response B (a second version that needs to be evaluated). "
    "You must decide whether the candidate response is \"PASS\" or \"FAIL\". This is not a test of "
    "exact wording or stylistic similarity; focus on whether the candidate response addresses all "
    "key points or needs expressed in the user message. If it does, it passes. If it fails to "
    "address one or more key points, it fails.\n"
    "Input will contain three tagged sections: <user_message>, <response_A>, and <response_B>.\n"
    "Return JSON with the result and a one-line explanation: "
    "{\"Result\": \"PASS/FAIL\", \"Reason\": YOUR_EXPLANATION}"
)

#: Default OpenRouter models. ``openai/gpt-4o`` is the paper's judge; EDIT ME to retarget either.
DEFAULT_RESPONSE_MODEL = "openai/gpt-4o"
DEFAULT_JUDGE_MODEL = "openai/gpt-4o"

#: Manual logic version for the utility caches; bump to force a full recompute (see caching.py).
ANSWER_UTILITY_VERSION = "1"


def _response_prompt(text) -> str:
    """The prompt handed to F: a conversation cell's user turns joined by real newlines instead of
    the on-disk ``\\n===\\n`` delimiter, so the response model sees clean text.

    Delegates to :func:`prompt_anonymity.utility._parsing.render_turns` with both of its flags off,
    which is byte-for-byte what this function has always produced. That matters more than it looks:
    this rendered text is the *cache key* for a store of paid completions, so any change here --
    including dropping whitespace-only turns -- would silently orphan existing entries.
    """
    return render_turns(text)


def _judge_input(user_message: str, response_a: str, response_b: str) -> str:
    """The judge's user message: the three tagged sections the rubric expects. Built by direct
    string join (not ``{{}}`` templating) so braces in the prompt or responses are never mangled."""
    return (
        f"<user_message>\n{user_message}\n</user_message>\n"
        f"<response_A>\n{response_a}\n</response_A>\n"
        f"<response_B>\n{response_b}\n</response_B>"
    )


def _parse_verdict(raw: str) -> tuple[bool | None, str]:
    """Parse a judge reply into ``(passed, reason)``.

    ``passed`` is ``True`` for PASS, ``False`` for FAIL, and ``None`` when the reply cannot be
    parsed (the caller counts that as a conservative FAIL, matching the paper's strict predicate).
    Tolerates markdown code fences and a stray reasoning preamble: parse the JSON object if present,
    else fall back to a ``"Result": PASS/FAIL`` field and finally to a bare PASS/FAIL token.
    """
    text = strip_code_fence(raw)

    result: bool | None = None
    reason = ""
    lowered = json_object(text)
    if lowered is not None:
        reason = str(lowered.get("reason", "") or "")
        verdict = lowered.get("result")
        if isinstance(verdict, str):
            match = re.search(r"pass|fail", verdict, re.IGNORECASE)
            if match:
                result = match.group(0).lower() == "pass"

    if result is None:  # JSON missing/garbled -> scan the raw text for the verdict token
        match = re.search(r'"?result"?\s*:\s*"?\s*(pass|fail)', text, re.IGNORECASE)
        if match is None:
            match = re.search(r"\b(pass|fail)\b", text, re.IGNORECASE)
        if match is not None:
            result = match.group(1).lower() == "pass"

    if result is None:
        return None, f"UNPARSED: {text[:200]}"
    return result, reason


@dataclass
class AnswerUtilityResult(UtilityResult):
    """Utility-utility score for one (original, defended) comparison.

    Attributes
    ----------
    pass_rate : float
        Fraction of scored turns the judge marked PASS (utility preserved), in ``[0, 1]``.
    n, n_pass, n_fail, n_unparsed : int
        Number of scored turns, and how many passed / failed / had an unparseable verdict (the last
        are counted as FAIL in ``pass_rate``, matching the paper's strict predicate -- note this
        differs from :class:`~prompt_anonymity.utility.prompt_judge.ConversationUtilityResult`,
        which excludes unparsed rows because imputing a floor on a 1-5 scale distorts a mean far
        more than it distorts a rate).
    sampled_from : int or None
        Full split size when ``limit`` was used, else ``None``.
    table : pandas.DataFrame
        Per-turn detail (``conv_index, turn_index, original, defended, response_a, response_b,
        result, reason``) for spot-checking.
    """

    pass_rate: float
    n: int
    n_pass: int
    n_fail: int
    n_unparsed: int
    table: pd.DataFrame
    sampled_from: int | None = None

    sort_column: str = field(default="result", repr=False)

    def summary(self) -> str:
        return (
            f"{self.sample_note()}Utility utility  n={self.n}  pass_rate={self.pass_rate:.4f}  "
            f"(PASS={self.n_pass}, FAIL={self.n_fail}, unparsed={self.n_unparsed})"
        )

    def _sort_values(self):
        """FAIL rows first: the ``result`` column is text, so map it to an order the base's
        ascending sort reads as worst-first."""
        return self.table["result"].map({"FAIL": 0, "PASS": 1}).fillna(0)


class AnswerUtility(UtilityMetric):
    """Score defense utility the paper's way: F answers original vs. defended, a judge rules PASS/FAIL.

    The response and judge clients are built lazily on first real need, so a fully-cached run makes
    no API calls and needs no ``OPENROUTER_API_KEY``. The active models and system prompts are part
    of the cache key, so swapping either re-caches automatically.

    Parameters
    ----------
    response_model, judge_model : str
        OpenRouter model ids for F and the judge (default ``openai/gpt-4o``).
    response_system_prompt, judge_system_prompt : str
        F's persona and the judge rubric (default: neutral assistant / the paper's App. E rubric).
    response_max_tokens, judge_max_tokens : int
        Output budgets; F needs room for a full answer, the judge only a short JSON verdict.
    """

    name = "utility_judge"
    version = ANSWER_UTILITY_VERSION

    def __init__(self, *, response_model: str = DEFAULT_RESPONSE_MODEL,
                 judge_model: str = DEFAULT_JUDGE_MODEL,
                 response_system_prompt: str = RESPONSE_SYSTEM_PROMPT,
                 judge_system_prompt: str = ANSWER_JUDGE_SYSTEM_PROMPT,
                 response_max_tokens: int = 1024, judge_max_tokens: int = 512):
        self.response_model = response_model
        self.judge_model = judge_model
        self.response_system_prompt = response_system_prompt
        self.judge_system_prompt = judge_system_prompt
        self.response_max_tokens = response_max_tokens
        self.judge_max_tokens = judge_max_tokens
        self._response_client: OpenRouterChat | None = None
        self._judge_client: OpenRouterChat | None = None

    def _generate(self, prompts: list[str]) -> list[str]:
        if self._response_client is None:
            self._response_client = OpenRouterChat(
                self.response_model, self.response_system_prompt, max_tokens=self.response_max_tokens
            )
        return self._response_client.complete_batch(prompts)

    def _judge(self, inputs: list[str]) -> list[str]:
        if self._judge_client is None:
            self._judge_client = OpenRouterChat(
                self.judge_model, self.judge_system_prompt, max_tokens=self.judge_max_tokens
            )
        return self._judge_client.complete_batch(inputs)

    def params(self) -> dict:
        # The judge stage's key. Must stay exactly these two entries: it addresses a populated cache
        # of paid judge replies, and any change to the dict re-namespaces and orphans it.
        return {"judge_model": self.judge_model,
                "judge_system_prompt": self.judge_system_prompt}

    def _response_cache(self, cache_dir):
        # The response stage gets its own namespace, keyed by prompt text and by F's model+persona
        # rather than the judge's, so an identical prompt is answered once no matter which judge
        # scores it. Same byte-for-byte params dict as before the base-class refactor, for the same
        # cache-preservation reason as `params`.
        return self._cache(
            cache_dir, name="responses",
            params={"response_model": self.response_model,
                    "response_system_prompt": self.response_system_prompt},
        )

    def _judge_cache(self, cache_dir):
        return self._cache(cache_dir)

    def score(self, data, *, cache_dir, reference, side: str = "unknown",
              limit: int | None = None, seed: int = DEFAULT_SEED) -> AnswerUtilityResult:
        """Score the utility utility of ``data`` (post-defense) against ``reference`` (pre-defense).

        Parameters
        ----------
        data : AttackData
            The defended split; ``data.{side}_texts`` are the rewritten prompts.
        cache_dir : str or pathlib.Path
            Cache root; responses and verdicts live under ``<cache_dir>/utility``.
        reference : AttackData
            The loaded (pre-defense) split -- typically the loader's output; ``reference.{side}_texts``
            are the original prompts. Rows must align with ``data`` by position.
        side : {"unknown", "known"}
            Which side to score. Defaults to ``"unknown"`` (the side a defense rewrites).
        limit : int, optional
            Score only a seeded random sample of this many conversations. Note this counts
            *conversations*, not API calls -- each one costs roughly three calls per changed turn,
            so a handful goes a long way here.
        seed : int
            Seed for that sample.

        Returns
        -------
        AnswerUtilityResult
            The per-turn pass rate and a per-turn table (see the module docstring for the per-turn
            scoring convention and the whole-cell fallback for structure-changing defenses).
        """
        sides = self._load_sides(data, reference, side, limit=limit, seed=seed)
        original, defended = sides.original, sides.defended
        n_conv = len(original)

        # Score PER USER TURN. A conversation cell is a user's turns joined by TURN_DELIM, and the
        # per-turn rewrite defenses preserve that structure (they defend each turn and re-join), so
        # turn j of the original aligns with turn j of the defended cell. Split both and pair the
        # turns positionally. If a defense did NOT preserve the structure -- e.g. the combined
        # StyleRemix+OpenAnonymity defense scrubs the whole joined conversation, so the turn counts
        # differ -- fall back to scoring that conversation as a single whole-cell unit (this is how
        # such per-conversation defenses were scored in the research sandbox).
        units: list[tuple[int, int, str, str]] = []  # (conv_index, turn_index, orig_turn, def_turn)
        for i in range(n_conv):
            o_turns = split_turns(original[i])
            d_turns = split_turns(defended[i])
            if len(o_turns) == len(d_turns):
                paired = list(enumerate(zip(o_turns, d_turns)))
            else:
                paired = [(0, (original[i], defended[i]))]  # structure changed -> whole cell
            # Record the row's position in the FULL split, not its offset within a sample, so a
            # limited run's table still joins against a full one.
            conv_index = sides.indices[i]
            for j, (o_turn, d_turn) in paired:
                if o_turn.strip():  # skip blank turns (a delimiter artifact; nothing to answer)
                    units.append((conv_index, j, o_turn, d_turn))
        n = len(units)

        # A turn the defense left unchanged trivially preserves utility (identical prompt -> the
        # judge would always PASS), so short-circuit it to PASS with no API call and only run F +
        # the judge on the turns the defense actually changed.
        results: list[bool | None] = [True] * n           # default PASS (covers unchanged turns)
        reasons: list[str] = ["unchanged prompt"] * n
        resp_a: list[str] = [""] * n
        resp_b: list[str] = [""] * n
        changed = [u for u, (_, _, o_turn, d_turn) in enumerate(units) if d_turn != o_turn]

        if changed:
            orig_prompts = [_response_prompt(units[u][2]) for u in changed]
            def_prompts = [_response_prompt(units[u][3]) for u in changed]
            # One cache call over both sides so identical turns (across sides, conversations, or
            # defenses) are generated once; split back into A (original) and B (defended) responses.
            resp_cache = self._response_cache(cache_dir)
            responses = resp_cache.apply_batch(orig_prompts + def_prompts, self._generate)
            answers_a = responses[: len(changed)]
            answers_b = responses[len(changed):]

            judge_inputs = [
                _judge_input(orig_prompts[k], answers_a[k], answers_b[k])
                for k in range(len(changed))
            ]
            judge_cache = self._judge_cache(cache_dir)
            verdicts = judge_cache.apply_batch(judge_inputs, self._judge)

            for k, u in enumerate(changed):
                passed, reason = _parse_verdict(verdicts[k])
                results[u] = passed
                reasons[u] = reason
                resp_a[u] = answers_a[k]
                resp_b[u] = answers_b[k]

        n_unparsed = sum(1 for r in results if r is None)
        n_pass = sum(1 for r in results if r is True)
        n_fail = n - n_pass  # unparsed -> counted as FAIL (conservative, matches the strict predicate)
        pass_rate = n_pass / n if n else 0.0

        table = pd.DataFrame({
            "conv_index": [u[0] for u in units],
            "turn_index": [u[1] for u in units],
            "original": [u[2] for u in units],
            "defended": [u[3] for u in units],
            "response_a": resp_a,
            "response_b": resp_b,
            "result": ["PASS" if r else "FAIL" for r in results],
            "reason": reasons,
        })
        return AnswerUtilityResult(
            pass_rate=pass_rate, n=n, n_pass=n_pass, n_fail=n_fail, n_unparsed=n_unparsed,
            table=table, sampled_from=sides.sampled_from,
        )


def answer_utility(data, *, cache_dir, reference, side: str = "unknown",
                     limit: int | None = None, seed: int = DEFAULT_SEED,
                     **kwargs) -> AnswerUtilityResult:
    """Convenience wrapper: score ``data`` vs. ``reference`` with a default :class:`AnswerUtility`.

    Mirrors :func:`prompt_anonymity.defenses.apply_defense` /
    :func:`prompt_anonymity.features.apply_featurizer`. Any :class:`AnswerUtility` constructor
    argument (``response_model``, ``judge_model``, ``*_system_prompt``, ``*_max_tokens``) may be
    passed through ``kwargs``.
    """
    return AnswerUtility(**kwargs).score(
        data, cache_dir=cache_dir, reference=reference, side=side, limit=limit, seed=seed
    )
