"""Utility: did the defense keep what mattered?

.. note::
   **Nothing currently calls this subpackage.** Its only entry point was the ``--fidelity`` flag
   of the fixed-split experiment runner, deleted on 2026-08-04 (this package was called
   ``fidelity`` then, and was renamed with it); today's ``experiments/run_experiment.py`` reads
   precomputed vectors and never sees prompt text, so it has nowhere to put a text-level judge.
   The code below is unchanged and working. Reaching the measurement again means a small driver
   over the original/defended parquet pair ``prompt_anonymity.data.apply_defenses`` writes, which
   is where a defense's before-and-after text now lives.

A linkage defense is only worth using if rewriting a prompt does not ruin what the user was trying
to get. This subpackage measures that -- the utility axis complementing the privacy axis the attacks
measure -- with two metrics that ask the question from opposite ends:

``answer`` (:mod:`.answer_judge`)
    The predicate from "Operationalizing Data Minimization for Privacy-Preserving LLM Prompting"
    (ICLR 2026, App. E). A response model ``F`` answers the original prompt (reference ``A``) and the
    defended prompt (candidate ``B``), and a judge rules **PASS** (``B`` still addresses every key
    point) or **FAIL**. Scored **per user turn**; utility is the fraction that PASS. Grounded in
    real answers, but blind to anything that only shows up across turns, and PASS/FAIL puts every
    defense that mostly works in one bucket.

``conversation`` (:mod:`.prompt_judge`)
    A judge reads the original and defended conversations **whole, side by side** and scores 1-5 --
    1 unusable, 5 all nuance preserved. Sees cross-turn breakage the per-turn predicate cannot
    (dropped turns, back-references that no longer resolve) and has the resolution to rank defenses,
    at one call per conversation instead of ~3 per turn. Judges the prompts rather than the answers,
    so it does not verify that a model still answers well.

They are complements, not substitutes -- run both and expect different numbers but broadly
consistent *rankings*. A defense with a high pass rate and a mean score of 2 means one of the two
rubrics is miscalibrated.

Both run on OpenRouter and cache every call with the package's content-addressed
:class:`~prompt_anonymity.caching.TransformCache` under ``<cache_dir>/utility``, so re-runs and
text shared across defenses cost nothing, and conversations a defense left unchanged short-circuit
with no API call at all. Scoring pairs a defended split with its pre-defense ``reference``, so the
flow a driver would implement is ``defend -> run_utility(defended, reference=original)``.

Writing a new metric: subclass :class:`~prompt_anonymity.utility.base.UtilityMetric`, add a
lowercase wrapper, and register it in :data:`UTILITY_METRICS`; :func:`run_utility` then reaches it
by name. See :mod:`.base` for the contract (and for why cache invalidation is opt-in here rather
than source-driven).

Example
-------
>>> from prompt_anonymity.defenses import apply_defense
>>> from prompt_anonymity.utility import run_utility
>>> defended = apply_defense("qwen_rewrite", data, cache_dir=".cache")
>>> # limit= scores a seeded sample -- calibrate a rubric for cents before a full run
>>> result = run_utility("conversation", defended, cache_dir=".cache", reference=data, limit=50)
>>> print(result.summary())
"""

from ._openrouter import OpenRouterChat
from .answer_judge import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_RESPONSE_MODEL,
    ANSWER_UTILITY_VERSION,
    RESPONSE_SYSTEM_PROMPT,
    ANSWER_JUDGE_SYSTEM_PROMPT,
    AnswerUtility,
    AnswerUtilityResult,
    answer_utility,
)
from .base import DEFAULT_SEED, UtilityMetric, UtilityResult
from .prompt_judge import (
    CONVERSATION_UTILITY_VERSION,
    CONVERSATION_JUDGE_SYSTEM_PROMPT,
    DEFAULT_CONVERSATION_JUDGE_MODEL,
    USABLE_SCORE_THRESHOLD,
    ConversationUtility,
    ConversationUtilityResult,
    conversation_utility,
)

# Registry so callers can select a utility metric by name (e.g. from a CLI argument), mirroring
# prompt_anonymity.attacks.ATTRIBUTION_ATTACKS. Registering a metric here is all it takes to make it selectable
# from the experiment driver.
UTILITY_METRICS = {
    "answer": answer_utility,              # answer-level PASS/FAIL, per turn (the paper's predicate)
    "conversation": conversation_utility,  # prompt-level 1-5, whole conversation
}


def get_utility(name: str):
    """Look up a registered utility metric by name."""
    try:
        return UTILITY_METRICS[name]
    except KeyError:
        raise ValueError(
            f"unknown utility metric {name!r}; available: {sorted(UTILITY_METRICS)}"
        ) from None


def run_utility(name: str, data, *, cache_dir, reference, side: str = "unknown",
                 limit: int | None = None, seed: int = DEFAULT_SEED, **kwargs):
    """Score ``data`` against ``reference`` with the named metric.

    Parameters
    ----------
    name : str
        A key of :data:`UTILITY_METRICS`.
    data, reference : AttackData
        The defended split and the loader's original (rows align by position).
    cache_dir : str or pathlib.Path
        Cache root; entries live under ``<cache_dir>/utility``.
    side : {"unknown", "known"}
        Which side to score.
    limit : int, optional
        Score only a seeded random sample of this many **conversations** -- not API calls. The two
        metrics differ sharply in calls per conversation (``conversation`` makes one; ``answer``
        makes roughly three per changed turn), so size it per metric.
    seed : int
        Seed for that sample.
    **kwargs
        Passed to the metric (e.g. ``judge_model``).

    Returns
    -------
    UtilityResult
        A metric-specific subclass; all of them expose ``summary()`` and ``to_csv()``, so callers
        need not know which metric ran.
    """
    return get_utility(name)(
        data, cache_dir=cache_dir, reference=reference, side=side, limit=limit, seed=seed, **kwargs
    )


__all__ = [
    "answer_utility",
    "AnswerUtility",
    "AnswerUtilityResult",
    "conversation_utility",
    "ConversationUtility",
    "ConversationUtilityResult",
    "UtilityMetric",
    "UtilityResult",
    "UTILITY_METRICS",
    "get_utility",
    "run_utility",
    "OpenRouterChat",
    "RESPONSE_SYSTEM_PROMPT",
    "ANSWER_JUDGE_SYSTEM_PROMPT",
    "CONVERSATION_JUDGE_SYSTEM_PROMPT",
    "DEFAULT_RESPONSE_MODEL",
    "DEFAULT_JUDGE_MODEL",
    "DEFAULT_CONVERSATION_JUDGE_MODEL",
    "USABLE_SCORE_THRESHOLD",
    "DEFAULT_SEED",
    "ANSWER_UTILITY_VERSION",
    "CONVERSATION_UTILITY_VERSION",
]
