"""Utility: did the defense keep what mattered?

A linkage defense is only worth using if rewriting a prompt does not ruin what the user was trying
to get. This subpackage measures that -- the utility axis complementing the privacy axis the
attacks measure -- with three metrics, split by what they cost:

``conversation`` (:mod:`.prompt_judge`) -- **paid**
    A judge reads the original and defended conversations **whole, side by side** and scores 1-5 --
    1 unusable, 5 all nuance preserved. Judging the conversation as a unit is what lets it see
    cross-turn breakage (dropped turns, back-references that no longer resolve) that a per-turn
    predicate structurally cannot, and the 1-5 scale is what lets it *rank* defenses rather than
    bucketing everything that mostly works. It judges the prompts rather than the answers, so it
    does not verify that a model still answers well. One API request per conversation.

``semantic`` (:mod:`.semantic`) -- free, local, GPU-shaped
    Multilingual NLI entailment and BERTScore-recall, both run in the recall direction so that
    material the rewrite *adds* cannot cost points. No API, so it covers a whole split where the
    judge realistically covers a sample.

``fluency`` (:mod:`.fluency`) -- free, local, GPU-shaped
    Perplexity of the rewrite over perplexity of the original: is the output still well-formed
    text, independent of whether it kept the content?

**The intended workflow is to combine them**: score the whole corpus with the two local metrics,
score a sample with the judge, and correlate -- which turns the paid judge from *the* measurement
into a calibration set for the free ones. :mod:`experiments.eval_utility` supports exactly that,
merging every metric's per-conversation scores into one file per (source, defense).

Judging runs on **DeepSeek** (:mod:`._deepseek`), reached through the ``openai`` library against
its OpenAI-compatible endpoint; ``DEEPSEEK_BASE_URL`` and ``DEEPSEEK_API_KEY`` come from the
``.env``. Every call is cached with the package's content-addressed
:class:`~prompt_anonymity.caching.TransformCache` under ``<cache_dir>/utility``, so re-runs and
text shared across defenses cost nothing, and conversations a defense left unchanged short-circuit
with no API call at all. The local metrics deliberately do **not** cache: a miss there is
GPU-minutes rather than a re-billed request. Scoring pairs a defended split with its pre-defense
``reference``, so the flow is ``defend -> eval_utility(defended, reference=original)`` -- which is
what :mod:`experiments.eval_utility` does over the parquet pair
:mod:`prompt_anonymity.data.apply_defenses` writes.

Writing a new metric: subclass :class:`~prompt_anonymity.evaluation.utility.base.UtilityMetric`, add
a lowercase wrapper, declare which of its result columns are scores
(:attr:`~prompt_anonymity.evaluation.utility.base.UtilityResult.score_columns`), and register it in
:data:`UTILITY_METRICS`; :func:`eval_utility` then reaches it by name and the driver merges its
columns into the score file without knowing anything else about it. See :mod:`.base` for the
contract (and for why cache invalidation is opt-in here rather than source-driven).

Example
-------
>>> from prompt_anonymity.evaluation.utility import eval_utility
>>> # limit= scores a seeded sample -- calibrate a rubric for cents before a full run
>>> result = eval_utility("conversation", defended, cache_dir=".cache", reference=original, limit=3)
>>> print(result.summary())
>>> result.scores()  # conv_id + this metric's score columns, ready to merge
"""

from importlib import import_module

from ._deepseek import DeepSeekJudge
from .base import DEFAULT_SEED, UtilityMetric, UtilityResult
from .prompt_judge import (
    CONVERSATION_UTILITY_VERSION,
    CONVERSATION_JUDGE_SYSTEM_PROMPT,
    DEFAULT_CONVERSATION_JUDGE_MODEL,
    DEFAULT_JUDGE_REASONING_EFFORT,
    DEFAULT_JUDGE_TEMPERATURE,
    DEFAULT_JUDGE_TOP_P,
    USABLE_SCORE_THRESHOLD,
    ConversationUtility,
    ConversationUtilityResult,
    conversation_utility,
)

# Registry: metric name -> "module:attribute", resolved on demand by `get_utility`.
#
# The values are import paths rather than the callables themselves, which is a departure from the
# package's other registries (DEFENSES, FEATURIZERS, ATTRIBUTION_ATTACKS all hold real objects).
# The reason is import cost: `semantic` and `fluency` pull in torch and transformers, and
# `experiments/eval_utility.py` imports this package to read `sorted(UTILITY_METRICS)` for its
# --metric choices before it knows which metric was asked for. Holding the callables would mean a
# deep-learning stack loading on every run, including one that only wants the API judge.
UTILITY_METRICS = {
    "conversation": "prompt_judge:conversation_utility",  # 1-5 LLM judge over the whole conversation
    "semantic": "semantic:semantic_utility",              # local multilingual NLI + BERTScore-recall
    "fluency": "fluency:fluency_utility",                 # local perplexity ratio (well-formedness)
}


def get_utility(name: str):
    """Look up a registered utility metric by name, importing its module on first use."""
    try:
        target = UTILITY_METRICS[name]
    except KeyError:
        raise ValueError(
            f"unknown utility metric {name!r}; available: {sorted(UTILITY_METRICS)}"
        ) from None
    module_name, attribute = target.split(":")
    module = import_module(f".{module_name}", __package__)
    return getattr(module, attribute)


def eval_utility(name: str, data, *, cache_dir, reference, side: str = "unknown",
                 limit: int | None = None, seed: int = DEFAULT_SEED, **kwargs):
    """Score ``data`` against ``reference`` with the named metric.

    Parameters
    ----------
    name : str
        A key of :data:`UTILITY_METRICS`.
    data, reference : AttackData
        The defended split and its pre-defense original (rows align by position).
    cache_dir : str or pathlib.Path
        Cache root; entries live under ``<cache_dir>/utility``.
    side : {"unknown", "known"}
        Which side to score.
    limit : int, optional
        Score only a seeded random sample of this many **conversations**. One conversation is one
        API request, so this is also the request count -- size it against what a calibration run
        should cost.
    seed : int
        Seed for that sample.
    **kwargs
        Passed to the metric (e.g. ``judge_model``, ``judge_reasoning_effort``).

    Returns
    -------
    UtilityResult
        A metric-specific subclass; all of them expose ``summary()`` and ``scores()``, so callers
        need not know which metric ran.
    """
    return get_utility(name)(
        data, cache_dir=cache_dir, reference=reference, side=side, limit=limit, seed=seed, **kwargs
    )


__all__ = [
    "conversation_utility",
    "ConversationUtility",
    "ConversationUtilityResult",
    "UtilityMetric",
    "UtilityResult",
    "UTILITY_METRICS",
    "get_utility",
    "eval_utility",
    "DeepSeekJudge",
    "CONVERSATION_JUDGE_SYSTEM_PROMPT",
    "DEFAULT_CONVERSATION_JUDGE_MODEL",
    "DEFAULT_JUDGE_REASONING_EFFORT",
    "DEFAULT_JUDGE_TEMPERATURE",
    "DEFAULT_JUDGE_TOP_P",
    "USABLE_SCORE_THRESHOLD",
    "DEFAULT_SEED",
    "CONVERSATION_UTILITY_VERSION",
]
