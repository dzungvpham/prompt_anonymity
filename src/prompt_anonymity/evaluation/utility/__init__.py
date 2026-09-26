"""Utility: did the defense keep what mattered?

A linkage defense is only worth using if rewriting a prompt does not ruin what the user was trying
to get. This subpackage measures that -- the utility axis complementing the privacy axis the
attacks measure -- with three metrics, split by what they cost:

``conversation`` (:mod:`.prompt_judge`) -- **paid**
    A judge reads the original and defended conversations whole, side by side, and scores 1-5
    (1 unusable, 5 all nuance preserved). Judging as a whole conversation catches cross-turn
    breakage a per-turn predicate can't see. One API request per conversation.

``semantic`` (:mod:`.semantic`) -- free, local, GPU-shaped
    Multilingual NLI entailment and BERTScore-recall, both run in the recall direction so material
    the rewrite *adds* cannot cost points. Cheap enough to cover a whole split.

``fluency`` (:mod:`.fluency`) -- free, local, GPU-shaped
    Perplexity of the rewrite over perplexity of the original: is the output still well-formed,
    independent of whether it kept the content?

**The intended workflow is to combine them**: score the whole corpus with the two local metrics,
score a sample with the judge, and correlate -- turning the paid judge into a calibration set for
the free ones. :mod:`experiments.eval_utility` merges every metric's per-conversation scores into
one file per (source, defense).

Judging runs on **DeepSeek** (:mod:`._deepseek`) by default via its OpenAI-compatible endpoint
(``DEEPSEEK_BASE_URL`` / ``DEEPSEEK_API_KEY`` from ``.env``); ``judge_backend="local"`` sends the
same rubric to a self-hosted vLLM server instead (:mod:`._vllm_judge`), at no cost. Every judge
call is cached under ``<cache_dir>/utility`` (:class:`~prompt_anonymity.caching.TransformCache`),
and a conversation a defense left unchanged short-circuits with no API call. The local metrics
deliberately do **not** cache -- a miss there is GPU-minutes, not a re-billed request. Scoring
pairs a defended split with its pre-defense ``reference``: ``defend -> eval_utility(defended,
reference=original)``, which is what :mod:`experiments.eval_utility` does over the parquet pair
:mod:`prompt_anonymity.data.apply_defenses` writes.

Writing a new metric: subclass :class:`~prompt_anonymity.evaluation.utility.base.UtilityMetric`, add
a lowercase wrapper, declare which of its result columns are scores
(:attr:`~prompt_anonymity.evaluation.utility.base.UtilityResult.score_columns`), and register it in
:data:`UTILITY_METRICS`. See :mod:`.base` for the contract.

Example
-------
>>> from prompt_anonymity.evaluation.utility import eval_utility
>>> # limit= scores a seeded sample -- calibrate a rubric for cents before a full run
>>> result = eval_utility("conversation", defended, cache_dir=".cache", reference=original, limit=3)
>>> print(result.summary())
>>> result.scores()  # conv_id + this metric's score columns, ready to merge
"""

from importlib import import_module

from ._deepseek import DeepSeekJudge, OpenAICompatibleJudge
from ._vllm_judge import VLLMJudge
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
# Import paths rather than the callables themselves (unlike DEFENSES/FEATURIZERS/
# ATTRIBUTION_ATTACKS): `semantic` and `fluency` pull in torch and transformers, and this module is
# imported just to read `sorted(UTILITY_METRICS)` before any metric is chosen.
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
    "OpenAICompatibleJudge",
    "VLLMJudge",
    "CONVERSATION_JUDGE_SYSTEM_PROMPT",
    "DEFAULT_CONVERSATION_JUDGE_MODEL",
    "DEFAULT_JUDGE_REASONING_EFFORT",
    "DEFAULT_JUDGE_TEMPERATURE",
    "DEFAULT_JUDGE_TOP_P",
    "USABLE_SCORE_THRESHOLD",
    "DEFAULT_SEED",
    "CONVERSATION_UTILITY_VERSION",
]
