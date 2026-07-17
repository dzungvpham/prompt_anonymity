"""Fidelity: does a defended prompt still get an equally-useful answer?

A linkage defense is only worth using if rewriting a prompt does not ruin the answer the user would
have gotten. This subpackage measures that with the utility predicate from "Operationalizing Data
Minimization for Privacy-Preserving LLM Prompting" (ICLR 2026, App. E): a response model ``F``
answers the original prompt (reference ``A``) and the defended prompt (candidate ``B``), and a judge
model rules **PASS** (``B`` still addresses every key point of the user's message) or **FAIL**. A
defense's *fidelity* is the fraction of prompts that PASS -- the utility axis that complements the
privacy axis the attacks measure.

Both ``F`` and the judge run on OpenRouter (default ``openai/gpt-4o``, the paper's judge). Scoring
takes the loaded split and its defended copy -- mirroring
:func:`prompt_anonymity.features.apply_featurizer`, which also pairs post-defense ``data`` with the
pre-defense ``reference`` -- so it slots into the driver flow
``load -> apply_defense -> utility_fidelity(defended, reference=loaded) -> featurize -> attack``.
Responses and verdicts are cached with the package's content-addressed
:class:`~prompt_anonymity.caching.TransformCache` under ``<cache_dir>/fidelity``, so re-runs and
prompts shared across defenses cost no API calls, and unchanged rows short-circuit to PASS.

Example
-------
>>> from prompt_anonymity.defenses import apply_defense
>>> from prompt_anonymity.fidelity import utility_fidelity
>>> defended = apply_defense("qwen_rewrite", data, cache_dir=".cache")
>>> result = utility_fidelity(defended, cache_dir=".cache", reference=data)
>>> print(result.summary())
"""

from ._openrouter import OpenRouterChat
from .judge import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_RESPONSE_MODEL,
    FIDELITY_VERSION,
    RESPONSE_SYSTEM_PROMPT,
    UTILITY_JUDGE_SYSTEM_PROMPT,
    FidelityResult,
    UtilityFidelity,
    utility_fidelity,
)

__all__ = [
    "utility_fidelity",
    "UtilityFidelity",
    "FidelityResult",
    "OpenRouterChat",
    "RESPONSE_SYSTEM_PROMPT",
    "UTILITY_JUDGE_SYSTEM_PROMPT",
    "DEFAULT_RESPONSE_MODEL",
    "DEFAULT_JUDGE_MODEL",
    "FIDELITY_VERSION",
]
