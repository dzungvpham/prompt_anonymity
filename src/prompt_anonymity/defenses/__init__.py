"""Defenses: transformations applied to a loaded dataset to resist linkage.

A *defense* models an anonymization countermeasure the user applies to their prompts before
releasing them. It runs *after* loading and *before* the attack, so the same attack/metrics
can be evaluated with and without a defense. Two kinds are supported:

* **Plain** defenses -- a callable ``AttackData -> AttackData`` (e.g. :func:`no_defense`).
* **Cached** defenses -- subclasses of :class:`CachedDefense` whose expensive transform is
  cached on disk with automatic, safe invalidation (see :mod:`prompt_anonymity.caching`). Use
  these for costly text transforms such as translation.

To add a defense, write either kind and register it in ``DEFENSES`` so it is selectable by
name. Run a defense via :func:`apply_defense` (pass ``cache_dir`` for cached ones).
"""

from __future__ import annotations

from typing import Callable, Union

from ..caching import IndexedRowCache, TransformCache, logic_hash, params_hash, source_digest
from ..core import AttackData
from .argos import ArgosRTTDefense
from .base import CachedDefense, CachedTextRewriteDefense
from .dp_mlm import DPMLMDefense
from .examples import ExampleTextNormalizationDefense, RoundTripTranslationDefense
from .openanonymity import OpenAnonymityDefense
from .qwen_rewrite import QwenRewriteDefense
from .styleremix import StyleRemixDefense
from .styleremix_openanon import StyleRemixOpenAnonymityDefense

# A defense is a plain callable or a (callable) CachedDefense instance.
Defense = Union[Callable[[AttackData], AttackData], CachedDefense]


def no_defense(data: AttackData) -> AttackData:
    """Identity defense: leave the dataset unchanged (the 'no anonymization' baseline)."""
    return data


# Registry of ready-to-use defenses, selectable by name (e.g. from a CLI argument). The
# model-backed defenses build their (heavy) backend lazily on first use, so registering them here
# is free -- selecting one never loads a model, and a fully-cached run loads none either.
# RoundTripTranslationDefense is intentionally absent: it needs a translation model supplied by the
# caller, so it cannot be a zero-config registry entry.
DEFENSES: dict[str, Defense] = {
    "none": no_defense,
    "example_normalization": ExampleTextNormalizationDefense(),
    "rtt_argos": ArgosRTTDefense(),
    "qwen_rewrite": QwenRewriteDefense(),
    "styleremix": StyleRemixDefense(),
    "openanonymity": OpenAnonymityDefense(),
    "styleremix_openanon": StyleRemixOpenAnonymityDefense(),
    "dp_mlm": DPMLMDefense(),
    "dp_mlm_pii": DPMLMDefense(pii=True),
}

#: DP-MLM per-word privacy budgets exposed as a sweep. Each registers a ``dp_mlm_eps<eps>`` defense
#: selectable via ``--defense``. Because run_experiment.py's output_tag embeds the defense name, every
#: epsilon writes to its OWN results dir (headline_results.csv / sweep_results.csv / ...), and each
#: caches separately since epsilon is in ``params()``. The paper's set is {10,25,50,100,250}; the
#: higher values are added because DP-MLM only becomes near-readable at large epsilon (weaker privacy
#: -- the point of sweeping). Keep this in sync with experiments/run_dpmlm_sweep.sh.
DPMLM_SWEEP_EPSILONS = (10, 25, 50, 100, 250, 500, 1000)
for _eps in DPMLM_SWEEP_EPSILONS:
    DEFENSES[f"dp_mlm_eps{_eps}"] = DPMLMDefense(epsilon=_eps)
del _eps


def get_defense(name: str) -> Defense:
    """Look up a registered defense by name."""
    try:
        return DEFENSES[name]
    except KeyError:
        raise ValueError(f"unknown defense {name!r}; available: {sorted(DEFENSES)}") from None


def apply_defense(name: str, data: AttackData, *, cache_dir=None) -> AttackData:
    """Apply the named defense to ``data``.

    Cached defenses (:class:`CachedDefense`) require ``cache_dir``; plain defenses ignore it.
    """
    defense = get_defense(name)
    if isinstance(defense, CachedDefense):
        if cache_dir is None:
            raise ValueError(f"defense {name!r} caches to disk; provide cache_dir.")
        return defense(data, cache_dir=cache_dir)
    return defense(data)


__all__ = [
    "Defense",
    "no_defense",
    "DEFENSES",
    "get_defense",
    "apply_defense",
    "CachedDefense",
    "CachedTextRewriteDefense",
    "TransformCache",
    "IndexedRowCache",
    "logic_hash",
    "params_hash",
    "source_digest",
    "ExampleTextNormalizationDefense",
    "RoundTripTranslationDefense",
    "ArgosRTTDefense",
    "QwenRewriteDefense",
    "StyleRemixDefense",
    "OpenAnonymityDefense",
    "StyleRemixOpenAnonymityDefense",
    "DPMLMDefense",
]
