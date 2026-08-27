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
from .afr import AFR_RESIDUALS, AgenticFootprintDefense
from .argos import ArgosRTTDefense
from .base import CachedDefense, CachedTextRewriteDefense
from .collision_seeding import SWE_CHAT_MARKERS, CollisionSeedingDefense
from .dp_mlm import DPMLMDefense
from .examples import ExampleTextNormalizationDefense, RoundTripTranslationDefense
from .frame_pad import FramePadDefense
from .frame_shift import SINGLE_FRAMING_KEY, FrameShiftDefense
from .loo_unlink import LOO_UNLINK_BUDGETS, LOOUnlinkDefense
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
# is free -- selecting one never loads a model, and a fully-cached run loads none either. The same
# goes for frame_pad's passage bank: it is resolved (and if absent, generated) on first use, so
# importing this registry never reads a file or needs an API key.
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
    "collision_seeding": CollisionSeedingDefense(marker_keys=SWE_CHAT_MARKERS),
    "frame_shift": FrameShiftDefense(),
    "frame_pad": FramePadDefense(),
}

#: Frame shift's single-frame ablation: the whole corpus is rewritten into ONE scene instead of
#: drawing from the 50-entry codebook. It is the convergence-vs-dilution control, and a real
#: contender rather than a straw man -- the default dilutes each author across 50 surface registers,
#: while this converges every document onto one, the way ``styleremix`` and ``qwen_rewrite`` converge
#: on one style. It is also the control that separates "the framing diluted the author" from
#: "everything simply got longer", since both arms inflate length the same way.
#: Cheaper to run than the default, too: with one frame shared by every row, turn-level cache dedup
#: is fully restored (see :meth:`~.frame_shift.FrameShiftDefense._rewrite_side`).
DEFENSES["frame_shift_single"] = FrameShiftDefense(single_framing=SINGLE_FRAMING_KEY)

#: Frame pad's single-scene ablation, the same control one level down: every document's appended turn
#: is drawn from ONE scene's passages instead of from all 50 scenes', so the corpus circulates P pads
#: rather than K x P. Read against ``frame_pad`` it asks whether pad *diversity* matters or only pad
#: presence; read against ``frame_shift_single`` it holds the scene fixed and varies only whether the
#: user's text was rewritten.
DEFENSES["frame_pad_single"] = FramePadDefense(single_framing=SINGLE_FRAMING_KEY)

#: Collision-seeding variants. Unlike every other defense here this one is pure Python string work
#: (no model, no GPU, seconds not hours), so a variant costs nothing to add and nothing to run --
#: only the featurize and attack stages after it are expensive.
#:
#: The two K variants sweep the privacy knob: the codebook size sets the expected collision group at
#: ``n_authors / K``, so k4 buys larger groups (stronger anonymity, more text touched per group) and
#: k24 smaller ones. The two ablations exist to be *compared against*, not deployed:
#:
#: * ``_full`` applies each marker to 100% of an author's documents. The prediction is that it does
#:   WORSE than the default despite being a bigger edit, because perfect consistency is a perfectly
#:   reliable feature -- which is the premise the 40-70% rate rests on.
#: * ``_indep`` draws markers per author independently instead of from the codebook, giving ~C(M,4)
#:   possible signatures. The prediction is that it does worse than NO defense, because a
#:   near-unique marker combination is a fingerprint. It is the control that shows the codebook is
#:   doing the work.
#: Every variant is wired to the audited marker set (see :data:`SWE_CHAT_MARKERS`) rather than the
#: full 98-marker inventory. Without this the run would apply markers the audit rejected for having
#: a base rate of exactly zero on this corpus -- the "perfect group indicator with no background to
#: hide in" case the audit exists to catch. Re-point these at a ``WILDCHAT_MARKERS`` before running
#: the WildChat arm.
DEFENSES["collision_seeding_k4"] = CollisionSeedingDefense(
    n_profiles=4, marker_keys=SWE_CHAT_MARKERS)
DEFENSES["collision_seeding_k24"] = CollisionSeedingDefense(
    n_profiles=24, marker_keys=SWE_CHAT_MARKERS)
DEFENSES["collision_seeding_full"] = CollisionSeedingDefense(
    rate_min=1.0, rate_max=1.0, marker_keys=SWE_CHAT_MARKERS)
DEFENSES["collision_seeding_indep"] = CollisionSeedingDefense(
    independent=True, marker_keys=SWE_CHAT_MARKERS)

#: DP-MLM per-word privacy budgets exposed as a sweep. Each registers a ``dp_mlm_eps<eps>`` defense
#: selectable via ``--defense``. Because run_experiment.py's output_tag embeds the defense name, every
#: epsilon writes to its OWN results directory, and each caches separately since epsilon is in
#: ``params()``. The paper's set is {10,25,50,100,250}; the
#: higher values are added because DP-MLM only becomes near-readable at large epsilon (weaker privacy
#: -- the point of sweeping). ``dp_mlm`` itself defaults to eps=100, the readable end of the paper's
#: set, so it and ``dp_mlm_eps100`` produce the same output and share a cache entry (the cache is
#: keyed on the defense's ``name`` + ``params()``, not on the registry key).
#: Keep this in sync with experiments/run_dpmlm_sweep.sh.
DPMLM_SWEEP_EPSILONS = (10, 25, 50, 100, 250, 500, 1000)
for _eps in DPMLM_SWEEP_EPSILONS:
    DEFENSES[f"dp_mlm_eps{_eps}"] = DPMLMDefense(epsilon=_eps)
del _eps

#: DP-MLM adaptive-length variants (the paper's Algorithm 3): at the default eps, each eligible word
#: is deleted with probability ``del_prob`` and followed by an extra DP-drawn word with probability
#: ``add_prob``, so the rewrite no longer preserves word count. Registered as ``dp_mlm_var_a<A*100>``
#: at the paper's Appendix C grid (A in {0.1, 0.25}, D = 0.05); the plain ``dp_mlm`` and the epsilon
#: sweep above stay fixed-length, so "same eps, with vs without length variability" is a clean A/B.
DPMLM_VARLEN_ADD_PROBS = (0.1, 0.25)
DPMLM_VARLEN_DEL_PROB = 0.05
for _add in DPMLM_VARLEN_ADD_PROBS:
    DEFENSES[f"dp_mlm_var_a{int(round(_add * 100))}"] = DPMLMDefense(
        add_prob=_add, del_prob=DPMLM_VARLEN_DEL_PROB
    )
del _add

#: Leave-one-out content unlinkability, one entry per linkage budget. Unlike every sweep above, the
#: axis here is a *target* rather than a mechanism parameter: ``loo_unlink_b30`` edits each prompt
#: until its similarity to its author's other prompts has fallen 30%, or until the utility allowance
#: runs out. So the arms are not equally expensive, and a prompt is allowed to fail its budget
#: rather than be destroyed reaching for it -- ``edits.jsonl`` records which did.
#: ``loo_unlink`` itself is ``loo_unlink_b30`` and shares its cache (the key is name + params, not
#: the registry key), exactly as ``dp_mlm`` and ``dp_mlm_eps100`` do.
DEFENSES["loo_unlink"] = LOOUnlinkDefense()
for _budget in LOO_UNLINK_BUDGETS:
    DEFENSES[f"loo_unlink_b{int(round(_budget * 100)):02d}"] = LOOUnlinkDefense(budget=_budget)
del _budget

#: Agentic footprint reduction, one entry per residual-linkage level. The axis is a target like
#: ``loo_unlink``'s, but an ABSOLUTE one: ``afr_a00`` edits each prompt until it is no closer to its
#: author's earlier prompts than a stranger's prompt is, and ``afr_a50`` until half that excess is
#: gone. So the arms are not equally expensive, and a prompt is allowed to miss its target rather
#: than be destroyed reaching for one -- ``edits_a<NN>.jsonl`` records which did.
#:
#: ``afr_stage1`` is the control the whole defense stands on. It runs the same first-pass abstraction
#: with the same model and then stops, so ``afr`` vs ``afr_stage1`` isolates *the measurement loop*
#: rather than the model -- the ``none`` baseline is what isolates the model. If the loop buys
#: nothing over the abstraction pass, that is the finding, and this entry is how it gets reported.
#: ``afr`` itself is ``afr_a00`` and shares its cache (the key is name + params, not the registry
#: key), exactly as ``dp_mlm``/``dp_mlm_eps100`` and ``loo_unlink``/``loo_unlink_b30`` do.
DEFENSES["afr"] = AgenticFootprintDefense()
DEFENSES["afr_stage1"] = AgenticFootprintDefense(max_probes=0)
for _alpha in AFR_RESIDUALS:
    DEFENSES[f"afr_a{int(round(_alpha * 100)):02d}"] = AgenticFootprintDefense(alpha=_alpha)
del _alpha


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
    "CollisionSeedingDefense",
    "FrameShiftDefense",
    "FramePadDefense",
    "LOOUnlinkDefense",
    "LOO_UNLINK_BUDGETS",
    "AgenticFootprintDefense",
    "AFR_RESIDUALS",
]
