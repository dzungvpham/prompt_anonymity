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
from .embad import EmBadDefense
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


# Registry of ready-to-use defenses, selectable by name (e.g. from a CLI argument). Model-backed
# defenses build their heavy backend lazily on first use, so registering them here is free and a
# fully-cached run loads no model.
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

#: Frame shift's single-frame ablation: the whole corpus is rewritten into one scene instead of
#: drawing from the codebook. It's the convergence-vs-dilution control -- the default dilutes each
#: author across many surface registers, this converges every document onto one -- and separates
#: "the framing diluted the author" from "everything simply got longer", since both arms inflate
#: length the same way.
DEFENSES["frame_shift_single"] = FrameShiftDefense(single_framing=SINGLE_FRAMING_KEY)

#: Frame pad's single-scene ablation, the same control one level down: every document's appended turn
#: is drawn from one scene's passages instead of from all of them. Read against ``frame_pad`` it asks
#: whether pad *diversity* matters or only pad presence.
DEFENSES["frame_pad_single"] = FramePadDefense(single_framing=SINGLE_FRAMING_KEY)

#: Collision-seeding variants. Pure Python string work (no model, no GPU), so a variant costs nothing
#: to add or run.
#:
#: The two K variants sweep the privacy knob: the codebook size sets the expected collision group at
#: ``n_authors / K``, so k4 buys larger groups and k24 smaller ones. The two ablations are controls,
#: not meant to be deployed:
#:
#: * ``_full`` applies each marker to 100% of an author's documents -- predicted to do WORSE than the
#:   default despite being a bigger edit, since perfect consistency is a perfectly reliable feature.
#: * ``_indep`` draws markers per author independently instead of from the codebook, giving a
#:   near-unique signature per author -- predicted to do worse than no defense at all, since a
#:   near-unique combination is a fingerprint. It's the control showing the codebook does the work.
#:
#: Every variant is wired to the audited marker set (:data:`SWE_CHAT_MARKERS`) rather than the full
#: inventory, since an unaudited marker can have zero base rate on this corpus and become a perfect
#: group indicator with no background to hide in. Re-point these at a ``WILDCHAT_MARKERS`` before
#: running the WildChat arm.
DEFENSES["collision_seeding_k4"] = CollisionSeedingDefense(
    n_profiles=4, marker_keys=SWE_CHAT_MARKERS)
DEFENSES["collision_seeding_k24"] = CollisionSeedingDefense(
    n_profiles=24, marker_keys=SWE_CHAT_MARKERS)
DEFENSES["collision_seeding_full"] = CollisionSeedingDefense(
    rate_min=1.0, rate_max=1.0, marker_keys=SWE_CHAT_MARKERS)
DEFENSES["collision_seeding_indep"] = CollisionSeedingDefense(
    independent=True, marker_keys=SWE_CHAT_MARKERS)

#: DP-MLM per-word privacy budgets exposed as a sweep. Each registers a ``dp_mlm_eps<eps>`` defense
#: selectable via ``--defense``, writing to its own results directory and caching separately since
#: epsilon is in ``params()``. ``dp_mlm`` itself defaults to eps=100, so it and ``dp_mlm_eps100``
#: produce the same output and share a cache entry (keyed on ``name`` + ``params()``, not the
#: registry key). Keep this in sync with experiments/run_dpmlm_sweep.sh.
DPMLM_SWEEP_EPSILONS = (10, 25, 50, 100, 250, 500, 1000)
for _eps in DPMLM_SWEEP_EPSILONS:
    DEFENSES[f"dp_mlm_eps{_eps}"] = DPMLMDefense(epsilon=_eps)
del _eps

#: DP-MLM adaptive-length variants: at the default eps, each eligible word is deleted with
#: probability ``del_prob`` and followed by an extra DP-drawn word with probability ``add_prob``, so
#: the rewrite no longer preserves word count. Registered as ``dp_mlm_var_a<A*100>``; the plain
#: ``dp_mlm`` and the epsilon sweep above stay fixed-length, so this isolates length variability
#: as its own axis.
DPMLM_VARLEN_ADD_PROBS = (0.1, 0.25)
DPMLM_VARLEN_DEL_PROB = 0.05
for _add in DPMLM_VARLEN_ADD_PROBS:
    DEFENSES[f"dp_mlm_var_a{int(round(_add * 100))}"] = DPMLMDefense(
        add_prob=_add, del_prob=DPMLM_VARLEN_DEL_PROB
    )
del _add

#: Leave-one-out content unlinkability, one entry per linkage budget. Unlike the sweeps above, the
#: axis here is a *target*, not a mechanism parameter: ``loo_unlink_b30`` edits each prompt until its
#: similarity to its author's other prompts has fallen 30%, or until the utility allowance runs out --
#: so a prompt may fail its budget rather than be destroyed reaching for it (``edits.jsonl`` records
#: which did). ``loo_unlink`` itself is ``loo_unlink_b30`` and shares its cache.
DEFENSES["loo_unlink"] = LOOUnlinkDefense()
for _budget in LOO_UNLINK_BUDGETS:
    DEFENSES[f"loo_unlink_b{int(round(_budget * 100)):02d}"] = LOOUnlinkDefense(budget=_budget)
del _budget

#: Agentic footprint reduction, one entry per residual-linkage level. Like ``loo_unlink``'s, the axis
#: is a target, but an absolute one: ``afr_a00`` edits each prompt until it is no closer to its
#: author's earlier prompts than a stranger's prompt is, and ``afr_a50`` until half that excess is
#: gone (``edits_a<NN>.jsonl`` records prompts that missed their target).
#:
#: ``afr_stage1`` runs the same first-pass abstraction with the same model and then stops, isolating
#: the measurement loop from the model -- the ``none`` baseline is what isolates the model itself.
#: ``afr`` itself is ``afr_a00`` and shares its cache.
DEFENSES["afr"] = AgenticFootprintDefense()
DEFENSES["afr_stage1"] = AgenticFootprintDefense(max_probes=0)
for _alpha in AFR_RESIDUALS:
    DEFENSES[f"afr_a{int(round(_alpha * 100)):02d}"] = AgenticFootprintDefense(alpha=_alpha)
del _alpha


#: EmBad evolves an appended turn -- an island-model MAP-Elites search over natural-language
#: passages, scored on the local embedding ensemble. The registry name carries the ensemble because
#: it is in the filename: a defended split is ``<split>_<defense>.parquet``, and a turn evolved
#: against one set of encoders is not the same artifact as one evolved against another.
EMBAD_ENSEMBLE = ("harrier", "embeddinggemma_300m", "jina_v5_nano")

#: One registered name per **objective**, since the objective is the arm being compared and the
#: defended file is named ``<split>_<defense>.parquet`` -- three arms under one name would overwrite
#: each other's parquet even though their searches cache separately.
#:
#: ``embad_gemini`` scores against the target encoder over the network and **bills real money**
#: (capped by ``DEFAULT_REMOTE_BUDGET``). Constructing it costs nothing (the objective is built
#: lazily); running it spends.
DEFENSES["embad"] = EmBadDefense()
DEFENSES["embad_summary"] = EmBadDefense(objective="summary")
DEFENSES["embad_gemini"] = EmBadDefense(objective="remote")

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
    "EmBadDefense",
    "CollisionSeedingDefense",
    "FrameShiftDefense",
    "FramePadDefense",
    "LOOUnlinkDefense",
    "LOO_UNLINK_BUDGETS",
    "AgenticFootprintDefense",
    "AFR_RESIDUALS",
]
