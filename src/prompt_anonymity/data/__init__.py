"""Dataset loaders, split rules, and the dataset build pipeline.

This package holds two related but separate families of modules.

**Loaders** (this module, :mod:`~prompt_anonymity.data.splits`,
:mod:`~prompt_anonymity.data.wildchat`, :mod:`~prompt_anonymity.data.swe_chat`) feed the attack
pipeline: each reads a dataset from disk, builds identities, applies a dataset-specific
known/unknown split, and returns an :class:`~prompt_anonymity.core.AttackData` ready for a defense
and an attack. Add a dataset by writing a loader (reusing the split helpers in
:mod:`prompt_anonymity.data.splits`) and registering it in ``DATASET_LOADERS``.

**The build pipeline** produces the unified public dataset those loaders (and the experiment
runners) consume, from the raw upstream corpora. Its entry points are meant to be run as scripts,
in this order:

    python -m prompt_anonymity.data.build_dataset       # raw sources  -> data/dist/*.parquet
    python -m prompt_anonymity.data.validate_dataset    # integrity checks on what was built
    python -m prompt_anonymity.data.compute_features    # -> data/dist/<split>_<feature>.parquet

with :mod:`~prompt_anonymity.data.sources_wildchat` / :mod:`~prompt_anonymity.data.sources_swe_chat`
adapting one upstream corpus each, and :mod:`~prompt_anonymity.data.text_cleaning`,
:mod:`~prompt_anonymity.data.identity`, :mod:`~prompt_anonymity.data.dedup` and
:mod:`~prompt_anonymity.data.language_detection` implementing the shared stages.
:mod:`~prompt_anonymity.data.download` mirrors the published dataset back down, and
:mod:`~prompt_anonymity.data.find_fragments` is a one-off study of identity fragmentation.

:mod:`~prompt_anonymity.data.apply_defenses` is an optional fourth step, run between the last two:
it rewrites a built split's conversations with one registered defense
(:mod:`prompt_anonymity.defenses`) and writes them back in the *same schema* at
``data/dist/defended/<defense>/<split>.parquet`` -- a drop-in replacement, so
``compute_features --dist-dir data/dist/defended/<defense>`` featurizes the defended text with no
further changes and the attack pipeline can then be run against it.

**Code lives here; data does not.** Where the raw inputs are read from and where the outputs are
written is configuration, not a constant -- see :mod:`prompt_anonymity.data.config`, which resolves
both (falling back to downloading the raw sources from HuggingFace) so the build runs on any
machine.
"""

from __future__ import annotations

from ..core import AttackData
from .splits import (
    KNOWN,
    UNKNOWN,
    attackable_mask,
    build_attack_data,
    split_by_last_conversation,
    split_by_model,
)
from .swe_chat import load_swe_chat
from .wildchat import load_wildchat

# Registry so callers can select a dataset by name (e.g. from a CLI argument).
DATASET_LOADERS = {
    "wildchat": load_wildchat,
    "swe_chat": load_swe_chat,
}


def load_dataset(name: str, data_dir, **options) -> AttackData:
    """Load a dataset by name, forwarding dataset-specific keyword options to its loader.

    Parameters
    ----------
    name : str
        Dataset key in ``DATASET_LOADERS`` (e.g. ``"wildchat"`` or ``"swe_chat"``).
    data_dir : str or pathlib.Path
        Directory holding that dataset's files.
    **options
        Loader-specific options, e.g. ``language=`` for WildChat or ``model_owner=`` for
        SWE-chat (both take ``feature=``).
    """
    try:
        loader = DATASET_LOADERS[name]
    except KeyError:
        raise ValueError(f"unknown dataset {name!r}; available: {sorted(DATASET_LOADERS)}") from None
    return loader(data_dir, **options)


__all__ = [
    "AttackData",
    "DATASET_LOADERS",
    "load_dataset",
    "load_wildchat",
    "load_swe_chat",
    "split_by_model",
    "split_by_last_conversation",
    "attackable_mask",
    "build_attack_data",
    "KNOWN",
    "UNKNOWN",
]
