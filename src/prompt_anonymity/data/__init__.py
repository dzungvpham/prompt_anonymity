"""Dataset loaders and split rules.

Each loader reads a dataset from disk, builds identities, applies a dataset-specific
known/unknown split, and returns an :class:`~prompt_anonymity.core.AttackData` ready for
a defense and an attack. Add a dataset by writing a loader (reusing the split helpers in
:mod:`prompt_anonymity.data.splits`) and registering it in ``DATASET_LOADERS``.
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
    "swe-chat": load_swe_chat,
}


def load_dataset(name: str, data_dir, **options) -> AttackData:
    """Load a dataset by name, forwarding dataset-specific keyword options to its loader.

    Parameters
    ----------
    name : str
        Dataset key in ``DATASET_LOADERS`` (e.g. ``"wildchat"`` or ``"swe-chat"``).
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
