"""Featurizers: turn (possibly defended) conversation text into attack-ready vectors.

Featurization is the third pipeline stage -- it runs **after** the defense, so it sees the
current text -- and it owns the distance metric the attack will use. Two layers are exposed:

* the :class:`Featurizer` classes themselves (:class:`StyloMetrixFeaturizer`,
  :class:`CharacterStatisticsFeaturizer`), each a cached ``texts -> ndarray`` transform; and
* :func:`apply_featurizer`, which featurizes both sides of a loaded
  :class:`~prompt_anonymity.core.AttackData` and returns a copy carrying the vectors and the
  featurizer's metric.

Reusing precomputed features
----------------------------
Loaders return an :class:`AttackData` whose embeddings are the committed precomputed features
for the *original* text. Pass that loaded object to :func:`apply_featurizer` as ``reference``:
text a defense left unchanged keeps its precomputed vector (no recompute), and only text a
defense rewrote is sent to the featurizer (and cached on disk by content hash). With no
defense this is a pure pass-through -- no GPU, exact reproduction of the committed numbers.

Add a featurizer by writing a :class:`Featurizer` and registering its class in
``FEATURIZERS``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Union

import numpy as np

from ..core import AttackData
from .base import Featurizer
from .character import CharacterStatisticsFeaturizer
from .stylometrix import StyloMetrixFeaturizer
from .function_words import FunctionWordFeaturizer
from .stylometrix_func import StyloFuncFeaturizer

# Registry of featurizer classes, selectable by name (e.g. from a CLI argument). Values are
# classes (not instances) because a featurizer may need configuration -- e.g. StyloMetrix's
# language_code -- supplied when it is built; see :func:`get_featurizer`.
FEATURIZERS: dict[str, type[Featurizer]] = {
    "stylometrix": StyloMetrixFeaturizer,
    "character_statistics": CharacterStatisticsFeaturizer,
    "function_words": FunctionWordFeaturizer,
    "stylometrix_func": StyloFuncFeaturizer,
}


def get_featurizer(name: str, **options) -> Featurizer:
    """Build a registered featurizer by name, forwarding ``options`` to its constructor.

    For example ``get_featurizer("stylometrix", language_code="ru")``.
    """
    try:
        featurizer_class = FEATURIZERS[name]
    except KeyError:
        raise ValueError(f"unknown featurizer {name!r}; available: {sorted(FEATURIZERS)}") from None
    return featurizer_class(**options)


def _featurize_side(featurizer, open_cache, texts, reference_texts, reference_embeddings) -> np.ndarray:
    """Featurize one side (known or unknown), reusing reference vectors for unchanged text.

    With no usable reference, every text is featurized (cached by content). Otherwise only
    texts that differ from their reference are recomputed; the rest keep their reference
    vector. The recomputed columns must match the reference's feature dimension. ``open_cache``
    is a zero-arg callable opening the (shared) on-disk cache, invoked only when something
    actually needs recomputing -- so a pure reference-reuse run never touches the cache dir.
    """
    texts = list(texts)
    if reference_texts is None or reference_embeddings is None:
        return featurizer.transform(texts, open_cache())

    reference_texts = list(reference_texts)
    reference_embeddings = np.asarray(reference_embeddings, dtype=float)
    if len(reference_texts) != len(texts):
        # A defense changed the row count, so positions no longer line up with the reference;
        # fall back to featurizing everything.
        return featurizer.transform(texts, open_cache())

    changed = [i for i, text in enumerate(texts) if text != reference_texts[i]]
    if not changed:
        return reference_embeddings.copy()

    recomputed = featurizer.transform([texts[i] for i in changed], open_cache())
    if recomputed.shape[1] != reference_embeddings.shape[1]:
        raise ValueError(
            f"featurizer {featurizer.name!r} produced {recomputed.shape[1]} features but the loaded "
            f"reference has {reference_embeddings.shape[1]}; the featurizer must match the loaded "
            f"feature space (use the same feature the dataset was loaded with)."
        )
    out = reference_embeddings.copy()
    out[changed] = recomputed
    return out


def apply_featurizer(
    featurizer: Union[str, Featurizer],
    data: AttackData,
    *,
    cache_dir,
    reference: AttackData | None = None,
) -> AttackData:
    """Featurize both sides of ``data`` and return a copy with the vectors and the metric.

    Parameters
    ----------
    featurizer : str or Featurizer
        A registered name (built with defaults) or a ready :class:`Featurizer` instance.
    data : AttackData
        The split to featurize. Its ``known_texts`` / ``unknown_texts`` (the current,
        post-defense text) are required.
    cache_dir : str or pathlib.Path
        Where to cache freshly computed vectors (under ``<cache_dir>/features``).
    reference : AttackData, optional
        The loaded (pre-defense) split, whose embeddings are reused for any text a defense left
        unchanged -- typically the same object returned by the loader. Omit to featurize every
        conversation from scratch.

    Returns
    -------
    AttackData
        A copy of ``data`` with ``known_embeddings`` / ``unknown_embeddings`` set to the
        featurizer's output and ``metric`` set to the featurizer's metric.
    """
    featurizer = featurizer if isinstance(featurizer, Featurizer) else get_featurizer(featurizer)
    if data.known_texts is None or data.unknown_texts is None:
        raise ValueError("featurization needs known_texts and unknown_texts; load the dataset with text.")

    # Open the cache lazily and at most once: a no-defense run reuses the reference for both
    # sides and so never creates a cache namespace.
    cache_box: list = []

    def open_cache():
        if not cache_box:
            cache_box.append(featurizer.open_cache(cache_dir))
        return cache_box[0]

    # Only reuse committed embeddings when the requested featurizer is the same
    # feature space as the committed dataset embeddings.
    reuse_reference = (
        reference is not None
        and featurizer.name == "stylometrix"
    )

    reference_known_texts = reference.known_texts if reuse_reference else None
    reference_unknown_texts = reference.unknown_texts if reuse_reference else None
    reference_known_embeddings = reference.known_embeddings if reuse_reference else None
    reference_unknown_embeddings = reference.unknown_embeddings if reuse_reference else None
    known = _featurize_side(
        featurizer, open_cache, data.known_texts, reference_known_texts, reference_known_embeddings
    )
    unknown = _featurize_side(
        featurizer, open_cache, data.unknown_texts, reference_unknown_texts, reference_unknown_embeddings
    )
    # replace() re-runs AttackData validation, so a featurizer returning the wrong shape is
    # caught here rather than silently corrupting the attack.
    return replace(data, known_embeddings=known, unknown_embeddings=unknown, metric=featurizer.metric)


__all__ = [
    "Featurizer",
    "StyloMetrixFeaturizer",
    "CharacterStatisticsFeaturizer",
    "FEATURIZERS",
    "get_featurizer",
    "apply_featurizer",
    "FunctionWordFeaturizer",
    "StyloFuncFeaturizer",
]
