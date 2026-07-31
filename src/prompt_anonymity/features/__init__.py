"""Featurizers: turn (possibly defended) conversation text into attack-ready vectors.

Featurization is the third pipeline stage: it runs **after** the defense, so it sees the
current (possibly rewritten) text. Two layers are exposed:

* the :class:`Featurizer` classes themselves (:class:`StyloMetrixFeaturizer`,
  :class:`FunctionWordFeaturizer`, :class:`CharacterStatisticsFeaturizer`,
  :class:`GeminiEmbedding2Featurizer`), each a cached ``texts -> ndarray`` transform; and
* :func:`apply_featurizer`, which featurizes both sides of a loaded
  :class:`~prompt_anonymity.core.AttackData` and returns a copy carrying the new vectors
  (the caller-chosen ``metric`` already on the data is left untouched).

Reusing precomputed features
----------------------------
Loaders return an :class:`AttackData` whose embeddings are the committed precomputed features
for the *original* text. Pass that loaded object to :func:`apply_featurizer` as ``reference``:
text a defense left unchanged keeps its precomputed vector (no recompute), and only text a
defense rewrote is sent to the featurizer (and cached on disk by content hash). With no
defense this is a pure pass-through -- no GPU, exact reproduction of the committed numbers.

Combining featurizers
---------------------
:func:`apply_featurizer` accepts a single featurizer **or a list of them**. Given several, it
featurizes both sides with each one *independently* -- so every featurizer reuses its own
on-disk cache and, for the featurizer whose space matches the committed vectors, the loaded
``reference`` -- and then concatenates their per-conversation vectors column-wise (in the order
listed) into one wider representation. This is the supported way to stack signals (e.g.
StyloMetrix + function words): there is no need to write a bespoke "combined" featurizer
class, and each component's cache is reused rather than recomputed. The distance metric is not
a featurizer concern -- it is carried on the :class:`AttackData` and chosen by the caller.

Add a featurizer by writing a :class:`Featurizer` and registering its class in
``FEATURIZERS``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence, Union

import numpy as np

from ..core import AttackData
from .base import Featurizer
from .character import CharacterStatisticsFeaturizer
from .stylometrix import StyloMetrixFeaturizer
from .function_words import FunctionWordFeaturizer
from .char_ngram_tfidf import CharNgramTfidfFeaturizer
from .gemini_embedding import GeminiEmbedding2Featurizer
from .style_distance import StyleDistanceFeaturizer

# Registry of featurizer classes, selectable by name (e.g. from a CLI argument). Values are
# classes (not instances) because a featurizer may need configuration -- e.g. StyloMetrix's
# language_code -- supplied when it is built; see :func:`get_featurizer`.
FEATURIZERS: dict[str, type[Featurizer]] = {
    "stylometrix": StyloMetrixFeaturizer,
    "character_statistics": CharacterStatisticsFeaturizer,
    "function_words": FunctionWordFeaturizer,
    "char_ngram_tfidf": CharNgramTfidfFeaturizer,
    "style_distance": StyleDistanceFeaturizer,
    "gemini_embedding_2": GeminiEmbedding2Featurizer,
}

# A loaded AttackData carries committed features for exactly one representation (StyloMetrix
# today), so only a featurizer producing that same space can reuse them as a ``reference``;
# every other featurizer -- including ``gemini_embedding_2``, whose vectors are computed by
# ``prompt_anonymity.data.compute_features`` and stored in their own parquet -- is recomputed
# from its own on-disk cache. Extend this when a loader starts returning another representation.
_REFERENCE_FEATURE = "stylometrix"


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


def _featurize_both_sides(featurizer, data, cache_dir, reference) -> tuple[np.ndarray, np.ndarray]:
    """Featurize the known and unknown sides with one featurizer, reusing caches.

    Returns ``(known_vectors, unknown_vectors)``. This featurizer's on-disk cache -- namespaced
    by its own ``name``, so combined featurizers never share entries -- is opened at most once,
    and only when something must actually be recomputed (a pure reference-reuse run never touches
    disk). The loaded ``reference`` embeddings are reused only by the featurizer whose space
    matches them (:data:`_REFERENCE_FEATURE`); every other featurizer recomputes from its own
    cache.
    """
    # Open the cache lazily and at most once for this featurizer: a no-defense StyloMetrix run
    # reuses the reference for both sides and so never creates a cache namespace.
    cache_box: list = []

    def open_cache():
        if not cache_box:
            cache_box.append(featurizer.open_cache(cache_dir))
        return cache_box[0]

    # Only reuse the committed embeddings when this featurizer produces that same feature space.
    reuse_reference = reference is not None and featurizer.name == _REFERENCE_FEATURE
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
    return known, unknown


def apply_featurizer(
    featurizers: Union[str, Featurizer, Sequence[Union[str, Featurizer]]],
    data: AttackData,
    *,
    cache_dir,
    reference: AttackData | None = None,
) -> AttackData:
    """Featurize both sides of ``data`` and return a copy carrying the new feature vectors.

    Parameters
    ----------
    featurizers : str, Featurizer, or sequence of them
        One featurizer, or several to combine. A name is built with defaults via
        :func:`get_featurizer`; a :class:`Featurizer` instance is used as given. Given several,
        each featurizes both sides independently (reusing its own on-disk cache and, where it
        matches, the committed ``reference``) and their vectors are concatenated column-wise --
        in the order listed -- into one wider representation.
    data : AttackData
        The split to featurize. Its ``known_texts`` / ``unknown_texts`` (the current,
        post-defense text) are required.
    cache_dir : str or pathlib.Path
        Where to cache freshly computed vectors (each featurizer under its own
        ``<cache_dir>/features/<name>`` namespace).
    reference : AttackData, optional
        The loaded (pre-defense) split, whose committed embeddings are reused for any text a
        defense left unchanged -- typically the same object returned by the loader. Only the
        featurizer matching the committed feature space (StyloMetrix) reuses it; omit to
        featurize every conversation from scratch.

    Returns
    -------
    AttackData
        A copy of ``data`` with ``known_embeddings`` / ``unknown_embeddings`` set to the
        featurizer output (concatenated column-wise when several are given). ``metric`` is left
        unchanged -- it is an attack-level choice, not derived from the featurizer.
    """
    if isinstance(featurizers, (str, Featurizer)):
        featurizers = [featurizers]
    featurizers = [f if isinstance(f, Featurizer) else get_featurizer(f) for f in featurizers]
    if not featurizers:
        raise ValueError("apply_featurizer needs at least one featurizer.")
    if data.known_texts is None or data.unknown_texts is None:
        raise ValueError("featurization needs known_texts and unknown_texts; load the dataset with text.")

    # Featurize each side with every featurizer, then stack the vectors column-wise. Each
    # featurizer reuses its own cache and the committed reference (see _featurize_both_sides).
    known_parts, unknown_parts = [], []
    for featurizer in featurizers:
        known, unknown = _featurize_both_sides(featurizer, data, cache_dir, reference)
        known_parts.append(known)
        unknown_parts.append(unknown)

    known = known_parts[0] if len(known_parts) == 1 else np.concatenate(known_parts, axis=1)
    unknown = unknown_parts[0] if len(unknown_parts) == 1 else np.concatenate(unknown_parts, axis=1)
    # replace() re-runs AttackData validation, so a featurizer returning the wrong shape is
    # caught here rather than silently corrupting the attack. The metric is deliberately left
    # untouched: it is set at the experiment level, not by the featurizer.
    return replace(data, known_embeddings=known, unknown_embeddings=unknown)


__all__ = [
    "Featurizer",
    "StyloMetrixFeaturizer",
    "FunctionWordFeaturizer",
    "CharacterStatisticsFeaturizer",
    "GeminiEmbedding2Featurizer",
    "FEATURIZERS",
    "get_featurizer",
    "apply_featurizer",
    "StyleDistanceFeaturizer",
    "_REFERENCE_FEATURE",
]
