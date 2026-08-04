"""Featurizers: turn (possibly defended) conversation text into attack-ready vectors.

Featurization is a **build** stage now: it runs over a built (and optionally defended) split and
writes a feature parquet, which is all the experiment runner ever reads. What this package
exposes is therefore one layer -- the :class:`Featurizer` classes themselves
(:class:`StyloMetrixFeaturizer`, :class:`FunctionWordFeaturizer`,
:class:`CharacterStatisticsFeaturizer`, :class:`GeminiEmbedding2Featurizer`, ...), each a cached
``texts -> ndarray`` transform, plus the :data:`FEATURIZERS` registry and :func:`get_featurizer`
to build one by name.

The driver is :mod:`prompt_anonymity.data.compute_features`::

    python -m prompt_anonymity.data.compute_features --source swe_chat --feature stylometrix

which shards, caches every vector by content hash, and combines features by being run once per
feature (the runner's ``--feature`` then names the parquet to attack).

There used to be a second layer here, ``apply_featurizer``: it featurized both sides of an
in-memory :class:`~prompt_anonymity.core.AttackData`, reusing the committed vectors for text a
defense had left unchanged and concatenating several featurizers column-wise. Its only caller was
the fixed-split experiment runner, and it was removed with it on 2026-08-04 -- the reuse
optimisation it existed for is now structural rather than clever, because defended text gets its
own parquet and is featurized once, offline. Recover it from git history if an in-memory path is
ever wanted again.

Add a featurizer by writing a :class:`Featurizer` and registering its class in
``FEATURIZERS``.
"""

from __future__ import annotations


from .base import Featurizer
from .character import CharacterStatisticsFeaturizer
from .stylometrix import StyloMetrixFeaturizer
from .function_words import FunctionWordFeaturizer
from .char_ngram_tfidf import CharNgramTfidfFeaturizer
from .gemini_embedding import GeminiEmbedding001Featurizer, GeminiEmbedding2Featurizer
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
    "gemini_embedding_001": GeminiEmbedding001Featurizer,
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


__all__ = [
    "Featurizer",
    "StyloMetrixFeaturizer",
    "FunctionWordFeaturizer",
    "CharacterStatisticsFeaturizer",
    "GeminiEmbedding2Featurizer",
    "GeminiEmbedding001Featurizer",
    "FEATURIZERS",
    "get_featurizer",
    "StyleDistanceFeaturizer",
]
