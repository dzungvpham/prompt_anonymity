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

Add a featurizer by writing a :class:`Featurizer` and registering its class in
``FEATURIZERS``.

The one exception to "featurization is a build stage" is :data:`KNOWN_SIDE_FEATURES`: features
whose vocabulary and weights are *fitted* (``char_ngram_tfidf``). Fitting them offline would fit
on the test documents too, so the experiment runner fits them itself, per known configuration.
"""

from __future__ import annotations


from .base import Featurizer
from .character import CharacterStatisticsFeaturizer
from .stylometrix import StyloMetrixFeaturizer
from .function_words import FunctionWordFeaturizer
from .char_ngram_tfidf import CharNgramTfidf, KnownSideTfidf
from .gemini_embedding import GeminiEmbedding001Featurizer, GeminiEmbedding2Featurizer
from .harrier import (HarrierFeaturizer, HarrierImperativeFeaturizer, HarrierPlainFeaturizer)
from .sentence_transformer import (EmbeddingGemma300mFeaturizer, Harrier270mFeaturizer,
                                   JinaV5NanoFeaturizer, SentenceTransformerFeaturizer)
from .style_distance import StyleDistanceFeaturizer
from .luar import LuarFeaturizer

# Registry of featurizer classes, selectable by name. Values are classes (not instances) because
# a featurizer may need configuration (e.g. StyloMetrix's language_code); see :func:`get_featurizer`.
FEATURIZERS: dict[str, type[Featurizer]] = {
    "gemini_embedding_2": GeminiEmbedding2Featurizer,
    "stylometrix": StyloMetrixFeaturizer,
    "character_statistics": CharacterStatisticsFeaturizer,
    "function_words": FunctionWordFeaturizer,
    "style_distance": StyleDistanceFeaturizer,
    "luar": LuarFeaturizer,
    # Local, offline, instruction-conditioned encoders used by the leave-one-out defense; see
    # harrier.py.
    "harrier": HarrierFeaturizer,
    "harrier_imperative": HarrierImperativeFeaturizer,
    "harrier_plain": HarrierPlainFeaturizer,
    "harrier_270m": Harrier270mFeaturizer,
    "embeddinggemma_300m": EmbeddingGemma300mFeaturizer,
    "jina_v5_nano": JinaV5NanoFeaturizer,
}

#: Features that are **fitted** rather than computed per document, so they cannot be precomputed
#: into a parquet without letting the test documents shape the feature space. The experiment
#: runner builds these itself, fitting one per known configuration on that configuration's known
#: documents only (see :mod:`.char_ngram_tfidf`). Keep the names disjoint from ``FEATURIZERS``.
KNOWN_SIDE_FEATURES: dict[str, type] = {
    "char_ngram_tfidf": CharNgramTfidf,
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
    "KNOWN_SIDE_FEATURES",
    "CharNgramTfidf",
    "KnownSideTfidf",
    "get_featurizer",
    "StyleDistanceFeaturizer",
    "LuarFeaturizer",
    "HarrierFeaturizer",
    "HarrierImperativeFeaturizer",
    "HarrierPlainFeaturizer",
    "_REFERENCE_FEATURE",
]
