"""Base class for featurizers: turning conversation text into attack-ready feature vectors.

A *featurizer* runs **after** any defense, so it always sees the current (possibly rewritten)
text and produces the vectors the attack compares. Featurization is the natural place for the
distance metric to be decided -- it is a property of the representation, not the dataset --
so a featurizer also declares the ``metric`` its vectors call for.

A developer adds a featurizer by subclassing :class:`Featurizer`, setting :attr:`name`,
:attr:`metric` and (optionally) :attr:`version`, overriding :meth:`params` for any
output-affecting configuration, and implementing :meth:`featurize` -- a **batch** transform
that turns a list of texts into a 2-D array. Expensive results are cached on disk,
content-addressed by text, with the same automatic, safe invalidation as defenses (see
:mod:`prompt_anonymity.caching`); the developer never touches cache keys.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..caching import TransformCache, logic_hash, params_hash


class Featurizer:
    """Maps conversation text to fixed-length feature vectors, with automatic caching.

    Subclasses set :attr:`name` (required) and :attr:`metric`, optionally :attr:`version`,
    may override :meth:`params`, and implement :meth:`featurize`. The cache auto-invalidates
    whenever the subclass's (or a base class's) source code, :attr:`version`, or :meth:`params`
    change.
    """

    #: Cache namespace for this featurizer; must be set by the subclass.
    name: str = ""
    #: Manual logic version. Bump it when behavior changes in a way source hashing cannot see
    #: (e.g. an upgraded StyloMetrix / spaCy model that shifts the feature columns).
    version: str = ""
    #: Distance metric the produced vectors call for ("euclidean" for StyloMetrix, "cosine"
    #: for L2-normalizable embeddings); written onto the :class:`AttackData` by the featurize
    #: step so the attack uses the right metric.
    metric: str = "euclidean"

    def params(self) -> dict:
        """Configuration that affects the output (e.g. ``{"language_code": "ru"}``).

        Included in the cache key so different configurations are cached separately, and
        recorded so a run is reproducible. Must be JSON-serializable.
        """
        return {}

    def featurize(self, texts) -> np.ndarray:
        """Compute a ``(len(texts), n_features)`` array from a sequence of texts.

        This is the expensive **batch** op, implemented by subclasses (e.g. one StyloMetrix /
        spaCy pass over all texts). Only cache-missing texts are passed in, so it never needs
        its own caching. Column order must be stable across calls so cached and freshly
        computed rows share a feature space.
        """
        raise NotImplementedError

    def _logic_hash(self) -> str:
        # Hash the whole class hierarchy's source (so edits to a base class also invalidate)
        # plus the explicit version.
        classes = [cls for cls in type(self).__mro__ if cls is not object]
        return logic_hash(classes, version=self.version)

    def open_cache(self, cache_dir) -> TransformCache:
        """Open this featurizer's on-disk cache under ``cache_dir`` (namespaced by name, logic
        version and params). Featurizer caches live under ``<cache_dir>/features``."""
        if not self.name:
            raise ValueError(f"{type(self).__name__} must set a non-empty class attribute `name`.")
        return TransformCache(
            Path(cache_dir) / "features", self.name, self._logic_hash(), params_hash(self.params())
        )

    def transform(self, texts, cache: TransformCache) -> np.ndarray:
        """Featurize ``texts`` (a sequence), caching each text's vector by content hash.

        Returns a ``(len(texts), n_features)`` float array. Only texts absent from ``cache``
        are sent to :meth:`featurize`; the rest are served from disk.
        """
        texts = list(texts)
        if not texts:
            return np.empty((0, 0), dtype=float)

        def batch(missing_texts):
            vectors = np.asarray(self.featurize(missing_texts), dtype=float)
            return [row.tolist() for row in vectors]  # JSON-serializable per-text outputs

        outputs = cache.apply_batch(texts, batch, key=str)
        return np.asarray(outputs, dtype=float)
