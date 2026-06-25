"""Base classes that give a defense automatic, auto-invalidating caching.

A developer writing a new defense subclasses :class:`CachedDefense` (or the text-rewrite
convenience :class:`CachedTextRewriteDefense`); the base classes compute the cache
fingerprint from the subclass's source + ``version`` + ``params()`` and hand the subclass a
ready :class:`~prompt_anonymity.caching.TransformCache`. The developer never touches cache
keys or invalidation -- editing the defense's code (or bumping ``version``) is enough to
invalidate its cache.

A defense transforms **text only**. Features are computed afterwards by a separate stage
(:mod:`prompt_anonymity.features`), so a text-rewrite defense returns an
:class:`~prompt_anonymity.core.AttackData` whose embeddings are now stale -- the caller must
re-featurize before attacking (the experiment driver does this automatically).
"""

from __future__ import annotations

from dataclasses import replace

from pathlib import Path

import numpy as np

from ..caching import TransformCache, logic_hash, params_hash
from ..core import AttackData


class CachedDefense:
    """Base class for defenses whose transform is expensive enough to cache on disk.

    Subclasses set :attr:`name` (required), optionally :attr:`version`, may override
    :meth:`params` to declare configuration that affects the output, and implement
    :meth:`transform` using the provided cache. The cache auto-invalidates whenever the
    subclass's (or a base class's) source code, :attr:`version`, or :meth:`params` change.
    """

    #: Cache namespace for this defense; must be set by the subclass.
    name: str = ""
    #: Manual logic version. Bump it when behavior changes in a way source hashing can't see
    #: (e.g. a helper module or an external model the defense calls was updated).
    version: str = ""

    def params(self) -> dict:
        """Configuration that affects the output (e.g. ``{"pivot_language": "fr"}``).

        Included in the cache key so different configurations are cached separately, and
        recorded so a run is reproducible. Must be JSON-serializable.
        """
        return {}

    def transform(self, data: AttackData, cache: TransformCache) -> AttackData:
        """Produce the defended :class:`AttackData`, using ``cache`` for the expensive work
        (typically ``cache.apply(items, expensive_fn)``). Implemented by subclasses."""
        raise NotImplementedError

    def _logic_hash(self) -> str:
        # Hash the whole class hierarchy's source (so edits to a base class also invalidate)
        # plus the explicit version.
        classes = [cls for cls in type(self).__mro__ if cls is not object]
        return logic_hash(classes, version=self.version)

    def __call__(self, data: AttackData, *, cache_dir) -> AttackData:
        """Apply the defense, caching under ``<cache_dir>/defenses``."""
        if not self.name:
            raise ValueError(f"{type(self).__name__} must set a non-empty class attribute `name`.")
        cache = TransformCache(
            Path(cache_dir) / "defenses", self.name, self._logic_hash(), params_hash(self.params())
        )
        return self.transform(data, cache)


class CachedTextRewriteDefense(CachedDefense):
    """Convenience base for defenses that rewrite each conversation's text (e.g. translation).

    The subclass implements just :meth:`rewrite_text` -- the expensive per-text op, which is
    cached per item. By default only the anonymous *unknown* side is rewritten (the *known*
    side is the adversary's untouched reference); set :attr:`rewrite_known` to rewrite both.

    A defense does **not** featurize. After it runs, the returned :class:`AttackData` carries
    the rewritten text but its embeddings are stale, so the rewritten side must be
    re-featurized by :func:`prompt_anonymity.features.apply_featurizer` before the attack (the
    experiment driver always does this).
    """

    #: Rewrite the known side too (keep ``False`` for the usual threat model, where the
    #: adversary's known conversations are left as released).
    rewrite_known: bool = False

    def rewrite_text(self, text: str) -> str:
        """The expensive per-conversation transformation (e.g. a translation call). Cached."""
        raise NotImplementedError

    def transform(self, data: AttackData, cache: TransformCache) -> AttackData:
        if data.unknown_texts is None:
            raise ValueError(f"defense {self.name!r} needs unknown_texts; load the dataset with text.")
        changes = {"unknown_texts": np.asarray(cache.apply(list(data.unknown_texts), self.rewrite_text), dtype=object)}
        if self.rewrite_known:
            if data.known_texts is None:
                raise ValueError(f"defense {self.name!r} has rewrite_known=True but no known_texts.")
            changes["known_texts"] = np.asarray(cache.apply(list(data.known_texts), self.rewrite_text), dtype=object)
        # Only text changes; embeddings are intentionally left stale for the featurize stage to
        # recompute (replace() re-validates lengths so a rewrite that drops rows is caught).
        return replace(data, **changes)
