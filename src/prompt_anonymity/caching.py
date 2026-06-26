"""Safe, auto-invalidating on-disk cache for expensive per-item transforms.

Two stages of the pipeline do costly per-conversation work that is wasteful to repeat:
defenses (e.g. translating each prompt) and featurizers (e.g. running StyloMetrix on GPU).
Both reuse this cache, which stores per-item outputs on disk while guaranteeing a stale
result is never served:

* **Content-addressed** -- each item is keyed by a hash of its input, so a cached output
  always corresponds to the exact input it was computed from (e.g. a prompt rewritten by a
  defense is cached under the hash of that rewritten text).
* **Auto-invalidating on logic change** -- the cache path is namespaced by a hash of the
  producer's *source code* (its whole class hierarchy) plus an optional explicit ``version``.
  Edit the defense/featurizer and the namespace changes, so old entries are ignored; the
  stale namespaces are also pruned from disk ("wiped"). Correctness never depends on the
  pruning -- a stale entry is unreachable by key regardless.
* **Parameter-aware** -- different configurations (e.g. target language) get separate
  sub-namespaces under the same logic version, so they coexist instead of evicting each
  other.
* **Crash / concurrency safe** -- writes go to a temp file and are atomically renamed, and
  an unreadable entry is treated as a miss and recomputed (never a crash, never a partial
  read).

What source hashing can and cannot see: editing any method of the producer's own class
hierarchy flips its namespace automatically. It does *not* see changes inside helper
functions/modules it calls or in external models (e.g. an upgraded translation model or a
new StyloMetrix release) -- bump ``version`` for those.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import tempfile
import textwrap
import time
from pathlib import Path

# Length (hex chars) of the short hashes used for directory names; 16 = 64 bits, ample to
# avoid accidental collisions while keeping paths readable.
_SHORT = 16


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_digest(obj) -> str:
    """Hash of a callable's or class's source, normalized so cosmetic indentation does not
    churn the cache but logic (and comment/docstring) edits do.

    Falls back to the qualified name when the source is unavailable (e.g. a builtin or an
    object defined interactively); such objects then invalidate only via ``version``.
    """
    try:
        source = inspect.getsource(obj)
    except (TypeError, OSError):
        return _sha256(f"{getattr(obj, '__module__', '?')}.{getattr(obj, '__qualname__', repr(obj))}")
    return _sha256(textwrap.dedent(source).strip())


def logic_hash(sources, *, version: str = "") -> str:
    """Short hash identifying a producer's *logic version*: the source of each object in
    ``sources`` (typically a defense's or featurizer's class hierarchy) plus an explicit
    ``version`` string. Any change flips the hash, so the cache moves to a fresh namespace.
    """
    joined = "\x00".join(source_digest(s) for s in sources) + "\x00" + version
    return _sha256(joined)[:_SHORT]


def params_hash(params: dict | None) -> str:
    """Short hash of a producer's configuration. Different params produce different hashes so
    their caches coexist under the same logic version."""
    return _sha256(json.dumps(params or {}, sort_keys=True, default=str))[:_SHORT]


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON to ``path`` atomically: serialize to a temp file in the same directory,
    flush+fsync, then rename over the target (rename is atomic on a POSIX filesystem)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)  # atomic; readers see either the old or the new file, never a partial one
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class TransformCache:
    """Per-item, content-addressed cache for one producer run (one logic version + params).

    A *producer* is whatever computes the cached outputs -- a defense's text rewrite or a
    featurizer. Each is given a cache scoped to its own ``name``, logic version and params, so
    producers never see each other's entries.

    Parameters
    ----------
    root : str or pathlib.Path
        Base cache directory for this producer category (e.g. ``<cache_dir>/defenses`` or
        ``<cache_dir>/features``). Categories live under different roots so they never share a
        ``name`` namespace (which would let one prune the other's stale-version cleanup).
    name : str
        Producer name; the top-level namespace under ``root``.
    logic_hash, params_hash : str
        The producer's logic-version and parameter hashes (see :func:`logic_hash` /
        :func:`params_hash`). The cache lives at ``root/name/logic_hash/params_hash/``.
    prune_stale : bool, default True
        On open, delete sibling *logic-version* namespaces for this producer (their entries
        are unreachable anyway), so a changed producer leaves no stale files behind. Parameter
        namespaces under the current logic version are kept.

    Use :meth:`apply` (per item) or :meth:`apply_batch` (compute all misses in one call) to run
    a transform over items with caching. After a call, ``self.hits`` and ``self.misses`` report
    cache reuse for the last batch.
    """

    def __init__(self, root, name, logic_hash, params_hash, *, prune_stale: bool = True):
        self.name = name
        self.logic_hash = logic_hash
        self.params_hash = params_hash
        self._logic_dir = Path(root) / name / logic_hash
        self.dir = self._logic_dir / params_hash
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self._write_meta()
        if prune_stale:
            self._prune_stale_logic_versions()

    def _write_meta(self) -> None:
        _atomic_write_json(
            self.dir / "meta.json",
            {"name": self.name, "logic_hash": self.logic_hash, "params_hash": self.params_hash,
             "created_at": time.time()},
        )

    def _prune_stale_logic_versions(self) -> None:
        # Remove other logic-version namespaces for this producer (best-effort: ignore races
        # with a concurrent run). Correctness does not depend on this -- those entries are
        # already unreachable because the logic hash differs.
        producer_dir = self._logic_dir.parent
        for child in producer_dir.iterdir() if producer_dir.exists() else []:
            if child.is_dir() and child.name != self.logic_hash:
                shutil.rmtree(child, ignore_errors=True)

    def _entry_path(self, key: str) -> Path:
        return self.dir / f"{_sha256(key)}.json"

    def _read(self, path: Path):
        """Return the cached output at ``path``, or ``None`` if missing or unreadable."""
        try:
            with open(path, encoding="utf-8") as handle:
                return json.load(handle)["output"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError, OSError):
            return None  # treat a corrupt/partial entry as a miss and recompute

    def apply(self, items, transform, *, key=str) -> list:
        """Return ``[transform(item) for item in items]``, computing only the items not
        already cached and persisting new outputs.

        Parameters
        ----------
        items : sequence
            Inputs to transform (e.g. conversation texts).
        transform : callable
            The expensive per-item function. Its output must be JSON-serializable.
        key : callable, default ``str``
            Maps an item to the string used as its content-addressed cache key. The default
            uses the item itself (suitable for text); override it when the item is not its
            own identity.

        Notes
        -----
        Repeated identical items within one call are computed once (the first write makes the
        rest hits). ``self.hits`` / ``self.misses`` are reset and updated per call. Use
        :meth:`apply_batch` instead when the transform vectorizes over many items at once.
        """
        self.hits = 0
        self.misses = 0
        outputs = []
        for item in items:
            path = self._entry_path(key(item))
            cached = self._read(path)
            if cached is not None:
                self.hits += 1
                outputs.append(cached)
            else:
                output = transform(item)
                _atomic_write_json(path, {"key": key(item), "output": output})
                self.misses += 1
                outputs.append(output)
        return outputs

    def apply_batch(self, items, batch_transform, *, key=str) -> list:
        """Like :meth:`apply`, but computes all cache-missing items in a single batched call.

        ``batch_transform(missing_items)`` must return one JSON-serializable output per input,
        in order. Use this for transforms that are far cheaper in bulk -- e.g. a featurizer
        running spaCy's ``pipe`` over many texts at once -- so only the uncached items hit the
        expensive path. Duplicate missing items within the batch are computed once.

        Parameters
        ----------
        items : sequence
            Inputs to transform.
        batch_transform : callable
            Maps the list of (de-duplicated) cache-missing items to a list of outputs aligned
            to it. Called at most once; not called at all when everything is cached.
        key : callable, default ``str``
            Maps an item to its content-addressed cache key (see :meth:`apply`).

        Notes
        -----
        ``self.misses`` is the number of distinct items actually computed; ``self.hits`` is the
        rest (items already on disk, and later duplicates of an item computed in this batch).
        This matches :meth:`apply`, where the first write of a repeated item makes the rest hits.
        """
        self.hits = 0
        self.misses = 0
        keys = [key(item) for item in items]
        cached = [self._read(self._entry_path(k)) for k in keys]

        # Unique cache-missing items, in order of first appearance, so the batch computes each
        # distinct input exactly once.
        seen: set = set()
        missing_keys: list = []
        missing_items: list = []
        for item, k, value in zip(items, keys, cached):
            if value is None and k not in seen:
                seen.add(k)
                missing_keys.append(k)
                missing_items.append(item)

        computed: dict = {}
        if missing_items:
            results = list(batch_transform(missing_items))
            if len(results) != len(missing_items):
                raise ValueError(
                    f"batch transform returned {len(results)} outputs for {len(missing_items)} inputs."
                )
            for k, output in zip(missing_keys, results):
                _atomic_write_json(self._entry_path(k), {"key": k, "output": output})
                computed[k] = output

        outputs = [value if value is not None else computed[k] for k, value in zip(keys, cached)]
        self.misses = len(missing_items)  # distinct items computed this call
        self.hits = len(items) - self.misses
        return outputs


__all__ = [
    "source_digest",
    "logic_hash",
    "params_hash",
    "TransformCache",
]
