"""Safe, auto-invalidating on-disk cache for expensive per-item transforms.

Two stages of the pipeline do costly per-conversation work that is wasteful to repeat:
defenses (e.g. translating each prompt) and featurizers (e.g. running a local encoder on GPU).
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
new release of a third-party feature library) -- bump ``version`` for those.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import os
import shutil
import sys
import tempfile
import textwrap
import time
from pathlib import Path


def _raise_csv_field_limit() -> None:
    """Lift csv's per-field size cap (default 131072) as high as this platform's C long allows.

    A defense caches whole conversations in the ``source``/``output`` columns, which routinely
    exceed the default limit. Without this, ``csv.reader`` raises ``field larger than field
    limit`` on a long row -- which :meth:`IndexedRowCache._read_table` catches and treats as an
    unreadable table, silently recomputing (re-billing) the entire cache on every run.
    """
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:  # too big for the platform's C long; back off and retry
            limit //= 10


_raise_csv_field_limit()

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

        ``batch_transform(missing_items)`` must return one JSON-serializable output per input, in
        order. Use this when the transform is far cheaper in bulk (e.g. a featurizer running
        a model over many texts at once). Duplicate missing items are computed once.

        **Nothing is persisted until the call returns**, so a process killed partway through
        buys nothing. Fine when the batched call itself is the unit of work (a batch API
        submission); use :meth:`apply_streaming` instead for a long run of independently
        expensive items.

        Parameters
        ----------
        items : sequence
            Inputs to transform.
        batch_transform : callable
            Maps the list of (de-duplicated) cache-missing items to a list of outputs aligned
            to it. Called at most once; not called at all when everything is cached.
        key : callable, default ``str``
            Maps an item to its content-addressed cache key (see :meth:`apply`).
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

    def _missing(self, items, key):
        """``(keys, cached, missing_keys, missing_items)`` -- the shared front half of the batch
        methods. Missing items are de-duplicated, in order of first appearance, so a repeated
        input is computed exactly once however it is delivered."""
        keys = [key(item) for item in items]
        cached = [self._read(self._entry_path(k)) for k in keys]

        seen: set = set()
        missing_keys: list = []
        missing_items: list = []
        for item, k, value in zip(items, keys, cached):
            if value is None and k not in seen:
                seen.add(k)
                missing_keys.append(k)
                missing_items.append(item)
        return keys, cached, missing_keys, missing_items

    def apply_streaming(self, items, stream_transform, *, key=str, flush_every=1,
                        on_progress=None) -> list:
        """Like :meth:`apply_batch`, but **persists results as they arrive** rather than at the end.

        Use this for a long run of independently expensive items (hours of GPU work, many paid
        API calls): a killed or preempted run resumes from whatever was already flushed instead
        of starting over.

        ``stream_transform(missing_items)`` yields ``(index, output)`` pairs, where ``index``
        positions the output in ``missing_items``, in **completion order** rather than input
        order -- so a concurrent transform can report each result as it lands while keeping every
        worker busy.

        Parameters
        ----------
        items : sequence
            Inputs to transform.
        stream_transform : callable
            Maps the list of (de-duplicated) cache-missing items to an iterable of
            ``(index, output)``. Called once; not called at all when everything is cached. Each
            output must be JSON-serializable. Raising partway through is fine and is the point:
            what was yielded first is already persisted.
        key : callable, default ``str``
            Maps an item to its content-addressed cache key (see :meth:`apply`).
        flush_every : int, default ``1``
            How many results to hold before writing them. ``1`` is right when losing an item is
            expensive (a paid API call); a higher number amortizes writes for a cheap transform.
        on_progress : callable or None
            Called as ``on_progress(done, total)`` after each flush, counting *distinct missing
            items* -- otherwise there is no way to tell a slow job from a hung one.

        Notes
        -----
        ``self.hits`` / ``self.misses`` follow :meth:`apply_batch`. An index out of range, or a
        transform that ends early, raises rather than silently returning a hole.
        """
        self.hits = 0
        self.misses = 0
        keys, cached, missing_keys, missing_items = self._missing(items, key)

        computed: dict = {}
        if missing_items:
            pending: list = []
            done = 0

            def flush():
                for k, output in pending:
                    _atomic_write_json(self._entry_path(k), {"key": k, "output": output})
                pending.clear()
                if on_progress is not None:
                    on_progress(done, len(missing_items))

            for index, output in stream_transform(missing_items):
                if not 0 <= index < len(missing_items):
                    raise ValueError(
                        f"stream transform yielded index {index} for {len(missing_items)} inputs."
                    )
                k = missing_keys[index]
                if k in computed:
                    raise ValueError(f"stream transform yielded index {index} twice.")
                computed[k] = output
                pending.append((k, output))
                done += 1
                if len(pending) >= max(1, int(flush_every)):
                    flush()
            if pending:
                flush()

            if done != len(missing_items):
                raise ValueError(
                    f"stream transform yielded {done} outputs for {len(missing_items)} inputs."
                )

        outputs = [value if value is not None else computed[k] for k, value in zip(keys, cached)]
        self.misses = len(missing_items)
        self.hits = len(items) - self.misses
        return outputs


class IndexedRowCache:
    """Ordered, index-keyed cache for a defense's per-row (per-conversation) outputs.

    Where :class:`TransformCache` is content-addressed (a pile of hash-named entries, deduped and
    unordered), a *defense* caches one row per source conversation, aligned to the reference
    dataset's row order. This stores one CSV table per labeled side (e.g. ``unknown.csv`` /
    ``known.csv``) with columns ``id, source, output``, so the cache is human-readable and each row
    traces back to the dataset it came from.

    ``id`` is the identifier the *original* dataset gives the row -- ``session_id`` for SWE-chat,
    ``idx`` for WildChat -- passed in by the caller; it falls back to the row's position when a
    loader supplies no ids. Keying on the dataset's own identifier (rather than on position in this
    particular split) means a cached rewrite still hits after the pool is re-ordered or re-subsetted.
    A row is reused only when its stored ``source`` still matches the current input, so an edited row
    always recomputes while unchanged rows -- and whole re-runs -- are served from disk.

    Namespaced by ``name`` + logic version + params exactly like :class:`TransformCache` (the cache
    lives at ``root/name/logic_hash/params_hash/``), with the same stale-logic-version pruning.
    After a call, ``self.hits`` / ``self.misses`` report reuse for the last side.
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
        _atomic_write_json(
            self.dir / "meta.json",
            {"name": self.name, "logic_hash": self.logic_hash, "params_hash": self.params_hash,
             "created_at": time.time()},
        )
        if prune_stale:
            # Mirror TransformCache: drop sibling logic-version namespaces (unreachable anyway).
            producer_dir = self._logic_dir.parent
            for child in producer_dir.iterdir() if producer_dir.exists() else []:
                if child.is_dir() and child.name != self.logic_hash:
                    shutil.rmtree(child, ignore_errors=True)

    def _table_path(self, label: str) -> Path:
        return self.dir / f"{label}.csv"

    def _read_table(self, path: Path) -> tuple[dict, dict]:
        """Return ``({id: (source, output)}, {position: (source, output)})`` from a table, or two
        empty dicts if it is missing/unreadable.

        The second (positional) map lets tables written before rows carried dataset ids -- whose
        ``id`` column held the row's position -- keep serving hits. Both maps are only ever consulted
        alongside a ``source`` equality check, so a stale positional match can never be reused.
        """
        if not path.exists():
            return {}, {}
        by_id: dict = {}
        by_position: dict = {}
        try:
            with open(path, newline="", encoding="utf-8") as handle:
                for position, record in enumerate(csv.DictReader(handle)):
                    # "index" is the legacy column name for the same field.
                    row_id = record.get("id", record.get("index"))
                    if row_id is None:
                        continue  # skip a malformed row rather than crash
                    row = (record.get("source") or "", record.get("output") or "")
                    by_id[str(row_id)] = row
                    by_position[position] = row
        except (OSError, csv.Error):
            return {}, {}  # unreadable table -> recompute everything (never a crash, never a partial read)
        return by_id, by_position

    def _write_table(self, path: Path, ids: list, sources: list, outputs: list) -> None:
        """Atomically (temp file + rename) write the ``id, source, output`` table in row order."""
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["id", "source", "output"])
                for row_id, source, output in zip(ids, sources, outputs):
                    writer.writerow([row_id, source, "" if output is None else output])
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)  # atomic; readers see the old or the new file, never a partial one
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def apply(self, label: str, sources, compute, *, ids=None, checkpoint_every=None) -> list:
        """Return one output per source (in order), recomputing only rows whose ``source`` changed.

        Parameters
        ----------
        label : str
            Table name for this side (e.g. ``"unknown"`` / ``"known"``); its own CSV under this
            cache's namespace.
        sources : sequence of str
            The reference-ordered source rows (e.g. conversation texts).
        compute : callable
            ``compute(missing_sources) -> outputs`` over the DISTINCT cache-missing sources, in
            order, one text output each. Called once when ``checkpoint_every`` is None, else once per
            batch of that many distinct sources. Not called at all when every row is cached.
        ids : sequence, optional
            The originating dataset's identifier for each row (SWE-chat ``session_id``, WildChat
            ``idx``), used as the cache key and written to the table's ``id`` column. Defaults to
            each row's position when the loader carries no ids.
        checkpoint_every : int, optional
            When set, compute the distinct missing sources in batches of this size and flush
            completed rows to disk after each batch, so a long producer is crash-safe and
            resumable. ``None`` (default) keeps the single-compute, single-write behaviour.
        """
        sources = [str(s) for s in sources]
        ids = [str(i) for i in ids] if ids is not None else [str(i) for i in range(len(sources))]
        if len(ids) != len(sources):
            raise ValueError(f"ids has {len(ids)} entries but sources has {len(sources)}.")
        path = self._table_path(label)
        cached_by_id, cached_by_position = self._read_table(path)

        outputs: list = [None] * len(sources)
        missing_positions: list = []
        for i, source in enumerate(sources):
            # Prefer the dataset id; fall back to position so pre-id tables still hit. Either way
            # the stored source must still match, so a wrong match degrades to a recompute.
            row = cached_by_id.get(ids[i]) or cached_by_position.get(i)
            if row is not None and row[0] == source:
                outputs[i] = row[1]
            else:
                missing_positions.append(i)

        # Compute distinct missing sources (once, or in checkpointed batches), first-appearance order.
        distinct: dict = {}
        for i in missing_positions:
            distinct.setdefault(sources[i], None)
        if distinct:
            distinct_list = list(distinct)
            step = checkpoint_every if checkpoint_every and checkpoint_every > 0 else len(distinct_list)
            computed: dict = {}
            for start in range(0, len(distinct_list), step):
                batch = distinct_list[start:start + step]
                results = list(compute(batch))
                if len(results) != len(batch):
                    raise ValueError(
                        f"compute returned {len(results)} outputs for {len(batch)} distinct sources."
                    )
                computed.update(zip(batch, results))
                if step < len(distinct_list):  # intermediate checkpoint: only rows done so far.
                    self._write_checkpoint(path, ids, sources, outputs, computed)
            for i in missing_positions:
                outputs[i] = str(computed[sources[i]])

        self.misses = len(distinct)                         # distinct rows actually computed
        self.hits = len(sources) - len(missing_positions)   # rows served from disk
        self._write_table(path, ids, sources, outputs)
        return outputs

    def _write_checkpoint(self, path: Path, ids, sources, outputs, computed) -> None:
        """Write only the rows finished so far (cached hits + freshly computed), omitting the rest.

        Omitting not-yet-done rows (rather than writing them blank) is what makes resume correct: on
        re-run a missing row is simply absent from the table and gets recomputed, while a genuinely
        empty output is present and read back as a hit."""
        ck_ids, ck_sources, ck_outputs = [], [], []
        for row_id, source, output in zip(ids, sources, outputs):
            if output is not None:
                value = output
            elif source in computed:
                value = str(computed[source])
            else:
                continue  # not done yet -> leave it out so it recomputes on resume.
            ck_ids.append(row_id)
            ck_sources.append(source)
            ck_outputs.append(value)
        self._write_table(path, ck_ids, ck_sources, ck_outputs)


__all__ = [
    "source_digest",
    "logic_hash",
    "params_hash",
    "TransformCache",
    "IndexedRowCache",
]
