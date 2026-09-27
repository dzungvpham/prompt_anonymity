"""StyloMetrix stylometric featurizer.

Produces the same feature space as the committed StyloMetrix CSVs, so when a defense leaves text
unchanged the featurize step can reuse those precomputed vectors and only recompute what a defense
actually rewrote: truncate each text to the first ``max_len`` chars (default :data:`MAX_LEN`; an
empty text becomes a harmless placeholder), run ``stylo_metrix.StyloMetrix(language_code)``, drop
the echoed ``text`` column, and replace NaN with 0.

``max_len`` is per-featurizer configuration, reported by :meth:`StyloMetrixFeaturizer.params`, so
each window caches under its own namespace and reading whole documents (``max_len=0``) doesn't
evict a truncated run's vectors.

**Parallelism.** StyloMetrix parses one document at a time with no batching, and its English model
is a transformer, so a large corpus is slow in a single CPU process. ``workers`` fans documents out
across worker processes, each pinned to a single compute thread and holding its own model (see
:func:`_init_worker`); the pool is sized from the allocation's CPU and memory budget, not the
machine's core count. **On a GPU the default is a single process** -- workers sharing one card
serialize in the driver, so a pool would mostly buy duplicate CUDA contexts and models; see
:meth:`StyloMetrixFeaturizer.resolve_workers` and :data:`AUTO_GPU_WORKERS`. Vectors are identical
either way, so ``workers`` is deliberately **absent from** :meth:`params`.
"""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np

from ..resources import available_cpus, available_memory_bytes
from .base import Featurizer

# Default number of leading characters fed to StyloMetrix; matches the committed feature CSVs.
MAX_LEN = 2048

# Placeholder for empty/whitespace-only text so StyloMetrix never sees an empty document; its
# features come out ~0 and carry no signal.
_EMPTY_PLACEHOLDER = "n a"

# spaCy's default per-document character guard (``nlp.max_length``), mirrored here so the parent
# process can report an untruncated document that exceeds it without loading a model of its own
# (in parallel mode only the workers hold models).
SPACY_DEFAULT_MAX_LENGTH = 1_000_000

# Upper bound on auto-selected worker processes: throughput flattens well before this, so a large
# core count should not turn into a large memory bill. An explicit ``workers=N`` is never capped.
MAX_AUTO_WORKERS = 12

# Memory budget per CPU worker, and what the parent needs alongside them, sized for
# ``en_core_web_trf``. The parent additionally holds the corpus and the assembled feature matrix.
CPU_WORKER_MEMORY_BYTES = 2 * 2 ** 30
PARENT_MEMORY_RESERVE_BYTES = 2 * 2 ** 30

# A GPU run auto-sizes to a **single process**: workers sharing one card serialize their kernels
# in the driver anyway, while each extra process still costs a full CUDA context and a full copy
# of torch/spaCy/the model in host RAM -- easy to OOM an allocation sized for one process.
# ``workers=N`` is still honoured for anyone who wants to try more on a large allocation.
AUTO_GPU_WORKERS = 1


# --- worker processes -------------------------------------------------------
# Module-level so a spawned worker can import them, and so each process builds its (expensive)
# model exactly once rather than once per document.

_WORKER_MODEL = None  # this process's stylo_metrix.StyloMetrix, set by _init_worker


def _init_worker(language_code: str, threads_per_worker: int = 1, use_gpu: bool = False) -> None:
    """Prepare one worker process: put it on the right device, then build its StyloMetrix model.

    ``use_gpu`` must mirror the parent's decision: this asks spaCy to *require* the GPU rather
    than prefer it, so a device that can't be acquired fails loudly instead of silently falling
    back to a much slower CPU model.

    Thread pinning matters for CPU workers -- left alone, every worker would try to use all cores
    for its own matrix multiplies, oversubscribing the allocation. One compute thread per process,
    parallelism from the process count instead, is what actually scales.
    """
    global _WORKER_MODEL
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(threads_per_worker)
    try:
        import torch
        torch.set_num_threads(threads_per_worker)
    except Exception:  # torch absent or already configured; the env vars above still apply
        pass

    if use_gpu:
        import spacy

        spacy.require_gpu()

    import stylo_metrix as sm
    import stylo_metrix.stylo_metrix as stylo_metrix_module

    # StyloMetrix draws a progress bar inside every transform() call. With one call per document
    # across N processes that is thousands of interleaved single-item bars on stderr; the caller
    # reports progress instead. Scoped to this worker process.
    stylo_metrix_module.tqdm = lambda iterable, **kwargs: iterable
    _WORKER_MODEL = sm.StyloMetrix(language_code)


def _featurize_one(task) -> list:
    """Featurize one prepared text in a worker; returns its metric row as a list of floats.

    ``task`` is ``(text, max_length)``, the character limit this batch needs spaCy to accept (see
    :meth:`StyloMetrixFeaturizer._allow_long_documents`). Only the numeric row travels back to the
    parent -- StyloMetrix echoes the input text as a column, which is dropped here.
    """
    text, max_length = task
    model = _WORKER_MODEL
    if max_length > model.nlp.max_length:
        model.nlp.max_length = max_length
    frame = model.transform([text])
    if "text" in frame.columns:
        frame = frame.drop(columns="text")
    return frame.to_numpy(dtype=np.float32)[0].tolist()


class StyloMetrixFeaturizer(Featurizer):
    """StyloMetrix features (the committed CSVs' feature space), computed from text.

    **Needs a source install of StyloMetrix** and uses GPU spaCy when available, falling back to
    CPU with a warning. spaCy/StyloMetrix are imported lazily inside :meth:`featurize`, so the
    featurizer can be constructed -- and cached vectors reused -- even where neither is available;
    they run only when text a defense rewrote must be recomputed.

    The feature column order is fixed by the installed StyloMetrix version. If the metric set or
    model changes, bump :attr:`version` so the cache invalidates instead of silently mismatching
    the committed CSVs.

    Parameters
    ----------
    language_code : str, default "en"
        StyloMetrix language model code (e.g. ``"en"``, ``"ru"``). Must match the language of
        the committed CSVs being reused.
    max_len : int, default :data:`MAX_LEN` (2048)
        Leading characters of each text to read; ``0`` (or ``None``) reads the whole document.
        The default reproduces the committed CSVs -- pairing a different window would mix vectors
        computed over different amounts of text.
    workers : int, optional
        Worker processes to featurize with. ``1`` runs in this process; ``None`` (default) sizes
        the pool from the allocation's cores and memory on CPU, and is a single process on a GPU
        -- see :meth:`resolve_workers`. Vectors are identical either way, so this is **not** part
        of :meth:`params`.
    """

    name = "stylometrix"
    version = "1"

    def __init__(self, *, language_code: str = "en", max_len: int | None = MAX_LEN,
                 workers: int | None = None):
        self.language_code = language_code
        self.max_len = int(max_len) if max_len else 0  # 0/None: read whole documents
        self.workers = None if workers is None else max(1, int(workers))
        self._model = None  # lazily constructed stylo_metrix.StyloMetrix (holds the spaCy pipeline)
        self._pool = None   # lazily constructed worker pool, reused across featurize() calls
        self._gpu = None    # tri-state: None until the device is probed, then True/False

    def params(self) -> dict:
        # `workers` excluded: it changes scheduling, never the vectors.
        return {"language_code": self.language_code, "max_len": self.max_len}

    def _gpu_enabled(self) -> bool:
        """Whether spaCy will use a GPU here, probed once and remembered.

        Probing *activates* the GPU for this process, which the in-process model needs; workers
        repeat the activation in their own processes. The CPU-fallback warning fires from here so
        it's emitted once per featurizer on either path.
        """
        if self._gpu is None:
            import spacy

            self._gpu = bool(spacy.prefer_gpu())
            if not self._gpu:
                warnings.warn(
                    "StyloMetrix is running on CPU because spaCy could not find a GPU; "
                    "featurizing many conversations this way can take a very long time. "
                    "Install GPU spaCy (see README.md) for a large speedup.",
                    stacklevel=3,
                )
        return self._gpu

    def _ensure_model(self):
        """Construct the in-process StyloMetrix model on first use (GPU when available)."""
        if self._model is None:
            import stylo_metrix as sm

            self._gpu_enabled()  # activate the device (and warn once) before building the model
            self._model = sm.StyloMetrix(self.language_code)
        return self._model

    def _allow_long_documents(self, texts, current_limit: int) -> int:
        """Return the ``nlp.max_length`` needed for ``texts``, warning when it exceeds the guard.

        spaCy refuses a document longer than ``nlp.max_length`` as a memory safeguard, and
        StyloMetrix's ``transform`` turns that refusal into a silent all-NaN row -- a corrupt
        feature vector indistinguishable from a real one. Reading whole (untruncated) documents is
        a deliberate choice, so the limit is raised to fit instead, with a warning about the
        memory cost. Truncated runs never reach this.
        """
        longest = max((len(text) for text in texts), default=0)
        if longest <= current_limit:
            return current_limit
        warnings.warn(
            f"raising spaCy's nlp.max_length from {current_limit:,} to {longest:,} "
            f"characters to featurize an untruncated document; this can use significant memory.",
            stacklevel=3,
        )
        return longest + 1

    def resolve_workers(self) -> int:
        """Worker-process count for this run, resolving (and caching) the ``None`` default.

        Called automatically on first use; call it earlier to report the decision in a run log.
        On GPU this is :data:`AUTO_GPU_WORKERS` (one process; see its docstring). On CPU it's the
        smallest of: available cores, :data:`MAX_AUTO_WORKERS`, and how many workers the host
        memory budget can fit (:func:`~prompt_anonymity.resources.available_memory_bytes` minus
        :data:`PARENT_MEMORY_RESERVE_BYTES`, divided by :data:`CPU_WORKER_MEMORY_BYTES`) -- the
        memory term keeps a many-core, small-memory allocation from spawning a pool the scheduler
        will OOM-kill. Cores/memory are read from the *allocation* (cgroup/scheduler), not the
        machine.
        """
        if self.workers is not None:
            return self.workers

        if self._gpu_enabled():
            self.workers = AUTO_GPU_WORKERS
        else:
            cpu_limit = max(1, min(available_cpus(), MAX_AUTO_WORKERS))
            memory = available_memory_bytes()
            fits = (max(0, memory - PARENT_MEMORY_RESERVE_BYTES) // CPU_WORKER_MEMORY_BYTES
                    if memory else cpu_limit)
            self.workers = max(1, min(cpu_limit, int(fits)))
        return self.workers

    def _ensure_pool(self) -> ProcessPoolExecutor:
        """Start (once) the worker pool, each process holding its own single-threaded model.

        The pool lives on the featurizer so models load once for the whole run rather than once
        per :meth:`featurize` call. Workers are **spawned**, not forked: a forked child inherits
        the parent's already-initialized native thread pools, a well-known way to deadlock a
        transformer pipeline.
        """
        if self._pool is None:
            workers = self.resolve_workers()
            print(f"[{self.name}] featurizing across {workers} worker process(es)")
            self._pool = ProcessPoolExecutor(
                max_workers=workers, mp_context=get_context("spawn"),
                initializer=_init_worker, initargs=(self.language_code, 1, self._gpu_enabled()),
            )
        return self._pool

    def close(self) -> None:
        """Shut down the worker pool, if one was started. Safe to call more than once."""
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def featurize(self, texts) -> np.ndarray:
        prepared = [
            ((text[:self.max_len] if self.max_len else text) if text and text.strip()
             else _EMPTY_PLACEHOLDER)
            for text in texts
        ]
        if self.resolve_workers() > 1:
            rows = self._featurize_parallel(prepared)
        else:
            model = self._ensure_model()
            model.nlp.max_length = self._allow_long_documents(prepared, model.nlp.max_length)
            frame = model.transform(prepared)
            if "text" in frame.columns:  # StyloMetrix echoes the input text as a column; drop it
                frame = frame.drop(columns="text")
            rows = frame.to_numpy(dtype=np.float32)
        # StyloMetrix can emit NaN (e.g. ratios with a zero denominator); treat those as 0.
        return np.nan_to_num(np.asarray(rows, dtype=np.float32), nan=0.0)

    def _featurize_parallel(self, prepared) -> np.ndarray:
        """Featurize prepared texts across the worker pool, preserving input order.

        Documents are handed out **one at a time** (``chunksize=1``) rather than in equal slices:
        cost grows faster than linearly with length and document lengths in these corpora vary
        wildly, so a static split would leave one worker parsing a very long document while the
        rest idle.
        """
        max_length = self._allow_long_documents(prepared, SPACY_DEFAULT_MAX_LENGTH)
        pool = self._ensure_pool()
        tasks = [(text, max_length) for text in prepared]
        return np.asarray(list(pool.map(_featurize_one, tasks, chunksize=1)), dtype=np.float32)
