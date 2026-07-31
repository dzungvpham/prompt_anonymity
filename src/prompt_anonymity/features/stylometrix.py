"""StyloMetrix stylometric featurizer.

Produces the same feature space as the committed StyloMetrix CSVs, so when a defense leaves
text unchanged the featurize step can reuse those precomputed vectors and only recompute the
text a defense actually rewrote. Mirrors ``wildchat/stylometrix.py`` and
``swe-chat/stylometrix.py``: truncate each text to the first ``max_len`` chars (default
:data:`MAX_LEN`, the value those scripts used; an empty text becomes a harmless token), run
``stylo_metrix.StyloMetrix(language_code)``, drop the echoed ``text`` column, and replace NaN
with 0.

``max_len`` is per-featurizer configuration rather than a fixed constant, so a caller can read
whole documents (``max_len=0``) instead of a prefix. Because it is reported by
:meth:`StyloMetrixFeaturizer.params`, each window caches under its own namespace and two
windows can be computed and compared without evicting each other.

**Parallelism.** StyloMetrix parses one document at a time in a plain Python loop -- it exposes
no batching or ``n_process`` knob -- and its English model is the ``en_core_web_trf``
transformer, so **on CPU** a large corpus takes hours in one process. ``workers`` therefore fans
the documents out across worker processes, each holding its own model and pinned to a single
compute thread (see :func:`_init_worker`), which roughly doubles CPU throughput over the default
all-threads-one-process arrangement. Because every worker holds a full model, the pool is sized
from the *allocation's* CPU and memory budget rather than from the machine's core count. **On a
GPU the default is a single process**: workers sharing one card serialize in the driver, so the
pool would mostly buy CUDA contexts and duplicate models -- see
:meth:`StyloMetrixFeaturizer.resolve_workers`, :data:`AUTO_GPU_WORKERS` and
:mod:`prompt_anonymity.resources`. Vectors are unaffected either way (every document is parsed by
the same model, one at a time), so ``workers`` is deliberately **absent from** :meth:`params` and
does not split the cache.
"""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np

from ..resources import available_cpus, available_memory_bytes
from .base import Featurizer

# Default number of leading characters fed to StyloMetrix; matches the committed feature CSVs
# (the MAX_LEN = 2048 in wildchat/stylometrix.py and swe-chat/stylometrix.py). The effective
# value is per-featurizer (``max_len``) and reported by params(), so each window gets its own
# cache namespace rather than silently reusing another window's vectors.
MAX_LEN = 2048

# Placeholder for empty/whitespace-only text so StyloMetrix never sees an empty document
# (matches swe-chat/stylometrix.py); its features come out ~0 and carry no signal.
_EMPTY_PLACEHOLDER = "n a"

# spaCy's default per-document character guard (``nlp.max_length``), mirrored here so the parent
# process can report an untruncated document that exceeds it without loading a model of its own
# (in parallel mode only the workers hold models).
SPACY_DEFAULT_MAX_LENGTH = 1_000_000

# Upper bound on auto-selected worker processes. Measured on this project's corpora, throughput
# flattens well before this (16 workers matched 12 on a 16-core allocation), so a large core
# count should not turn into a large memory bill. An explicit ``workers=N`` is never capped.
MAX_AUTO_WORKERS = 12

# Memory one CPU worker needs, and what the parent needs alongside them. Measured with
# ``en_core_web_trf``: ~0.7 GiB of imports, ~1.2 GiB once the model is built, peaking at ~1.5 GiB
# while parsing the longest SWE-chat session (117K characters). 2 GiB per worker leaves room for a
# longer document; the parent holds the corpus and the assembled feature matrix.
CPU_WORKER_MEMORY_BYTES = 2 * 2 ** 30
PARENT_MEMORY_RESERVE_BYTES = 2 * 2 ** 30

# A GPU run auto-sizes to a **single process**. Several workers sharing one card serialize their
# kernels in the driver anyway, so the only thing extra processes buy is the overlap of
# StyloMetrix's per-document Python work with GPU compute -- while each one costs a full CUDA
# context (VRAM) *and* a full Python process holding torch, spaCy and the model in host RAM. That
# host cost is the one that bites: a 128-task WildChat array on 4 CPUs / 8 GiB / an 11 GiB card
# auto-sized to 3 GPU workers on free VRAM alone and every task was OOM-killed by the scheduler
# within two minutes. One process per card is the arrangement that reliably fits; ``workers=N`` is
# still honoured for anyone who wants to try more on a large allocation.
AUTO_GPU_WORKERS = 1


# --- worker processes -------------------------------------------------------
# Module-level so a spawned worker can import them, and so each process builds its (expensive)
# model exactly once rather than once per document.

_WORKER_MODEL = None  # this process's stylo_metrix.StyloMetrix, set by _init_worker


def _init_worker(language_code: str, threads_per_worker: int = 1, use_gpu: bool = False) -> None:
    """Prepare one worker process: put it on the right device, then build its StyloMetrix model.

    ``use_gpu`` must mirror the parent's decision. A worker that does not activate the GPU builds
    a CPU model instead and quietly runs an order of magnitude slower, so this asks spaCy to
    *require* the GPU rather than prefer it -- on a run the parent already sized for GPU workers,
    a device that cannot be acquired is an error worth seeing, not a silent fallback.

    Thread pinning is the whole game for CPU workers: left alone, every worker would try to use
    all cores for its own matrix multiplies, and N such workers oversubscribe the allocation badly
    enough to be slower than running serially. One compute thread per process, with the
    parallelism coming from the process count, is the arrangement that actually scales.
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

    ``task`` is ``(text, max_length)``, the second being the character limit this batch needs
    spaCy to accept (see :meth:`StyloMetrixFeaturizer._allow_long_documents`). Only the numeric
    row travels back to the parent -- StyloMetrix echoes the input text as a column, and sending
    it home again would double the corpus across the process boundary for nothing.
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

    **Needs a source install of StyloMetrix** (see ``README.md``) and uses GPU spaCy when one is
    available, falling back to CPU with a warning (CPU is far slower). spaCy and StyloMetrix are
    imported lazily inside :meth:`featurize`, so the featurizer can be constructed -- and cached
    vectors reused, including the whole no-defense run off the committed CSVs -- even where
    StyloMetrix or a GPU is unavailable; StyloMetrix runs only when text a defense rewrote must
    be recomputed.

    The feature column order is fixed by the installed StyloMetrix version, so freshly computed
    rows line up with the committed CSVs. If you change the StyloMetrix metric set or model,
    bump :attr:`version` (source hashing cannot see an external package upgrade) so the cache
    -- and any reuse of the now-mismatched committed CSVs -- invalidates.

    Parameters
    ----------
    language_code : str, default "en"
        StyloMetrix language model code (e.g. ``"en"``, ``"ru"``). Must match the language of
        the committed CSVs being reused (WildChat: ``"en"``/``"ru"``; SWE-chat: ``"en"``).
    max_len : int, default :data:`MAX_LEN` (2048)
        Leading characters of each text to read; ``0`` (or ``None``) reads the whole document.
        The default reproduces the committed CSVs, so **reuse of those precomputed vectors is
        only valid at the default**: pairing a different window with a loaded ``reference``
        would mix vectors computed over different amounts of text. Cost grows faster than
        linearly with document length, so widening the window on a corpus with a long tail is
        much more expensive than the character count alone suggests.
    workers : int, optional
        Worker processes to featurize with. ``1`` runs everything in this process; ``None`` (the
        default) sizes the pool from the *allocation's* cores and memory on CPU, and is a single
        process on a GPU -- see :meth:`resolve_workers`. The vectors are identical either way, so
        this is **not** part of :meth:`params` and does not split the cache.
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
        # `workers` is deliberately excluded: it changes how the work is scheduled, never the
        # vectors, so serial and parallel runs share a single cache namespace.
        return {"language_code": self.language_code, "max_len": self.max_len}

    def _gpu_enabled(self) -> bool:
        """Whether spaCy will use a GPU here, probed once and remembered.

        Probing *activates* the GPU for this process (``spacy.prefer_gpu``), which is what the
        in-process model needs; workers repeat the activation in their own processes. The
        CPU-fallback warning is emitted from here so it fires once per featurizer, on the serial
        and the parallel path alike.
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

        spaCy refuses a document longer than ``nlp.max_length`` (1,000,000 characters by
        default) as a memory safeguard. StyloMetrix's ``transform`` catches that error and emits
        an all-NaN row, which :meth:`featurize` would turn into an all-zero vector -- a corrupt
        feature vector indistinguishable from a real one. An uncapped run is a deliberate choice
        to read whole documents, so the limit is raised to fit the longest one and the memory
        cost is warned about rather than hidden. Truncated runs never reach this.

        The caller applies the returned value to whichever model(s) will do the parsing -- its
        own in serial mode, each worker's in parallel mode -- so the warning is emitted once, in
        this process, rather than once per worker.
        """
        longest = max((len(text) for text in texts), default=0)
        if longest <= current_limit:
            return current_limit
        warnings.warn(
            f"raising spaCy's nlp.max_length from {current_limit:,} to {longest:,} "
            f"characters to featurize an untruncated document; parsing a document this long "
            f"needs on the order of 1 GB of memory per 100,000 characters.",
            stacklevel=3,
        )
        return longest + 1

    def resolve_workers(self) -> int:
        """Worker-process count for this run, resolving (and caching) the ``None`` default.

        Called automatically on first use; call it earlier to report the decision in a run log.
        What the pool is sized from depends on which device does the parsing:

        * **GPU**: :data:`AUTO_GPU_WORKERS` -- one process, whatever the card and the allocation.
          Processes sharing a card serialize in the driver, so a pool buys little there while
          costing a CUDA context and a full Python process each; see the constant.
        * **CPU**: cores, :data:`MAX_AUTO_WORKERS`, and the host-memory budget
          (:func:`~prompt_anonymity.resources.available_memory_bytes` minus
          :data:`PARENT_MEMORY_RESERVE_BYTES`, divided by :data:`CPU_WORKER_MEMORY_BYTES`),
          whichever is smallest. Every worker holds a full model, so the memory term is what keeps
          a many-core, small-memory allocation from spawning a pool the scheduler will OOM-kill.

        The CPU budget comes from :mod:`prompt_anonymity.resources`, which reads the *allocation*
        (cgroup / scheduler) rather than the machine.
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

        The pool lives on the featurizer, so the models are loaded once for the whole run rather
        than once per :meth:`featurize` call -- loading ``en_core_web_trf`` costs ~20 s per
        process. Workers are **spawned**, not forked: a forked child inherits the parent's
        already-initialized native thread pools, which is a well-known way to deadlock a
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
        StyloMetrix's cost grows faster than linearly with length and these corpora are wildly
        uneven (a median SWE-chat session is ~340 characters, the longest ~117,000), so a static
        split would leave one worker parsing a novel while the rest idle. Per-document handoff
        costs a pickled string and ~200 floats back, which is nothing beside the parsing.
        """
        max_length = self._allow_long_documents(prepared, SPACY_DEFAULT_MAX_LENGTH)
        pool = self._ensure_pool()
        tasks = [(text, max_length) for text in prepared]
        return np.asarray(list(pool.map(_featurize_one, tasks, chunksize=1)), dtype=np.float32)
