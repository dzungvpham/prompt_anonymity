"""Shared helpers for the model-backed defenses ported from the DS_env research sandbox.

The defenses in this package rewrite a user's conversations before release to wash out the
per-user stylometric signal a linkage attack exploits. Three of them (Qwen rewrite, StyleRemix,
OpenAnonymity) share the same shape: a heavy backend that is far cheaper run in bulk, applied to
each conversation **one user turn at a time**. This module holds the pieces that shape reuses:

* the per-turn split/join convention (a WildChat conversation cell is a user's turns joined by a
  literal ``\\n===\\n`` delimiter), so each user message is defended on its own and the backend
  never sees the delimiter, and
* :class:`PerTurnBatchRewriteDefense`, a batching analogue of
  :class:`~prompt_anonymity.defenses.base.CachedTextRewriteDefense` that flattens every
  conversation's turns into one stream, rewrites the stream in bulk through the cache, and
  re-groups -- so identical turns shared across conversations are computed once and a GPU/vLLM
  backend stays saturated.

Like every defense here, these only rewrite text; features are recomputed by the featurize stage.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..caching import IndexedRowCache
from ..core import AttackData
from .base import CachedDefense

#: Delimiter joining a user's turns within one conversation cell. On disk the real newlines were
#: stored as the two-char sequence ``\n``, so the separator is the literal string ``\n===\n`` (four
#: visible characters around ``===``), NOT actual newlines -- matching the WildChat preprocessing.
TURN_DELIM = "\\n===\\n"


def split_turns(text: str) -> list[str]:
    """Split a conversation cell into its user turns on :data:`TURN_DELIM`.

    Returns ``[text]`` unchanged when the delimiter is absent, so a defense applied to text without
    the per-turn structure simply rewrites the whole string (the split is a no-op).
    """
    return str(text).split(TURN_DELIM)


def join_turns(turns: list[str]) -> str:
    """Re-join defended turns back into one conversation cell with :data:`TURN_DELIM`."""
    return TURN_DELIM.join(turns)


#: Turns this short (in stripped characters) are left alone by a defense that sets
#: ``min_defend_chars``. Measured on the 7,764 rewrites of a previous StyleRemix run: an input of
#: 1-5 characters came back more than 3x longer 11.6% of the time and 6-10 characters 7.8% of the
#: time -- the model inventing content because there was none to rewrite (``"```"`` became a
#: paragraph about a company's market strategy) -- while at 16+ characters that rate is 0.0%. A
#: passthrough turn is visibly *undefended*, which is a far better failure than fabricated text
#: silently entering the dataset, and it is cheap: turns under 16 characters are 17.5% of SWE-chat's
#: and 9.6% of WildChat's, but only ~0.6% and ~0.05% of their text.
MIN_DEFEND_CHARS = 16


def defend_conversations_per_turn(conversations: list[str], rewrite_batch,
                                  min_chars: int = 1) -> list[str]:
    """Defend whole conversations one user turn at a time, then re-join.

    Splits each conversation on :data:`TURN_DELIM`, flattens every turn into one stream, rewrites the
    distinct eligible turns in a single ``rewrite_batch`` call, then re-groups and re-joins one
    string per conversation -- so identical turns shared across conversations are computed once and
    turn structure (and count) is preserved.

    ``min_chars`` is the shortest turn worth defending: anything shorter passes through untouched
    and never reaches the backend. The default of 1 means only blank turns are skipped (there is
    nothing to rewrite, and an empty prompt just wastes a generation). A defense that sets it higher
    -- see :data:`MIN_DEFEND_CHARS` -- is protecting itself from the other end of the scale, where a
    contentless turn gives a generative rewriter nothing to work from and it invents something.

    Shared by :class:`PerTurnBatchRewriteDefense` and the combined StyleRemix+OpenAnonymity defense
    so both defend per turn identically; ``rewrite_batch`` is the backend's list-in/list-out op.
    """
    threshold = max(1, min_chars)
    turn_lists = [split_turns(c) for c in conversations]
    counts = [len(turns) for turns in turn_lists]
    flat = [turn for turns in turn_lists for turn in turns]

    def defendable(turn: str) -> bool:
        return len(turn.strip()) >= threshold

    # Distinct eligible turns, first-appearance order -> one backend call, no repeated work.
    distinct: dict = {}
    for turn in flat:
        if defendable(turn):
            distinct.setdefault(turn, None)
    rewritten = rewrite_batch(list(distinct)) if distinct else []
    mapping = dict(zip(distinct, rewritten))
    out_flat = [mapping[turn] if defendable(turn) else turn for turn in flat]

    results, pos = [], 0
    for n in counts:
        results.append(join_turns(out_flat[pos:pos + n]))
        pos += n
    return results


def shutdown_vllm(llm) -> None:
    """Terminate a vLLM engine and give its GPU memory back.

    Needed when one process runs two models in sequence -- the combined StyleRemix+OpenAnonymity
    defense restyles with Llama-3-8B and only then scrubs with gpt-oss -- because vLLM reserves a
    fixed *fraction* of the device up front and holds it for the engine's lifetime. Without this the
    second engine would find the first still resident and the two fractions would have to sum under
    1.0, which wastes memory in the common single-model case and still risks an OOM.

    vLLM v1 runs its engine core in a child process, so the shutdown that matters is telling that
    process to exit; the local collection afterwards releases whatever this process still holds.
    Best-effort by design: failing to reclaim memory must not lose a completed defense pass.
    """
    import gc

    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception as error:  # noqa: BLE001 - teardown must not mask the work already done
        print(f"note: vLLM engine shutdown raised {type(error).__name__}: {error}")
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - torch may be absent or already torn down
        pass


def resolve_model_path(model: str) -> str:
    """Resolve a HuggingFace hub *cache* directory to the checkpoint inside it; pass anything else
    through untouched.

    A hub cache entry (``models--meta-llama--Meta-Llama-3-8B/``) is not itself loadable -- the
    checkpoint sits in ``snapshots/<commit>/`` -- so a path pointing at one is followed to that
    snapshot, with ``refs/main`` preferred when it names a snapshot that is actually present. It
    often is not: mirroring a repo without its refs, or pruning an old snapshot, leaves the ref
    dangling, and in that case the snapshot on disk is the one the user means. A plain checkpoint
    directory, or a repo id to download, is returned unchanged.
    """
    path = Path(model)
    if not path.is_dir() or (path / "config.json").exists():
        return model  # a repo id, or already a checkpoint directory
    snapshots = path / "snapshots"
    if not snapshots.is_dir():
        return model
    available = sorted((p for p in snapshots.iterdir() if (p / "config.json").exists()),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    if not available:
        raise RuntimeError(f"{path} looks like a HuggingFace cache entry but holds no snapshot with "
                           f"a config.json; point the model setting at a checkpoint directory.")
    ref = path / "refs" / "main"
    if ref.exists():
        pinned = snapshots / ref.read_text().strip()
        if pinned in available:
            return str(pinned)
        print(f"note: {ref} names snapshot {pinned.name} but it is not on disk; "
              f"using {available[0].name} instead.")
    return str(available[0])


def find_bundled_cuda_toolkit() -> Path | None:
    """The CUDA toolkit pip installed into this environment, if there is one.

    ``nvidia-cuda-nvcc-cuXX`` ships a complete toolkit as a Python package --
    ``site-packages/nvidia/cu13/`` with ``bin/nvcc``, ``include/`` and ``nvvm/libdevice`` -- which
    is how an environment can have a working compiler while ``/usr/local/cuda`` does not exist and
    ``nvcc`` is nowhere on ``PATH``. Located through the installed ``nvidia`` namespace package
    rather than a hard-coded path, so it is found wherever the environment lives; the newest
    ``cuNN`` directory wins if several are installed.
    """
    spec = importlib.util.find_spec("nvidia")
    if spec is None:
        return None
    for location in spec.submodule_search_locations or []:
        candidates = sorted(Path(location).glob("cu*/bin/nvcc")) + \
            sorted(Path(location).glob("cuda_nvcc/bin/nvcc"))
        for nvcc in reversed(candidates):  # newest CUDA major version first
            if nvcc.is_file():
                return nvcc.parent.parent
    return None


def configure_cuda_toolkit() -> None:
    """Prepare the environment a vLLM-backed defense is about to import vLLM into. Two things:

    **The sampler is pinned to vLLM's native one** (``VLLM_USE_FLASHINFER_SAMPLER=0``). vLLM
    otherwise prefers FlashInfer's top-k/top-p kernel, which it JIT-compiles during the startup
    memory profile -- and which these defenses never use, because they all sample greedily
    (temperature 0), and greedy decoding takes neither top-k nor top-p. So the kernel is pure
    startup risk: on a node with no compiler the build cannot run at all, and with CUDA 13 it fails
    to compile anyway (FlashInfer vendors its own CCCL headers, whose version guard rejects a CUDA
    13 ``nvcc``: *"CUDA compiler and CUDA toolkit headers are incompatible"*). Either way the engine
    dies before finishing startup, having gained nothing. The native path is identical in result
    under greedy decoding.

    **A CUDA compiler is made reachable** when one is not already, for every *other* JIT path
    (torch extensions, other vLLM kernels). ``nvidia-cuda-nvcc`` installs a full toolkit into
    site-packages (:func:`find_bundled_cuda_toolkit`), so an environment can have a working
    compiler while ``/usr/local/cuda`` does not exist; exporting ``CUDA_HOME`` and extending
    ``PATH`` is what torch's ``cpp_extension`` -- and so vLLM -- looks at.

    Neither step overrides a value already in the environment, so an explicit
    ``VLLM_USE_FLASHINFER_SAMPLER=1`` or a pre-set ``CUDA_HOME`` wins. Must run *before* vLLM is
    imported, since vLLM reads its environment at import time.
    """
    if "VLLM_USE_FLASHINFER_SAMPLER" not in os.environ:
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

    if shutil.which("nvcc") or os.environ.get("CUDA_HOME"):
        return
    toolkit = find_bundled_cuda_toolkit()
    if toolkit is not None:
        os.environ["CUDA_HOME"] = str(toolkit)
        os.environ["PATH"] = f"{toolkit / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"
        print(f"note: no system CUDA toolkit; using the one bundled in this environment ({toolkit}).")


def gpu_dtype(torch, *, prefer_bf16: bool = True):
    """Best generation dtype for the visible device: bf16 on Ampere+ (A100/H100 -- same throughput
    as fp16 with no overflow risk), fp16 on older GPUs, fp32 on CPU."""
    if not torch.cuda.is_available():
        return torch.float32
    if prefer_bf16 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


class PerTurnBatchRewriteDefense(CachedDefense):
    """Base for a text-rewrite defense whose backend is far cheaper in bulk, applied per turn.

    A subclass implements :meth:`rewrite_batch` -- the expensive op over a *list* of turns -- plus
    :attr:`name`, an optional :attr:`version`, and :meth:`params`. This base handles everything
    else, at two levels:

    * **Caching** is per CONVERSATION, keyed by the row's index in the reference dataset (see
      :class:`~prompt_anonymity.caching.IndexedRowCache`): the on-disk cache is an ordered
      ``index, source, output`` table, one defended conversation per source row.
    * **Compute** for the cache-missing conversations is per TURN: each missing conversation is
      split on :data:`TURN_DELIM`, all their turns are flattened into one stream (deduped, blanks
      passed through untouched) and rewritten in one :meth:`rewrite_batch` call so a GPU/vLLM
      backend stays saturated, then re-grouped and re-joined.

    A fully-cached side never calls :meth:`rewrite_batch`, so a lazily-built backend (see the
    subclasses) loads no model on a cached run.

    Like :class:`~prompt_anonymity.defenses.base.CachedTextRewriteDefense`, only the anonymous
    *unknown* side is rewritten unless :attr:`rewrite_known` is set. Embeddings are left stale for
    the featurize stage to recompute.
    """

    #: Rewrite the known side too (default keeps the usual threat model: the adversary's known
    #: conversations are left as released).
    rewrite_known: bool = False

    #: When set, flush finished conversations to the cache every this many computed conversations, so
    #: a long/expensive backend is crash-safe and resumable (see :meth:`IndexedRowCache.apply`).
    #: ``None`` keeps the default single-write-at-the-end behaviour used by the fast rewriters.
    checkpoint_every: int | None = None

    #: Shortest turn (stripped characters) this defense will rewrite; anything shorter passes through
    #: untouched. ``0``/``1`` defends everything non-blank, which is the default because skipping is
    #: not universally safe: a *style* rewriter gains from it (see :data:`MIN_DEFEND_CHARS`), while a
    #: redaction defense must not skip short turns -- a 13-character turn can be an email address.
    min_defend_chars: int = 0

    def rewrite_batch(self, texts: list[str]) -> list[str]:
        """Rewrite a list of turns, returning one output per input in order. Implemented by
        subclasses; called only on the cache-missing, de-duplicated, non-blank turns."""
        raise NotImplementedError

    def transform(self, data: AttackData, cache: IndexedRowCache) -> AttackData:
        if data.unknown_texts is None:
            raise ValueError(f"defense {self.name!r} needs unknown_texts; load the dataset with text.")
        changes = {
            "unknown_texts": self._rewrite_side("unknown", data.unknown_texts, cache, data.unknown_ids)
        }
        if self.rewrite_known:
            if data.known_texts is None:
                raise ValueError(f"defense {self.name!r} has rewrite_known=True but no known_texts.")
            changes["known_texts"] = self._rewrite_side("known", data.known_texts, cache, data.known_ids)
        # Only text changes; replace() re-validates row counts so a rewrite that drops/adds rows is
        # caught (defended conversations stay row-aligned to the reference).
        return replace(data, **changes)

    def _rewrite_side(self, label: str, texts, cache: IndexedRowCache, ids=None) -> np.ndarray:
        # Cache per conversation, keyed by the originating dataset's row id (SWE-chat session_id,
        # WildChat idx); the compute for cache-missing conversations runs the backend per turn
        # across all of them at once.
        outputs = cache.apply(
            label, [str(t) for t in texts], self._defend_conversations, ids=ids,
            checkpoint_every=self.checkpoint_every,
        )
        return np.asarray(outputs, dtype=object)

    def _defend_conversations(self, conversations: list[str]) -> list[str]:
        """Defend a list of whole conversations, per turn (see :func:`defend_conversations_per_turn`)."""
        return defend_conversations_per_turn(conversations, self.rewrite_batch,
                                             min_chars=self.min_defend_chars or 1)

    def close(self) -> None:
        """Release the backend's resources (a GPU-resident model, typically) and drop it.

        Safe to call when the backend was never built -- a fully-cached run never loads one -- and
        safe to call twice. What makes it necessary is running two model-backed defenses in one
        process: see :class:`~prompt_anonymity.defenses.styleremix_openanon.StyleRemixOpenAnonymityDefense`,
        which hands the GPU from one stage to the next. A later ``rewrite_batch`` rebuilds the
        backend from scratch, so closing is never fatal, only expensive.
        """
        backend = getattr(self, "_backend", None)
        if backend is not None and hasattr(backend, "close"):
            backend.close()
        self._backend = None
