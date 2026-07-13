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

from dataclasses import replace

import numpy as np

from ..caching import TransformCache
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
    else: splitting each conversation on :data:`TURN_DELIM`, flattening all turns across all
    conversations into one stream, running that stream through :meth:`TransformCache.apply_batch`
    (so duplicate turns are computed once and only cache misses reach the backend), then re-grouping
    and re-joining. Blank turns pass through untouched, so the backend is never handed an empty
    message.

    Batching plus caching means a fully-cached side never calls :meth:`rewrite_batch` at all, so a
    lazily-built backend (see the subclasses) loads no model on a cached run.

    Like :class:`~prompt_anonymity.defenses.base.CachedTextRewriteDefense`, only the anonymous
    *unknown* side is rewritten unless :attr:`rewrite_known` is set. Embeddings are left stale for
    the featurize stage to recompute.
    """

    #: Rewrite the known side too (default keeps the usual threat model: the adversary's known
    #: conversations are left as released).
    rewrite_known: bool = False

    def rewrite_batch(self, texts: list[str]) -> list[str]:
        """Rewrite a list of turns, returning one output per input in order. Implemented by
        subclasses; called only on the cache-missing, de-duplicated, non-blank turns."""
        raise NotImplementedError

    def transform(self, data: AttackData, cache: TransformCache) -> AttackData:
        if data.unknown_texts is None:
            raise ValueError(f"defense {self.name!r} needs unknown_texts; load the dataset with text.")
        changes = {"unknown_texts": self._rewrite_side(data.unknown_texts, cache)}
        if self.rewrite_known:
            if data.known_texts is None:
                raise ValueError(f"defense {self.name!r} has rewrite_known=True but no known_texts.")
            changes["known_texts"] = self._rewrite_side(data.known_texts, cache)
        # Only text changes; replace() re-validates row counts so a rewrite that drops/adds turns is
        # caught (join_turns keeps the row count fixed regardless).
        return replace(data, **changes)

    def _rewrite_side(self, texts, cache: TransformCache) -> np.ndarray:
        turn_lists = [split_turns(t) for t in texts]
        counts = [len(turns) for turns in turn_lists]
        flat = [turn for turns in turn_lists for turn in turns]

        # Defend only non-blank turns; blanks (e.g. from a trailing delimiter) pass through so the
        # backend never sees an empty message. apply_batch dedupes identical turns and skips the
        # backend entirely when nothing is missing.
        defend_idx = [i for i, turn in enumerate(flat) if turn.strip()]
        defended = cache.apply_batch([flat[i] for i in defend_idx], self.rewrite_batch)
        out_flat = list(flat)
        for i, value in zip(defend_idx, defended):
            out_flat[i] = value

        results, pos = [], 0
        for n in counts:
            results.append(join_turns(out_flat[pos:pos + n]))
            pos += n
        return np.asarray(results, dtype=object)
