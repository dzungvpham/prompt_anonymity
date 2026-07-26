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


def defend_conversations_per_turn(conversations: list[str], rewrite_batch) -> list[str]:
    """Defend whole conversations one user turn at a time, then re-join.

    Splits each conversation on :data:`TURN_DELIM`, flattens every turn into one stream, rewrites the
    distinct non-blank turns in a single ``rewrite_batch`` call (blanks pass through untouched), then
    re-groups and re-joins one string per conversation -- so identical turns shared across
    conversations are computed once and turn structure (and count) is preserved.

    Shared by :class:`PerTurnBatchRewriteDefense` and the combined StyleRemix+OpenAnonymity defense
    so both scrub per turn identically; ``rewrite_batch`` is the backend's list-in/list-out op.
    """
    turn_lists = [split_turns(c) for c in conversations]
    counts = [len(turns) for turns in turn_lists]
    flat = [turn for turns in turn_lists for turn in turns]

    # Distinct non-blank turns, first-appearance order -> one backend call, no repeated work.
    distinct: dict = {}
    for turn in flat:
        if turn.strip():
            distinct.setdefault(turn, None)
    rewritten = rewrite_batch(list(distinct)) if distinct else []
    mapping = dict(zip(distinct, rewritten))
    out_flat = [mapping[turn] if turn.strip() else turn for turn in flat]

    results, pos = [], 0
    for n in counts:
        results.append(join_turns(out_flat[pos:pos + n]))
        pos += n
    return results


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
        return defend_conversations_per_turn(conversations, self.rewrite_batch)
