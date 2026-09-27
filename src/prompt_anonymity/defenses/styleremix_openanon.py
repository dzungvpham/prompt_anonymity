"""Combined StyleRemix + OpenAnonymity defense, applied PER USER TURN.

Chains the two rewrite backends into one defense: StyleRemix restyles every prompt toward the fixed
:data:`~prompt_anonymity.defenses.styleremix.STYLEREMIX_SLIDERS` target style, then the
OpenAnonymity scrubber redacts identifiers and de-identifies residual style, both per turn.
Style-convergence runs first and identifier redaction last, so OA's ``[PERSON_1]``/``[ORG_1]``
placeholders survive intact rather than being reworded by StyleRemix.

Both stages are per turn, so turn structure and count are preserved end to end -- which keeps the
defended conversation aligned turn-for-turn with the original, as per-turn utility scoring needs.
StyleRemix reuses its own standalone per-turn cache; this defense's own cache holds the stage-2
result keyed by the *styled* text, so a stage-1 change re-caches stage 2.

The stages run strictly in sequence, never concurrently: stage 1 restyles the whole dataset, its
vLLM engine is shut down, and only then does stage 2 load the scrubber. Both models are large and
each vLLM engine reserves a fixed fraction of the device for its lifetime, so running them at once
would mean splitting one GPU between two models never used at the same moment.

Not a simple per-text rewrite, so it implements :meth:`__call__`/:meth:`transform` directly rather
than subclassing the text-rewrite spine. Bump :attr:`version` when the OpenAnonymity backend's
behavior changes (its source is not in this class's hash).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from ..caching import IndexedRowCache, params_hash
from ..core import AttackData
from ._backends import defend_conversations_per_turn
from .base import CachedDefense
from .openanonymity import (OPENANON_API_MODEL, OPENANON_MODEL, OPENANON_SYSTEM_PROMPT,
                            _OpenAnonAPIBackend, _OpenAnonBackend)
from .styleremix import STYLEREMIX_BASE_MODEL, STYLEREMIX_SLIDERS, StyleRemixDefense


class StyleRemixOpenAnonymityDefense(CachedDefense):
    """StyleRemix restyle then OpenAnonymity redact, both per user turn.

    Both backends are built lazily: a run whose StyleRemix and combined caches are already complete
    loads neither model and makes no API calls. Side selection follows :attr:`rewrite_known`.
    """

    name = "styleremix_openanon"
    version = "2"
    rewrite_known: bool = False

    def __init__(self, *, sliders: dict | None = None, base_model: str = STYLEREMIX_BASE_MODEL,
                 oa_model: str = OPENANON_MODEL, oa_system_prompt: str = OPENANON_SYSTEM_PROMPT,
                 oa_api_model: str | None = None):
        self.sliders = dict(STYLEREMIX_SLIDERS if sliders is None else sliders)
        self.base_model = base_model
        self.oa_model = oa_model
        self.oa_system_prompt = oa_system_prompt
        #: Redact through OpenRouter rather than a local checkpoint -- see
        #: :data:`~.openanonymity.OPENANON_API_MODEL`. Also sidesteps needing one big GPU for both
        #: stages, since the redactor becomes a network call.
        self.oa_api_model = oa_api_model if oa_api_model is not None else OPENANON_API_MODEL
        self._redactor = None

    def params(self) -> dict:
        # Included for reproducibility and so an OA model/prompt swap re-caches stage 2 (their
        # source isn't covered by this class's logic hash).
        base = {
            "sliders": self.sliders, "base_model": self.base_model,
            "oa_model": self.oa_model, "oa_system_prompt": self.oa_system_prompt,
        }
        # Only included when set, so an API run and a local run (different weights) don't share a
        # cache entry, while existing local caches keep hitting.
        return {**base, "oa_api_model": self.oa_api_model} if self.oa_api_model else base

    def _get_redactor(self):
        if self._redactor is None:
            self._redactor = (_OpenAnonAPIBackend(self.oa_api_model, self.oa_system_prompt)
                              if self.oa_api_model
                              else _OpenAnonBackend(self.oa_model, self.oa_system_prompt))
        return self._redactor

    def __call__(self, data: AttackData, *, cache_dir) -> AttackData:
        # Strictly sequential: stage 1 restyles the whole dataset and its engine is shut down before
        # stage 2 loads and scrubs the restyled text. Stage 2 reads stage 1's output, not the
        # original, so this ordering is the definition of the defense.
        restyle = StyleRemixDefense(sliders=self.sliders, base_model=self.base_model)
        restyle.rewrite_known = self.rewrite_known
        try:
            styled = restyle(data, cache_dir=cache_dir)
        finally:
            # Free the restyler even if it failed part-way: a fully-cached stage 1 loaded nothing,
            # and anything it did load has no further use.
            restyle.close()

        # Stage 2: OpenAnonymity redact PER TURN over the restyled text, in this defense's own
        # index-keyed cache (keyed by the styled text, so a stage-1 change re-caches stage 2).
        cache = IndexedRowCache(
            Path(cache_dir) / "defenses", self.name, self._logic_hash(), params_hash(self.params())
        )
        try:
            return self.transform(styled, cache)
        finally:
            self.close()

    def close(self) -> None:
        """Release the stage-2 scrubber (the restyler is released as soon as stage 1 finishes)."""
        if self._redactor is not None and hasattr(self._redactor, "close"):
            self._redactor.close()
        self._redactor = None

    def transform(self, data: AttackData, cache: IndexedRowCache) -> AttackData:
        if data.unknown_texts is None:
            raise ValueError(f"defense {self.name!r} needs unknown_texts; load the dataset with text.")
        changes = {
            "unknown_texts": self._redact_side("unknown", data.unknown_texts, cache, data.unknown_ids)
        }
        if self.rewrite_known:
            if data.known_texts is None:
                raise ValueError(f"defense {self.name!r} has rewrite_known=True but no known_texts.")
            changes["known_texts"] = self._redact_side("known", data.known_texts, cache, data.known_ids)
        return replace(data, **changes)

    def _redact_side(self, label: str, texts, cache: IndexedRowCache, ids=None) -> np.ndarray:
        # Scrub each styled conversation per turn (split on TURN_DELIM, scrub the distinct turns in
        # bulk, re-join), cached per conversation by the originating dataset's row id. The
        # OpenAnonymity backend token-chunks any oversized turn, so no length cap is needed here.
        redactor = self._get_redactor()

        def scrub_per_turn(conversations):
            return defend_conversations_per_turn(conversations, redactor.rewrite_batch)

        redacted = cache.apply(label, [str(t) for t in texts], scrub_per_turn, ids=ids)
        return np.asarray(redacted, dtype=object)
