"""Combined StyleRemix + OpenAnonymity defense, applied PER USER TURN.

Chains the two rewrite backends into one defense: StyleRemix restyles every prompt toward the fixed
:data:`~prompt_anonymity.defenses.styleremix.STYLEREMIX_SLIDERS` target style, then the
OpenAnonymity scrubber redacts identifiers and de-identifies residual style. Both stages run PER
TURN. Order rationale: style-convergence first, identifier redaction LAST, so OA's
``[PERSON_1]``/``[ORG_1]`` placeholders survive intact rather than being reworded by StyleRemix.

Granularity: both stages are per turn, so every user turn is defended on its own and the turn
structure (and count) is preserved end to end. That keeps the defended conversation aligned
turn-for-turn with the original -- which is what per-turn utility scoring needs -- and lets
StyleRemix reuse the standalone StyleRemix defense's per-turn cache. This defense's own cache holds
the stage-2 result keyed per conversation (by the *styled* text, so a stage-1 change re-caches
stage 2), and OA fragments any oversized turn so no length cap is needed.

Sequencing: the stages are not concurrent. Stage 1 restyles the entire dataset, then its vLLM engine
is shut down and the GPU handed to stage 2, which loads the scrubber and works from stage 1's
output. Both models are large (Llama-3-8B and gpt-oss-120b) and each vLLM engine reserves a fixed
fraction of the device for its lifetime, so overlapping them would mean splitting the GPU between
two models that are never used at the same moment.

Not a simple per-text rewrite, so it implements :meth:`__call__`/:meth:`transform` directly and
composes the two backends rather than subclassing the text-rewrite spine. Bump :attr:`version` when
the OpenAnonymity backend's behavior changes (its source is not in this class's hash; a StyleRemix
change re-caches automatically because it alters the stage-1 text that keys stage 2).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from ..caching import IndexedRowCache, params_hash
from ..core import AttackData
from ._backends import defend_conversations_per_turn
from .base import CachedDefense
from .openanonymity import OPENANON_MODEL, OPENANON_SYSTEM_PROMPT, _OpenAnonBackend
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
                 oa_model: str = OPENANON_MODEL, oa_system_prompt: str = OPENANON_SYSTEM_PROMPT):
        self.sliders = dict(STYLEREMIX_SLIDERS if sliders is None else sliders)
        self.base_model = base_model
        self.oa_model = oa_model
        self.oa_system_prompt = oa_system_prompt
        self._redactor = None

    def params(self) -> dict:
        # Stage 2 is keyed by the styled conversation text, so a slider/base-model change re-caches
        # automatically; include them anyway for reproducibility, plus the OA model + prompt (whose
        # source this class's hash does not cover) so an OA swap re-caches.
        return {
            "sliders": self.sliders, "base_model": self.base_model,
            "oa_model": self.oa_model, "oa_system_prompt": self.oa_system_prompt,
        }

    def _get_redactor(self) -> _OpenAnonBackend:
        if self._redactor is None:
            self._redactor = _OpenAnonBackend(self.oa_model, self.oa_system_prompt)
        return self._redactor

    def __call__(self, data: AttackData, *, cache_dir) -> AttackData:
        # The two stages run STRICTLY IN SEQUENCE, and only one model is resident at a time. Stage 1
        # restyles the whole dataset and its engine is then shut down, handing the GPU to stage 2,
        # which scrubs the restyled text. That ordering is the definition of the defense -- stage 2
        # reads stage 1's output, not the original -- and running the models concurrently would need
        # their memory fractions to sum under 1.0, which neither 8B + 120B nor the defaults allow.
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
        # Scrub each styled conversation PER TURN (split on TURN_DELIM -> scrub the distinct turns in
        # bulk -> re-join), so turn structure is preserved, cached per conversation by the originating
        # dataset's row id (SWE-chat session_id, WildChat idx). The OpenAnonymity backend token-chunks
        # any oversized turn, so no length cap is needed (matching the other per-turn defenses; the
        # featurize stage truncates later).
        redactor = self._get_redactor()

        def scrub_per_turn(conversations):
            return defend_conversations_per_turn(conversations, redactor.rewrite_batch)

        redacted = cache.apply(label, [str(t) for t in texts], scrub_per_turn, ids=ids)
        return np.asarray(redacted, dtype=object)
