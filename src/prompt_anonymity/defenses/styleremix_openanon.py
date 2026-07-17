"""Combined StyleRemix + OpenAnonymity defense, applied PER USER TURN.

Chains the two rewrite backends into one defense: StyleRemix restyles every prompt toward the fixed
:data:`~prompt_anonymity.defenses.styleremix.STYLEREMIX_SLIDERS` target style, then the
OpenAnonymity scrubber redacts identifiers and de-identifies residual style. Both stages run PER
TURN. Order rationale: style-convergence first, identifier redaction LAST, so OA's
``[PERSON_1]``/``[ORG_1]`` placeholders survive intact rather than being reworded by StyleRemix.

Granularity: both stages are per turn, so every user turn is defended on its own and the turn
structure (and count) is preserved end to end. That keeps the defended conversation aligned
turn-for-turn with the original -- which is what per-turn utility/fidelity scoring needs -- and lets
StyleRemix reuse the standalone StyleRemix defense's per-turn cache. The trade-off is cost: the paid
OpenRouter API is now called once per distinct turn rather than once per conversation. This defense's
own cache holds the stage-2 result keyed per conversation (by the styled text), so the run stays
resumable, and OA token-chunks any oversized turn so no length cap is needed.

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
        # Stage 1: StyleRemix restyle PER TURN, via the standalone defense so its cache is shared
        # (already-restyled turns are reused, not recomputed).
        restyle = StyleRemixDefense(sliders=self.sliders, base_model=self.base_model)
        restyle.rewrite_known = self.rewrite_known
        styled = restyle(data, cache_dir=cache_dir)

        # Stage 2: OpenAnonymity redact PER CONVERSATION, in this defense's own index-keyed cache.
        cache = IndexedRowCache(
            Path(cache_dir) / "defenses", self.name, self._logic_hash(), params_hash(self.params())
        )
        return self.transform(styled, cache)

    def transform(self, data: AttackData, cache: IndexedRowCache) -> AttackData:
        if data.unknown_texts is None:
            raise ValueError(f"defense {self.name!r} needs unknown_texts; load the dataset with text.")
        changes = {"unknown_texts": self._redact_side("unknown", data.unknown_texts, cache)}
        if self.rewrite_known:
            if data.known_texts is None:
                raise ValueError(f"defense {self.name!r} has rewrite_known=True but no known_texts.")
            changes["known_texts"] = self._redact_side("known", data.known_texts, cache)
        return replace(data, **changes)

    def _redact_side(self, label: str, texts, cache: IndexedRowCache) -> np.ndarray:
        # Scrub each styled conversation PER TURN (split on TURN_DELIM -> scrub the distinct turns in
        # bulk -> re-join), so turn structure is preserved, cached per conversation by its
        # reference-row index. The OpenAnonymity backend token-chunks any oversized turn, so no length
        # cap is needed (matching the other per-turn defenses; the featurize stage truncates later).
        redactor = self._get_redactor()

        def scrub_per_turn(conversations):
            return defend_conversations_per_turn(conversations, redactor.rewrite_batch)

        redacted = cache.apply(label, [str(t) for t in texts], scrub_per_turn)
        return np.asarray(redacted, dtype=object)
