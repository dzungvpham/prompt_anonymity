"""Combined StyleRemix + OpenAnonymity defense at MIXED granularity.

Chains the two rewrite backends into one defense: StyleRemix restyles every prompt toward the fixed
:data:`~prompt_anonymity.defenses.styleremix.STYLEREMIX_SLIDERS` target style PER TURN, then the
OpenAnonymity scrubber redacts identifiers and de-identifies residual style on the FULLY-JOINED
conversation. Order rationale: style-convergence first, identifier redaction LAST, so OA's
``[PERSON_1]``/``[ORG_1]`` placeholders survive intact rather than being reworded by StyleRemix.

Granularity rationale (cost): StyleRemix stays PER TURN so it *reuses the standalone StyleRemix
defense's cache* (keyed by per-turn text) and keeps each input inside its 2048-token window.
OpenAnonymity runs PER CONVERSATION so the paid OpenRouter API is called once per conversation
instead of once per turn -- a large cut. This defense's own cache holds the stage-2 result keyed by
the STYLED, capped conversation, so the run is resumable and OA is paid once per unique styled
conversation.

Not a simple per-text rewrite, so it implements :meth:`__call__`/:meth:`transform` directly and
composes the two backends rather than subclassing the text-rewrite spine. Bump :attr:`version` when
the OpenAnonymity backend's behavior changes (its source is not in this class's hash; a StyleRemix
change re-caches automatically because it alters the stage-1 text that keys stage 2).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from ..caching import TransformCache, params_hash
from ..core import AttackData
from .base import CachedDefense
from .openanonymity import OPENANON_MODEL, OPENANON_SYSTEM_PROMPT, _OpenAnonBackend
from .styleremix import STYLEREMIX_BASE_MODEL, STYLEREMIX_SLIDERS, StyleRemixDefense

#: Char cap on each styled conversation before the OA call. The featurize stage truncates to this
#: anyway (so anything past it is discarded by the attack), and uncapped conversations reach ~10^5+
#: tokens, which stalls/times out the per-conversation OA call.
MAX_LEN = 2048


class StyleRemixOpenAnonymityDefense(CachedDefense):
    """StyleRemix restyle (per turn) then OpenAnonymity redact (per conversation).

    Both backends are built lazily: a run whose StyleRemix and combined caches are already complete
    loads neither model and makes no API calls. Side selection follows :attr:`rewrite_known`.
    """

    name = "styleremix_openanon"
    version = "1"
    rewrite_known: bool = False

    def __init__(self, *, sliders: dict | None = None, base_model: str = STYLEREMIX_BASE_MODEL,
                 oa_model: str = OPENANON_MODEL, oa_system_prompt: str = OPENANON_SYSTEM_PROMPT,
                 max_len: int = MAX_LEN):
        self.sliders = dict(STYLEREMIX_SLIDERS if sliders is None else sliders)
        self.base_model = base_model
        self.oa_model = oa_model
        self.oa_system_prompt = oa_system_prompt
        self.max_len = max_len
        self._redactor = None

    def params(self) -> dict:
        # Stage 2 is keyed by the styled conversation text, so a slider/base-model change re-caches
        # automatically; include them anyway for reproducibility, plus the OA model + prompt (whose
        # source this class's hash does not cover) so an OA swap re-caches.
        return {
            "sliders": self.sliders, "base_model": self.base_model,
            "oa_model": self.oa_model, "oa_system_prompt": self.oa_system_prompt,
            "max_len": self.max_len,
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

        # Stage 2: OpenAnonymity redact PER CONVERSATION, in this defense's own cache.
        cache = TransformCache(
            Path(cache_dir) / "defenses", self.name, self._logic_hash(), params_hash(self.params())
        )
        return self.transform(styled, cache)

    def transform(self, data: AttackData, cache: TransformCache) -> AttackData:
        if data.unknown_texts is None:
            raise ValueError(f"defense {self.name!r} needs unknown_texts; load the dataset with text.")
        changes = {"unknown_texts": self._redact_side(data.unknown_texts, cache)}
        if self.rewrite_known:
            if data.known_texts is None:
                raise ValueError(f"defense {self.name!r} has rewrite_known=True but no known_texts.")
            changes["known_texts"] = self._redact_side(data.known_texts, cache)
        return replace(data, **changes)

    def _redact_side(self, texts, cache: TransformCache) -> np.ndarray:
        # Cap each styled conversation, then scrub the WHOLE conversation once (per-conversation OA).
        capped = [str(t)[:self.max_len] for t in texts]
        redactor = self._get_redactor()
        redacted = cache.apply_batch(capped, redactor.rewrite_batch)
        return np.asarray(redacted, dtype=object)
