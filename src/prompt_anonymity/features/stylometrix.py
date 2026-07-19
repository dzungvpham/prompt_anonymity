"""StyloMetrix stylometric featurizer.

Produces the same feature space as the committed StyloMetrix CSVs, so when a defense leaves
text unchanged the featurize step can reuse those precomputed vectors and only recompute the
text a defense actually rewrote. Mirrors ``wildchat/stylometrix.py`` and
``swe-chat/stylometrix.py`` exactly: truncate each text to the first :data:`MAX_LEN` chars
(an empty text becomes a harmless token), run ``stylo_metrix.StyloMetrix(language_code)``,
drop the echoed ``text`` column, and replace NaN with 0.
"""

from __future__ import annotations

import warnings

import numpy as np

from .base import Featurizer

# First N characters fed to StyloMetrix; matches the committed feature CSVs (the
# MAX_LEN = 2048 in wildchat/stylometrix.py and swe-chat/stylometrix.py). Kept in params() so
# changing it invalidates the cache (its value is invisible to source hashing).
MAX_LEN = 2048

# Placeholder for empty/whitespace-only text so StyloMetrix never sees an empty document
# (matches swe-chat/stylometrix.py); its features come out ~0 and carry no signal.
_EMPTY_PLACEHOLDER = "n a"


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
    """

    name = "stylometrix"
    version = "1"

    def __init__(self, *, language_code: str = "en"):
        self.language_code = language_code
        self._model = None  # lazily constructed stylo_metrix.StyloMetrix (holds the spaCy pipeline)

    def params(self) -> dict:
        return {"language_code": self.language_code, "max_len": MAX_LEN}

    def _ensure_model(self):
        """Construct the StyloMetrix model on first use (lazy; GPU when available, else CPU)."""
        if self._model is None:
            import spacy
            import stylo_metrix as sm

            # Use the GPU if spaCy can reach one; otherwise fall back to CPU (much slower).
            if not spacy.prefer_gpu():
                warnings.warn(
                    "StyloMetrix is running on CPU because spaCy could not find a GPU; "
                    "featurizing many conversations this way can take a very long time. "
                    "Install GPU spaCy (see README.md) for a large speedup.",
                    stacklevel=2,
                )
            self._model = sm.StyloMetrix(self.language_code)
        return self._model

    def featurize(self, texts) -> np.ndarray:
        model = self._ensure_model()
        prepared = [
            (text[:MAX_LEN] if text and text.strip() else _EMPTY_PLACEHOLDER) for text in texts
        ]
        frame = model.transform(prepared)
        if "text" in frame.columns:  # StyloMetrix echoes the input text as a column; drop it
            frame = frame.drop(columns="text")
        # StyloMetrix can emit NaN (e.g. ratios with a zero denominator); treat those as 0.
        return np.nan_to_num(frame.to_numpy(dtype=np.float32), nan=0.0)
