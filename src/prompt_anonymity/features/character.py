"""A tiny, dependency-free featurizer for tests and examples.

:class:`CharacterStatisticsFeaturizer` needs no GPU or external models, so it lets the whole
load -> defense -> featurize -> attack pipeline run end-to-end anywhere (and pairs with the
example text-rewrite defense). It is **not** a serious stylometric signal -- use
:class:`~prompt_anonymity.features.stylometrix.StyloMetrixFeaturizer` for real experiments.
"""

from __future__ import annotations

import string

import numpy as np

from .base import Featurizer

_PUNCTUATION = set(string.punctuation)


class CharacterStatisticsFeaturizer(Featurizer):
    """Fixed-length vector of surface statistics per text: character count, uppercase / digit /
    whitespace / punctuation ratios, mean word length, and word count (7 features)."""

    name = "character_statistics"
    version = "1"

    def featurize(self, texts) -> np.ndarray:
        rows = []
        for text in texts:
            text = text or ""
            length = len(text) or 1  # avoid divide-by-zero on empty text
            words = text.split()
            rows.append([
                len(text),
                sum(char.isupper() for char in text) / length,
                sum(char.isdigit() for char in text) / length,
                sum(char.isspace() for char in text) / length,
                sum(char in _PUNCTUATION for char in text) / length,
                (sum(len(word) for word in words) / len(words)) if words else 0.0,
                len(words),
            ])
        return np.asarray(rows, dtype=float)
