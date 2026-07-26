"""Character n-gram TF-IDF featurizer for stylometric linkage.

Character n-grams (typically n=3-4) are a long-standing, strong signal in authorship
attribution literature -- they capture sub-word stylistic habits (spelling, punctuation
patterns, morphology) that word-level or POS-level features miss. This featurizer fits
a TfidfVectorizer over character n-grams and reduces dimensionality with truncated SVD
so it produces a fixed-length, dense vector compatible with the rest of the pipeline.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from .base import Featurizer
from sklearn.preprocessing import normalize

class CharNgramTfidfFeaturizer(Featurizer):
    """TF-IDF over character n-grams, reduced to a fixed-length dense vector via SVD."""

    name = "char_ngram_tfidf"
    version = "2"
    metric = "cosine"

    def __init__(self, ngram_range=(3, 4), max_features=5000, n_components=128, seed=47):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.n_components = n_components
        self.seed = seed
        self._vectorizer = None
        self._svd = None

    def params(self) -> dict:
        return {
            "ngram_range": list(self.ngram_range),
            "max_features": self.max_features,
            "n_components": self.n_components,
        }

    def featurize(self, texts) -> np.ndarray:
        texts = [t or "" for t in texts]

        if self._vectorizer is None:
            self._vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=self.ngram_range,
                max_features=self.max_features,
            )
            tfidf = self._vectorizer.fit_transform(texts)
            n_components = min(
                self.n_components,
                tfidf.shape[1] - 1,
                tfidf.shape[0] - 1,
            )
            self._svd = TruncatedSVD(
                n_components=n_components,
                random_state=self.seed,
            )

            vecs = self._svd.fit_transform(tfidf)
            return normalize(vecs, norm="l2")

        tfidf = self._vectorizer.transform(texts)
        vecs = self._svd.transform(tfidf)
        return normalize(vecs, norm="l2")