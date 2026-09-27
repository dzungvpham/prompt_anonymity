"""POS n-gram TF-IDF featurizer for stylometric linkage.

Part-of-speech tag sequences capture syntactic habits (clause structure, word-order preferences)
largely orthogonal to the sub-word signal char n-grams capture. Expects each input string to
already be a space-joined POS tag sequence, and fits a word-level TfidfVectorizer over POS
n-grams, reduced to a fixed-length dense vector via SVD.

Deliberately **not** registered in :data:`~prompt_anonymity.features.FEATURIZERS`: a TF-IDF
vectorizer fits on the first batch it sees, so a corpus-wide precompute pass would leak
vocabulary from documents on both sides of any known/unknown split. Callers must construct a
fresh instance and fit it on the known slice only, per window.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from .base import Featurizer


class POSNgramTfidfFeaturizer(Featurizer):
    """TF-IDF over POS-tag n-grams, reduced to a fixed-length dense vector via SVD."""

    name = "pos_ngram_tfidf"
    version = "1"
    metric = "cosine"

    def __init__(self, ngram_range=(1, 3), max_features=5000, n_components=128, seed=47):
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
        """``texts`` are space-joined POS tag sequences, one per document.

        First call fits the vectorizer and SVD (intended to be the known-side call); every later
        call only transforms. Reuse one instance per split only -- not across splits.
        """
        texts = [t or "" for t in texts]

        if self._vectorizer is None:
            self._vectorizer = TfidfVectorizer(
                analyzer="word",
                token_pattern=r"(?u)\S+",
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
