"""Character n-gram TF-IDF, fitted on the attacker's known side and nowhere else.

Character n-grams capture sub-word habits -- spelling, punctuation, spacing, casing, morphology --
that word- and POS-level features miss. This follows Koppel, Schler & Argamon's "Authorship
attribution in the wild" (LRE 2011): character 4-grams, TF-IDF weighting, cosine similarity, no
dimensionality reduction.

**Why this is not a** :class:`~prompt_anonymity.features.base.Featurizer`. Every other feature is
a per-document function, computed once offline into a parquet. TF-IDF is *fitted*: which n-grams
are kept and how each is weighted are statistics of a corpus, so fitting on the whole split would
let held-out test documents shape the feature space the attacker compares them in. Instead
``experiments/run_experiment.py`` fits it **per known configuration**, on that configuration's
known documents only (and again per tuning fold), so no scored document influences its own
representation.

The work splits in two so the fitted part stays cheap to repeat:

1. :meth:`CharNgramTfidf.count` -- raw counts of every distinct character n-gram, computed once
   per run; a document's count depends only on itself.
2. :class:`KnownSideTfidf` -- the fitted part: keep the ``max_features`` most frequent n-grams
   *on the fitted rows*, weight by IDF over those rows, L2-normalise.

The count vocabulary spans the whole split, which is not a leak: an n-gram absent from the known
side has known-side frequency zero and can never be selected or weighted.

Notable choices: ``analyzer="char"`` (not ``"char_wb"``) so n-grams run across word boundaries,
capturing inter-word spacing/punctuation; case is kept (capitalisation is a writing habit, not
noise); no SVD, since a variance-maximizing projection tends to discard the rare, idiosyncratic
n-grams that identify a writer; rows are L2-normalised. The output is **already scaled**, so the
runner must not z-score it -- a per-column z-score would cancel the IDF weighting (see
``run_experiment.py``'s ``--standardize``).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize


class KnownSideTfidf(BaseEstimator, TransformerMixin):
    """TF-IDF over a precomputed n-gram count matrix, fitted only on the rows given to :meth:`fit`.

    Keeps the ``max_features`` n-grams with the largest total count over the fitted rows, weighted
    by smoothed IDF over those rows. :meth:`transform` returns a dense ``float32``
    ``[n_documents x max_features]`` array with unit-L2 rows.

    A plain sklearn transformer, so a fresh copy can be fitted per tuning fold with
    :func:`sklearn.base.clone`.
    """

    def __init__(self, max_features: int = 3072):
        self.max_features = max_features

    def fit(self, counts, y=None):
        """Choose the kept n-grams and their IDF weights from ``counts`` (a sparse count matrix)."""
        counts = sp.csr_matrix(counts)
        total_counts = np.asarray(counts.sum(axis=0)).ravel()
        present = np.flatnonzero(total_counts)
        # Stable sort so a refit on the same rows always keeps the same columns.
        order = present[np.argsort(-total_counts[present], kind="stable")]
        self.columns_ = np.sort(order[:self.max_features])
        kept = counts[:, self.columns_]
        document_frequency = np.bincount(kept.indices, minlength=len(self.columns_))
        n_documents = counts.shape[0]
        self.idf_ = np.log((1.0 + n_documents) / (1.0 + document_frequency)) + 1.0
        return self

    def transform(self, counts) -> np.ndarray:
        """Weight and normalise ``counts`` with the fitted columns and IDF; dense ``float32``."""
        weighted = sp.csr_matrix(counts)[:, self.columns_].multiply(self.idf_).tocsr()
        return normalize(weighted, norm="l2").astype(np.float32).toarray()


class CharNgramTfidf:
    """Character ``n``-gram counts plus a :class:`KnownSideTfidf` to fit on them.

    ``name`` is the runner's ``--feature`` value and the feature part of a results directory.
    :attr:`already_scaled` tells the runner to skip its z-score (see the module docstring).
    """

    name = "char_ngram_tfidf"
    already_scaled = True

    def __init__(self, ngram_length: int = 4, max_features: int = 3072, lowercase: bool = False):
        self.ngram_length = ngram_length
        self.max_features = max_features
        self.lowercase = lowercase

    def params(self) -> dict:
        """The configuration that decides the vectors, for printing next to a run."""
        return {"ngram_length": self.ngram_length, "max_features": self.max_features,
                "analyzer": "char", "lowercase": self.lowercase}

    def count(self, *text_blocks) -> list[sp.csr_matrix]:
        """Raw n-gram counts for each block of texts, all in **one** shared column space.

        Several blocks exist when the known and unknown sides come from different text (a
        defended query against an undefended history, ``--known-defense``): the columns must mean
        the same n-gram in both, so one vocabulary is built over every block together.
        """
        vectorizer = CountVectorizer(analyzer="char", lowercase=self.lowercase,
                                     ngram_range=(self.ngram_length, self.ngram_length),
                                     dtype=np.int32)
        stacked = vectorizer.fit_transform([text or "" for block in text_blocks for text in block])
        bounds = np.cumsum([0, *(len(block) for block in text_blocks)])
        return [stacked[start:stop] for start, stop in zip(bounds[:-1], bounds[1:])]

    def transformer(self) -> KnownSideTfidf:
        """A fresh, unfitted weighting step, to be fitted on one known side."""
        return KnownSideTfidf(max_features=self.max_features)
