"""Character n-gram TF-IDF, fitted on the attacker's known side and nowhere else.

Character n-grams are the long-standing workhorse of authorship attribution: they capture
sub-word habits -- spelling, punctuation, spacing, casing, morphology -- that word- and POS-level
features miss. The closest precedent at this project's scale is Koppel, Schler & Argamon,
"Authorship attribution in the wild" (LRE 2011): character 4-grams, TF-IDF weighting and cosine
similarity over ~10,000 blog authors, with no dimensionality reduction. This is that
representation.

**Why this is not a** :class:`~prompt_anonymity.features.base.Featurizer`. Every other feature
is a per-document function, so it is computed once, offline, into a parquet the runner reads.
TF-IDF is *fitted*: which n-grams are kept and how each is weighted are statistics of a corpus.
Fitting them on the whole split -- which is what the earlier SVD-based featurizer did -- lets the
held-out test documents shape the feature space the attacker compares them in. Here the fit is
done by ``experiments/run_experiment.py`` **per known configuration**, on that configuration's
known documents only, and again on each tuning fold's training block, so no document the attack
is scored on influences its own representation.

The work is split in two so that the fitted part is cheap enough to repeat that often:

1. :meth:`CharNgramTfidf.count` -- **raw counts** of every distinct character n-gram in every
   document, computed once per run. A count depends on its own document only.
2. :class:`KnownSideTfidf` -- the fitted part: keep the ``max_features`` n-grams most frequent
   *on the rows it is fitted on*, weight them by inverse document frequency over those rows, and
   L2-normalise each document. A column slice and two vector products on a sparse matrix.

The count vocabulary does span the whole split, and that is not a leak: an n-gram seen only in
unknown documents has a known-side frequency of zero, so it can never be selected and never
receives a weight. Selecting from the whole split's vocabulary is therefore exactly equivalent to
fitting ``sklearn.feature_extraction.text.TfidfVectorizer(analyzer="char", ngram_range=(n, n),
lowercase=False, max_features=...)`` on the known documents and transforming the rest (up to the
order of n-grams tied at the ``max_features`` boundary, which sklearn does not fix either).

Choices worth knowing before comparing numbers:

* ``analyzer="char"``, not ``"char_wb"``: n-grams run across word boundaries, so spacing and
  punctuation between words (``", th"``, ``") {"``) are features. They are much of the stylistic
  signal, and ``char_wb`` pads each word with spaces and cannot see them.
* **Case is kept** (``lowercase=False``): capitalisation is a writing habit, not noise.
* **No SVD.** A projection keeps the directions of largest variance, which are largely topic, and
  can discard exactly the rare-but-idiosyncratic n-grams that identify a writer.
* **Rows are L2-normalised**, sklearn's TF-IDF default. It makes every document count equally in
  an author centroid and keeps long documents from dominating a fit.
* The output is **already scaled**, so the runner does not z-score it: a per-column z-score
  divides every column by its own spread, which cancels the IDF weighting outright (IDF *is* a
  per-column scale). See ``run_experiment.py``'s ``--standardize``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize


class KnownSideTfidf(BaseEstimator, TransformerMixin):
    """TF-IDF over a precomputed n-gram count matrix, fitted only on the rows given to :meth:`fit`.

    ``max_features`` columns are kept -- the n-grams with the largest total count over the fitted
    rows, which is the selection rule of sklearn's ``TfidfVectorizer(max_features=...)``. Each is
    weighted by sklearn's smoothed inverse document frequency, ``ln((1 + n) / (1 + df)) + 1``,
    over the same rows; raw counts are the term frequency. :meth:`transform` returns a dense
    ``float32`` ``[n_documents x max_features]`` array with unit-L2 rows (a document containing
    none of the kept n-grams stays all zero).

    Being a plain sklearn transformer, a fresh copy can be fitted per tuning fold with
    :func:`sklearn.base.clone`.
    """

    def __init__(self, max_features: int = 3072):
        self.max_features = max_features

    def fit(self, counts, y=None):
        """Choose the kept n-grams and their IDF weights from ``counts`` (a sparse count matrix)."""
        counts = sp.csr_matrix(counts)
        total_counts = np.asarray(counts.sum(axis=0)).ravel()
        present = np.flatnonzero(total_counts)
        # Most frequent first; a stable sort breaks ties by column index, so a refit on the same
        # rows always keeps the same columns.
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
