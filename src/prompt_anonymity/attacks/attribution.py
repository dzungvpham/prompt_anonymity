"""Supervised authorship-attribution attacks: fit on the known side, score unknown documents.

The second attack family in this package. The one in
:func:`~prompt_anonymity.attacks.nearest_neighbor_attack` is a fixed *distance* between
conversations; these are **models trained on the attacker's own labelled data**, which is a
fair thing to assume -- the known side is the attacker's, so its labels, document counts and
covariance structure are all available to them. That extra information is worth a lot: on
swe-chat it roughly doubles top-1 over the same features scored by cosine distance.

Every attack here is fitted on the known side only -- known documents plus their author labels
-- and then applied to unknown documents using nothing but their feature vectors. No attack
reads an unknown label, and the standardiser, covariances and classifiers are all estimated
from known data, so nothing leaks from the set being re-identified.

Interface
---------
``fit(embeddings, labels) -> self`` then ``score(embeddings) -> (n_docs, n_authors)`` where
**higher means "more likely this author"**, with the author order in ``self.authors``. Note the
orientation: this is a *score* matrix over authors, not the ``[n_unknown x n_known]`` distance
matrix over conversations that :data:`~prompt_anonymity.attacks.ATTACKS` produces, which is why
the two families have separate registries. Identification is ``argmax``; the out-of-set decision
is a threshold on the rejection score built by
:func:`~prompt_anonymity.attacks.rejection.rejection_score` from the same matrix. Score it with
:mod:`prompt_anonymity.metrics.ranking` and :mod:`prompt_anonymity.metrics.detection`.

Register a new one in :data:`ATTRIBUTION_ATTACKS` and it becomes selectable by name.

Why the discriminative attacks win here
---------------------------------------
Plain cosine to an author centroid treats every StyloMetrix direction as equally informative.
It is not: some columns vary wildly *within* one author (noise) and some separate authors
(signal). With ~124 candidate authors, that mistake is fatal -- the best of 124 impostor
distances lands closer than a genuine match, so the nearest centroid is usually the wrong one.
The methods below fix it in two different ways, and on swe-chat the discriminative one wins:

=====================  ============  ==========  ===========
attack                 top-1 (75%)   OOD AUROC   DIR@10%
=====================  ============  ==========  ===========
nearest_neighbor       0.154         0.588       0.018
cosine                 0.129         0.523       0.005
wccn                   0.184         0.496       0.005
lda                    0.168         0.518       0.013
plda                   0.133         0.481       0.001
logistic (balanced)    0.252         0.600       0.094
**logistic**           **0.258**     **0.631**   **0.106**
=====================  ============  ==========  ===========

(75% known window, 124 known authors, 997 unknown documents, 17.9% out-of-set; scores
cohort-normalised, all attacks at their default settings.)

Among the discriminative attacks the trees win, and by more than the linear ones differ from
each other. Averaged over all eight rolling windows, with **every hyper-parameter chosen per
window on that window's own known side** (``run_experiment_v2.py --tune``, measured over the
exhaustive grid that ``--tune`` used before it moved to successive halving -- the ranking is the
finding, the third decimal place is not):

=====================  ========  =======  ======  ======  =====
attack                 top-1     macro    MRR     MAP     ECE
=====================  ========  =======  ======  ======  =====
logistic               0.265     0.181    0.388   0.168   0.165
svm                    0.279     0.189    0.400   0.255   0.374
**xgboost**            **0.311**  0.198   0.424   0.268   0.199
=====================  ========  =======  ======  ======  =====

Three things that table is saying, none of them "trees are just better":

* **The gain is concentrated on prolific authors.** Micro top-1 improves by 0.046 from logistic
  to xgboost but the author-averaged version improves by only 0.017. Trees exploit the document
  count, which is real signal but means the typical user is much less affected than the headline
  suggests.
* **Logistic is far worse at retrieval than at identification** (MAP 0.168 against 0.255-0.268).
  Per-class bias terms shift a whole score column, which leaves row-wise ``argmax`` untouched but
  scrambles the column-wise ranking that :mod:`prompt_anonymity.metrics.retrieval` measures.
* **The SVM's confidence is meaningless** (ECE 0.374, roughly double the others). Expected: its
  one-vs-one margins are not posteriors, so read its calibration column as "not applicable"
  rather than "poorly calibrated". Only ``logistic`` and ``xgboost`` emit anything posterior-like.

The choice of *attack* was made the same way as the choice of hyper-parameters: the known-side
CV preferred xgboost in 7 of the 8 windows, and it went on to win 8 of 8 on the unknown side, so
the preference was predicted rather than observed. Selecting the attack per window on known-side
CV scores 0.309, against 0.311 for always using xgboost and 0.311 for a (not legitimate) oracle.

**How firm is "xgboost wins"?** Firm on the average, softer per window than 8-of-8 suggests.
Re-running the identical comparison under the successive-halving search that ``--tune`` now uses
(a different, equally legitimate way to pick each attack's hyper-parameters) gives xgboost 0.308,
svm 0.291, logistic 0.254 -- the same order, and the same gap between trees and linear models --
but the per-window tally becomes 5 xgboost, 2 svm, 1 logistic. A clean sweep was partly the luck
of one search; the mean is what replicates.

Two things that did *not* help, both worth not re-trying blind: a learned rejector over the
score-vector shape (margin, entropy, peakedness) overfits the simulation and loses to the plain
cohort-normalised score (DIR@10% 0.055 vs 0.106), and clustering the unknown side to pool
evidence fails because pairwise same-author AUROC is only 0.64 even in the whitened space
(adjusted Rand index 0.16 at best).
"""

from __future__ import annotations

import numpy as np
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances_chunked
from sklearn.svm import SVC


# --- shared linear algebra ---------------------------------------------------

def unit_rows(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalise each row, leaving all-zero rows alone."""
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    return embeddings / np.where(norms > 0, norms, 1.0)


def class_means(embeddings: np.ndarray, codes: np.ndarray, n_classes: int) -> np.ndarray:
    """Mean vector per integer-coded class."""
    sums = np.zeros((n_classes, embeddings.shape[1]))
    np.add.at(sums, codes, embeddings)
    return sums / np.bincount(codes, minlength=n_classes)[:, None]


def within_class_covariance(embeddings: np.ndarray, codes: np.ndarray, n_classes: int,
                            shrinkage: float = 0.1) -> np.ndarray:
    """Pooled within-author covariance, shrunk toward a scaled identity.

    This is the matrix plain cosine ignores. Directions along which a single author's own
    documents scatter widely should count for *less* when comparing documents, not the same.
    Shrinkage keeps the estimate invertible when authors are few or documents are short.
    """
    centred = embeddings - class_means(embeddings, codes, n_classes)[codes]
    covariance = centred.T @ centred / max(len(embeddings) - n_classes, 1)
    scale = np.trace(covariance) / covariance.shape[0]
    return (1 - shrinkage) * covariance + shrinkage * scale * np.eye(covariance.shape[0])


def inverse_sqrt(matrix: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    """Symmetric inverse square root, with eigenvalues floored for numerical safety."""
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, floor * values.max())
    return vectors @ np.diag(values ** -0.5) @ vectors.T


# --- scoring methods ---------------------------------------------------------

class NearestNeighbor:
    """The original attack: an author scores as well as their single closest known document.

    Kept as a registered method so the document-level baseline appears in the same
    ``headline_results.csv`` as everything else. Note what the author-level aggregation buys:
    ranking *authors* by their nearest document is exactly the ranking
    :class:`~prompt_anonymity.evaluation.LinkageRanking` already derives for ``id_acc``, but it
    also makes ``conv_acc`` mean "the true author is among the top k **authors**" rather than
    "among the authors of the top k **documents**". The latter is a harder question at the same
    k -- the 5 nearest documents cover only ~4 distinct authors on swe-chat, and the 10 nearest
    only ~7 -- so this aggregation is what makes top-k comparable across methods.

    ``linkage="max"`` (the default, meaning maximum similarity / minimum distance) is the
    nearest-neighbour attack proper; ``"mean"`` averages over all of an author's documents.
    Mean linkage is *almost* :class:`CentroidCosine` -- it is exactly that attack without the
    final re-normalisation of the centroid (see :meth:`fit`) -- but the difference is real
    rather than cosmetic, so both are worth running.

    Scaling
    -------
    Sized for author pools in the tens of thousands. Three things make that work:

    * **Cosine is computed as a matrix product, not by** :func:`scipy.spatial.distance.cdist`.
      ``cdist``'s cosine is a naive C loop over pairs that also forces float64; normalising both
      sides once and calling BLAS is the same arithmetic ~150x faster (measured 115s vs 0.77s on
      2,000 x 20,000 x 3,072, agreeing to 6e-8). Any other ``metric`` falls back to
      :func:`~sklearn.metrics.pairwise_distances_chunked`, which is BLAS-backed for
      ``"euclidean"`` and chunks a per-block ``cdist`` for everything else.
    * **The pairwise matrix is never materialised.** Queries are processed in row blocks sized to
      ``working_memory_mb`` and each block is reduced to author scores before the next is built.
      At 50,000 unknown against 100,000 known the full float64 distance matrix would be 40 GB.
    * **Documents are float32 by default.** Halves both the GEMM cost and the score matrix
      (3 GB rather than 6 GB at 50,000 x 15,000). Cosine rankings are unaffected at that
      precision; pass ``dtype=np.float64`` to reproduce older runs bit-for-bit.

    End to end at 50,000 unknown x 100,000 known x 3,072 dimensions over 15,000 authors: about
    two minutes on 36 CPU cores, against roughly four hours for the ``cdist`` formulation. This
    is *exact* nearest-neighbour search, deliberately -- an approximate index would trade recall
    for time, and a missed neighbour is a false negative in precisely the hard cases that
    separate one attack from another, which would bias the headline re-identification number
    downward and non-uniformly.
    """

    name = "nearest_neighbor"

    def __init__(self, metric: str = "cosine", linkage: str = "max",
                 working_memory_mb: int = 2048, dtype=np.float32):
        self.metric = metric
        self.linkage = linkage
        self.working_memory_mb = working_memory_mb
        self.dtype = dtype

    def fit(self, embeddings, labels):
        self.authors, codes = np.unique(labels, return_inverse=True)
        known = np.asarray(embeddings, dtype=self.dtype)

        # Sort the known side by author so every author's documents form one contiguous block.
        # ``ufunc.reduceat`` can then aggregate all authors in a single pass over a score block,
        # replacing a per-author Python loop that rebuilt an n_known boolean mask once per author.
        order = np.argsort(codes, kind="stable")
        self._known = np.ascontiguousarray(known[order])
        sorted_codes = codes[order]
        # Codes come from np.unique over the labels, so every author owns at least one document:
        # the block boundaries are strictly increasing and no block is empty, which is exactly
        # the precondition reduceat needs.
        self._starts = np.searchsorted(sorted_codes, np.arange(len(self.authors)))
        # Held in the working dtype rather than as integers: these are only ever a divisor for
        # mean linkage, and NumPy would promote a float32 score block divided by an int64 count
        # back to float64, quietly doubling the size of the score matrix.
        self._counts = np.bincount(sorted_codes, minlength=len(self.authors)).astype(self.dtype)

        if self._uses_cosine:
            self._known_unit = unit_rows(self._known)
            if self.linkage == "mean":
                # Cosine distance is affine in the second vector, so averaging it over an
                # author's documents commutes with the dot product:
                #     mean_j (1 - x.y_j) = 1 - x.(mean_j y_j)
                # One centroid per author therefore reproduces mean linkage *exactly* while
                # shrinking the known side from n_known vectors to n_authors.
                #
                # The centroid is deliberately left un-normalised, which is the whole difference
                # from CentroidCosine: its length records how tightly the author's documents
                # cluster, so a diffuse author is penalised. Re-normalising here would change the
                # top-1 pick on a non-trivial fraction of documents.
                self._centroids = (np.add.reduceat(self._known_unit, self._starts, axis=0)
                                   / self._counts[:, None])
        return self

    @property
    def _uses_cosine(self) -> bool:
        """Whether the BLAS fast path applies (it is written for cosine only)."""
        return self.metric == "cosine"

    def _query_block_rows(self, n_known: int) -> int:
        """Number of query rows whose score block fits the memory budget."""
        row_bytes = max(n_known * np.dtype(self.dtype).itemsize, 1)
        return max(1, int(self.working_memory_mb * 1024 * 1024 // row_bytes))

    def score(self, embeddings):
        query = np.asarray(embeddings, dtype=self.dtype)
        if self._uses_cosine:
            return self._score_cosine(query)

        # General path: any other cdist-compatible metric. sklearn picks the block size from
        # its own working_memory setting and hands each block to reduce_func, so the full
        # pairwise matrix is never held either.
        def reduce_func(block, start):
            if self.linkage == "mean":
                return np.add.reduceat(block, self._starts, axis=1) / self._counts
            return np.minimum.reduceat(block, self._starts, axis=1)

        with sklearn.config_context(working_memory=self.working_memory_mb):
            blocks = list(pairwise_distances_chunked(
                query, self._known, metric=self.metric, reduce_func=reduce_func))
        return -np.vstack(blocks)  # negate: higher must mean "more likely this author"

    def _score_cosine(self, query):
        """Cosine scoring via BLAS, with the aggregation folded into the distance computation.

        Both linkages are rewritten in terms of cosine *similarity* so the work is a matrix
        product: the cosine distance ``1 - s`` is decreasing in the similarity ``s``, so the
        minimum distance over an author's documents is their maximum similarity. The trailing
        ``- 1`` restores the negated-distance scale the caller expects; it is a constant shift
        across the whole matrix and so leaves every ranking untouched, but keeping it means these
        scores stay numerically comparable with the general path above.
        """
        query_unit = unit_rows(query)
        if self.linkage == "mean":
            return query_unit @ self._centroids.T - 1.0

        scores = np.empty((len(query_unit), len(self.authors)), dtype=self.dtype)
        for start in range(0, len(query_unit), self._query_block_rows(len(self._known_unit))):
            stop = min(start + self._query_block_rows(len(self._known_unit)), len(query_unit))
            similarity = query_unit[start:stop] @ self._known_unit.T
            np.maximum.reduceat(similarity, self._starts, axis=1, out=scores[start:stop])
        return scores - 1.0


class CentroidCosine:
    """Baseline: cosine similarity to each author's mean direction.

    The attack the rolling-window experiment shipped with. Kept as the reference point every
    supervised method below is measured against.
    """

    name = "cosine"

    def fit(self, embeddings, labels):
        self.authors, codes = np.unique(labels, return_inverse=True)
        self.centroids = unit_rows(class_means(unit_rows(embeddings), codes, len(self.authors)))
        return self

    def score(self, embeddings):
        return unit_rows(embeddings) @ self.centroids.T


class WhitenedCentroid:
    """Cosine to the centroid after whitening by the within-author covariance (WCCN).

    Speaker verification's standard first move: whitening equalises within-author noise across
    directions, so what remains of the distance is between-author structure. Length-normalising
    after whitening keeps the transformed vectors well-behaved enough for a cosine to mean
    something.
    """

    name = "wccn"

    def __init__(self, shrinkage: float = 0.2):
        self.shrinkage = shrinkage

    def fit(self, embeddings, labels):
        self.authors, codes = np.unique(labels, return_inverse=True)
        self.mean = embeddings.mean(axis=0)
        covariance = within_class_covariance(embeddings - self.mean, codes, len(self.authors),
                                             self.shrinkage)
        self.transform = inverse_sqrt(covariance)
        whitened = unit_rows((embeddings - self.mean) @ self.transform)
        self.centroids = unit_rows(class_means(whitened, codes, len(self.authors)))
        return self

    def project(self, embeddings):
        """Map documents into the whitened space (useful for clustering experiments)."""
        return unit_rows((embeddings - self.mean) @ self.transform)

    def score(self, embeddings):
        return self.project(embeddings) @ self.centroids.T


class LDACentroid:
    """Project onto the LDA discriminant subspace, then cosine to the centroid.

    LDA maximises between-author over within-author scatter, which both denoises and discards
    the StyloMetrix directions that carry no author information at all.

    ``n_components`` is a *ceiling*: LDA cannot produce more than ``n_authors - 1`` discriminants,
    so a request for more is silently clamped rather than raised. That matters when the same
    setting is reused across differently sized author pools -- a hyper-parameter search over
    subsampled folds would otherwise fail on the small ones for a reason that has nothing to do
    with the setting's quality.
    """

    name = "lda"

    def __init__(self, n_components: int | None = None):
        self.n_components = n_components

    def fit(self, embeddings, labels):
        self.authors, codes = np.unique(labels, return_inverse=True)
        available = min(len(self.authors) - 1, embeddings.shape[1])
        n_components = min(self.n_components, available) if self.n_components else available
        self.model = LinearDiscriminantAnalysis(
            solver="eigen", shrinkage="auto", n_components=n_components
        ).fit(embeddings, codes)
        self.centroids = unit_rows(class_means(unit_rows(self.model.transform(embeddings)),
                                               codes, len(self.authors)))
        return self

    def project(self, embeddings):
        return unit_rows(self.model.transform(embeddings))

    def score(self, embeddings):
        return self.project(embeddings) @ self.centroids.T


class LogisticAttribution:
    """Multinomial logistic regression over the known authors; the score is the class logit.

    The best method measured on swe-chat, and the reason is worth stating: the generative
    methods above model *where each author sits*, while this learns *what separates them*. With
    196 noisy features and ~124 authors, the discriminative objective spends its capacity on the
    directions that actually discriminate, which nearly doubles top-1 over cosine.

    ``class_weight="balanced"`` matters here -- known authors range from 1 to 325 documents, and
    without it the handful of prolific authors dominate the objective.
    """

    name = "logistic"

    def __init__(self, C: float = 1.0, class_weight: str | None = "balanced", max_iter: int = 3000):
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter

    def fit(self, embeddings, labels):
        self.authors, codes = np.unique(labels, return_inverse=True)
        self.model = LogisticRegression(
            C=self.C, max_iter=self.max_iter, class_weight=self.class_weight
        ).fit(embeddings, codes)
        return self

    def score(self, embeddings):
        return self.model.decision_function(embeddings)


class SupportVectorAttribution:
    """Support vector machine over the known authors; the score is the decision function.

    The other classical discriminative answer alongside :class:`LogisticAttribution`, and worth
    having because it optimises a different thing: logistic regression fits the whole conditional
    distribution, while an SVM only cares about the documents near each boundary. With ~25
    documents per author that focus tends to pay, and the RBF kernel additionally buys
    non-linearity, which no other attack here has.

    Uses ``sklearn.svm.SVC`` rather than ``LinearSVC`` even for ``kernel="linear"``, and the
    reason is purely practical: ``SVC`` is one-vs-one, so it trains ~7,600 tiny pairwise problems,
    whereas ``LinearSVC`` is one-vs-rest and trains 124 problems each against the entire corpus.
    Measured on a 2,992-document known side, that is 1.8 seconds against 299.

    ``decision_function_shape="ovr"`` folds the pairwise votes back into one column per author, so
    the output has the same shape as every other attack's. Those margins are *not* posteriors --
    :func:`prompt_anonymity.metrics.max_softmax_confidence` will report a near-uniform confidence
    for them, so read this attack's calibration numbers as meaningless rather than as bad.
    """

    name = "svm"

    def __init__(self, C: float = 1.0, kernel: str = "rbf", gamma: str | float = "scale"):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma

    def fit(self, embeddings, labels):
        self.authors, codes = np.unique(labels, return_inverse=True)
        self.model = SVC(C=self.C, kernel=self.kernel, gamma=self.gamma,
                         decision_function_shape="ovr").fit(embeddings, codes)
        return self

    def score(self, embeddings):
        return self.model.decision_function(embeddings)


class GradientBoostedTrees:
    """Gradient-boosted decision trees (XGBoost) over the known authors.

    The only non-linear, non-metric attack here: every other one ultimately compares documents
    along straight lines in feature space. StyloMetrix features are heterogeneous -- ratios in
    [0, 1] next to raw counts, many near-zero for most documents -- and trees handle that mix
    natively, splitting on thresholds instead of weighting directions, and picking up interactions
    between features that a linear model cannot express.

    The score is the **log** class probability, not the raw probability. It is the same ranking
    either way, but log-space is what the downstream machinery expects: cohort normalisation
    z-scores across authors (meaningful for log-odds-like quantities, not for probabilities that
    sum to 1), and it makes ``softmax(score)`` recover the model's own posterior exactly, so the
    calibration metrics in :mod:`prompt_anonymity.metrics.detection` measure something real for
    this attack.

    ``xgboost`` is imported lazily so the rest of the package works without it installed.
    """

    name = "xgboost"

    def __init__(self, n_estimators: int = 300, max_depth: int = 3, learning_rate: float = 0.3,
                 subsample: float = 1.0, n_jobs: int = -1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.n_jobs = n_jobs

    def fit(self, embeddings, labels):
        from xgboost import XGBClassifier  # lazy: keeps xgboost an optional runtime dependency

        self.authors, codes = np.unique(labels, return_inverse=True)
        self.model = XGBClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, subsample=self.subsample,
            tree_method="hist", objective="multi:softprob", n_jobs=self.n_jobs, verbosity=0,
        ).fit(embeddings, codes)
        return self

    def score(self, embeddings):
        return np.log(np.clip(self.model.predict_proba(embeddings), 1e-12, None))


class PLDA:
    """Two-covariance (Gaussian) PLDA: the score is a same-author log-likelihood ratio.

    Included because it is the principled answer to the failure that motivated all this: a
    distance answers "how close?", while a likelihood ratio answers "how much more likely is
    this document under *this author* than under the population?" -- which is comparable across
    documents and accounts for how many documents each author was enrolled with (1 to 325 here).

    Both covariances are estimated on the known side and simultaneously diagonalised, so within-
    author variance becomes 1 and between-author variance becomes ``psi`` per dimension and the
    ratio has a closed diagonal form. It underperforms the logistic model on this data -- the
    Gaussian assumption is a poor fit for StyloMetrix ratios on 300-character documents -- but it
    is the right starting point if the features ever improve.
    """

    name = "plda"

    def __init__(self, shrinkage: float = 0.2):
        self.shrinkage = shrinkage

    def fit(self, embeddings, labels):
        self.authors, codes = np.unique(labels, return_inverse=True)
        n_authors = len(self.authors)
        self.mean = embeddings.mean(axis=0)
        centred = embeddings - self.mean

        whiten = inverse_sqrt(within_class_covariance(centred, codes, n_authors, self.shrinkage))
        means = class_means(centred @ whiten, codes, n_authors)
        counts = np.bincount(codes, minlength=n_authors)

        # Between-author covariance, corrected for the within-author noise still carried by a
        # class mean estimated from only n_a documents.
        spread = means - means.mean(axis=0)
        between = (spread.T @ spread) / max(n_authors - 1, 1) \
            - np.eye(embeddings.shape[1]) * np.mean(1 / counts)
        values, vectors = np.linalg.eigh(between)
        self.psi = np.maximum(values, 1e-6)
        self.transform = whiten @ vectors
        self.enrolled_means = class_means(centred @ self.transform, codes, n_authors)
        self.enrolled_counts = counts
        return self

    def score(self, embeddings):
        projected = (embeddings - self.mean) @ self.transform
        psi, counts = self.psi, self.enrolled_counts[:, None]

        posterior_variance = psi / (1.0 + counts * psi)
        posterior_mean = counts * psi / (1.0 + counts * psi) * self.enrolled_means
        same_variance = 1.0 + posterior_variance
        different_variance = 1.0 + psi

        # The quadratic form is expanded into matrix products so no (n_docs, n_authors, d)
        # array is ever materialised.
        inverse = 1.0 / same_variance
        quadratic = (projected ** 2) @ inverse.T \
            - 2.0 * projected @ (posterior_mean * inverse).T \
            + np.sum(posterior_mean ** 2 * inverse, axis=1)[None, :]
        same = -0.5 * (np.sum(np.log(same_variance), axis=1)[None, :] + quadratic)
        different = -0.5 * (np.sum(np.log(different_variance))
                            + (projected ** 2) @ (1.0 / different_variance))
        return same - different[:, None]


# Registry so callers can select a supervised attribution attack by name (e.g. from a CLI
# argument). Values are classes, not instances: each is constructed per fit, because threshold
# calibration refits the same configuration on many simulation folds.
ATTRIBUTION_ATTACKS = {
    "nearest_neighbor": NearestNeighbor,
    "cosine": CentroidCosine,
    "wccn": WhitenedCentroid,
    "lda": LDACentroid,
    "logistic": LogisticAttribution,
    "svm": SupportVectorAttribution,
    "xgboost": GradientBoostedTrees,
    "plda": PLDA,
}


def get_attribution_attack(name: str):
    """Look up a registered supervised attribution attack by name."""
    try:
        return ATTRIBUTION_ATTACKS[name]
    except KeyError:
        raise ValueError(
            f"unknown attribution attack {name!r}; available: {sorted(ATTRIBUTION_ATTACKS)}"
        ) from None
