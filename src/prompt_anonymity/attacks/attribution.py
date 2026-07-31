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
cohort-normalised. Selected on a known-side simulation, which preferred the winning row on all
three criteria, then measured once on the unknown window -- see ``benchmark_attribution.py``.)

Two things that did *not* help, both worth not re-trying blind: a learned rejector over the
score-vector shape (margin, entropy, peakedness) overfits the simulation and loses to the plain
cohort-normalised score (DIR@10% 0.055 vs 0.106), and clustering the unknown side to pool
evidence fails because pairwise same-author AUROC is only 0.64 even in the whitened space
(adjusted Rand index 0.16 at best).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression


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
    nearest-neighbour attack proper; ``"mean"`` averages over all of an author's documents,
    which is a distance-space cousin of :class:`CentroidCosine`.
    """

    name = "nearest_neighbor"

    def __init__(self, metric: str = "cosine", linkage: str = "max"):
        self.metric = metric
        self.linkage = linkage

    def fit(self, embeddings, labels):
        self.authors, self._codes = np.unique(labels, return_inverse=True)
        self._known = np.asarray(embeddings, dtype=float)
        return self

    def score(self, embeddings):
        distances = cdist(np.asarray(embeddings, dtype=float), self._known, metric=self.metric)
        aggregated = np.empty((len(distances), len(self.authors)))
        for index in range(len(self.authors)):
            columns = distances[:, self._codes == index]
            aggregated[:, index] = columns.mean(axis=1) if self.linkage == "mean" \
                else columns.min(axis=1)
        return -aggregated  # negate: higher must mean "more likely this author"


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
    """

    name = "lda"

    def __init__(self, n_components: int | None = None):
        self.n_components = n_components

    def fit(self, embeddings, labels):
        self.authors, codes = np.unique(labels, return_inverse=True)
        n_components = self.n_components or min(len(self.authors) - 1, embeddings.shape[1])
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
