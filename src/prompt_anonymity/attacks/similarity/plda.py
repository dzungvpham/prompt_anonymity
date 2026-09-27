"""Two-covariance (Gaussian) PLDA: a same-author log-likelihood ratio."""

from __future__ import annotations

import numpy as np

from ..common import class_means, inverse_sqrt, within_class_covariance


class PLDA:
    """Two-covariance (Gaussian) PLDA: the score is a same-author log-likelihood ratio.

    A distance answers "how close?", while a likelihood ratio answers "how much more likely is
    this document under *this author* than under the population?" -- comparable across documents,
    and it accounts for how many documents each author was enrolled with.

    Both covariances are estimated on the known side and simultaneously diagonalised, so
    within-author variance becomes 1 and between-author variance becomes ``psi`` per dimension,
    giving the ratio a closed diagonal form. The Gaussian assumption underperforms a discriminative
    classifier on features that aren't well-modeled by it, but it's the principled starting point.
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

