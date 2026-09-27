"""Numerical helpers shared across the attack families.

Two groups. The **linear algebra** is what separates the similarity attacks in
:mod:`prompt_anonymity.attacks.similarity` from plain cosine: :func:`class_means` and
:func:`within_class_covariance` describe where each author sits and how widely their own
documents scatter, and :func:`inverse_sqrt` turns the latter into the whitening transform that
:class:`~prompt_anonymity.attacks.similarity.WhitenedCentroid` and
:class:`~prompt_anonymity.attacks.similarity.PLDA` both build on.

The **author grouping** helper is a different kind of sharing, and it crosses families:
:class:`~prompt_anonymity.attacks.similarity.NearestNeighbor` and
:class:`~prompt_anonymity.attacks.multiclass.RegularizedLeastSquares` both need each author's
documents laid out contiguously -- one so a single ``ufunc.reduceat`` pass can aggregate every
author instead of a Python loop that rebuilds an ``n_documents`` boolean mask once per author,
the other so each author's rank-``n_a`` covariance update is a slice.

This module imports nothing from the attack packages, which is what lets every one of them
import it.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


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


class AuthorGroups(NamedTuple):
    """Row ordering that lays each author's documents out contiguously.

    Attributes
    ----------
    order : numpy.ndarray of shape (n_documents,)
        Permutation sorting the documents by author code. Apply it to the embeddings **and** to
        anything aligned with them before using ``starts``.
    starts : numpy.ndarray of shape (n_authors,)
        First row of each author's block in the reordered array. Strictly increasing, because
        every author owns at least one document -- exactly the precondition ``reduceat`` needs.
    stops : numpy.ndarray of shape (n_authors,)
        One past the last row of each author's block.
    counts : numpy.ndarray of shape (n_authors,)
        Documents per author, in author order.
    """

    order: np.ndarray
    starts: np.ndarray
    stops: np.ndarray
    counts: np.ndarray


def group_by_author(codes: np.ndarray, n_authors: int) -> AuthorGroups:
    """Sort integer author codes into contiguous blocks and return the block boundaries.

    ``codes`` is the ``return_inverse`` output of ``numpy.unique`` over the labels, so every
    author in ``0..n_authors-1`` owns at least one document by construction.
    """
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    starts = np.searchsorted(sorted_codes, np.arange(n_authors))
    counts = np.bincount(sorted_codes, minlength=n_authors)
    return AuthorGroups(order=order, starts=starts,
                        stops=np.append(starts[1:], len(codes)), counts=counts)
