"""Core data structure shared across the package.

:class:`AttackData` is the hand-off contract between the three stages of a linkage
experiment: dataset loaders (:mod:`prompt_anonymity.data`) produce it, defenses
(:mod:`prompt_anonymity.defenses`) map it to a transformed copy, and attacks
(:mod:`prompt_anonymity.attacks`) turn it into an unknown-vs-known distance matrix.
Kept in its own tiny module so every subpackage can depend on the contract without
depending on each other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AttackData:
    """A known/unknown split ready for a linkage attack.

    A linkage attack compares anonymous "unknown" conversations against labeled "known"
    conversations. This bundle holds one feature vector per conversation on each side,
    the identity (true author) of each conversation, and the distance metric the
    features call for. Rows are aligned by position: ``known_embeddings[i]`` is authored
    by ``known_labels[i]`` (with text ``known_texts[i]`` if present); likewise for the
    unknown side.

    Attributes
    ----------
    known_embeddings, unknown_embeddings : np.ndarray, shape (n, n_features)
        Feature vectors (semantic embeddings or stylometric features) for the labeled
        and anonymous conversations respectively. Both sides must share ``n_features``.
    known_labels, unknown_labels : np.ndarray, shape (n,)
        Identity (true author) of each known / unknown conversation.
    metric : str
        Distance metric the attack uses to compare these vectors (smaller = more
        similar). A property of the *attack*, not of the featurizer: chosen at the
        experiment level (default "cosine"), any ``scipy.spatial.distance.cdist``
        metric name.
    known_texts, unknown_texts : np.ndarray of str or None
        Optional raw prompt text per conversation, carried for inspection and for
        future text-level defenses; ``None`` when the loader does not provide it.
    known_ids, unknown_ids : np.ndarray or None
        Optional stable row identifier from the *original* dataset (e.g. ``session_id``
        for SWE-chat, ``idx`` for WildChat). Lets downstream artifacts like the defense
        cache key on the dataset's own row identity instead of position, so a cached
        rewrite survives re-ordering or re-subsetting the pool. ``None`` falls back to
        position as the identifier.
    """

    known_embeddings: np.ndarray
    unknown_embeddings: np.ndarray
    known_labels: np.ndarray
    unknown_labels: np.ndarray
    metric: str = "cosine"
    known_texts: np.ndarray | None = None
    unknown_texts: np.ndarray | None = None
    known_ids: np.ndarray | None = None
    unknown_ids: np.ndarray | None = None

    def __post_init__(self) -> None:
        # Validate up front so every downstream stage can trust the shapes.
        self.known_embeddings = np.asarray(self.known_embeddings)
        self.unknown_embeddings = np.asarray(self.unknown_embeddings)
        self.known_labels = np.asarray(self.known_labels)
        self.unknown_labels = np.asarray(self.unknown_labels)
        if self.known_embeddings.ndim != 2 or self.unknown_embeddings.ndim != 2:
            raise ValueError(
                "known/unknown embeddings must be 2-D (got shapes "
                f"{self.known_embeddings.shape} and {self.unknown_embeddings.shape})."
            )
        if self.known_embeddings.shape[1] != self.unknown_embeddings.shape[1]:
            raise ValueError(
                "known and unknown embeddings must share the feature dimension "
                f"(got {self.known_embeddings.shape[1]} and {self.unknown_embeddings.shape[1]})."
            )
        if len(self.known_labels) != len(self.known_embeddings):
            raise ValueError(
                f"known_labels ({len(self.known_labels)}) must match known_embeddings "
                f"rows ({len(self.known_embeddings)})."
            )
        if len(self.unknown_labels) != len(self.unknown_embeddings):
            raise ValueError(
                f"unknown_labels ({len(self.unknown_labels)}) must match unknown_embeddings "
                f"rows ({len(self.unknown_embeddings)})."
            )
        for name, values, n in [
            ("known_texts", self.known_texts, len(self.known_labels)),
            ("unknown_texts", self.unknown_texts, len(self.unknown_labels)),
            ("known_ids", self.known_ids, len(self.known_labels)),
            ("unknown_ids", self.unknown_ids, len(self.unknown_labels)),
        ]:
            if values is not None:
                values = np.asarray(values)
                if len(values) != n:
                    raise ValueError(f"{name} has {len(values)} entries but expected {n}.")
                setattr(self, name, values)

    @property
    def n_known(self) -> int:
        """Number of labeled (known) conversations."""
        return len(self.known_labels)

    @property
    def n_unknown(self) -> int:
        """Number of anonymous (unknown) conversations to re-identify."""
        return len(self.unknown_labels)

    @property
    def n_identities(self) -> int:
        """Number of distinct target identities (the candidate-pool size)."""
        return len(np.unique(self.unknown_labels))
