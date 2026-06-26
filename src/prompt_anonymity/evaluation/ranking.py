"""Rank once, evaluate many times.

:class:`LinkageRanking` sorts each unknown conversation's known candidates a single time
and caches the ranking, so top-k accuracy on any sub-pool of candidate users is a cheap
vectorized lookup instead of a fresh sort. This is the engine behind the pool-size
sweep, where the same ranking is scored against hundreds of random sub-pools.

For a one-off accuracy on the full pool, the stateless
:func:`prompt_anonymity.metrics.top_k_accuracy` is simpler; this class produces the same
numbers (verified) but pays the sort only once and scores sub-pools without re-sorting.

Implementation note
-------------------
Identities are integer-coded once, and for each unknown conversation we cache (a) its
known candidates ranked nearest-first as identity codes, and (b) the distinct identities
in ranked order together with the rank of the *true* identity within them. Restricting
to a sub-pool of candidate identities is then a single vectorized pass: an identity is
re-identified at top-k iff fewer than ``k`` *candidate* identities outrank its true
identity. This avoids the per-row Python filtering that made the naive version
unusably slow on the larger pools.
"""

from __future__ import annotations

import numpy as np

from .._ranking import rank_known_by_distance


class LinkageRanking:
    """Cached nearest-first ranking of known identities for each unknown conversation.

    Parameters
    ----------
    distances : pandas.DataFrame or array-like of shape (n_unknown, n_known)
        Distance matrix from an attack (smaller = more similar).
    known_labels : array-like of shape (n_known,)
        Identity owning each known conversation, aligned to the columns of ``distances``.
    unknown_labels : array-like of shape (n_unknown,)
        True identity of each unknown conversation, aligned to the rows of ``distances``.

    Notes
    -----
    Restricting to a sub-pool of ``candidate_identities`` means the adversary only holds
    those users' known conversations and only those users are targets: both the candidate
    columns and the evaluated rows are filtered to that set. This matches the semantics
    of building a fresh distance matrix for just those users.
    """

    def __init__(self, distances, known_labels, unknown_labels):
        self.known_labels = np.asarray(known_labels)
        self.unknown_labels = np.asarray(unknown_labels)
        distance_matrix = np.asarray(distances)
        if distance_matrix.shape != (len(self.unknown_labels), len(self.known_labels)):
            raise ValueError(
                f"distances shape {distance_matrix.shape} does not match "
                f"({len(self.unknown_labels)} unknown, {len(self.known_labels)} known) labels."
            )
        n_unknown = len(self.unknown_labels)

        # Integer-code identities once (shared across the known and unknown sides).
        identities, codes = np.unique(
            np.concatenate([self.known_labels, self.unknown_labels]), return_inverse=True
        )
        self._identities = identities
        self._n_ids = len(identities)
        self._code_of = {label: code for code, label in enumerate(identities)}
        known_codes = codes[: len(self.known_labels)]
        self._unknown_codes = codes[len(self.known_labels):]
        # Distinct target identity codes and each one's number of unknown conversations.
        self._unknown_identity_codes = np.unique(self._unknown_codes)
        self._unknown_counts = np.bincount(self._unknown_codes, minlength=self._n_ids)

        # Each unknown's known candidates ranked nearest-first, as identity codes.
        ranked = rank_known_by_distance(distance_matrix)  # (n_unknown, n_known)
        self._ranked_known_codes = known_codes[ranked]

        # Per unknown: the distinct candidate identities in ranked order (padded with a
        # sentinel code so rows align into a rectangular array), and the rank of the true
        # identity within that distinct ranking. The true identity is always present (an
        # attackable user has at least one known conversation).
        self._sentinel = self._n_ids
        self._ranked_distinct = np.full((n_unknown, self._n_ids), self._sentinel, dtype=np.int64)
        self._true_rank = np.empty(n_unknown, dtype=np.int64)
        for u in range(n_unknown):
            row = self._ranked_known_codes[u]
            _, first_occurrence = np.unique(row, return_index=True)
            distinct = row[np.sort(first_occurrence)]  # distinct codes in nearest-first order
            self._ranked_distinct[u, : len(distinct)] = distinct
            self._true_rank[u] = np.flatnonzero(distinct == self._unknown_codes[u])[0]

    def _candidate_membership(self, candidate_identities) -> np.ndarray | None:
        """Boolean membership over identity codes (length ``n_ids + 1``; the last slot is
        the always-False sentinel). ``None`` means the full pool."""
        if candidate_identities is None:
            return None
        membership = np.zeros(self._n_ids + 1, dtype=bool)
        for label in set(candidate_identities):
            code = self._code_of.get(label)
            if code is not None:
                membership[code] = True
        return membership

    def top_k_accuracy(self, k: int = 1, *, level: str = "identity", candidate_identities=None) -> float:
        """Top-k accuracy over the full pool or a sub-pool of candidate identities.

        See :func:`prompt_anonymity.metrics.top_k_accuracy` for the ``level`` semantics;
        results are identical to it when ``candidate_identities`` is ``None``, and match a
        freshly built distance matrix for the candidate users otherwise. Returns 0.0 if no
        candidate identity has an unknown conversation.
        """
        if k < 1:
            raise ValueError(f"k must be a positive integer (got {k}).")
        membership = self._candidate_membership(candidate_identities)
        if level == "identity":
            return self._identity_accuracy(k, membership)
        if level == "conversation":
            return self._conversation_accuracy(k, membership)
        raise ValueError(f"level must be 'identity' or 'conversation' (got {level!r}).")

    def _identity_accuracy(self, k: int, membership: np.ndarray | None) -> float:
        """Identity-level top-k accuracy (vectorized).

        An identity is re-identified if the true identity is among the first k *candidate*
        distinct identities of one of its unknown conversations -- i.e. fewer than k
        candidate identities outrank it. ``cumulative[u]`` counts candidate identities up
        to and including the true identity, so the hit condition is ``cumulative <= k``.
        """
        if membership is None:
            in_pool = np.ones(self._n_ids + 1, dtype=bool)
            in_pool[self._sentinel] = False
            evaluated = np.ones(len(self._unknown_codes), dtype=bool)
            n_candidates = len(self._unknown_identity_codes)
        else:
            in_pool = membership
            evaluated = in_pool[self._unknown_codes]  # unknowns whose true identity is a candidate
            n_candidates = int(in_pool[self._unknown_identity_codes].sum())
        if n_candidates == 0:
            return 0.0

        is_candidate = in_pool[self._ranked_distinct]  # (n_unknown, n_ids); sentinel -> False
        cumulative = np.cumsum(is_candidate, axis=1)[np.arange(len(self._unknown_codes)), self._true_rank]
        hit = evaluated & (cumulative <= k)
        n_reidentified = len(np.unique(self._unknown_codes[hit]))
        return n_reidentified / n_candidates

    def _conversation_accuracy(self, k: int, membership: np.ndarray | None) -> float:
        """Conversation-level top-k accuracy: fraction of (candidate) unknown conversations
        whose top-k nearest candidate known conversations include a true-identity match."""
        unknown_codes = self._unknown_codes
        if membership is None:
            top_k_codes = self._ranked_known_codes[:, :k]
            hit = np.any(top_k_codes == unknown_codes[:, None], axis=1)
            return float(np.mean(hit))
        # Sub-pool: keep only candidate known conversations, then take the top k of each
        # evaluated unknown. (Not used by the sweep, so a per-row pass is fine here.)
        rows = np.flatnonzero(membership[unknown_codes])
        if len(rows) == 0:
            return 0.0
        hits = 0
        for u in rows:
            candidate_known = self._ranked_known_codes[u][membership[self._ranked_known_codes[u]]]
            hits += bool(np.any(candidate_known[:k] == unknown_codes[u]))
        return hits / len(rows)

    def random_guessing_accuracy(self, k: int = 1, *, candidate_identities=None) -> float:
        """Uniform-random identity baseline for the same (sub-)pool.

        Matches :func:`prompt_anonymity.metrics.random_guessing_accuracy` over the unknown
        labels of the candidate pool, so the baseline shrinks with the pool the same way
        the measured accuracy does.
        """
        if k < 1:
            raise ValueError(f"k must be a positive integer (got {k}).")
        if candidate_identities is None:
            codes = self._unknown_identity_codes
        else:
            wanted = {self._code_of[label] for label in set(candidate_identities) if label in self._code_of}
            codes = self._unknown_identity_codes[np.isin(self._unknown_identity_codes, list(wanted))]
        n_candidates = len(codes)
        if n_candidates == 0:
            return 0.0
        conversation_counts = self._unknown_counts[codes]
        hit_probability_per_guess = min(k, n_candidates) / n_candidates
        per_identity = 1.0 - (1.0 - hit_probability_per_guess) ** conversation_counts
        return float(np.sum(per_identity) / n_candidates)
