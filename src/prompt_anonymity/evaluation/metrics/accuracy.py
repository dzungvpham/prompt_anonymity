"""Top-k accuracy and the random-guessing baseline for linkage attacks.

These stateless metrics consume the distance matrix produced by an attack (e.g.
:class:`prompt_anonymity.attacks.NearestNeighbor`) together with the identity
labels of the known and unknown conversations, and report how often the attack
re-identifies the correct user.

For repeated evaluation on many sub-pools of candidate users, prefer
:class:`prompt_anonymity.evaluation.LinkageRanking`, which ranks once and reuses the ranking
instead of re-sorting on every call.

Glossary
--------
identity (label)
    The user a conversation belongs to -- in WildChat the
    ``hashed_ip|accept_language|device_info`` tuple, in SWE-chat the user_id/repo. A
    user usually owns several known and several unknown conversations.
known conversation (matrix column)
    A labeled conversation the adversary may match an unknown conversation against.
unknown conversation (matrix row)
    An anonymous conversation the adversary tries to re-identify.
"""

from __future__ import annotations

import numpy as np

from ..._ranking import first_k_distinct, rank_known_by_distance


def top_k_accuracy(
    distances,
    known_labels,
    unknown_labels,
    k: int = 1,
    *,
    level: str = "identity",
) -> float:
    """Fraction of targets the attack ranks within its top-k guesses.

    Parameters
    ----------
    distances : pandas.DataFrame or array-like of shape (n_unknown, n_known)
        Distance matrix from an attack; entry ``[u, k]`` is the distance between
        unknown conversation ``u`` and known conversation ``k`` (smaller = more
        similar), e.g. the output of :class:`~prompt_anonymity.attacks.NearestNeighbor`.
    known_labels : array-like of shape (n_known,)
        Identity owning each known conversation, aligned to the *columns* of
        ``distances`` by position.
    unknown_labels : array-like of shape (n_unknown,)
        True identity of each unknown conversation, aligned to the *rows* of
        ``distances`` by position.
    k : int, default 1
        Number of top-ranked guesses that count as a success (the "top-k").
    level : {"identity", "conversation"}, default "identity"
        ``"identity"`` (re-identification rate) -- fraction of distinct target
        identities that are re-identified. The per-conversation ranking is first
        collapsed to distinct candidate identities (each identity keeps its
        best-ranked known conversation); an identity counts as re-identified if at
        least one of its unknown conversations ranks the true identity within the
        top-k distinct identities.
        ``"conversation"`` -- fraction of unknown conversations whose top-k nearest
        *known conversations* include at least one owned by the true identity. At
        ``k=1`` this equals the identity level when every identity has a single
        unknown conversation.

    Returns
    -------
    float
        Accuracy in ``[0, 1]``. Compare against :func:`random_guessing_accuracy` (same
        ``k`` and target labels) to measure the adversary's advantage over chance.
    """
    known_labels = np.asarray(known_labels)
    unknown_labels = np.asarray(unknown_labels)
    distance_matrix = np.asarray(distances)

    n_unknown, n_known = distance_matrix.shape
    if len(unknown_labels) != n_unknown:
        raise ValueError(
            f"unknown_labels has {len(unknown_labels)} entries but distances has "
            f"{n_unknown} rows (one per unknown conversation)."
        )
    if len(known_labels) != n_known:
        raise ValueError(
            f"known_labels has {len(known_labels)} entries but distances has "
            f"{n_known} columns (one per known conversation)."
        )
    if k < 1:
        raise ValueError(f"k must be a positive integer (got {k}).")

    # Known identities ranked nearest-first for every unknown conversation.
    ranked_labels = known_labels[rank_known_by_distance(distance_matrix)]  # (n_unknown, n_known)

    if level == "conversation":
        # An unknown conversation is a hit if any of its k nearest known conversations
        # is owned by its true identity.
        conversation_hits = [
            bool(np.any(row_labels[:k] == true_identity))
            for row_labels, true_identity in zip(ranked_labels, unknown_labels)
        ]
        return float(np.mean(conversation_hits))

    if level == "identity":
        # Collapse each ranking to distinct identities and check whether the true
        # identity falls in the top-k. A user is re-identified if ANY of its unknown
        # conversations succeeds, so we accumulate the set of re-identified identities
        # and divide by the number of distinct target identities.
        reidentified_identities: set = set()
        for row_labels, true_identity in zip(ranked_labels, unknown_labels):
            if true_identity in first_k_distinct(row_labels, k):
                reidentified_identities.add(true_identity)
        n_target_identities = len(np.unique(unknown_labels))
        return len(reidentified_identities) / n_target_identities

    raise ValueError(f"level must be 'identity' or 'conversation' (got {level!r}).")


def random_guessing_accuracy(
    unknown_labels,
    k: int = 1,
    *,
    n_candidates: int | None = None,
) -> float:
    """Expected top-k identity accuracy of an adversary that guesses uniformly at random.

    This is the chance baseline for the ``"identity"`` level of :func:`top_k_accuracy`.
    The adversary short-lists ``k`` distinct candidate identities uniformly at random
    per guess, so one guess names any given identity with probability
    ``min(k, n) / n``, where ``n`` is the candidate-pool size. A user with ``c`` unknown
    conversations gets ``c`` independent guesses and is re-identified if any of them
    succeeds, with probability ``1 - (1 - min(k, n) / n) ** c``. Averaging that over all
    target identities gives the expected accuracy.

    Parameters
    ----------
    unknown_labels : array-like of shape (n_unknown,)
        True identity of each unknown conversation -- the same labels passed to
        :func:`top_k_accuracy`. The per-identity unknown-conversation counts (the ``c``
        above) are derived from these.
    k : int, default 1
        Top-k cutoff, matching the ``k`` used for the measured accuracy.
    n_candidates : int, optional
        Size of the candidate identity pool the adversary guesses among. Defaults to the
        number of distinct identities in ``unknown_labels`` (the WildChat / SWE-chat
        setup, where every unknown user also appears in the known set). Override it when
        the pool differs, e.g. for a sub-sampled candidate set.

    Returns
    -------
    float
        Expected accuracy in ``[0, 1]`` under uniform-random guessing.
    """
    unknown_labels = np.asarray(unknown_labels)
    if k < 1:
        raise ValueError(f"k must be a positive integer (got {k}).")

    # c = number of unknown conversations per distinct target identity.
    _, conversation_counts = np.unique(unknown_labels, return_counts=True)
    n_target_identities = len(conversation_counts)
    pool_size = n_target_identities if n_candidates is None else n_candidates
    if pool_size < 1:
        raise ValueError(f"n_candidates must be a positive integer (got {pool_size}).")

    # Probability that a single uniform top-k guess names a given identity.
    hit_probability_per_guess = min(k, pool_size) / pool_size
    # An identity with c unknown conversations is re-identified if any of its c guesses hits.
    per_identity_accuracy = 1.0 - (1.0 - hit_probability_per_guess) ** conversation_counts
    return float(np.sum(per_identity_accuracy) / n_target_identities)
