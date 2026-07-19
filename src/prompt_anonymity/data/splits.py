"""Split rules: how each user's conversations are divided into known vs. unknown.

This is the real per-dataset difference. A *split rule* assigns every conversation a
role -- ``KNOWN`` (labeled), ``UNKNOWN`` (anonymous, to be re-identified), or NA
(dropped) -- as a pandas Series aligned to the input frame. WildChat splits by model
(:func:`split_by_model`); SWE-chat splits by time (:func:`split_by_last_conversation`).

:func:`attackable_mask` then drops identities that lack a counterpart on either side,
and :func:`build_attack_data` assembles the aligned arrays into an
:class:`~prompt_anonymity.core.AttackData`. New datasets implement (or reuse) a split
rule and call these two helpers, so loaders stay thin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import AttackData

# Role values a split rule assigns to each conversation.
KNOWN = "known"
UNKNOWN = "unknown"


def split_by_model(frame, *, known_model, unknown_model, model_col="model") -> pd.Series:
    """Split by which model produced the conversation (WildChat).

    Conversations from ``known_model`` form the labeled set and those from
    ``unknown_model`` are the anonymous set to re-identify; any other model is left
    unassigned (NA) and dropped downstream.
    """
    role = pd.Series(pd.NA, index=frame.index, dtype="object")
    role[frame[model_col] == known_model] = KNOWN
    role[frame[model_col] == unknown_model] = UNKNOWN
    return role


def split_by_last_conversation(
    frame, *, identity_col="identity", time_col="timestamp", tiebreak_col=None
) -> pd.Series:
    """Split by time: each identity's last conversation is unknown, earlier ones known (SWE-chat).

    Within each identity the conversations are ordered by ``time_col`` (then
    ``tiebreak_col`` if given, to break equal timestamps deterministically); the single
    latest is the unknown and all earlier ones are known. An identity with one
    conversation gets only an unknown and is dropped by :func:`attackable_mask`.
    """
    sort_cols = [identity_col, time_col] + ([tiebreak_col] if tiebreak_col else [])
    latest_index = frame.sort_values(sort_cols).groupby(identity_col, sort=False).tail(1).index
    role = pd.Series(KNOWN, index=frame.index, dtype="object")
    role[latest_index] = UNKNOWN
    return role


def attackable_mask(identities, role) -> np.ndarray:
    """Boolean mask keeping conversations whose identity has BOTH a known and an unknown.

    Re-identification is only defined for users that appear on both sides, so this drops
    identities present in only one role (e.g. a SWE-chat user with a single session, or
    a WildChat identity that used only one of the two models). Operates on positionally
    aligned arrays and returns a mask over them.
    """
    identities = np.asarray(identities)
    role = np.asarray(role)
    known_identities = set(identities[role == KNOWN])
    unknown_identities = set(identities[role == UNKNOWN])
    attackable = known_identities & unknown_identities
    return np.isin(identities, list(attackable))


def build_attack_data(*, identities, role, embeddings, metric="cosine", texts=None) -> AttackData:
    """Assemble positionally aligned per-conversation arrays into an :class:`AttackData`.

    ``identities``, ``role``, ``embeddings`` (and optional ``texts``) must all be in the
    same row order. Identities without a counterpart on both sides are dropped via
    :func:`attackable_mask`; the rest are partitioned into the known and unknown sides.

    ``metric`` is the distance the attack will use to compare vectors; it is an attack-level
    choice (default "cosine"), not a property of the features, so loaders need not set it.
    """
    identities = np.asarray(identities)
    role = np.asarray(role)
    embeddings = np.asarray(embeddings)
    texts = None if texts is None else np.asarray(texts)

    attackable = attackable_mask(identities, role)
    known_selector = attackable & (role == KNOWN)
    unknown_selector = attackable & (role == UNKNOWN)
    return AttackData(
        known_embeddings=embeddings[known_selector],
        unknown_embeddings=embeddings[unknown_selector],
        known_labels=identities[known_selector],
        unknown_labels=identities[unknown_selector],
        metric=metric,
        known_texts=None if texts is None else texts[known_selector],
        unknown_texts=None if texts is None else texts[unknown_selector],
    )
