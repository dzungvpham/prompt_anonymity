"""SWE-chat loader.

Identity comes from ``user_id``, with repo-based recovery for id-less sessions; the
known/unknown split is by time (each user's chronologically last session is the
unknown, earlier ones are known). Mirrors the data prep in
``swe-chat/analyze_swe_chat.py`` (identity recovery + dedup), then delegates the split
to :func:`~prompt_anonymity.data.splits.split_by_last_conversation`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..core import AttackData
from .splits import build_attack_data, split_by_last_conversation

# Filenames inside the SWE-chat data directory.
SESSIONS_CSV = "swe_chat_sessions.csv"
STYLOMETRIX_CSV = "swe_chat_stylometrix_en_2048.csv"
MIN_AFFIX_LEN = 50  # prefix/suffix length for near-duplicate dedup (matches WildChat filter.py)


def _recover_identity(sessions: pd.DataFrame) -> pd.Series:
    """Assign each session an identity, recovering id-less sessions from their repo.

    user_id when present; an id-less session on a single-user repo merges into that
    repo's sole user; an id-less session on an orphan repo (no labeled user anywhere)
    becomes its own identity (the repo_id); an id-less session on a repo shared by more
    than one user has no valid identifier and is left NA (dropped by the caller).
    """
    has_user_id = sessions["user_id"].notna() & (sessions["user_id"].astype(str).str.lower() != "nan")
    labeled = sessions[has_user_id]
    users_per_repo = labeled.groupby("repo_id")["user_id"].nunique()
    sole_user_per_repo = labeled.groupby("repo_id")["user_id"].first()
    repo_user_count = sessions["repo_id"].map(users_per_repo).fillna(0).astype(int)

    is_idless = ~has_user_id
    identity = sessions["user_id"].astype("object").where(has_user_id)
    identity = identity.mask(is_idless & (repo_user_count == 1), sessions["repo_id"].map(sole_user_per_repo))
    identity = identity.mask(is_idless & (repo_user_count == 0), sessions["repo_id"])
    return identity  # (>1-user-repo id-less sessions stay NA)


def _dedup_per_identity(sessions: pd.DataFrame) -> pd.DataFrame:
    """Drop exact-duplicate and near-duplicate (shared prefix/suffix) sessions per identity,
    keeping the first (chronological) of each duplicate group."""
    data = sessions.sort_values(["identity", "timestamp", "session_id"]).reset_index(drop=True)
    data = data.drop_duplicates(["identity", "content"], keep="first")
    affix = data.assign(
        _prefix=data["content"].str[:MIN_AFFIX_LEN],
        _suffix=data["content"].str[-MIN_AFFIX_LEN:],
        _length=data["content"].str.len(),
    )
    long_enough = affix["_length"] >= MIN_AFFIX_LEN
    is_affix_dup = (long_enough & affix.duplicated(["identity", "_prefix"], keep="first")) | (
        long_enough & affix.duplicated(["identity", "_suffix"], keep="first")
    )
    return data[~is_affix_dup].reset_index(drop=True)


def load_swe_chat(data_dir, *, feature: str = "stylometrix", model_owner: str = "Anthropic") -> AttackData:
    """Load a SWE-chat known/unknown split as an :class:`AttackData`.

    Parameters
    ----------
    data_dir : str or pathlib.Path
        Directory holding ``swe_chat_sessions.csv`` and the StyloMetrix feature CSV
        (the repo's ``swe-chat/`` directory).
    feature : {"stylometrix"}, default "stylometrix"
        Conversation representation (only StyloMetrix is available for SWE-chat).
    model_owner : str, default "Anthropic"
        Restrict the pool to one agent provider (the ``model_owner`` column), or
        ``"all"`` to keep every provider.
    """
    data_dir = Path(data_dir)

    sessions = pd.read_csv(data_dir / SESSIONS_CSV)
    if model_owner and model_owner.lower() != "all":
        sessions = sessions[sessions["model_owner"] == model_owner]
    sessions["timestamp"] = pd.to_datetime(sessions["timestamp"], format="ISO8601", utc=True)
    sessions["content"] = sessions["content"].fillna("").astype(str)

    feature_df = pd.read_csv(data_dir / STYLOMETRIX_CSV).set_index("session_id")
    feature_df = feature_df[feature_df.index.isin(sessions["session_id"])]
    feature_cols = feature_df.columns.tolist()

    sessions["identity"] = _recover_identity(sessions)
    sessions = sessions[sessions["identity"].notna()].copy()
    sessions["identity"] = sessions["identity"].astype(str)
    data = _dedup_per_identity(sessions)

    role = split_by_last_conversation(
        data, identity_col="identity", time_col="timestamp", tiebreak_col="session_id"
    )
    # Features aligned to `data` row order via the session_id index; NaN -> 0.
    embeddings = np.nan_to_num(
        feature_df.loc[data["session_id"], feature_cols].to_numpy(dtype=np.float32), nan=0.0
    )
    # metric is an attack-level choice (default "cosine" on AttackData); the loader leaves it.
    return build_attack_data(
        identities=data["identity"].to_numpy(),
        role=role,
        embeddings=embeddings,
        texts=data["content"].to_numpy(),
    )
