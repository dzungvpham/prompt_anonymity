"""WildChat loader.

Identity = ``hashed_ip|accept_language|device_info``; the known/unknown split is by
model (one model's conversations are labeled, another's are anonymous). The filtered
CSV holds only the two models and only identities that used both, so every unknown user
has a known counterpart. The StyloMetrix CSV is row-aligned to ``df[language][>=2
models]`` (see ``wildchat/stylometrix.py``), so we reproduce that exact filter order and
index the features positionally.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..core import AttackData
from .splits import build_attack_data, split_by_model

# Filenames inside the WildChat data directory.
FILTERED_CSV = "wildchat_filtered_4o20240806_41mini20250414_device_deduped.csv"
STYLOMETRIX_CSV = "wildchat_embeddings/wildchat_filtered_{language_code}_2048_stylometrix.csv"

KNOWN_MODEL = "gpt-4o-2024-08-06"          # labeled set
UNKNOWN_MODEL = "gpt-4.1-mini-2025-04-14"  # to re-identify
LANGUAGE_CODES = {"English": "en", "Russian": "ru"}
IDENTITY_COLUMNS = ["hashed_ip", "accept_language", "device_info"]


def load_wildchat(data_dir, *, feature: str = "stylometrix", language: str = "English") -> AttackData:
    """Load a WildChat known/unknown split as an :class:`AttackData`.

    Parameters
    ----------
    data_dir : str or pathlib.Path
        Directory holding the filtered CSV and the ``wildchat_embeddings/`` features
        (the repo's ``wildchat/`` directory).
    feature : {"stylometrix"}, default "stylometrix"
        Conversation representation. Only StyloMetrix is wired up; Gemini embeddings are
        a planned addition.
    language : {"English", "Russian"}, default "English"
        Language subset to attack (selects the matching StyloMetrix feature CSV).
    """
    if language not in LANGUAGE_CODES:
        raise ValueError(f"language must be one of {sorted(LANGUAGE_CODES)} (got {language!r}).")
    data_dir = Path(data_dir)
    language_code = LANGUAGE_CODES[language]

    frame = pd.read_csv(data_dir / FILTERED_CSV)
    frame = frame[frame["language"] == language]
    # Keep identities that used >=2 distinct models. The CSV holds only the known and
    # unknown models, so this guarantees each kept user appears in both sets; it also
    # reproduces the row set/order the StyloMetrix CSV was generated from.
    frame = frame[
        frame.groupby(IDENTITY_COLUMNS)["model"].transform("nunique").ge(2)
    ].reset_index(drop=True)
    identities = (frame["hashed_ip"] + "|" + frame["accept_language"] + "|" + frame["device_info"]).to_numpy()

    stylometrix_csv = data_dir / STYLOMETRIX_CSV.format(language_code=language_code)
    features = pd.read_csv(stylometrix_csv).drop(columns="text")
    if len(features) != len(frame):
        raise RuntimeError(
            f"StyloMetrix rows ({len(features)}) != filtered conversations ({len(frame)}) for "
            f"{language}; the feature CSV is out of sync with the filtered dataset."
        )
    # StyloMetrix can emit NaN (e.g. ratios with a zero denominator); treat those as 0.
    embeddings = np.nan_to_num(features.to_numpy(dtype=np.float32), nan=0.0)

    role = split_by_model(frame, known_model=KNOWN_MODEL, unknown_model=UNKNOWN_MODEL)
    # metric is an attack-level choice (default "cosine" on AttackData); the loader leaves it.
    return build_attack_data(
        identities=identities,
        role=role,
        embeddings=embeddings,
        texts=frame["conversation"].to_numpy(),
        ids=frame["idx"].to_numpy(),  # original-dataset row id, used as the defense cache key
    )
