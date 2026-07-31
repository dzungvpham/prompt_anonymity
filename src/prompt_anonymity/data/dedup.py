"""Unified near-duplicate / boilerplate removal, applied identically to both sources.

The two original pipelines deduped slightly differently: WildChat (``wildchat/filter.py``)
removed affixes shared across *different* identities and capped repeats within an identity,
keying on the *first turn*; SWE-chat (``data/swe_chat.py``) dropped per-identity exact and affix
duplicates keying on the *whole* text, with no cross-identity step and no cap. This module
reconciles them into one policy, run over the whole cleaned corpus:

1. **Within-identity exact duplicates** -- identical cleaned ``text`` from the same author
   collapses to its earliest occurrence (no information in a verbatim repeat).
2. **Cross-identity boilerplate** -- if a document's leading or trailing
   ``affix_len``-char slice is shared by *two or more distinct identities*, every document
   carrying that slice is dropped. Such shared openings/closings are templates (pasted
   jailbreaks, form letters, tool preambles) that would forge false authorship links.
3. **Within-identity affix cap** -- among what remains, keep the ``max_per_affix`` *earliest*
   documents per (identity, leading-slice) and per (identity, trailing-slice), dropping the
   rest. This bounds inflation from one author's repeated template while preserving a couple of
   instances of a habitual opening as genuine style signal.

Documents shorter than ``affix_len`` have no stable affix and are exempt from steps 2-3
(they still go through step 1). The corpus is expected to be pre-sorted so that "earliest"
is well defined; see :func:`deduplicate`.

The affix-based steps (2 and 3) can be turned off with ``affix_dedup=False``, leaving only
the exact-duplicate step. SWE-chat runs this way: it strips injected non-human templates at
the *turn* level (see ``sources_swe_chat.py``) rather than dropping a whole conversation just
because it shares a templated opening/closing, so only step 1 applies to it. WildChat keeps
all three steps (its cross-author templating is heavy and not template-stripped per turn).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Leading/trailing slice length. Both legacy pipelines used 50; 100 is deliberately more
# conservative, because the window has to reach past a template's boilerplate opening into the
# variable payload before two documents stop colliding. At 50 chars, two people who merely began
# with the same standard idiom ("#include <bits/stdc++.h>\nusing namespace std;\nbool") collided
# and *both* lost their document; doubling the window makes an accidental collision far less
# likely while still catching the pasted preambles the rule exists for.
MIN_AFFIX_LEN = 100
# Per-identity cap on documents sharing an affix: keep the earliest two, drop the rest. Two is
# enough to preserve a habitual opening as genuine style signal (one document alone could not
# establish it as a habit) without letting one author's repeated template dominate their profile.
MAX_PER_AFFIX = 2


def deduplicate(
    frame: pd.DataFrame,
    *,
    identity_col: str = "identity",
    text_col: str = "text",
    order_cols: tuple[str, ...] = ("identity", "timestamp", "doc_id"),
    affix_len: int = MIN_AFFIX_LEN,
    max_per_affix: int = MAX_PER_AFFIX,
    affix_dedup: bool = True,
) -> pd.DataFrame:
    """Return ``frame`` with exact, cross-identity boilerplate, and capped affix duplicates removed.

    ``order_cols`` fixes what "earliest" means (typically identity, then timestamp, then a
    stable id tiebreak); rows are sorted by it up front so the result is deterministic.
    Operates on the cleaned ``text_col``. Returns a new, re-indexed frame.

    With ``affix_dedup=False`` only the exact-duplicate step (1) runs; the two affix-based
    steps (2 cross-identity boilerplate, 3 within-identity cap) are skipped, so no document is
    dropped merely for sharing a leading/trailing slice. SWE-chat uses this (it strips injected
    templates per turn instead); WildChat keeps the default ``True``.
    """
    data = frame.sort_values(list(order_cols)).reset_index(drop=True)

    # Step 1: within-identity exact duplicates -> keep earliest.
    data = data.drop_duplicates([identity_col, text_col], keep="first").reset_index(drop=True)

    if not affix_dedup:  # exact-dedup only; skip the affix-based steps below
        return data

    text = data[text_col].astype(str)
    length = text.str.len()
    has_affix = length >= affix_len
    prefix = text.str[:affix_len].where(has_affix)
    suffix = text.str[-affix_len:].where(has_affix)

    # Step 2: cross-identity boilerplate -> drop every row whose prefix or suffix is used
    # by more than one distinct identity.
    drop = pd.Series(False, index=data.index)
    for affix in (prefix, suffix):
        present = affix.notna()
        n_ids = data[present].groupby(affix[present])[identity_col].transform("nunique")
        shared = pd.Series(False, index=data.index)
        shared.loc[present] = (n_ids >= 2).to_numpy()
        drop |= shared
    data = data[~drop].reset_index(drop=True)

    # Step 3: within-identity affix cap -> keep the max_per_affix earliest per (identity, affix).
    # `data` is still in `order_cols` order here, so cumcount ranks each group chronologically and
    # dropping rank >= max_per_affix keeps exactly the earliest documents.
    text = data[text_col].astype(str)
    has_affix = text.str.len() >= affix_len
    keep = pd.Series(True, index=data.index)
    for slicer in (lambda s: s.str[:affix_len], lambda s: s.str[-affix_len:]):
        affix = slicer(text).where(has_affix)
        present = affix.notna()
        rank = data[present].groupby([data[identity_col][present], affix[present]]).cumcount()
        over = pd.Series(False, index=data.index)
        over.loc[present] = (rank >= max_per_affix).to_numpy()
        keep &= ~over
    return data[keep].reset_index(drop=True)
