"""Explode the filtered WildChat CSV into one row per conversation turn.

Each conversation cell concatenates a user's turns joined by a delimiter that
preprocess.py stored in ESCAPED form: real newlines were turned into the literal
two-character sequence ``\\n``, so the turn separator on disk is the literal
string ``\\n===\\n`` (backslash, n, =, =, =, backslash, n) -- NOT real newlines.
We split on that literal string and emit one row per turn, copying every column
from the parent conversation row and adding:

  * ``conv_id``  -- stable id of the source conversation (its ``idx``).
  * ``turn_idx`` -- 0-based position of the turn within its conversation.

Turns are kept AS-IS: no length filtering, no stripping, empty turns retained.

Row order matters: ``stylometrix.py`` re-embeds in CSV order without sorting,
while ``stylometric_attacks.load_data`` sorts by
[hashed_ip, accept_language, device_info, timestamp]. Turns of one conversation
share a timestamp, so we add ``turn_idx`` as a stable tiebreaker here and the
loader must add it too (see note printed at the end) to keep the data CSV and
the regenerated embeddings CSV positionally aligned.
"""

import os

import pandas as pd

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DS_DIR = os.path.join(REPO_DIR, "DS_env")
SRC_CSV = os.path.join(DS_DIR, "wildchat_filtered_4o20240806_41mini20250414_device_deduped.csv")
DST_CSV = os.path.join(DS_DIR, "wildchat_filtered_4o20240806_41mini20250414_device_deduped_turns.csv")

# Literal escaped delimiter as written by wildchat/preprocess.py (NOT real \n).
TURN_DELIM = "\\n===\\n"
# Same identity+time ordering load_data uses, plus turn_idx as the tiebreaker.
SORT_KEYS = ["hashed_ip", "accept_language", "device_info", "timestamp"]


def main():
    df = pd.read_csv(SRC_CSV)
    n_conv = len(df)

    # Stable conversation key. The source already carries a unique per-row idx.
    df = df.reset_index(drop=True)
    df["conv_id"] = df["idx"] if "idx" in df.columns else df.index

    # Split each conversation into its turns (literal delimiter, kept as-is).
    parts = df["conversation"].fillna("").astype(str).str.split(TURN_DELIM, regex=False)
    df["conversation"] = parts
    df["turn_idx"] = parts.map(lambda p: list(range(len(p))))

    exploded = df.explode(["conversation", "turn_idx"], ignore_index=True)
    exploded["turn_idx"] = exploded["turn_idx"].astype(int)

    # Deterministic order: identity+timestamp, then turn order within a convo.
    # Stable sort so equal-key rows keep their (already turn-ordered) sequence.
    exploded = exploded.sort_values(SORT_KEYS + ["turn_idx"], kind="stable").reset_index(drop=True)

    exploded.to_csv(DST_CSV, index=False)

    n_turns = len(exploded)
    n_empty = int((exploded["conversation"].astype(str).str.len() == 0).sum())
    print(f"Source conversations : {n_conv}")
    print(f"Exploded turns       : {n_turns}  (x{n_turns / n_conv:.2f} per conversation)")
    print(f"Empty turns retained : {n_empty}")
    print(f"Wrote -> {DST_CSV}")
    print()
    print("Next steps to use this as the default:")
    print("  1. Re-run wildchat/stylometrix.py against this CSV to regenerate the")
    print("     per-turn embeddings CSV (the two files are positionally paired).")
    print("  2. Point DATA_CSV / EMBEDDINGS_CSV in DS_env/stylometric_attacks.py at")
    print("     the new files, and add 'turn_idx' to load_data()'s sort_values keys")
    print("     so the loader's order matches this CSV's stored order.")


if __name__ == "__main__":
    main()
