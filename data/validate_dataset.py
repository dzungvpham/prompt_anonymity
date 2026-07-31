"""Integrity + consistency checks for the built unified prompt dataset.

Run after ``build_dataset.py``:  ``PYTHONPATH=. python -m data.validate_dataset``.
The dataset ships as **one parquet per source** (a HuggingFace split): ``wildchat.parquet`` and
``swe_chat.parquet``. This validates each file on its own (it holds exactly its own source,
with source-prefixed author ids and unique doc_ids) and the two combined -- schema/
pseudonymity, turn-list integrity, the ``min_docs`` floor, the SWE-chat human-turn invariants,
and (most importantly) that both sources were scrubbed with the *same* placeholder vocabulary so
``source`` does not leak through raw identifiers. (No known/unknown split is validated -- the
dataset no longer ships a ``split_role`` column.) Prints a PASS/FAIL line per check and exits
non-zero if any fail.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

from .build_dataset import CONSECUTIVE_DUP_MAX_LEN
from .sources_swe_chat import HUMAN_TURN_MAX_LEN, is_scaffolding_turn

DIST = Path(__file__).with_name("dist")
# source -> split/parquet file name (must match build_dataset.SPLIT_NAMES).
SPLIT_FILES = {"wildchat": "wildchat.parquet", "swe-chat": "swe_chat.parquet"}
EXPECTED_COLUMNS = [
    "doc_id", "source", "author_id",
    "turns", "num_turns",
    "language_primary", "language_secondary", "model", "model_owner", "agent", "started_at", "ended_at",
]
RAW_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
RAW_URL = re.compile(r"https?://\S+")


def _consecutive_dup_stats(turn_lists) -> tuple[int, int]:
    """Return ``(max_len, n_scaffold)`` over *surviving* consecutive-duplicate turns.

    A surviving consecutive duplicate is any turn equal to the one immediately before it.
    ``max_len`` is the longest such turn (verifies the length rule held); ``n_scaffold`` counts
    how many are framework scaffolding messages (verifies the scaffolding rule held). Both are 0
    when the consecutive-turn dedup left nothing it should have removed.
    """
    max_len = 0
    n_scaffold = 0
    for turns in turn_lists:
        turns = list(turns)
        for k in range(1, len(turns)):
            if turns[k] and turns[k] == turns[k - 1]:
                max_len = max(max_len, len(turns[k]))
                if is_scaffolding_turn(turns[k]):
                    n_scaffold += 1
    return max_len, n_scaffold


def main(dist: str | Path = DIST) -> int:
    dist = Path(dist)
    checks: list[tuple[str, bool, str]] = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    # --- per-file: exists, holds exactly its own source, prefixed ids, locally unique ---
    per_source: dict[str, pd.DataFrame] = {}
    for source, fname in SPLIT_FILES.items():
        path = dist / fname
        if not path.exists():
            check(f"[{fname}] file exists", False, str(path))
            continue
        df = pd.read_parquet(path)
        per_source[source] = df
        check(f"[{fname}] file exists", True)
        check(f"[{fname}] holds exactly source={source!r}",
              set(df["source"].unique()) == {source}, str(sorted(df["source"].unique())))
        check(f"[{fname}] author_id prefixed '{source}-'",
              df["author_id"].str.startswith(f"{source}-").all())
        check(f"[{fname}] doc_id unique within file", df["doc_id"].is_unique)
        # Schema + language columns are checked per file, so a source still on the old single
        # `languages` list (a not-yet-rebuilt source, e.g. WildChat during the transition) is
        # flagged as pending rather than crashing the combined checks below.
        if "language_primary" not in df.columns and "languages" in df.columns:
            check(f"[{fname}] columns match schema", False, "old `languages` schema — rerun pending")
        else:
            missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
            extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
            check(f"[{fname}] columns match schema", list(df.columns) == EXPECTED_COLUMNS,
                  str(missing + extra))
            check(f"[{fname}] language_primary non-null string",
                  df["language_primary"].map(lambda x: isinstance(x, str)).all())
            check(f"[{fname}] language_secondary string-or-null",
                  df["language_secondary"].map(lambda x: x is None or isinstance(x, str) or pd.isna(x)).all())

    if not per_source:
        print("no per-source files found in", dist)
        return 1

    df = pd.concat(per_source.values(), ignore_index=True)
    joined = df["turns"].map(lambda ts: "\n".join(ts))  # cleaned turns concatenated (for scans)

    # --- schema & pseudonymity --- (per-file column-schema checks are in the loop above)
    check("no raw identity/repo/user/original columns",
          not ({"identity", "repo_id", "user_id", "user_agent", "header", "turns_raw", "turns_original"} & set(df.columns)))
    check("doc_id globally unique", df["doc_id"].is_unique)
    check("author_id source-prefixed & opaque",
          df["author_id"].str.match(r"^(wildchat|swe-chat)-[0-9a-f]{16}$").all())
    check("no null in core fields",
          df[["doc_id", "source", "author_id"]].notna().all().all())

    # --- turn-list integrity ---
    check("every doc has >= 1 turn", df["turns"].map(len).ge(1).all())
    check("num_turns == len(turns)", (df["num_turns"] == df["turns"].map(len)).all())
    check("no empty documents (>=1 non-empty turn)",
          df["turns"].map(lambda ts: any(t for t in ts)).all())

    # --- min docs per author (a count filter, not a length filter) ---
    sizes = df.groupby("author_id")["doc_id"].size()
    check("every author has >= 2 documents", (sizes >= 2).all(), f"min={int(sizes.min())}")

    swe = df[df.source == "swe-chat"]

    # SWE-chat consecutive-turn dedup invariants: the always-per-turn rules (length cap and
    # scaffolding) must leave nothing behind. (The turn_id / no-agent / spammed-in-one-doc rules
    # depend on source metadata or corpus counts and are not re-derivable from the output alone.)
    max_dup_len, n_scaffold_dups = _consecutive_dup_stats(swe["turns"])
    check(f"SWE-chat: no consecutive dup >= {CONSECUTIVE_DUP_MAX_LEN} chars",
          max_dup_len < CONSECUTIVE_DUP_MAX_LEN, f"max_dup_len={max_dup_len}")
    check("SWE-chat: no consecutive-dup scaffolding turns",
          n_scaffold_dups == 0, f"n={n_scaffold_dups}")

    # SWE-chat human-only invariants: filter_to_human_turns must leave no agent/framework turns in
    # the output -- no orchestration messages, tool-I/O echoes, framework wrappers, /compact
    # summaries, or over-long pasted/injected turns.
    swe_turns = [t for ts in swe["turns"] for t in ts]
    check("SWE-chat: no <teammate-message> turns",
          not any("<teammate-message" in t for t in swe_turns))
    check("SWE-chat: no <system_instruction> wrappers",
          not any("<system_instruction>" in t for t in swe_turns))
    check("SWE-chat: no tool-io / task-notification turns",
          not any(t.lstrip().startswith((
              "<task-notification>", "<bash-input>", "<bash-stdout>", "<bash-stderr>",
              "<ide_", "<local-command")) for t in swe_turns))
    check("SWE-chat: no /compact continuation turns",
          not any(t.lstrip().startswith(("This session is being continued",
                                         "Caveat: The messages below")) for t in swe_turns))
    check("SWE-chat: no plan-mode ('Implement the following plan') turns",
          not any(t.lstrip().startswith("Implement the following plan") for t in swe_turns))
    longest_swe = max((len(t) for t in swe_turns), default=0)
    check(f"SWE-chat: no turn >= {HUMAN_TURN_MAX_LEN} chars (human-length cap)",
          longest_swe < HUMAN_TURN_MAX_LEN, f"longest={longest_swe}")

    # --- scrubbing consistency (the anti-leak invariant) ---
    for src in df["source"].unique():
        m = df["source"] == src
        raw_email = joined[m].str.contains(RAW_EMAIL).mean()
        raw_url = joined[m].str.contains(RAW_URL).mean()
        check(f"[{src}] cleaned turns have ~no raw emails", raw_email < 0.005, f"{raw_email:.3%}")
        check(f"[{src}] cleaned turns have ~no raw http URLs", raw_url < 0.005, f"{raw_url:.3%}")
        rate = joined[m].str.contains(r"<URL>|<EMAIL>|<IP>|<PATH>|<REPO>|<USER>|<ID>|<HOST>", regex=True).mean()
        check(f"[{src}] some docs carry placeholders", rate > 0.0, f"{rate:.1%} of docs")

    # --- report ---
    width = max(len(n) for n, _, _ in checks)
    n_fail = sum(not ok for _, ok, _ in checks)
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    print(f"\n{len(checks) - n_fail}/{len(checks)} checks passed.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
