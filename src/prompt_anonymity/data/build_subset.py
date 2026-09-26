r"""Cut a small, reproducible working subset out of a built split.

The leave-one-out defense (:mod:`prompt_anonymity.defenses.loo_unlink`) is roughly quadratic in span
count per document and re-scores after every edit, and the budget sweep multiplies that further.
Running it against the full corpus is not a thing to attempt before the method is known to work at
all, so the whole v1 build happens on a subset sized for iteration speed rather than statistical
power.

This is **not** a second dataset build. It reads the split
:mod:`prompt_anonymity.data.build_dataset` already wrote, filters and samples it, and writes a
parquet with the identical schema -- so ``compute_features``, ``apply_defenses``,
``run_experiment.py`` and ``eval_utility.py`` all read it exactly like the split it came from,
without re-running the source build to reproduce rows already on disk.

Two filters, in this order:

1. **Authors with at least ``--k-min`` documents** (default 5). Same rule as
   :func:`~prompt_anonymity.data.build_dataset.filter_min_docs`, whose own default is 2 -- the
   defense needs each author to have enough *other* prompts for a top-m similarity baseline to mean
   anything, and m is 3.
2. **A seeded sample of ``--n-authors`` of the survivors** (default 200). Sampling **authors**, never
   documents: cutting documents at random would leave authors holding one or two prompts and quietly
   undo filter 1.

Every number reported off this subset has to be traceable to a specific build, so a manifest goes
next to the parquet **and** into a committed copy under ``experiments/manifests/``. It records the
source file's sha256, both filter parameters, the seed, the resulting counts, and the git commit --
enough to tell whether two results came from the same subset without diffing parquets.

Token counts are printed because the paid stages downstream are billed per token: the Gemini
featurizer's cost is (documents x tokens per document), and this is the last point before that where
the number is cheap to look at.

Usage::

    python -m prompt_anonymity.data.build_subset                      # wildchat -> wildchat_small
    python -m prompt_anonymity.data.build_subset --k-min 5 --n-authors 200
    python -m prompt_anonymity.data.build_subset --source swe_chat --out-source swe_chat_small
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .compute_features import write_parquet
from .config import dist_dir as default_dist_dir, hf_dir

#: Rows per batch in the second pass. The parent split's ``turns`` column is essentially the whole
#: corpus -- several GiB of Python strings on WildChat -- so the full table is never materialized;
#: batches are filtered down to the chosen authors and only the survivors are kept. This is the same
#: reason ``compute_features`` does not import ``build_dataset``: the heavy thing is avoided rather
#: than paid for and thrown away.
READ_BATCH_ROWS = 5000

#: Default author-count floor. Deliberately above ``build_dataset``'s ``--min-docs`` default of 2:
#: with m=3 for the top-m author-similarity baseline, an author holding 2 documents gives the
#: defense a 1-document reference and a linkage score that is mostly noise.
DEFAULT_K_MIN = 5

#: Default number of authors kept, sized so one budget point runs in minutes rather than hours.
DEFAULT_N_AUTHORS = 200

#: The project's standard RNG seed (``run_experiment.py --seed``), used here so the subset draw and
#: every downstream sampling step share one number.
DEFAULT_SEED = 47

#: Appended where ``--max-chars`` cut inside a turn, so a truncated document is visibly truncated.
TRUNCATION_MARK = " [...truncated]"

#: Characters per document for ``--max-chars``, when it is passed without a value. Sized against the
#: window the local defenses serve (``afr``'s context budget), assuming the dense CJK case of
#: ~2 characters/token rather than English's ~4, so it still fits the multilingual corpus.
DEFAULT_MAX_CHARS = 60_000

#: Where the committed copy of the manifest goes. ``data/`` is gitignored, so the manifest beside the
#: parquet is not a record anybody else can see; this one is.
COMMITTED_MANIFEST_DIR = Path(__file__).resolve().parents[3] / "experiments" / "manifests"


def file_digest(path: Path) -> str:
    """sha256 of a file, read in chunks (these parquets are gigabytes)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str | None:
    """The current commit, or ``None`` outside a git checkout. Recorded so a manifest identifies the
    code that produced it, not just its inputs."""
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                cwd=Path(__file__).resolve().parent, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def select_authors(index: pd.DataFrame, *, k_min: int, n_authors: int | None,
                   seed: int) -> set:
    """The authors that survive both filters, given a ``doc_id``/``author_id`` index of the split."""
    sizes = index.groupby("author_id")["doc_id"].transform("size")
    eligible = index[sizes >= k_min]
    if eligible.empty:
        raise SystemExit(
            f"no author has {k_min} or more documents in this split; lower --k-min."
        )

    authors = np.sort(eligible["author_id"].unique())
    if n_authors is not None and n_authors < len(authors):
        # Seeded choice over a SORTED author array: `unique()` returns first-appearance order, which
        # depends on the parent split's row order, so sorting first is what makes the same seed pick
        # the same authors after an unrelated rebuild reorders rows.
        rng = np.random.default_rng(seed)
        authors = np.sort(rng.choice(authors, size=n_authors, replace=False))
    return set(authors)


def read_authors(path: Path, authors: set) -> pd.DataFrame:
    """Every row of ``path`` belonging to ``authors``, with the file's own schema and row order.

    Streamed in batches (see :data:`READ_BATCH_ROWS`) so the parent split's ``turns`` column is
    never fully materialized. Row order is preserved rather than grouped by author: every downstream
    stage re-sorts by ``ended_at`` anyway, and keeping source order makes a diff against the parent
    readable.
    """
    kept: list[pd.DataFrame] = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=READ_BATCH_ROWS):
        frame = batch.to_pandas()
        match = frame[frame["author_id"].isin(authors)]
        if not match.empty:
            kept.append(match)
    if not kept:
        raise SystemExit("the filters selected no documents; loosen --k-min or --n-authors.")
    return pd.concat(kept, ignore_index=True)


def cap_document(turns, max_chars: int) -> list:
    """One document's turns, cut down to ``max_chars`` total characters.

    Whole turns are kept while they fit and the rest are dropped, so turn boundaries stay real
    conversation boundaries; a *single* turn longer than the cap is truncated in place, because the
    alternative is emitting an empty document. At least one turn always survives.

    The cut is marked so a truncated document is never mistaken for a complete one -- by a reader,
    by the judge that scores utility, or by anyone diffing this split against its parent.
    """
    kept: list[str] = []
    used = 0
    for turn in turns:
        text = str(turn)
        if used + len(text) <= max_chars:
            kept.append(text)
            used += len(text)
            continue
        if not kept:            # the first turn alone is over the cap: cut inside it
            kept.append(text[:max_chars] + TRUNCATION_MARK)
        break
    return kept or [str(turns[0])[:max_chars] + TRUNCATION_MARK]


def cap_documents(subset: pd.DataFrame, max_chars: int) -> tuple[pd.DataFrame, dict]:
    """Apply :func:`cap_document` across the subset, returning it with a summary of what was cut.

    WHY THIS IS A CORPUS FILTER AND NOT A DEFENSE ONE. ``afr`` cannot rewrite a document whose
    prompt exceeds the served context window; its own escape hatch emits such documents
    **undefended**, which is not admissible in an evaluation -- an untouched document inside the
    defended split inflates the attack against the defense that never got to touch it.

    Cutting here instead means every arm (``base``, ``afr_stage1``, ``afr``) reads the *same* text,
    so the comparison stays honest and the truncation becomes a stated property of the split rather
    than an artifact of one defense. Cutting inside the defense would do the opposite: only the
    defended documents would be shorter, and shorter text carries less authorship signal, so the
    defense would look effective for a reason that has nothing to do with the defense.
    """
    lengths = subset["turns"].map(lambda items: sum(len(str(turn)) for turn in items))
    over = int((lengths > max_chars).sum())
    capped = subset.copy()
    capped["turns"] = subset["turns"].map(lambda items: cap_document(list(items), max_chars))
    after = capped["turns"].map(lambda items: sum(len(str(turn)) for turn in items))
    return capped, {
        "max_chars": int(max_chars),
        "documents_truncated": over,
        "share_truncated": round(over / len(subset), 4) if len(subset) else 0.0,
        "characters_before": int(lengths.sum()),
        "characters_after": int(after.sum()),
        "longest_before": int(lengths.max()) if len(lengths) else 0,
        "longest_after": int(after.max()) if len(after) else 0,
    }


def token_estimate(turns: pd.Series) -> dict:
    """Rough per-document token statistics for the documents in ``turns``.

    Characters over four, not a real tokenizer: this exists to size a bill before it is incurred,
    and loading tiktoken to put a second significant figure on an estimate that is then multiplied
    by an unknown number of retries would be false precision. The Gemini featurizer does its own
    exact accounting when it actually spends.
    """
    characters = turns.map(lambda items: sum(len(str(turn)) for turn in items))
    tokens = (characters / 4.0).round()
    return {
        "mean": float(tokens.mean()),
        "median": float(tokens.median()),
        "p95": float(tokens.quantile(0.95)),
        "max": float(tokens.max()),
        "total": int(tokens.sum()),
    }


def build_manifest(*, source: str, out_source: str, source_path: Path, parent: pd.DataFrame,
                   subset: pd.DataFrame, k_min: int, n_authors: int | None, seed: int) -> dict:
    """Everything needed to tell whether two results came from the same subset."""
    return {
        "out_source": out_source,
        "source": source,
        "source_path": str(source_path),
        "source_sha256": file_digest(source_path),
        "git_commit": git_commit(),
        "filters": {"k_min": k_min, "n_authors": n_authors, "seed": seed},
        "parent_counts": {
            "documents": int(len(parent)),
            "authors": int(parent["author_id"].nunique()),
        },
        "counts": {
            "documents": int(len(subset)),
            "authors": int(subset["author_id"].nunique()),
            "turns": int(subset["num_turns"].sum()),
            "documents_per_author_min": int(subset.groupby("author_id").size().min()),
            "documents_per_author_median": float(subset.groupby("author_id").size().median()),
            "documents_per_author_max": int(subset.groupby("author_id").size().max()),
        },
        "tokens_per_document_estimate": token_estimate(subset["turns"]),
        "languages": {
            str(name): int(count)
            for name, count in subset["language_primary"].value_counts().head(10).items()
        },
    }


def report(manifest: dict) -> None:
    """Print the manifest's headline numbers, including the ones that cost money downstream."""
    counts, tokens = manifest["counts"], manifest["tokens_per_document_estimate"]
    print(f"\n[{manifest['out_source']}] {counts['documents']:,} documents / "
          f"{counts['authors']:,} authors / {counts['turns']:,} turns")
    print(f"  documents per author: min {counts['documents_per_author_min']}, "
          f"median {counts['documents_per_author_median']:.0f}, "
          f"max {counts['documents_per_author_max']}")
    print(f"  est. tokens per document: median {tokens['median']:,.0f}, "
          f"mean {tokens['mean']:,.0f}, p95 {tokens['p95']:,.0f}, max {tokens['max']:,.0f}")
    print(f"  est. total tokens: {tokens['total']:,}")
    # The one number that turns into a bill. Gemini embeddings are $0.20/M tokens, one call per
    # document per condition, so this is what a single condition costs to embed.
    print(f"  -> gemini_embedding_2 at $0.20/M tokens: "
          f"${tokens['total'] * 0.20 / 1e6:.2f} per condition (undefended + one per budget)")
    print(f"  languages: {manifest['languages']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="wildchat",
                        help="built split to cut down (default: wildchat)")
    parser.add_argument("--out-source", default=None,
                        help="name of the subset split, which is also its parquet stem and the "
                             "--source every downstream stage takes (default: <source>_small)")
    parser.add_argument("--k-min", type=int, default=DEFAULT_K_MIN,
                        help=f"keep authors with at least this many documents "
                             f"(default: {DEFAULT_K_MIN})")
    parser.add_argument("--n-authors", type=int, default=DEFAULT_N_AUTHORS,
                        help=f"keep a seeded sample of this many authors; pass 0 for all of them "
                             f"(default: {DEFAULT_N_AUTHORS})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"seed for the author sample (default: {DEFAULT_SEED})")
    parser.add_argument("--data-dir", default=None,
                        help="directory holding <source>.parquet (default: the published mirror, "
                             "data/hf)")
    parser.add_argument("--out-dir", default=None,
                        help="where the subset parquet goes (default: data/dist)")
    parser.add_argument("--max-chars", type=int, nargs="?", const=DEFAULT_MAX_CHARS, default=None,
                        metavar="N",
                        help=f"cap every document at N characters, dropping whole trailing turns "
                             f"and marking any cut (default: no cap; bare --max-chars uses "
                             f"{DEFAULT_MAX_CHARS:,}). Use this when a local defense has a context "
                             f"window: afr cannot rewrite a document that exceeds it and would "
                             f"otherwise emit it UNDEFENDED into the defended split, which is not "
                             f"admissible in an evaluation. Capping here keeps every arm on the "
                             f"same text")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing subset parquet. Without this an existing file "
                             "is left alone, so re-running the pipeline's sbatch is cheap and "
                             "cannot silently reshuffle a subset that results already reference")
    args = parser.parse_args()

    out_source = args.out_source or f"{args.source}_small"
    n_authors = args.n_authors if args.n_authors and args.n_authors > 0 else None
    data_dir = Path(args.data_dir) if args.data_dir else hf_dir()
    out_dir = Path(args.out_dir) if args.out_dir else default_dist_dir()
    source_path = data_dir / f"{args.source}.parquet"
    out_path = out_dir / f"{out_source}.parquet"
    manifest_path = out_dir / f"{out_source}_manifest.json"

    if out_path.exists() and not args.force:
        print(f"{out_path} already exists; leaving it alone (pass --force to rebuild).")
        if manifest_path.exists():
            report(json.loads(manifest_path.read_text()))
        return

    if not source_path.exists():
        raise SystemExit(
            f"{source_path} does not exist. Build it first with "
            f"'python -m prompt_anonymity.data.build_dataset', or download the published mirror "
            f"with 'python -m prompt_anonymity.data.download'."
        )

    # Pass 1: the join keys only. The `turns` column is gigabytes and the filters below don't
    # need a single character of text, so it's never read here.
    print(f"reading {source_path}")
    index = pq.read_table(source_path, columns=["doc_id", "author_id"]).to_pandas()
    authors = select_authors(index, k_min=args.k_min, n_authors=n_authors, seed=args.seed)
    print(f"  {len(authors):,} of {index['author_id'].nunique():,} authors selected "
          f"(k_min={args.k_min}, n_authors={n_authors}, seed={args.seed})")

    # Pass 2: the full rows for those authors, streamed.
    subset = read_authors(source_path, authors)

    # Cap BEFORE the manifest, so its statistics describe the split that is actually written.
    truncation = None
    if args.max_chars:
        subset, truncation = cap_documents(subset, args.max_chars)
        print(f"  capped at {args.max_chars:,} characters: "
              f"{truncation['documents_truncated']:,} of {len(subset):,} documents truncated "
              f"({truncation['share_truncated']:.1%}), longest now "
              f"{truncation['longest_after']:,} characters")

    manifest = build_manifest(source=args.source, out_source=out_source, source_path=source_path,
                              parent=index, subset=subset, k_min=args.k_min,
                              n_authors=n_authors, seed=args.seed)
    # Recorded with the split because it changes what the corpus IS: every downstream number is
    # conditional on it, and the committed manifest is where that has to be discoverable.
    manifest["truncation"] = truncation

    write_parquet(subset, out_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    COMMITTED_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    (COMMITTED_MANIFEST_DIR / f"{out_source}.json").write_text(json.dumps(manifest, indent=2) + "\n")

    report(manifest)
    print(f"\nwrote {out_path}")
    print(f"      {manifest_path}")
    print(f"      {COMMITTED_MANIFEST_DIR / f'{out_source}.json'}  (commit this)")
    print(f"\nNext: python -m prompt_anonymity.data.compute_features --source {out_source} "
          f"--feature harrier --dist-dir {out_dir}")


if __name__ == "__main__":
    main()
