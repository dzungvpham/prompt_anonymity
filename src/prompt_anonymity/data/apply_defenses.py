r"""Rewrite a built dataset split's conversations with a defense -- one defended parquet per defense.

Third companion to :mod:`prompt_anonymity.data.build_dataset` (which writes the documents) and
:mod:`prompt_anonymity.data.compute_features` (which writes their feature vectors). This script
sits between the two: it takes a built split, runs every conversation through one registered
**defense** -- the anonymization countermeasure a user would apply to their prompts before
releasing them -- and writes the defended conversations back out as a parquet with the *same
schema*, so everything downstream can read it exactly like the original split.

Defenses are **not reimplemented here**. The script drives the registered defenses of the
installed package (``prompt_anonymity.defenses.DEFENSES``) -- the same registry
``run_experiment.py --defense`` reads -- so ``--defense`` accepts whatever that registry exposes
and a newly registered defense becomes available with no change to this file.

Where the output goes
---------------------

A defended split is written to ``dist/<split>_<defense>.parquet`` -- ``dist/swe_chat_openanonymity.parquet``
-- beside the split it came from, carrying the split's own schema so anything that reads the
original can read it.

Note this shares a namespace with the feature files ``compute_features`` writes
(``dist/swe_chat_stylometrix.parquet``): both are ``<split>_<name>.parquet``, distinguished only by
whether ``<name>`` is a registered defense or a registered featurizer. The two registries are
disjoint today. What tells them apart on disk is their columns -- a defended split has ``turns``,
a feature file has ``doc_id``/``author_id`` plus feature columns.

To featurize defended text, point ``compute_features`` at the defended file with its ``--defense``
flag::

    python -m prompt_anonymity.data.apply_defenses   --source swe-chat --defense openanonymity
    python -m prompt_anonymity.data.compute_features --source swe-chat --defense openanonymity \
        --feature stylometrix        # reads swe_chat_openanonymity.parquet
                                     # writes swe_chat_openanonymity_stylometrix.parquet

What a defense sees: one user turn at a time
--------------------------------------------

Each document is a list of user turns, and **each turn is defended on its own**, then the results
are re-assembled into a list of the same length -- so ``num_turns`` and the turn boundaries survive
the rewrite, and a document's defended turns line up one-for-one with its originals.

That is also the granularity the caching works at: the package's defense machinery caches one row
per input it is handed (:class:`~prompt_anonymity.caching.IndexedRowCache`, keyed by the id passed
in and verified against the source text), so handing it turns rather than whole conversations means
a turn repeated across documents is computed once, and an interrupted run resumes at turn
granularity instead of re-doing a 422-turn session from the top. Turn ids are ``<doc_id>#<n>``.

Note this differs from ``run_experiment.py --defense``, which splits a WildChat conversation *cell*
on the legacy ``\n===\n`` marker to recover its turns. The unified dataset stores turns as a real
list, so no delimiter is involved and no text can be mistaken for one.

Cost, caching and re-runs
-------------------------

Every defense here runs a model on this machine -- ``styleremix``, ``qwen_rewrite``, ``dp_mlm``,
and ``openanonymity``, which generates once per distinct turn with a local gpt-oss-120b through
vLLM and so wants a GPU large enough to hold it. The run prints how many documents, turns and
distinct turns it is about to defend **before** starting, so the scale is visible up front, and
every defended turn is cached under ``--cache-dir``: a re-run recomputes nothing.

The cache is scoped per split **and per shard layout** -- ``<cache-dir>/defended/<split>[/<i>-of-<n>]``
-- because the defense cache is one table per side, rewritten whole on each run: two shards sharing
one table would each drop the other's rows. Two consequences worth knowing: applying the same
defense to two sources never mixes their caches, and re-running an array with a *different*
``--num-shards`` recomputes everything. Keep the array size fixed across re-runs, or pay the GPU
hours again.

Sharding (SLURM job arrays)
---------------------------

Sharding works exactly as in :mod:`~prompt_anonymity.data.compute_features` and shares its
implementation: ``--num-shards N --shard-index I`` defends every ``N``-th document, either flag is
filled in from ``SLURM_ARRAY_TASK_ID`` / ``SLURM_ARRAY_TASK_COUNT`` in an array job, shards land in
``dist/shards/``, and the last shard to finish concatenates them into the final split-ordered
parquet (or ``--merge`` does it on demand). Sharding is what makes WildChat tractable for a slow
defense, and it bounds the size of each task's cache table as well as its runtime.

Run (from the repo root):

    python -m prompt_anonymity.data.apply_defenses --defense openanonymity --limit 3   # smoke test
    python -m prompt_anonymity.data.apply_defenses --source swe-chat --defense openanonymity
    python -m prompt_anonymity.data.apply_defenses --source wildchat --defense dp_mlm_eps100 \
        --num-shards 32 --shard-index 0
    python -m prompt_anonymity.data.apply_defenses --source wildchat --defense dp_mlm_eps100 --merge
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from prompt_anonymity.core import AttackData
from prompt_anonymity.defenses import DEFENSES, apply_defense

from .compute_features import (
    READ_BATCH_ROWS,
    SPLIT_NAMES,
    load_split,
    merge_shards,
    resolve_sharding,
    select_documents,
    select_shard,
    shard_path,
    write_parquet,
)
from .config import cache_dir, dist_dir

# (Defended splits are written beside the split they came from, as <split>_<defense>.parquet; see
# `output_path` and the module docstring.)

# Where the per-turn defense cache is rooted under --cache-dir. The defense machinery adds its own
# ``defenses/<name>/<logic hash>/<params hash>/`` below this; what this level adds is the split (and
# shard), which that namespace does not carry and which would otherwise collide.
CACHE_SUBDIR = "defended"

# Separator between a document id and a turn's position within it, forming the cache key for one
# turn. Any character absent from the built doc_ids (``sc-<date>-<uuid>`` / ``wc-<...>``) works; '#'
# is chosen because it reads as a fragment reference and never appears in an id.
TURN_ID_SEPARATOR = "#"


# --- input ------------------------------------------------------------------

def split_columns(source: str, dist_dir: str | Path) -> list[str]:
    """The built split's column names, in file order, read from the parquet footer alone.

    Used to reproduce the split's exact schema in the defended output -- including where ``turns``
    sits among the other columns -- without reading a single row of data.

    Read from the **Arrow** schema, not the Parquet one: Parquet describes a list column by its
    leaf, so ``turns`` appears there as the element name (``element``), which is not a column
    anything can select.
    """
    path = Path(dist_dir) / f"{SPLIT_NAMES[source]}.parquet"
    if not path.exists():
        raise SystemExit(f"{path} not found -- build it first with "
                         f"`python -m prompt_anonymity.data.build_dataset`.")
    return list(pq.ParquetFile(path).schema_arrow.names)


def read_turns(source: str, dist_dir: str | Path, positions) -> list[list[str]]:
    """The turn lists of these split row positions, in the order given.

    The list-of-turns counterpart of :func:`~prompt_anonymity.data.compute_features.read_texts`,
    and streaming for the same reason: ``turns`` *is* essentially the whole dataset (WildChat's
    parquet is 412 MB on disk and several GiB once pandas has made Python lists of strings out of
    it), while a shard needs 1/N of it. Reading a batch at a time and keeping only the wanted rows
    makes a task's memory scale with its shard rather than with the corpus.
    """
    path = Path(dist_dir) / f"{SPLIT_NAMES[source]}.parquet"
    wanted = {int(position) for position in positions}
    turns: dict[int, list[str]] = {}
    first_row = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=READ_BATCH_ROWS, columns=["turns"]):
        column = batch.column(0)
        for offset in range(len(batch)):
            if first_row + offset in wanted:
                turns[first_row + offset] = [str(turn) for turn in column[offset].as_py()]
        first_row += len(batch)
    return [turns[int(position)] for position in positions]


# --- turn-level view of the documents ---------------------------------------

def flatten_turns(doc_ids, author_ids, turn_lists) -> tuple[list[str], list[str], list[str], list[int]]:
    """Explode documents into their turns: ``(texts, ids, authors, turns_per_document)``.

    The defense is applied to this flat stream so every turn is defended independently and cached
    under its own key (``<doc_id>#<n>``, see :data:`TURN_ID_SEPARATOR`); ``turns_per_document`` is
    what :func:`regroup_turns` needs to put the documents back together afterwards.
    """
    texts: list[str] = []
    ids: list[str] = []
    authors: list[str] = []
    counts: list[int] = []
    for doc_id, author_id, turns in zip(doc_ids, author_ids, turn_lists):
        counts.append(len(turns))
        for position, turn in enumerate(turns):
            texts.append(turn)
            ids.append(f"{doc_id}{TURN_ID_SEPARATOR}{position}")
            authors.append(author_id)
    return texts, ids, authors, counts


def regroup_turns(defended: list[str], counts: list[int]) -> list[list[str]]:
    """Cut the defended turn stream back into one list per document, using the original counts.

    A defense returns one output per input, so every document gets back exactly as many turns as it
    had -- ``num_turns`` and the turn boundaries are preserved by construction (a turn a defense
    emptied is still a turn). A length mismatch means a defense broke that contract, which would
    silently misalign every document after it, so it is raised rather than trimmed.
    """
    if len(defended) != sum(counts):
        raise ValueError(f"defense returned {len(defended):,} turns for {sum(counts):,} inputs; "
                         f"a defense must return exactly one output per turn, in order.")
    grouped: list[list[str]] = []
    position = 0
    for count in counts:
        grouped.append(defended[position:position + count])
        position += count
    return grouped


# --- running the defense ----------------------------------------------------

def defense_cache_dir(cache_root: str | Path, source: str,
                      shard_index: int, num_shards: int) -> Path:
    """Cache directory for this (split, shard) -- the scope the defense's own namespace lacks.

    A defense caches its rows in one table per side, named for the *side* and namespaced only by
    the defense's name/logic/params (see :class:`~prompt_anonymity.caching.IndexedRowCache`), and
    rewrites that whole table on each run. Two runs that share a table therefore overwrite each
    other's rows -- which is precisely what defending two different splits, or two shards of one
    split in parallel, would do. Giving each its own root keeps them apart.

    The cost is that the shard layout is part of the cache path, so re-running an array with a
    different ``--num-shards`` starts from an empty cache. That is the deliberate trade: a
    recomputation is expensive, but silently losing another task's rows is worse.
    """
    path = Path(cache_root) / CACHE_SUBDIR / SPLIT_NAMES[source]
    if num_shards > 1:
        path = path / f"{shard_index:04d}-of-{num_shards:04d}"
    return path


def as_attack_data(texts: list[str], ids: list[str], authors: list[str]) -> AttackData:
    """Wrap a stream of turns as the :class:`~prompt_anonymity.core.AttackData` a defense takes.

    A defense's input type is the experiment's known/unknown bundle, but a defense only ever reads
    and rewrites **text** -- features are computed afterwards, by a separate stage (here: a later
    ``compute_features`` run over the defended parquet). So the bundle handed over carries the
    turns, their cache ids and their authors, and zero-width embedding matrices: not placeholder
    vectors, but an explicit statement that no features exist yet.

    Everything goes on the *unknown* side, the side a defense rewrites by default (the threat model
    being that the adversary's known conversations are already out). Defending a whole dataset has
    no known side, so that side is empty.
    """
    n = len(texts)
    return AttackData(
        known_embeddings=np.zeros((0, 0), dtype=np.float32),
        unknown_embeddings=np.zeros((n, 0), dtype=np.float32),
        known_labels=np.empty(0, dtype=object),
        unknown_labels=np.asarray(authors, dtype=object),
        unknown_texts=np.asarray(texts, dtype=object),
        unknown_ids=np.asarray(ids, dtype=object),
    )


def report_workload(defense: str, n_documents: int, texts: list[str]) -> None:
    """Log the size of the job before any of it runs: documents, turns, distinct turns, characters.

    Distinct turns is the number that matters for a paid or slow defense -- identical turns are
    computed once (and cached), so it is the real call count for an uncached run -- and seeing it
    before the first request is the difference between noticing a mis-scoped run and paying for it.
    """
    distinct = len({text for text in texts if text.strip()})
    characters = sum(len(text) for text in texts)
    print(f"[{defense}] defending {n_documents:,} documents / {len(texts):,} turns "
          f"({distinct:,} distinct non-blank, {characters:,} characters); "
          f"cached turns are not recomputed")


def defend_documents(defense: str, doc_ids, author_ids, turn_lists,
                     *, cache_dir: str | Path) -> list[list[str]]:
    """Defend every turn of every document, returning the defended turn lists in input order."""
    texts, ids, authors, counts = flatten_turns(doc_ids, author_ids, turn_lists)
    if not texts:
        return []
    report_workload(defense, len(counts), texts)
    defended = apply_defense(defense, as_attack_data(texts, ids, authors), cache_dir=cache_dir)
    return regroup_turns([str(text) for text in defended.unknown_texts], counts)


# --- output -----------------------------------------------------------------

def build_defended_frame(metadata: pd.DataFrame, turn_lists: list[list[str]],
                         columns: list[str]) -> pd.DataFrame:
    """The defended documents in the split's own schema: every column, with ``turns`` replaced.

    Keeping the schema byte-for-byte comparable to the undefended split is what lets a defended
    file be read by anything that reads the original -- the loaders, the featurizer, the runners --
    with no special case for "this one is defended". ``num_turns`` needs no recomputation: a
    per-turn rewrite returns one turn per turn (see :func:`regroup_turns`).
    """
    frame = metadata.reset_index(drop=True).copy()
    frame["turns"] = pd.Series(turn_lists, dtype=object)
    return frame[columns]


def defended_stem(source: str, defense: str) -> str:
    """Name the defended split goes by: ``swe_chat_openanonymity``.

    Also the stem its shard files are named after, so a sharded run and an unsharded one agree.
    """
    return f"{SPLIT_NAMES[source]}_{defense}"


def output_path(out_dir: str | Path, source: str, defense: str) -> Path:
    """The defended split file, beside the split it came from: ``dist/<split>_<defense>.parquet``."""
    return Path(out_dir) / f"{defended_stem(source, defense)}.parquet"


def report_written(frame: pd.DataFrame, path: Path, kind: str = "defended documents") -> None:
    """Log what landed on disk: documents, total turns, file size."""
    turns = int(sum(len(row) for row in frame["turns"])) if len(frame) else 0
    print(f"wrote {len(frame):,} {kind} / {turns:,} turns -> {path} "
          f"({path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="swe-chat", choices=sorted(SPLIT_NAMES),
                   help="which built split to defend (default: swe-chat)")
    p.add_argument("--defense", default="openanonymity", choices=sorted(DEFENSES),
                   help="registered defense to apply (default: openanonymity). 'none' copies the "
                        "split through unchanged, as a control")
    p.add_argument("--language", default=None,
                   help="optional filter: keep only documents whose language_primary is this "
                        "(default: defend every document). A filtered run writes a SUBSET of the "
                        "split under the same filename, so do not mix the two")
    p.add_argument("--dist-dir", default=None,
                   help="directory holding the built parquets (default: the project's data/dist)")
    p.add_argument("--out-dir", default=None,
                   help="where to write the defended split (default: --dist-dir, i.e. beside the "
                        "split it came from, as <split>_<defense>.parquet)")
    p.add_argument("--cache-dir", default=None,
                   help="on-disk cache for defended turns, regenerable (default: data/.cache)")
    p.add_argument("--limit", type=int, default=None,
                   help="testing: defend only the first N selected documents")
    p.add_argument("--num-shards", type=int, default=None,
                   help="cut the selected documents into this many shards and defend only one of "
                        "them, for running a SLURM job array (default: SLURM_ARRAY_TASK_COUNT in "
                        "an array job, else no sharding)")
    p.add_argument("--shard-index", type=int, default=None,
                   help="which shard (0-based) this run defends (default: this array task's "
                        "SLURM_ARRAY_TASK_ID)")
    p.add_argument("--merge", action="store_true",
                   help="do not defend: concatenate the shard files an array already wrote into "
                        "the final defended parquet. Sharded runs also try this automatically "
                        "once they are the last shard to finish")
    p.add_argument("--no-auto-merge", action="store_true",
                   help="a sharded run writes only its shard, leaving the merge to an explicit "
                        "--merge")
    args = p.parse_args()

    language = None if (args.language or "all").lower() == "all" else args.language
    # Paths default to the project's data/ folder (see prompt_anonymity.data.config): the code
    # lives in the installed package, the data does not.
    dist = Path(args.dist_dir) if args.dist_dir else dist_dir()
    cache = Path(args.cache_dir) if args.cache_dir else cache_dir()
    out_dir = Path(args.out_dir) if args.out_dir else dist
    merged_path = output_path(out_dir, args.source, args.defense)
    stem = defended_stem(args.source, args.defense)

    # --merge only reassembles what an array already computed -- no defense is built, so no model is
    # loaded and no API key is needed -- and reads only the columns the selection needs.
    if args.merge:
        documents = select_documents(
            load_split(args.source, dist, columns=["doc_id", "language_primary"]),
            language=language, limit=args.limit,
        )
        merged = merge_shards(out_dir, stem, list(documents["doc_id"]))
        if merged is None:
            raise SystemExit("cannot merge yet: the shards listed above have not been defended. "
                             "Re-run those array tasks, then merge again.")
        write_parquet(merged, merged_path)
        report_written(merged, merged_path)
        return

    shard_index, num_shards = resolve_sharding(args.shard_index, args.num_shards)

    # Every column except `turns`, which is read separately (and only for this shard's rows) since
    # it is the bulk of the dataset; `columns` keeps the output in the split's own schema order.
    columns = split_columns(args.source, dist)
    frame = load_split(args.source, dist, columns=[c for c in columns if c != "turns"])
    documents = select_documents(frame, language=language, limit=args.limit)
    if documents.empty:
        raise SystemExit(f"no documents in {args.source} match --language {args.language}.")
    selected = f" (language_primary == {language!r})" if language else " (all languages)"
    print(f"[{args.source}] {len(documents):,} of {len(frame):,} documents selected{selected}")

    doc_order = list(documents["doc_id"])  # split order, for the merge
    shard = select_shard(documents, shard_index, num_shards)
    out_path = merged_path if num_shards == 1 else shard_path(out_dir, stem, shard_index, num_shards)
    if num_shards > 1:
        print(f"[shard {shard_index}/{num_shards}] defending {len(shard):,} of them "
              f"-> {out_path.name}")

    turn_lists = read_turns(args.source, dist, shard.index)
    defended = defend_documents(
        args.defense, shard["doc_id"], shard["author_id"], turn_lists,
        cache_dir=defense_cache_dir(cache, args.source, shard_index, num_shards),
    )
    documents_out = build_defended_frame(shard, defended, columns)
    write_parquet(documents_out, out_path)
    report_written(documents_out, out_path,
                   "defended documents" if num_shards == 1 else "defended shard documents")

    if num_shards > 1 and not args.no_auto_merge:
        # Every task tries this; only the last one to finish finds a complete set of shards, so the
        # array assembles its own final file with no follow-up job.
        merged = merge_shards(out_dir, stem, doc_order)
        if merged is not None:
            write_parquet(merged, merged_path)
            report_written(merged, merged_path)


if __name__ == "__main__":
    main()
