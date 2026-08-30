r"""Rewrite a built dataset split's conversations with a defense -- one defended parquet per defense.

Third companion to :mod:`prompt_anonymity.data.build_dataset` (which writes the documents) and
:mod:`prompt_anonymity.data.compute_features` (which writes their feature vectors). This script
sits between the two: it takes a built split, runs every conversation through one registered
**defense** -- the anonymization countermeasure a user would apply to their prompts before
releasing them -- and writes the defended conversations back out as a parquet with the *same
schema*, so everything downstream can read it exactly like the original split.

Defenses are **not reimplemented here**. The script drives the registered defenses of the
installed package (``prompt_anonymity.defenses.DEFENSES``) -- the same registry
``run_experiment.py --defense`` names its already-defended parquet from -- so ``--defense`` accepts
whatever that registry exposes
and a newly registered defense becomes available with no change to this file.

Where the output goes
---------------------

The split itself is READ from ``data/hf`` (the downloaded/published mirror, see
:func:`prompt_anonymity.data.config.hf_dir`) and a defended split is WRITTEN to
``dist/<split>_<defense>.parquet`` -- ``dist/swe_chat_openanonymity.parquet`` -- carrying the
split's own schema so anything that reads the original can read it. Keeping writes out of
``data/hf`` means running this locally never mutates the downloaded mirror.

Note this shares a namespace with the feature files ``compute_features`` writes
(``dist/swe_chat_stylometrix.parquet``): both are ``<split>_<name>.parquet``, distinguished only by
whether ``<name>`` is a registered defense or a registered featurizer. The two registries are
disjoint today. What tells them apart on disk is their columns -- a defended split has ``turns``,
a feature file has ``doc_id``/``author_id`` plus feature columns.

To featurize defended text, point ``compute_features`` at the defended file with its ``--defense``
flag -- and, since that file lives in ``dist/`` rather than ``compute_features``'s own default
read location (``data/hf``), also point ``--dist-dir`` there::

    python -m prompt_anonymity.data.apply_defenses   --source swe_chat --defense openanonymity
    python -m prompt_anonymity.data.compute_features --source swe_chat --defense openanonymity \
        --feature stylometrix --dist-dir data/dist   # reads swe_chat_openanonymity.parquet
                                                      # writes swe_chat_openanonymity_stylometrix.parquet

What a defense sees: one user turn at a time
--------------------------------------------

Each document is a list of user turns, and **each turn is defended on its own**, then the results
are re-assembled into a list of the same length -- so ``num_turns`` and the turn boundaries survive
the rewrite, and a document's defended turns line up one-for-one with its originals.

The one exception is a defense that *adds* turns rather than rewriting them (``frame_pad``, which
appends a shared off-topic turn to each document). Such a defense declares ``appends_turns = True``
and exposes ``extra_turns(doc_id)``; it never sees the existing turns, which are copied through
byte-identical, and it is applied per DOCUMENT rather than through the per-turn path below -- that
path structurally cannot add a turn, since it must return one output per input row. For those
defenses, and only those, the defended document has MORE turns than the original.

That is also the granularity the caching works at: the package's defense machinery caches one row
per input it is handed (:class:`~prompt_anonymity.caching.IndexedRowCache`, keyed by the id passed
in and verified against the source text), so handing it turns rather than whole conversations means
a turn repeated across documents is computed once, and an interrupted run resumes at turn
granularity instead of re-doing a 422-turn session from the top. Turn ids are ``<doc_id>#<n>``.

This is the only place a defense is applied to text. The removed fixed-split runner used to do it
inside the run, splitting a WildChat conversation *cell* on the legacy ``\n===\n`` marker to recover
its turns; the unified dataset stores turns as a real list, so no delimiter is involved here and no
text can be mistaken for one.

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
    python -m prompt_anonymity.data.apply_defenses --source swe_chat --defense openanonymity
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
from prompt_anonymity.defenses import DEFENSES, apply_defense, get_defense
from prompt_anonymity.defenses._backends import TURN_ID_SEPARATOR

from .compute_features import (
    READ_BATCH_ROWS,
    SOURCES,
    load_split,
    merge_shards,
    resolve_sharding,
    select_author_shard,
    select_documents,
    select_shard,
    shard_path,
    write_parquet,
)
from .config import cache_dir, dist_dir, hf_dir

# The defended file's columns. Just the key, its author, and the rewritten text: everything else in
# the split (timestamps, language, model, agent) is unchanged by a defense, so copying it would
# duplicate the split rather than describe the defense. Anything that needs those columns joins back
# to <split>.parquet on doc_id, which is what `compute_features --defense` does for the language
# filter.
DEFENDED_COLUMNS = ["doc_id", "author_id", "turns"]

# Where the per-turn defense cache is rooted under --cache-dir. The defense machinery adds its own
# ``defenses/<name>/<logic hash>/<params hash>/`` below this; what this level adds is the split (and
# shard), which that namespace does not carry and which would otherwise collide.
CACHE_SUBDIR = "defended"

# Separator between a document id and a turn's position within it, forming the cache key for one
# turn. Any character absent from the built doc_ids (``sc-<date>-<uuid>`` / ``wc-<...>``) works; '#'
# is chosen because it reads as a fragment reference and never appears in an id. Defined in
# `defenses._backends` (imported above, and re-exported here) so a defense can read the doc_id back
# out of a row id without importing this module, which imports the defense registry.


# --- input ------------------------------------------------------------------

def read_turns(source: str, dist_dir: str | Path, positions) -> list[list[str]]:
    """The turn lists of these split row positions, in the order given.

    The list-of-turns counterpart of :func:`~prompt_anonymity.data.compute_features.read_texts`,
    and streaming for the same reason: ``turns`` *is* essentially the whole dataset (WildChat's
    parquet is 412 MB on disk and several GiB once pandas has made Python lists of strings out of
    it), while a shard needs 1/N of it. Reading a batch at a time and keeping only the wanted rows
    makes a task's memory scale with its shard rather than with the corpus.
    """
    path = Path(dist_dir) / f"{source}.parquet"
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

    A defense that means to add a turn therefore cannot do it here: it declares ``appends_turns``
    and is applied by :func:`append_extra_turns` instead, on documents that this function has
    already put back together.
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
    path = Path(cache_root) / CACHE_SUBDIR / source
    if num_shards > 1:
        path = path / f"{shard_index:04d}-of-{num_shards:04d}"
    return path


def as_attack_data(texts: list[str], ids: list[str], authors: list[str],
                   reference=None) -> AttackData:
    """Wrap a stream of turns as the :class:`~prompt_anonymity.core.AttackData` a defense takes.

    A defense's input type is the experiment's known/unknown bundle, but a defense only ever reads
    and rewrites **text** -- features are computed afterwards, by a separate stage (here: a later
    ``compute_features`` run over the defended parquet). So the bundle handed over carries the
    turns, their cache ids and their authors, and zero-width embedding matrices: not placeholder
    vectors, but an explicit statement that no features exist yet.

    Everything goes on the *unknown* side, the side a defense rewrites by default (the threat model
    being that the adversary's known conversations are already out). Defending a whole dataset
    normally has no known side, so that side is empty.

    ``reference``, when given, is ``(texts, ids, authors)`` of read-only context documents that go
    on the **known** side -- "labeled reference conversations", which is what that side means. Only
    an author-sharded run needs it: the task holds a fraction of the split but a defense calibrating
    against "a median unrelated document" must see the same reference pool as every other shard, or
    each optimizes to a different target. Nothing on this side is rewritten or returned. It is the
    same per-turn stream as the unknown side, so the defense rebuilds documents from it identically.
    """
    n = len(texts)
    ref_texts, ref_ids, ref_authors = reference if reference else ([], [], [])
    return AttackData(
        known_embeddings=np.zeros((len(ref_texts), 0), dtype=np.float32),
        unknown_embeddings=np.zeros((n, 0), dtype=np.float32),
        known_labels=np.asarray(ref_authors, dtype=object),
        unknown_labels=np.asarray(authors, dtype=object),
        known_texts=np.asarray(ref_texts, dtype=object) if ref_texts else None,
        unknown_texts=np.asarray(texts, dtype=object),
        known_ids=np.asarray(ref_ids, dtype=object) if ref_ids else None,
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


def append_extra_turns(name: str, defense, doc_ids, turn_lists) -> list[list[str]]:
    """Apply a turn-ADDING defense: copy each document's turns through and append what it asks for.

    The path for a defense that declares ``appends_turns`` (``frame_pad``). It is document-level, so
    it skips the per-turn machinery entirely -- no flatten, no cache, no backend. That is not just an
    optimisation: the per-turn path caches one row per input turn and requires one output per input
    turn, so it can neither express "one more turn" nor gain anything from caching a defense whose
    transform is a dictionary lookup keyed by ``doc_id``.

    The existing turns are never handed to the defense, which is the guarantee this kind of defense
    is built on: whatever it appends, the user's own text is byte-identical to its input.
    """
    padded = [list(turns) + [str(turn) for turn in defense.extra_turns(doc_id)]
              for doc_id, turns in zip(doc_ids, turn_lists)]
    added = [new[len(old):] for new, old in zip(padded, turn_lists)]
    characters = sum(len(turn) for turns in added for turn in turns)
    print(f"[{name}] appending turns to {len(padded):,} documents: "
          f"{sum(len(turns) for turns in added):,} turns / {characters:,} characters added; "
          f"existing turns are copied through unchanged")
    report = getattr(defense, "report", None)
    if callable(report):
        report()
    return padded


def defend_documents(defense: str, doc_ids, author_ids, turn_lists,
                     *, cache_dir: str | Path, num_shards: int = 1,
                     reference=None) -> list[list[str]]:
    """Defend every turn of every document, returning the defended turn lists in input order.

    ``reference`` is the optional known-side context described in :func:`as_attack_data`, passed
    only for a defense that declares ``shardable_by = "author"``.
    """
    if not turn_lists:
        return []
    registered = get_defense(defense)
    # An author-aware defense measures a document against the author's OTHER documents, and
    # `select_shard` splits by document (interleaved), so a shard holds an arbitrary subset of each
    # author. `loo_unlink` would compute its linkage baseline against a truncated author; `afr`
    # would cascade from the wrong documents entirely. Neither failure is visible in the output --
    # both produce a plausible defended parquet that means something different per shard -- so this
    # refuses the run rather than trusting the operator to remember.
    #
    # `shardable_by = "author"` is the exemption: main() gave that defense whole authors via
    # `select_author_shard`, which is exact rather than merely tolerable.
    if (num_shards > 1 and not getattr(registered, "shardable", True)
            and getattr(registered, "shardable_by", None) != "author"):
        raise SystemExit(
            f"defense {defense!r} cannot be sharded: it measures each document against its "
            f"author's other documents, and --num-shards splits by document. Re-run without "
            f"--num-shards/--shard-index (or on a smaller --source)."
        )
    # A turn-adding defense (see append_extra_turns) works on documents, not on the per-turn stream.
    if getattr(registered, "appends_turns", False):
        return append_extra_turns(defense, registered, doc_ids, turn_lists)

    texts, ids, authors, counts = flatten_turns(doc_ids, author_ids, turn_lists)
    if not texts:
        return []
    report_workload(defense, len(counts), texts)
    defended = apply_defense(defense, as_attack_data(texts, ids, authors, reference),
                             cache_dir=cache_dir)
    return regroup_turns([str(text) for text in defended.unknown_texts], counts)


def reference_pool(defense, registered, documents: pd.DataFrame, source: str,
                   dist_dir: str | Path):
    """The known-side reference documents a sharded run owes the defense, as a per-turn stream.

    Selected over the **whole** ``documents`` frame, before any shard is taken, so every task in the
    array calibrates against the same "unrelated" -- see ``afr``'s ``reference_pool_ids``. Returns
    ``None`` for a defense that does not ask for one, which is every defense but ``afr``.
    """
    chooser = getattr(registered, "reference_pool_ids", None)
    if chooser is None:
        return None
    wanted = set(chooser(list(documents["doc_id"])))
    rows = documents[documents["doc_id"].isin(wanted)]
    if rows.empty:
        return None
    texts, ids, authors, _counts = flatten_turns(
        rows["doc_id"], rows["author_id"], read_turns(source, dist_dir, rows.index))
    print(f"[{defense}] reference pool: {len(rows):,} documents drawn from the full split "
          f"({len(texts):,} turns), identical in every shard")
    return texts, ids, authors


# --- output -----------------------------------------------------------------

def build_defended_frame(metadata: pd.DataFrame, turn_lists: list[list[str]]) -> pd.DataFrame:
    """The defended documents: ``doc_id``, ``author_id``, and the rewritten ``turns``.

    Only what the defense actually produces (see :data:`DEFENDED_COLUMNS`). The split's other
    columns -- timestamps, language, model, agent -- are untouched by a defense, so they stay in
    ``<split>.parquet`` and are joined back on ``doc_id`` by whoever needs them; ``num_turns`` is
    likewise recoverable, since a per-turn rewrite returns one turn per turn
    (see :func:`regroup_turns`) -- with the exception of a turn-ADDING defense
    (:func:`append_extra_turns`), whose documents carry more turns than the split records, so read
    the count from this file rather than from ``<split>.parquet``'s ``num_turns``.
    """
    frame = metadata.reset_index(drop=True).copy()
    frame["turns"] = pd.Series(turn_lists, dtype=object)
    return frame[DEFENDED_COLUMNS]


def defended_stem(source: str, defense: str) -> str:
    """Name the defended split goes by: ``swe_chat_openanonymity``.

    Also the stem its shard files are named after, so a sharded run and an unsharded one agree.
    """
    return f"{source}_{defense}"


def output_path(out_dir: str | Path, source: str, defense: str) -> Path:
    """The defended split file: ``dist/<split>_<defense>.parquet``."""
    return Path(out_dir) / f"{defended_stem(source, defense)}.parquet"


def report_written(frame: pd.DataFrame, path: Path, kind: str = "defended documents") -> None:
    """Log what landed on disk: documents, total turns, file size."""
    turns = int(sum(len(row) for row in frame["turns"])) if len(frame) else 0
    print(f"wrote {len(frame):,} {kind} / {turns:,} turns -> {path} "
          f"({path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="swe_chat", choices=sorted(SOURCES),
                   help="which built split to defend (default: swe_chat)")
    p.add_argument("--defense", default="openanonymity", choices=sorted(DEFENSES),
                   help="registered defense to apply (default: openanonymity). 'none' copies the "
                        "split through unchanged, as a control")
    p.add_argument("--language", default=None,
                   help="optional filter: keep only documents whose language_primary is this "
                        "(default: defend every document). A filtered run writes a SUBSET of the "
                        "split under the same filename, so do not mix the two")
    p.add_argument("--dist-dir", default=None,
                   help="directory holding the built parquets to read (default: the project's "
                        "data/hf; point this at data/dist to defend a local, unpublished build "
                        "instead)")
    p.add_argument("--out-dir", default=None,
                   help="where to write the defended split (default: the project's data/dist, as "
                        "<split>_<defense>.parquet)")
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
    dist = Path(args.dist_dir) if args.dist_dir else hf_dir()
    cache = Path(args.cache_dir) if args.cache_dir else cache_dir()
    out_dir = Path(args.out_dir) if args.out_dir else dist_dir()
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

    # Only the columns this needs: the two that are written out, plus the one --language filters on.
    # `turns` is read separately, and only for this shard's rows, since it is the bulk of the split.
    frame = load_split(args.source, dist, columns=["doc_id", "author_id", "language_primary"])
    documents = select_documents(frame, language=language, limit=args.limit)
    if documents.empty:
        raise SystemExit(f"no documents in {args.source} match --language {args.language}.")
    selected = f" (language_primary == {language!r})" if language else " (all languages)"
    print(f"[{args.source}] {len(documents):,} of {len(frame):,} documents selected{selected}")

    doc_order = list(documents["doc_id"])  # split order, for the merge

    # A defense that cascades over an author's timeline gets WHOLE AUTHORS; everything else gets the
    # row-interleaved split. Both are deterministic given the selection and the shard count.
    registered = get_defense(args.defense)
    by_author = getattr(registered, "shardable_by", None) == "author"
    shard = (select_author_shard(documents, shard_index, num_shards) if by_author
             else select_shard(documents, shard_index, num_shards))
    out_path = merged_path if num_shards == 1 else shard_path(out_dir, stem, shard_index, num_shards)
    if num_shards > 1:
        split_by = f" ({shard['author_id'].nunique():,} whole authors)" if by_author else ""
        print(f"[shard {shard_index}/{num_shards}] defending {len(shard):,} of them{split_by} "
              f"-> {out_path.name}")

    # Selected over `documents` (the whole split), NOT over `shard` -- the point of it.
    reference = (reference_pool(args.defense, registered, documents, args.source, dist)
                 if num_shards > 1 and by_author else None)

    turn_lists = read_turns(args.source, dist, shard.index)
    defended = defend_documents(
        args.defense, shard["doc_id"], shard["author_id"], turn_lists,
        cache_dir=defense_cache_dir(cache, args.source, shard_index, num_shards),
        num_shards=num_shards, reference=reference,
    )
    documents_out = build_defended_frame(shard, defended)
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
