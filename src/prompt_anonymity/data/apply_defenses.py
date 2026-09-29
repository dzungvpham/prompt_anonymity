r"""Rewrite a built dataset split's conversations with a defense -- one defended parquet per defense.

Third companion to :mod:`prompt_anonymity.data.build_dataset` (which writes the documents) and
:mod:`prompt_anonymity.data.compute_features` (which writes their feature vectors). This script
sits between the two: it takes a built split, runs every conversation through one registered
**defense** -- the anonymization countermeasure a user would apply to their prompts before
releasing them -- and writes the defended conversations back out as a parquet with the *same
schema*, so everything downstream can read it exactly like the original split.

Defenses are **not reimplemented here** -- the script drives the registered defenses of the
installed package (``prompt_anonymity.defenses.DEFENSES``), so a newly registered defense becomes
available with no change to this file.

Where the output goes
---------------------

The split is READ from ``data/hf`` (the downloaded/published mirror) and a defended split is
WRITTEN to ``dist/<split>_<defense>.parquet``, carrying the split's own schema. Keeping writes out
of ``data/hf`` means running this locally never mutates the downloaded mirror.

This shares a namespace with the feature files ``compute_features`` writes
(``dist/swe_chat_gemini_embedding_2.parquet``): both are ``<split>_<name>.parquet``, distinguished on disk
by their columns -- a defended split has ``turns``, a feature file has ``doc_id``/``author_id``
plus feature columns.

To featurize defended text, point ``compute_features`` at the defended file with its ``--defense``
flag -- and, since that file lives in ``dist/`` rather than ``compute_features``'s own default
read location (``data/hf``), also point ``--dist-dir`` there::

    python -m prompt_anonymity.data.apply_defenses   --source swe_chat --defense openanonymity
    python -m prompt_anonymity.data.compute_features --source swe_chat --defense openanonymity \
        --feature gemini_embedding_2 --dist-dir data/dist   # reads swe_chat_openanonymity.parquet
                                                      # writes swe_chat_openanonymity_gemini_embedding_2.parquet

What a defense sees: one user turn at a time
--------------------------------------------

Each document is a list of user turns, and **each turn is defended on its own**, then the results
are re-assembled into a list of the same length -- so ``num_turns`` and the turn boundaries survive
the rewrite, and a document's defended turns line up one-for-one with its originals.

The one exception is a defense that *adds* turns rather than rewriting them (``frame_pad``, which
appends a shared off-topic turn to each document). Such a defense declares ``appends_turns = True``
and is applied per DOCUMENT rather than through the per-turn path below, which structurally cannot
add a turn since it must return one output per input row.

There are two ways to be such a defense, depending on whether it needs to see the document:

* ``extra_turns(doc_id)`` returns the turns to append. The existing turns are copied through
  byte-identical (``frame_pad``).
* ``rewrite_document(doc_id, turns)`` returns the document's whole new turn list, so it may shorten
  the existing turns as well as add one.

:func:`append_extra_turns` prefers the second when a defense offers it.

The defense machinery caches one row per input it is handed
(:class:`~prompt_anonymity.caching.IndexedRowCache`), so handing it turns rather than whole
conversations means a turn repeated across documents is computed once, and an interrupted run
resumes at turn granularity. Turn ids are ``<doc_id>#<n>``.

Cost, caching and re-runs
-------------------------

Every model-backed defense here (``styleremix``, ``qwen_rewrite``, ``dp_mlm``, ``openanonymity``)
runs locally and wants a GPU large enough to hold its model. The run prints how many documents,
turns and distinct turns it is about to defend before starting, and every defended turn is cached
under ``--cache-dir``: a re-run recomputes nothing.

The cache is scoped per split **and per shard layout** -- ``<cache-dir>/defended/<split>[/<i>-of-<n>]``
-- because the defense cache is one table per side, rewritten whole on each run: two shards sharing
one table would each drop the other's rows. Re-running an array with a *different* ``--num-shards``
therefore recomputes everything, so keep the array size fixed across re-runs.

Sharding (SLURM job arrays)
---------------------------

Sharding works exactly as in :mod:`~prompt_anonymity.data.compute_features` and shares its
implementation: ``--num-shards N --shard-index I`` defends every ``N``-th document, either flag is
filled in from ``SLURM_ARRAY_TASK_ID`` / ``SLURM_ARRAY_TASK_COUNT`` in an array job, shards land in
``dist/shards/``, and the last shard to finish concatenates them into the final split-ordered
parquet (or ``--merge`` does it on demand).

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
from prompt_anonymity.defenses.embad import (
    DEFAULT_GENERATIONS, DEFAULT_SEARCH_MAX_CHARS, DEFAULT_SEARCH_MIN_CHARS,
    DEFAULT_SEARCH_SAMPLES, DEFAULT_SEARCH_SOURCE, DEFAULT_TOPICS_PER_DOCUMENT,
    DEFAULT_VALIDATION_SAMPLES, MUTATORS, EmBadDefense)

from .compute_features import (
    READ_BATCH_ROWS,
    SOURCES,
    TURN_SEPARATOR,
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

# The defended file's columns. Just the key, its author, and the rewritten text -- everything else
# in the split (timestamps, language, model, agent) is unchanged by a defense, so it stays in
# <split>.parquet and is joined back on doc_id by whoever needs it.
DEFENDED_COLUMNS = ["doc_id", "author_id", "turns"]

# Where the per-turn defense cache is rooted under --cache-dir. The defense machinery adds its own
# ``defenses/<name>/<logic hash>/<params hash>/`` below this; this level adds the split (and shard),
# which that namespace does not carry.
CACHE_SUBDIR = "defended"

# Separator between a document id and a turn's position within it, forming the cache key for one
# turn. Defined in `defenses._backends` (imported above, and re-exported here) so a defense can
# read the doc_id back out of a row id without importing this module, which imports the defense
# registry.


# --- input ------------------------------------------------------------------

def read_turns(source: str, dist_dir: str | Path, positions) -> list[list[str]]:
    """The turn lists of these split row positions, in the order given.

    The list-of-turns counterpart of :func:`~prompt_anonymity.data.compute_features.read_texts`,
    and streaming for the same reason: ``turns`` is essentially the whole dataset, while a shard
    needs only a fraction of it. Reading a batch at a time and keeping only the wanted rows makes a
    task's memory scale with its shard rather than with the corpus.
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
    had. A length mismatch means a defense broke that contract, which would silently misalign every
    document after it, so it is raised rather than trimmed.

    A defense that means to add a turn cannot do it here: it declares ``appends_turns`` and is
    applied by :func:`append_extra_turns` instead.
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

    A defense caches its rows in one table per side, namespaced only by the defense's
    name/logic/params (:class:`~prompt_anonymity.caching.IndexedRowCache`), and rewrites that whole
    table on each run. Two runs that share a table would overwrite each other's rows -- which is
    exactly what defending two different splits, or two shards of one split in parallel, would do
    without this. The cost: re-running an array with a different ``--num-shards`` starts from an
    empty cache, but that's cheaper than silently losing another task's rows.
    """
    path = Path(cache_root) / CACHE_SUBDIR / source
    if num_shards > 1:
        path = path / f"{shard_index:04d}-of-{num_shards:04d}"
    return path


def as_attack_data(texts: list[str], ids: list[str], authors: list[str],
                   reference=None) -> AttackData:
    """Wrap a stream of turns as the :class:`~prompt_anonymity.core.AttackData` a defense takes.

    A defense's input type is the experiment's known/unknown bundle, but a defense only ever reads
    and rewrites **text** -- features are computed afterwards, by a separate stage. So the bundle
    carries the turns, their cache ids and authors, and zero-width embedding matrices: an explicit
    statement that no features exist yet, not placeholder vectors.

    Everything goes on the *unknown* side, the side a defense rewrites by default (the threat model
    being that the adversary's known conversations are already out). Defending a whole dataset
    normally has no known side, so that side is empty.

    ``reference``, when given, is ``(texts, ids, authors)`` of read-only context documents that go
    on the **known** side. Only an author-sharded run needs it: the task holds a fraction of the
    split but a defense calibrating against "a median unrelated document" must see the same
    reference pool as every other shard. Nothing on this side is rewritten or returned.
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

    Distinct turns is the number that matters for a paid or slow defense: identical turns are
    computed once, so it's the real call count for an uncached run.
    """
    distinct = len({text for text in texts if text.strip()})
    characters = sum(len(text) for text in texts)
    print(f"[{defense}] defending {n_documents:,} documents / {len(texts):,} turns "
          f"({distinct:,} distinct non-blank, {characters:,} characters); "
          f"cached turns are not recomputed")


def append_extra_turns(name: str, defense, doc_ids, turn_lists) -> list[list[str]]:
    """Apply a turn-ADDING defense, document by document.

    The path for a defense that declares ``appends_turns`` (``frame_pad``). It is document-level,
    skipping the per-turn machinery entirely: that path caches and requires one output per input
    turn, so it can't express "one more turn".

    Two hooks, in order of preference (see this module's docstring):

    * ``rewrite_document(doc_id, turns)`` -> the document's whole new turn list. A defense that must
      see the document uses this; it may shorten the existing turns as well as add one.
    * ``extra_turns(doc_id)`` -> the turns to append, with the existing ones copied through
      byte-identical. The defense never sees the document at all.

    The summary line says which happened, and counts characters as a signed delta rather than an
    addition, because under the first hook a document can come out SHORTER than it went in.
    """
    rewrite = getattr(defense, "rewrite_document", None)
    if callable(rewrite):
        padded = [[str(turn) for turn in rewrite(doc_id, list(turns))]
                  for doc_id, turns in zip(doc_ids, turn_lists)]
    else:
        padded = [list(turns) + [str(turn) for turn in defense.extra_turns(doc_id)]
                  for doc_id, turns in zip(doc_ids, turn_lists)]

    # Both counts are SIGNED deltas: a defense that shortens a document to make room for what it
    # appends can drive either negative.
    turns_delta = sum(len(new) - len(old) for new, old in zip(padded, turn_lists))
    delta = (sum(len(turn) for turns in padded for turn in turns)
             - sum(len(str(turn)) for turns in turn_lists for turn in turns))
    shortened = sum(list(new[:len(old)]) != [str(turn) for turn in old]
                    for new, old in zip(padded, turn_lists))
    print(f"[{name}] document-level pass over {len(padded):,} documents: {turns_delta:+,} turns, "
          f"{delta:+,} characters net; "
          + (f"{shortened:,} document(s) had their OWN turns shortened to make room"
             if shortened else "existing turns are copied through unchanged"))
    report = getattr(defense, "report", None)
    if callable(report):
        report()
    return padded


def append_optimized_turns(defense: str, registered, doc_ids, author_ids, turn_lists,
                           *, cache_dir) -> list[list[str]]:
    """Apply a turn-ADDING defense that has to READ the document first.

    **Unreached today** -- no registered defense currently declares ``needs_document`` (``embad``
    used to, before it switched to fitting one universal trigger). Kept as the contract a
    document-reading turn-adder would use: :func:`append_extra_turns` never shows the defense any
    text, while the per-turn path in :func:`defend_documents` shows text but structurally cannot add
    a turn.

    Hands the defense one row per **document** (id ``doc_id``, source the joined turns) and appends
    what comes back. Two consequences: the cached unit is a document, not a turn, so a repeated turn
    is optimized once per document rather than once overall; and what the defense returns is the
    turn to append, not a rewritten document.
    """
    from prompt_anonymity.defenses import apply_defense

    texts = [TURN_SEPARATOR.join(turns) for turns in turn_lists]
    ids = [str(doc_id) for doc_id in doc_ids]
    authors = [str(author) for author in author_ids]
    report_workload(defense, len(texts), texts)
    result = apply_defense(defense, as_attack_data(texts, ids, authors), cache_dir=cache_dir)
    extra = [str(turn) for turn in result.unknown_texts]
    if len(extra) != len(turn_lists):
        raise ValueError(f"defense {defense!r} returned {len(extra)} turns for "
                         f"{len(turn_lists)} documents.")
    padded = [list(turns) + ([turn] if turn else []) for turns, turn in zip(turn_lists, extra)]
    characters = sum(len(turn) for turn in extra)
    print(f"[{defense}] appending an optimized turn to {len(padded):,} documents: "
          f"{characters:,} characters added; existing turns are copied through unchanged")
    report = getattr(registered, "report", None)
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
    # author -- silently producing a plausible but wrong defended parquet. Refuse rather than trust
    # the operator to remember. `shardable_by = "author"` is the exemption: main() gives that
    # defense whole authors via `select_author_shard`.
    if (num_shards > 1 and not getattr(registered, "shardable", True)
            and getattr(registered, "shardable_by", None) != "author"):
        raise SystemExit(
            f"defense {defense!r} cannot be sharded: it measures each document against its "
            f"author's other documents, and --num-shards splits by document. Re-run without "
            f"--num-shards/--shard-index (or on a smaller --source)."
        )
    # A turn-adding defense works on documents, not on the per-turn stream. Two flavours: one that
    # never sees the text (`frame_pad`) and one that has to (`needs_document`); no registered
    # defense takes the second branch today, but it's kept as the contract for one that would.
    if getattr(registered, "appends_turns", False):
        if getattr(registered, "needs_document", False):
            return append_optimized_turns(defense, registered, doc_ids, author_ids, turn_lists,
                                          cache_dir=cache_dir)
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
    array calibrates against the same reference pool. Returns ``None`` for a defense that does not
    ask for one (every defense but ``afr``).
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
    columns are untouched by a defense and joined back on ``doc_id`` by whoever needs them.
    ``num_turns`` is recoverable from a per-turn rewrite, but not from a turn-ADDING defense
    (:func:`append_extra_turns`), whose documents carry more turns than the split records -- read
    the count from this file instead.
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


def configure_embad(args):
    """Return the defense instance for this run, rebuilding ``embad`` from the ``--embad-*`` flags.

    ``embad`` is the one defense whose behaviour a user is expected to steer from the command line,
    so its registry entry is **replaced for this process** with an instance carrying the flags.
    Everything downstream still resolves it by name, so nothing else has to know this happened.

    A non-embad defense is returned untouched, flags and all -- they are documented as ignored.
    """
    registered = get_defense(args.defense)
    if not isinstance(registered, EmBadDefense):
        return registered

    # Keep whatever the registry entry configured, override only what the user asked for on the
    # command line. `objective` and `aggregation` are deliberately registry-only, with no flag: the
    # objective is the whole difference between the embad variants, and a flag would let the
    # search's objective disagree with the filename it writes.
    configured = EmBadDefense(
        aggregation=registered.aggregation,
        objective=registered.objective_kind,
        mutator=args.embad_mutator,
        document_aggregation=args.embad_document_aggregation,
        generations=args.embad_generations,
        topics_per_document=args.embad_topics_per_document,
        search_source=args.embad_search_source,
        search_samples=args.embad_search_samples,
        validation_samples=args.embad_validation_samples,
        search_min_chars=args.embad_search_min_chars,
        search_max_chars=args.embad_search_max_chars,
        search_data_dir=Path(args.dist_dir) if args.dist_dir else None,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        seed=args.embad_seed,
    )
    DEFENSES[args.defense] = configured
    return configured


def merge_one_source(source: str, args, *, language, dist: Path, out_dir: Path) -> None:
    """Reassemble one split's shards into its final parquet.

    Builds no defense, so no model is loaded and no API key is needed. Deliberately never resolves
    sharding: a merge is about what an array already wrote, not about which shard this process
    would have computed.
    """
    documents = select_documents(
        load_split(source, dist, columns=["doc_id", "language_primary"]),
        language=language, limit=args.limit,
    )
    merged = merge_shards(out_dir, defended_stem(source, args.defense), list(documents["doc_id"]))
    if merged is None:
        raise SystemExit(f"cannot merge {source} yet: the shards listed above have not been "
                         f"defended. Re-run those array tasks, then merge again.")
    merged_path = output_path(out_dir, source, args.defense)
    write_parquet(merged, merged_path)
    report_written(merged, merged_path)


def defend_one_source(source: str, args, *, language, dist: Path, cache: Path, out_dir: Path,
                      shard_index: int, num_shards: int) -> None:
    """Defend one split, writing ``<split>_<defense>.parquet`` (or this run's shard of it).

    One call is one corpus. With several ``--source`` values the caller loops, safe because
    ``num_shards`` is then guaranteed to be 1 -- every source is defended whole.
    """
    merged_path = output_path(out_dir, source, args.defense)
    stem = defended_stem(source, args.defense)

    # `turns` is read separately, and only for this shard's rows, since it is the bulk of the split.
    frame = load_split(source, dist, columns=["doc_id", "author_id", "language_primary"])
    documents = select_documents(frame, language=language, limit=args.limit)
    if documents.empty:
        raise SystemExit(f"no documents in {source} match --language {args.language}.")
    selected = f" (language_primary == {language!r})" if language else " (all languages)"
    print(f"[{source}] {len(documents):,} of {len(frame):,} documents selected{selected}")

    doc_order = list(documents["doc_id"])  # split order, for the merge

    # A defense that cascades over an author's timeline gets WHOLE AUTHORS; everything else gets
    # the row-interleaved split.
    registered = configure_embad(args)
    by_author = getattr(registered, "shardable_by", None) == "author"
    shard = (select_author_shard(documents, shard_index, num_shards) if by_author
             else select_shard(documents, shard_index, num_shards))
    out_path = merged_path if num_shards == 1 else shard_path(out_dir, stem, shard_index, num_shards)
    if num_shards > 1:
        split_by = f" ({shard['author_id'].nunique():,} whole authors)" if by_author else ""
        print(f"[shard {shard_index}/{num_shards}] defending {len(shard):,} of them{split_by} "
              f"-> {out_path.name}")

    # Selected over `documents` (the whole split), NOT over `shard` -- the point of it.
    reference = (reference_pool(args.defense, registered, documents, source, dist)
                 if num_shards > 1 and by_author else None)

    turn_lists = read_turns(source, dist, shard.index)
    defended = defend_documents(
        args.defense, shard["doc_id"], shard["author_id"], turn_lists,
        cache_dir=defense_cache_dir(cache, source, shard_index, num_shards),
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", nargs="+", default=["swe_chat"], choices=sorted(SOURCES),
                   metavar="SOURCE",
                   help="which built split(s) to defend; several are defended one after another, "
                        "each to its own <split>_<defense>.parquet. Sharding is refused with more "
                        "than one, since the corpora differ in size by ~40x and one shard layout "
                        "cannot serve both (default: swe_chat)")
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

    embad = p.add_argument_group(
        "embad",
        "Ignored unless --defense is an embad variant. embad searches ONE universal trigger "
        "against a pool of documents from its own optimization corpus -- never from --source -- "
        "and appends that same turn to every defended document. These flags control the search; "
        "any of them changes the cache key. What the search is SCORED AGAINST is not a flag: pick "
        "embad (local ensemble), embad_summary (summary bottleneck) or embad_gemini (the target "
        "encoder, billed) with --defense, so the objective and the output filename can never "
        "disagree.")
    embad.add_argument("--embad-mutator", default="claude", choices=sorted(MUTATORS),
                       help="who writes the candidate mechanisms -- the search's mutation "
                            "operator. 'claude' is a hosted model (bills per call, does not "
                            "reproduce); 'local' is Qwen3-1.7B through vLLM (free, seeded, "
                            "replays exactly) (default: claude)")
    embad.add_argument("--embad-search-source", default=DEFAULT_SEARCH_SOURCE,
                       choices=sorted(SOURCES),
                       help=f"corpus the search draws its documents from, independent of --source "
                            f"(default: {DEFAULT_SEARCH_SOURCE})")
    embad.add_argument("--embad-search-samples", type=int, default=DEFAULT_SEARCH_SAMPLES,
                       help=f"documents every candidate is scored against "
                            f"(default: {DEFAULT_SEARCH_SAMPLES})")
    embad.add_argument("--embad-validation-samples", type=int,
                       default=DEFAULT_VALIDATION_SAMPLES,
                       help=f"held-out documents the winner is CHOSEN on, disjoint from the "
                            f"search pool; 0 takes the search's own best "
                            f"(default: {DEFAULT_VALIDATION_SAMPLES})")
    embad.add_argument("--embad-generations", type=int, default=DEFAULT_GENERATIONS,
                       help=f"search rounds (default: {DEFAULT_GENERATIONS})")
    embad.add_argument("--embad-topics-per-document", type=int,
                       default=DEFAULT_TOPICS_PER_DOCUMENT,
                       help=f"decoy subjects each document is scored under per round, averaged out "
                            f"before aggregation to buy precision rather than change what is "
                            f"measured (default: {DEFAULT_TOPICS_PER_DOCUMENT})")
    embad.add_argument("--embad-document-aggregation", default="mean", choices=("mean", "worst"),
                       help="how a candidate's per-document cosines become one fitness: 'mean' "
                            "asks for a trigger that works on average, 'worst' for one with no "
                            "bad document (default: mean)")
    embad.add_argument("--embad-search-min-chars", type=int, default=DEFAULT_SEARCH_MIN_CHARS,
                       help=f"shortest document the search will sample, excluding a degenerate "
                            f"regime where the appended turn would be most of the text "
                            f"(default: {DEFAULT_SEARCH_MIN_CHARS})")
    embad.add_argument("--embad-search-max-chars", type=int, default=DEFAULT_SEARCH_MAX_CHARS,
                       help=f"longest document the search will sample "
                            f"(default: {DEFAULT_SEARCH_MAX_CHARS})")
    embad.add_argument("--embad-seed", type=int, default=0,
                       help="seeds both the document sample and the search (default: 0)")
    args = p.parse_args()

    language = None if (args.language or "all").lower() == "all" else args.language
    # Deduplicated but order-preserving: naming a split twice is a typo, not a request to defend it
    # twice into the same file.
    sources = list(dict.fromkeys(args.source))
    dist = Path(args.dist_dir) if args.dist_dir else hf_dir()
    cache = Path(args.cache_dir) if args.cache_dir else cache_dir()
    out_dir = Path(args.out_dir) if args.out_dir else dist_dir()

    # --merge only reassembles what an array already computed -- no defense is built, so no model
    # is loaded and no API key is needed.
    if args.merge:
        for source in sources:
            merge_one_source(source, args, language=language, dist=dist, out_dir=out_dir)
        return

    shard_index, num_shards = resolve_sharding(args.shard_index, args.num_shards)
    # Checked on the RESOLVED count, not on the flags, so a multi-source run submitted into a SLURM
    # array is caught too.
    if num_shards > 1 and len(sources) > 1:
        p.error(f"--source names {len(sources)} splits and this run is shard {shard_index} of "
                f"{num_shards}. A shard layout is per-corpus -- WildChat is ~40x swe_chat, so one "
                f"count cannot serve both, and the defense cache is scoped per (split, layout) so "
                f"a wrong one costs a full recompute. Shard one source per job, or defend several "
                f"unsharded.")

    for source in sources:
        defend_one_source(source, args, language=language, dist=dist, cache=cache,
                          out_dir=out_dir, shard_index=shard_index, num_shards=num_shards)


if __name__ == "__main__":
    main()
