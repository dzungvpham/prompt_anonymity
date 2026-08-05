r"""Compute feature vectors for a built dataset split -- one parquet per featurizer.

Companion to :mod:`prompt_anonymity.data.build_dataset`: that script writes the documents
(``dist/swe_chat.parquet``), this one writes the *features* for them
(``dist/swe_chat_stylometrix.parquet``), keyed by ``doc_id`` so the two join cleanly.

Featurization is **not reimplemented here**. The script drives the registered featurizers of
the installed ``prompt_anonymity`` package (``prompt_anonymity.features.FEATURIZERS``) -- the
same registry the experiment runner uses -- so ``--feature`` accepts whatever that registry
exposes and a newly registered featurizer becomes available with no change to this file. It
supersedes the one-off ``swe-chat/stylometrix.py`` / ``wildchat/stylometrix.py`` scripts, which
called StyloMetrix directly.

Differences from those older scripts, worth knowing before comparing numbers:

* **Turns are joined with a blank line** (:data:`TURN_SEPARATOR`, ``"\n\n"``) rather than the
  older ``"\n===\n"`` marker, which is not natural writing and left its own mark on the features.
* The input is the unified dataset's cleaned, identifier-scrubbed ``turns``, not the raw
  per-source session text.
* Vectors are cached per document (content-addressed), so a re-run recomputes nothing and an
  interrupted run resumes from the last completed chunk.

**Every document in the split is featurized by default**, with one language model
(``--language-code``, default ``en``) -- as the older per-source scripts did. The model is a
property of the featurizer, not a filter: a Japanese document still gets a feature vector, just
one an English model computed. ``--language`` is a separate, optional filter for the times you
want only one language's documents, and makes the output a *subset* of the split; either way,
join the result back on ``doc_id``.

How much of each document is read is the ``--max-len`` window, set **per source**
(:data:`MAX_LEN_BY_SOURCE`): SWE-chat is read **in full**, while other sources keep the
featurizer's own default (StyloMetrix: the first 2,048 characters). The split is deliberate --
StyloMetrix costs grow faster than linearly with document length, and SWE-chat's tail is short
enough to absorb that (longest session ~117K characters) where WildChat's is not (longest
~953K, at spaCy's 1,000,000-character parser guard). SWE-chat's uncapped window is a decision
rather than a default: ``--max-len`` cannot truncate it (see :func:`resolve_max_len`), because
the feature file name records no window and a truncated run would quietly overwrite the full-text
vectors with a different feature space under the same name.

Featurization runs across worker processes where the featurizer supports it (StyloMetrix does),
sized from the CPUs and memory this job is actually allocated rather than the machine's -- see
``--workers`` and :mod:`prompt_anonymity.resources`.

**Remote (paid) featurizers.** ``--feature gemini_embedding_2`` embeds the documents through
OpenRouter instead of computing anything locally, so it needs no GPU but does need an
``OPENROUTER_API_KEY`` in the environment or a ``.env``, and it costs money -- about $0.32 for
all of SWE-chat at ``gemini-embedding-2``'s $0.20/M tokens. Three things follow:

* ``--workers`` there means *concurrent HTTP requests*, not processes.
* ``--dimensions`` trims the embedding (768 / 1536 instead of the native 3072), which mostly
  matters for the size of the output parquet.
* ``--task`` picks what Embedding 2 is told the vector is *for* (``clustering`` by
  default, or ``sentence similarity`` / ``classification``). It genuinely changes the vectors -- the same
  document under two tasks embeds to cosine ~0.88 -- so a non-default task is appended to the
  output filename (``swe_chat_gemini_embedding_2_clustering.parquet``) and caches separately,
  letting two tasks sit side by side for comparison. ``gemini_embedding_001`` has no working
  task mechanism and rejects the flag rather than ignoring it.
* The vector cache is worth more than usual: every cache miss is a paid call, so a re-run, a
  resumed run and a re-run with different sharding all cost nothing for documents already done.
  The run prints what it actually spent when it finishes.

``--max-len`` does not apply to it. The embedding model reads a fixed window (8,192 tokens), so
each document is cut to just above that (10,000) and embedded **once** -- one document, one call,
one vector, no pooling -- which also caps what any single document can cost, however long it is.
Each input carries the model's task prefix (``task: sentence similarity | query: ...``), which is
how Embedding 2 is told what the vector is for. The vector therefore represents a document's
opening, not all of it; see :mod:`prompt_anonymity.features.gemini_embedding`.

**Sharding (SLURM array jobs).** A whole source can be more than one job's worth of work
(WildChat is ~172K documents), so a run can be restricted to one *shard* of the split --
``--num-shards N --shard-index I`` -- and N such runs launched as a SLURM job array, each on its
own node and GPU. Either flag is filled in from the array task's own environment when omitted
(``SLURM_ARRAY_TASK_ID`` / ``SLURM_ARRAY_TASK_COUNT``), so ``scripts/compute_features_slurm.sh``
needs no per-task bookkeeping. A sharded run writes ``dist/shards/<split>_<feature>.<I>-of-<N>.parquet``
rather than the final file, and the last shard to finish concatenates them all into it (in split
order) -- so the array produces exactly the file a single unsharded run would have. ``--merge``
does that concatenation on demand, without a GPU, for when a task had to be re-run.

Shards are independent by construction: each vector depends only on its own document, and the
on-disk cache is content-addressed with atomic writes, so concurrent tasks share one cache safely
(a document featurized by an earlier run is a hit for whichever shard now owns it).

Run (from the repo root):

    python -m prompt_anonymity.data.compute_features                                  # all swe-chat docs, stylometrix/en
    python -m prompt_anonymity.data.compute_features --source wildchat --language-code ru
    python -m prompt_anonymity.data.compute_features --feature function_words --language English
    python -m prompt_anonymity.data.compute_features --workers 4                      # cap the worker pool
    python -m prompt_anonymity.data.compute_features --source wildchat --num-shards 32 --shard-index 0
    python -m prompt_anonymity.data.compute_features --source wildchat --num-shards 32 --merge
    python -m prompt_anonymity.data.compute_features --feature gemini_embedding_2    # paid API, default task type is clustering
"""

from __future__ import annotations

import argparse
import inspect
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from prompt_anonymity.features import FEATURIZERS, get_featurizer
from prompt_anonymity.resources import describe_budget

from .config import cache_dir, dist_dir, hf_dir

# The sources, which are also their split and parquet base names -- one spelling per corpus, so
# there is no mapping here any more, only a vocabulary. Kept as a literal rather than imported
# from ``build_dataset`` (which it must match): featurizing does not otherwise need the build
# pipeline, and importing it would pull the whole raw-source and language-detection stack in
# behind ``--help``.
SOURCES = ("wildchat", "swe_chat")

# A document's text is its turns joined by a blank line. The earlier per-source scripts joined
# with "\n===\n"; that marker is not natural writing and leaves its own fingerprint in the
# features, so the unified dataset uses an ordinary paragraph break instead.
TURN_SEPARATOR = "\n\n"

# Documents per featurizer call. Each call's vectors are written to the on-disk cache before the
# next starts, so a long run checkpoints as it goes: smaller chunks checkpoint more often, larger
# ones amortize per-call overhead.
CHUNK_SIZE = 256

# Rows per batch when streaming `turns` back out of the parquet (see `read_texts`). Only one batch
# is decoded at a time, so this bounds that read's memory rather than the shard's.
READ_BATCH_ROWS = 2048

# dataset `language_primary` -> language model code, used only to pick a sensible default model
# when a run is restricted to one language (--language). StyloMetrix ships models for these five
# languages only; anything else needs an explicit --language-code (if it is supported at all).
LANGUAGE_CODES = {"English": "en", "German": "de", "Polish": "pl", "Russian": "ru", "Ukrainian": "ukr"}
DEFAULT_LANGUAGE_CODE = "en"

# Input window per source, in characters, for featurizers that read a prefix (0 = read whole
# documents). SWE-chat is uncapped: its longest session is ~117K characters, so reading every
# session in full is affordable, and truncating discards exactly the long sessions that carry the
# most style evidence. A source absent from this map keeps the featurizer's own default
# (StyloMetrix: 2,048 characters) -- WildChat deliberately does, since its length tail reaches
# ~953K characters and StyloMetrix's cost grows faster than linearly with length. Featurizers that
# take no character window at all (gemini_embedding_2 measures its own, in tokens) ignore this.
#
# A window of 0 here is a **decision, not a default**: the source is uncapped by design and
# `--max-len` may not override it (see `resolve_max_len`). Since the output filename does not
# encode the window, a truncated run would otherwise quietly overwrite the uncapped feature file
# with a differently-computed one under the same name. To genuinely change a source's window,
# change it here.
MAX_LEN_BY_SOURCE = {"swe_chat": 0}

# Where a sharded run parks its partial outputs, under the output directory. They are the array
# job's intermediate state, not a deliverable -- once merged, the shard files can be deleted.
SHARD_SUBDIR = "shards"

# Shard filename suffix: ``.<index>-of-<count>.parquet``, zero-padded so `ls` sorts them in order.
SHARD_SUFFIX = re.compile(r"\.(\d+)-of-(\d+)\.parquet$")

# Featurizers that cannot be sharded, because they *fit* on the documents they are given (a TF-IDF
# vocabulary and an SVD basis) instead of computing each vector independently. Sharding one would
# fit a different feature space per shard, and concatenating those would produce a file whose
# columns mean different things in different rows -- so it is refused rather than approximated.
# These are the same featurizers that require `--chunk-size 0`.
UNSHARDABLE_FEATURES = {"char_ngram_tfidf"}


# --- input ------------------------------------------------------------------

def split_stem(source: str, defense: str | None = None) -> str:
    """The document file's name without its suffix: ``swe_chat``, or ``swe_chat_<defense>``.

    A defense writes its output to ``dist/`` (:mod:`prompt_anonymity.data.apply_defenses`), so
    featurizing defended text is a matter of reading that file instead (point ``--dist-dir`` at
    ``dist/`` too) -- same schema, same row order, one name apart.
    """
    return f"{source}_{defense}" if defense else source


def split_path(source: str, dist_dir: str | Path, defense: str | None = None) -> Path:
    """Path of the document parquet to featurize -- the split, or a defended version of it."""
    return Path(dist_dir) / f"{split_stem(source, defense)}.parquet"


def load_split(source: str, dist_dir: str | Path, columns: list[str] | None = None,
               defense: str | None = None) -> pd.DataFrame:
    """Read the built parquet for ``source`` (``swe_chat`` -> ``swe_chat.parquet``).

    ``columns`` reads a subset of them, which is what makes ``--merge`` cheap: merging only needs
    each document's id and language, never the ``turns`` that dominate the file's size. ``defense``
    reads that defense's defended version of the split instead.
    """
    path = split_path(source, dist_dir, defense)
    if not path.exists():
        hint = (f"run `python -m prompt_anonymity.data.apply_defenses --source {source} "
                f"--defense {defense}` first." if defense else
                "build it first with `python -m prompt_anonymity.data.build_dataset`.")
        raise SystemExit(f"{path} not found -- {hint}")
    return pd.read_parquet(path, columns=columns)


def load_documents(source: str, dist_dir: str | Path, columns: list[str],
                   defense: str | None = None) -> pd.DataFrame:
    """The documents to featurize, with the metadata this script needs, defended or not.

    A defended file (:mod:`prompt_anonymity.data.apply_defenses`) carries only ``doc_id``,
    ``author_id`` and the rewritten ``turns`` -- a defense changes nothing else, so the rest stays
    in ``<split>.parquet`` rather than being duplicated. Any other requested column is therefore
    joined back from the undefended split on ``doc_id``, which is also what keeps this correct for
    a *subset* of the split (a ``--language`` or ``--limit`` run) rather than assuming the two files
    are row-aligned. The frame's index stays each row's position in the file being featurized, which
    is what :func:`read_texts` reads back against.
    """
    if defense is None:
        return load_split(source, dist_dir, columns=columns, defense=None)
    available = set(pq.ParquetFile(split_path(source, dist_dir, defense)).schema_arrow.names)
    frame = load_split(source, dist_dir, columns=[c for c in columns if c in available],
                       defense=defense)
    missing = [c for c in columns if c not in available]
    if missing:
        base = load_split(source, dist_dir, columns=["doc_id", *missing]).set_index("doc_id")
        for column in missing:
            frame[column] = frame["doc_id"].map(base[column])
    return frame[columns]


def select_documents(frame: pd.DataFrame, language: str | None = None,
                     limit: int | None = None) -> pd.DataFrame:
    """Pick the documents to featurize, keeping the split's row order.

    ``language`` keeps only documents whose ``language_primary`` matches it; the default
    (``None``) featurizes **every** document, whatever language it is in, since the featurizer's
    language model is a choice about how to read text rather than a reason to drop documents.
    ``limit`` truncates to the first N documents (for smoke tests).

    The index is deliberately left as each document's **row position in the split**, which is what
    :func:`read_texts` needs to go back to the parquet for the text of just these rows.
    """
    if language:
        frame = frame[frame["language_primary"] == language]
    if limit:
        frame = frame.head(limit)
    return frame


def read_texts(source: str, dist_dir: str | Path, positions, defense: str | None = None) -> list[str]:
    """Document text (``turns`` joined with :data:`TURN_SEPARATOR`) for these split row positions.

    Streams the ``turns`` column a batch at a time and materializes only the wanted rows, instead
    of reading the split and indexing into it. ``turns`` *is* essentially the whole dataset --
    WildChat's parquet is 412 MB on disk and ~3.8 GiB once pandas has turned it into Python lists
    of strings -- while an array task needs 1/N of it, so this makes a task's memory scale with its
    shard rather than with the corpus. That is the difference between fitting alongside the
    featurizer's worker processes in a small allocation and being OOM-killed by the scheduler
    (:meth:`~prompt_anonymity.features.stylometrix.StyloMetrixFeaturizer.resolve_workers` budgets
    only :data:`~prompt_anonymity.features.stylometrix.PARENT_MEMORY_RESERVE_BYTES` for this
    process).
    """
    path = split_path(source, dist_dir, defense)
    wanted = {int(position) for position in positions}
    texts: dict[int, str] = {}
    first_row = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=READ_BATCH_ROWS, columns=["turns"]):
        column = batch.column(0)
        for offset in range(len(batch)):
            if first_row + offset in wanted:
                texts[first_row + offset] = TURN_SEPARATOR.join(column[offset].as_py())
        first_row += len(batch)
    return [texts[int(position)] for position in positions]


# --- sharding (SLURM job arrays) --------------------------------------------

def slurm_array_shard() -> tuple[int | None, int | None]:
    """``(shard_index, num_shards)`` read from this SLURM array task, either possibly ``None``.

    Lets a job array script name its shard layout once (in the ``#SBATCH --array=`` line) instead
    of also threading a task id through to every invocation. **The array is taken to be 0-based**:
    the task id is the shard index as-is, and the count is ``SLURM_ARRAY_TASK_COUNT`` (falling back
    to the id range). A 1-based array's last task then asks for a shard that does not exist and
    :func:`resolve_sharding` rejects it -- deliberately, because the alternative (numbering from
    ``SLURM_ARRAY_TASK_MIN``) reads ``sbatch --array=7``, the way one failed task of an array is
    re-run, as "shard 0 of 1" and silently recomputes the whole split under the final output's
    name. Failing loudly on an unusual array beats quietly computing the wrong shard on a routine
    re-run.

    Both entries are ``None`` outside a job array (an ordinary ``sbatch``, ``srun`` or laptop run),
    which leaves the run unsharded. Either can be overridden by its flag, and a re-run of part of
    an array needs ``--num-shards`` to be passed, since the environment then describes only the
    tasks actually queued.
    """
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task is None:
        return None, None
    count = os.environ.get("SLURM_ARRAY_TASK_COUNT")
    last = os.environ.get("SLURM_ARRAY_TASK_MAX")
    total = int(count) if count else (int(last) + 1 if last else None)
    return int(task), total


def resolve_sharding(shard_index: int | None, num_shards: int | None) -> tuple[int, int]:
    """Settle which shard this run computes: explicit flags first, the SLURM array second.

    Returns ``(0, 1)`` -- the whole split, one shard -- when neither is given and this is not an
    array task, so an ordinary run is unaffected. A flag always wins over the environment, and a
    half-specified shard (an index with no count, or a count with no array task to take the index
    from) is an error rather than a guess, since either guess would produce a wrongly-named file.
    """
    detected_index, detected_count = slurm_array_shard()
    index = shard_index if shard_index is not None else detected_index
    total = num_shards if num_shards is not None else detected_count
    if index is None and total is None:
        return 0, 1
    if total is None:
        raise SystemExit("--shard-index needs --num-shards (how many shards the split is cut into).")
    if index is None:
        raise SystemExit("--num-shards needs --shard-index outside a SLURM array job "
                         "(no SLURM_ARRAY_TASK_ID in the environment to take it from).")
    if total < 1 or not 0 <= index < total:
        raise SystemExit(f"shard {index} is out of range for {total} shard(s) -- expected an index "
                         f"in [0, {total - 1}]. In an array job, check that the #SBATCH --array "
                         f"range matches --num-shards.")
    return index, total


def select_shard(documents: pd.DataFrame, shard_index: int, num_shards: int) -> pd.DataFrame:
    """This shard's documents: every ``num_shards``-th row starting at ``shard_index``.

    Interleaved rather than cut into contiguous blocks, for wall-clock balance. Featurization cost
    grows with document length, and the split is ordered (by source, then broadly by time), so
    contiguous blocks hand one task all of a long-document region and leave the array waiting on
    it; taking every N-th row spreads any such run across all shards. Which documents land in
    which shard does not otherwise matter -- the merge restores split order by ``doc_id``.

    The index still carries each document's split row position, for :func:`read_texts`.
    """
    return documents.iloc[shard_index::num_shards]


# --- featurization ----------------------------------------------------------

def resolve_max_len(source: str, requested: int | None) -> int | None:
    """The input window to featurize ``source`` with, refusing to truncate an uncapped source.

    ``--max-len`` (``requested``) normally wins over the per-source default. The exception is a
    source pinned to ``0`` in :data:`MAX_LEN_BY_SOURCE` -- uncapped by design -- where a truncated
    run is rejected rather than performed: the output filename carries no window, so it would
    overwrite that source's full-text feature file with vectors computed over a prefix, leaving
    two incompatible feature spaces indistinguishable on disk. Passing ``--max-len 0`` explicitly
    is still fine (it asks for what the source already does).
    """
    default = MAX_LEN_BY_SOURCE.get(source)
    if requested is None:
        return default
    if default == 0 and requested:
        raise SystemExit(
            f"--max-len {requested} refused: {source} is featurized over whole documents by "
            f"design, and the feature file name does not record the window, so a truncated run "
            f"would silently replace the uncapped vectors. Change MAX_LEN_BY_SOURCE in "
            f"prompt_anonymity/data/compute_features.py if the source's window should really change."
        )
    return requested


def build_featurizer(name: str, **options):
    """Build a registered featurizer, forwarding only the options its constructor accepts.

    The featurizer class comes from ``prompt_anonymity.features.FEATURIZERS``, so this script
    never hard-codes a feature implementation -- but featurizers do not take the same arguments
    (StyloMetrix has a language model and an input window; the embedding one has an output width
    and no language model; the surface-statistic ones take nothing). Options are therefore matched
    against the constructor's signature rather than a hard-coded list of names, and an option left
    as ``None`` is dropped so the featurizer keeps its own default. A flag a featurizer does not
    accept is silently ignored, which is what lets one CLI drive all of them.
    """
    featurizer_class = FEATURIZERS.get(name)
    accepted = inspect.signature(featurizer_class).parameters if featurizer_class is not None else {}
    return get_featurizer(name, **{key: value for key, value in options.items()
                                   if value is not None and key in accepted})


def report_window(featurizer, texts) -> None:
    """Log how much of each document the featurizer will actually read, when it reads a prefix.

    StyloMetrix reads the first ``max_len`` characters of a document (``0`` = all of it), so on a
    corpus of long agent sessions a narrow window can leave most of the text unread -- better
    seen in the run log than discovered later. A featurizer whose window is measured in *tokens*
    (``input_tokens``, the embedding ones) reports that instead: counting tokens for the whole
    split up front would be its own pass over the corpus, so the count of documents actually cut
    is left to the featurizer's own end-of-run summary.
    """
    parameters = featurizer.params()
    if not texts:
        return
    if parameters.get("input_tokens"):
        print(f"[{featurizer.name}] reads the first {parameters['input_tokens']:,} tokens of each "
              f"document (longest document: {max(len(text) for text in texts):,} characters)")
        return
    max_len = parameters.get("max_len")
    if max_len is None:
        return
    longest = max(len(text) for text in texts)
    if not max_len:
        print(f"[{featurizer.name}] reads every document in full "
              f"(longest: {longest:,} characters)")
        return
    n_long = sum(len(text) > max_len for text in texts)
    print(f"[{featurizer.name}] reads the first {max_len:,} characters of each document; "
          f"{n_long:,}/{len(texts):,} ({n_long / len(texts):.1%}) documents are longer than that "
          f"(longest: {longest:,})")


def compute_features(featurizer, texts, *, cache_dir, chunk_size: int = CHUNK_SIZE) -> np.ndarray:
    """Featurize ``texts`` in cache-checkpointed chunks; returns ``(len(texts), n_features)``.

    Vectors are cached on disk per document -- content-addressed, namespaced by the featurizer's
    name, source version and parameters (see :mod:`prompt_anonymity.caching`) -- so a re-run
    recomputes nothing and an interrupted run resumes from the last completed chunk. Pass
    ``chunk_size <= 0`` to featurize everything in a single call, which is required by
    featurizers that *fit* on the first batch they see (``char_ngram_tfidf``), where chunking
    would fit on the first chunk alone.
    """
    if not texts:
        raise ValueError("no documents to featurize.")
    cache = featurizer.open_cache(cache_dir)
    step = chunk_size if chunk_size and chunk_size > 0 else len(texts)
    chunks, computed, reused = [], 0, 0
    for start in tqdm(range(0, len(texts), step), desc=f"featurizing ({featurizer.name})"):
        chunks.append(featurizer.transform(texts[start:start + step], cache))
        computed, reused = computed + cache.misses, reused + cache.hits
    print(f"[{featurizer.name}] {computed:,} vectors computed, {reused:,} served from {cache.dir}")
    return np.concatenate(chunks, axis=0)


# --- output -----------------------------------------------------------------

def stylometrix_column_names(language_code: str) -> list[str] | None:
    """StyloMetrix's own metric codes (``POS_VERB``, ``L_TTR``, ...) in ``transform`` order.

    Read from the installed StyloMetrix's metric registry, which needs no spaCy model load:
    iterating a ``MetricGroup`` yields its metrics sorted by metric id, which is exactly the
    column order ``StyloMetrix.transform`` emits (verified against the header of the previously
    committed ``swe-chat/swe_chat_stylometrix_en_2048.csv``). Returns ``None`` when StyloMetrix
    is not installed or does not know the language, so the caller falls back to positional names.
    """
    try:
        from stylo_metrix.structures import Lang
        return [metric.code for metric in Lang.get_language(language_code).get_metrics()]
    except Exception:  # not installed, or a bare Exception for an unknown language definition
        return None


def positional_feature_names(n_features: int) -> list[str]:
    """``f0000``-style names for a feature space whose dimensions have no meaning of their own.

    Every name is zero-padded to the width of the **largest index**, inferred from the count --
    3072 features give ``f0000 .. f3071``, 196 give ``f000 .. f195``. Uniform width is the point:
    with ragged padding, sorting the column names (``f1000`` before ``f999``) silently permutes
    the dimensions, and a reader who sorts is not doing anything unreasonable.

    The names are deliberately *not* prefixed with the featurizer: an embedding dimension carries
    no interpretation, and the file it lives in already says which featurizer (and task) produced
    it. Two feature files therefore share column names by design -- join them side by side with an
    explicit suffix.
    """
    width = len(str(max(n_features - 1, 0)))
    return [f"f{i:0{width}d}" for i in range(n_features)]


def feature_column_names(featurizer, n_features: int) -> list[str]:
    """Column names for the feature matrix -- the featurizer's own wherever they are recoverable.

    Named columns keep the parquet self-describing (``POS_VERB`` rather than ``f0007``) for the
    featurizers whose dimensions mean something. A featurizer may publish them directly via an
    optional ``feature_names()`` method; otherwise the two whose layout is public are resolved
    here. Anything else -- an embedding, say -- and any name list whose length disagrees with the
    computed matrix falls back to :func:`positional_feature_names`, so the vectors stay the source
    of truth and can never be silently mislabeled.
    """
    names = None
    published = getattr(featurizer, "feature_names", None)
    if callable(published):
        names = published()
    elif featurizer.name == "stylometrix":
        names = stylometrix_column_names(featurizer.language_code)
    elif featurizer.name == "function_words":
        from prompt_anonymity.features.function_words import FUNCTION_WORDS
        names = list(FUNCTION_WORDS)
    if names is not None and len(names) != n_features:
        print(f"WARNING: {featurizer.name} reported {len(names)} feature names for {n_features} "
              f"computed features; falling back to positional names.")
        names = None
    return list(names) if names else positional_feature_names(n_features)


def build_feature_frame(doc_ids, author_ids, vectors: np.ndarray, columns: list[str]) -> pd.DataFrame:
    """``doc_id``, ``author_id``, then one float32 column per feature dimension, in split order.

    Carrying ``author_id`` alongside the key makes the feature file usable on its own for the
    authorship work it feeds -- grouping, per-author splits, attribution -- with no join back to
    the document table. It is copied verbatim from the split, so the two stay consistent.

    float32 is the precision the featurizers actually produce, so this narrows the file without
    discarding anything.
    """
    frame = pd.DataFrame(np.asarray(vectors, dtype=np.float32), columns=columns)
    frame.insert(0, "doc_id", list(doc_ids))
    frame.insert(1, "author_id", list(author_ids))
    return frame


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write ``frame`` to ``path`` atomically: a temp file beside it, then a rename.

    Array tasks write their shards while a sibling task may already be merging, and two tasks
    that finish together may both write the merged file. A rename is atomic on a POSIX
    filesystem, so a reader sees either no file or a complete one -- never a half-written parquet.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".parquet.tmp")
    os.close(fd)
    try:
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def feature_label(feature: str, task: str | None) -> str:
    """The name the output files carry: the featurizer, plus ``--task`` when it is not the default.

    A featurizer whose vectors depend on a task (``gemini_embedding_2``, whose task is written
    into the text) produces a *different feature space* per task, so two tasks must not land in
    one file. Only a **non-default** task is appended, which keeps the default run writing the
    plain ``<split>_<feature>.parquet`` that already exists and that the experiment runners name;
    asking for the default task explicitly is not a different file. The task is slugified
    (``"sentence similarity"`` -> ``sentence_similarity``) so the name stays a plain identifier.

    Takes the default from the registered *class*, so ``--merge`` -- which never builds a
    featurizer -- labels its files exactly like the run that wrote them.
    """
    default = getattr(FEATURIZERS.get(feature), "default_task", None)
    if task is None or task == default:
        return feature
    slug = re.sub(r"[^a-z0-9]+", "_", task.lower()).strip("_") if task else "no_task"
    return f"{feature}_{slug}"


def output_path(out_dir: str | Path, source: str, feature: str, defense: str | None = None) -> Path:
    """The merged feature file for a source/featurizer -- what an unsharded run writes.

    Features of defended text carry the defense in the name too
    (``swe_chat_openanonymity_stylometrix.parquet``), so a defended run's vectors never overwrite
    the undefended ones.
    """
    return Path(out_dir) / f"{split_stem(source, defense)}_{feature}.parquet"


def shard_path(out_dir: str | Path, stem: str, shard_index: int, num_shards: int) -> Path:
    """One shard's partial output file, under :data:`SHARD_SUBDIR`.

    ``stem`` is the merged file's name without its suffix (``swe_chat_stylometrix`` here; the
    defense pipeline in :mod:`~prompt_anonymity.data.apply_defenses` shards by the same rules with
    its own stem). The shard count is part of the name so that shards of a re-run with a different
    array size cannot be mistaken for each other (see :func:`discover_shards`), and both numbers
    are zero-padded so a directory listing sorts in shard order.
    """
    return Path(out_dir) / SHARD_SUBDIR / f"{stem}.{shard_index:04d}-of-{num_shards:04d}.parquet"


def discover_shards(out_dir: str | Path, stem: str) -> tuple[int, dict[int, Path]]:
    """``(num_shards, {shard_index: path})`` for the shard files on disk; ``(0, {})`` if none.

    The shard count is read back from the filenames rather than taken from the caller, so
    ``--merge`` needs no ``--num-shards`` and can never merge N shards under the belief there
    were M. Finding two different counts means shards of two different array sizes are sitting in
    the directory together -- an incoherent mix that would merge into a wrong file -- so it is a
    hard error asking for the stale ones to be removed.
    """
    directory = Path(out_dir) / SHARD_SUBDIR
    found: dict[int, Path] = {}
    counts: set[int] = set()
    for path in sorted(directory.glob(f"{stem}.*-of-*.parquet")):
        match = SHARD_SUFFIX.search(path.name)
        if not match:
            continue
        index, total = int(match.group(1)), int(match.group(2))
        counts.add(total)
        found[index] = path
    if len(counts) > 1:
        raise SystemExit(
            f"{directory} holds shards of {len(counts)} different array sizes "
            f"({', '.join(str(c) for c in sorted(counts))}) for {stem} -- they cannot be merged "
            f"together. Delete the stale ones and re-run that array."
        )
    return (counts.pop() if counts else 0), found


def merge_shards(out_dir: str | Path, stem: str, doc_order: list) -> pd.DataFrame | None:
    """Concatenate every shard file into one frame in split order, or ``None`` if any is missing.

    ``doc_order`` is the ``doc_id`` of each selected document, in the order the split has them, so
    the merged file is row-for-row what a single unsharded run would have written. Returning
    ``None`` for an incomplete set is what lets every task try to merge after writing its shard:
    all but the last one find work outstanding and simply do nothing.

    The merged documents are checked against ``doc_order`` as a set: a mismatch means the shards
    were computed over a different selection than this run is merging (a different ``--language``
    or ``--limit``, or a split rebuilt since), which would otherwise produce a quietly wrong file.
    """
    num_shards, found = discover_shards(out_dir, stem)
    if not num_shards:
        raise SystemExit(f"no shard files found in {Path(out_dir) / SHARD_SUBDIR} for {stem}.")
    missing = [i for i in range(num_shards) if i not in found]
    if missing:
        print(f"[merge] {len(found)}/{num_shards} shards present; still missing "
              f"{', '.join(str(i) for i in missing[:10])}{' ...' if len(missing) > 10 else ''}")
        return None

    frames = [pd.read_parquet(found[i]) for i in range(num_shards)]
    # Shards with no documents (more shards than documents) carry no feature columns, so they are
    # dropped before concatenating rather than widening the frame with all-null columns.
    merged = pd.concat([frame for frame in frames if len(frame)], ignore_index=True)
    if len(merged) != len(doc_order) or set(merged["doc_id"]) != set(doc_order):
        raise SystemExit(
            f"the {num_shards} shards hold {len(merged):,} documents but this run selected "
            f"{len(doc_order):,} -- they were computed over a different selection. Re-run the "
            f"array with matching options, or delete {Path(out_dir) / SHARD_SUBDIR}."
        )
    return merged.set_index("doc_id").reindex(doc_order).reset_index()


def report_written(frame: pd.DataFrame, path: Path, kind: str = "features") -> None:
    """Log what landed on disk: rows x feature dimensions (the two key columns are not features)."""
    print(f"wrote {len(frame):,} x {max(frame.shape[1] - 2, 0)} {kind} -> {path} "
          f"({path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="swe_chat", choices=sorted(SOURCES),
                   help="which built split to featurize (default: swe_chat)")
    p.add_argument("--feature", default="stylometrix", choices=sorted(FEATURIZERS),
                   help="registered featurizer to run (default: stylometrix)")
    p.add_argument("--language", default=None,
                   help="optional filter: keep only documents whose language_primary is this "
                        "(default: featurize every document; the output filename is the same "
                        "either way, so a filtered run overwrites an unfiltered one)")
    p.add_argument("--language-code", default=None,
                   help="model code for language-specific featurizers, e.g. en/ru (default: "
                        f"derived from --language, else {DEFAULT_LANGUAGE_CODE!r})")
    p.add_argument("--max-len", type=int, default=None,
                   help="characters of each document to read, for featurizers that read a prefix; "
                        "0 reads whole documents (default: the source's window -- swe_chat is "
                        "uncapped by design and refuses to be truncated here; other sources keep "
                        "the featurizer's own default, 2048 for StyloMetrix)")
    p.add_argument("--task", default=None,
                   help="task for featurizers that embed with one (gemini_embedding_2: "
                        "'clustering' (default), 'sentence similarity', 'classification'). A "
                        "non-default task is appended to the output filename, so tasks never "
                        "overwrite each other; featurizers with no task mechanism reject it")
    p.add_argument("--dimensions", type=int, default=None,
                   help="output width for embedding featurizers that support truncation "
                        "(gemini_embedding_2: 128-3072, e.g. 768/1536; default: the model's native 3072). "
                        "Ignored by featurizers with a fixed feature space")
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                   help=f"documents per featurizer call / cache checkpoint; 0 runs one call "
                        f"(default: {CHUNK_SIZE})")
    p.add_argument("--workers", type=int, default=None,
                   help="worker processes for featurizers that support it (StyloMetrix), or "
                        "concurrent API requests for remote ones (gemini_embedding_2); 1 runs "
                        "in-process (default: sized from the allocation's CPUs/memory, or free "
                        "GPU memory when running on a GPU)")
    p.add_argument("--defense", default=None,
                   help="featurize a DEFENDED version of the split instead: reads "
                        "<split>_<defense>.parquet (written by apply_defenses) and writes "
                        "<split>_<defense>_<feature>.parquet, so defended and undefended vectors "
                        "never overwrite each other")
    p.add_argument("--dist-dir", default=None,
                   help="directory holding the built parquets to read (default: the project's "
                        "data/hf; point this at data/dist to featurize a local, unpublished build "
                        "or defended split instead)")
    p.add_argument("--out-dir", default=None,
                   help="where to write the feature parquet (default: the project's data/dist)")
    p.add_argument("--cache-dir", default=None,
                   help="on-disk cache for computed vectors, regenerable (default: data/.cache)")
    p.add_argument("--limit", type=int, default=None,
                   help="testing: featurize only the first N selected documents")
    p.add_argument("--num-shards", type=int, default=None,
                   help="cut the selected documents into this many shards and compute only one of "
                        "them, for running a SLURM job array (default: SLURM_ARRAY_TASK_COUNT in "
                        "an array job, else no sharding)")
    p.add_argument("--shard-index", type=int, default=None,
                   help="which shard (0-based) this run computes (default: this array task's "
                        "SLURM_ARRAY_TASK_ID)")
    p.add_argument("--merge", action="store_true",
                   help="do not featurize: concatenate the shard files an array already wrote "
                        "into the final feature parquet (needs no GPU). Sharded runs also try "
                        "this automatically once they are the last shard to finish")
    p.add_argument("--no-auto-merge", action="store_true",
                   help="a sharded run writes only its shard, leaving the merge to an explicit "
                        "--merge")
    args = p.parse_args()

    language = None if (args.language or "all").lower() == "all" else args.language
    language_code = args.language_code or LANGUAGE_CODES.get(language, DEFAULT_LANGUAGE_CODE)
    if args.language_code is None and language and language not in LANGUAGE_CODES:
        print(f"note: no language model mapped for {language!r}; using {language_code!r} "
              f"(set --language-code to override)")

    # Paths default to the project's data/ folder (see prompt_anonymity.data.config): the code
    # lives in the installed package, the data does not.
    dist = Path(args.dist_dir) if args.dist_dir else hf_dir()
    cache = Path(args.cache_dir) if args.cache_dir else cache_dir()
    out_dir = Path(args.out_dir) if args.out_dir else dist_dir()
    # Output files are named for the feature *and* a non-default --task, since a task changes the
    # vectors: two tasks are two feature spaces and must not share a filename.
    label = feature_label(args.feature, args.task)
    merged_path = output_path(out_dir, args.source, label, args.defense)
    stem = merged_path.stem  # what the shard files are named after
    selected = f" (language_primary == {language!r})" if language else " (all languages)"

    # --merge only reassembles what an array already computed: no featurizer, no GPU, and only the
    # two columns the selection needs -- never the `turns` that make the split large.
    if args.merge:
        documents = select_documents(
            load_documents(args.source, dist, ["doc_id", "language_primary"], args.defense),
            language=language, limit=args.limit,
        )
        merged = merge_shards(out_dir, stem, list(documents["doc_id"]))
        if merged is None:
            raise SystemExit("cannot merge yet: the shards listed above have not been computed. "
                             "Re-run those array tasks, then merge again.")
        write_parquet(merged, merged_path)
        report_written(merged, merged_path)
        return

    shard_index, num_shards = resolve_sharding(args.shard_index, args.num_shards)
    if num_shards > 1 and args.feature in UNSHARDABLE_FEATURES:
        raise SystemExit(f"--num-shards cannot be used with {args.feature}: it fits its feature "
                         f"space (vocabulary, SVD basis) on the documents it is given, so each "
                         f"shard would produce a different one. Run it unsharded.")

    max_len = resolve_max_len(args.source, args.max_len)
    featurizer = build_featurizer(args.feature, language_code=language_code, max_len=max_len,
                                  workers=args.workers, dimensions=args.dimensions, task=args.task)
    # Which model this run uses: a language model for the local featurizers, a remote model id
    # for the API-backed ones (whose `model` attribute is an OpenRouter id).
    if getattr(featurizer, "language_code", None):
        model = f" (language model {language_code!r})"
    elif getattr(featurizer, "model", None):
        task = getattr(featurizer, "task", "")
        model = f" (model {featurizer.model!r}{f', task {task!r}' if task else ''})"
    else:
        model = ""
    print(f"[{args.feature}] featurizer ready{model}")
    print(f"[resources] {describe_budget()}")

    # Only the key/filter columns, never `turns`: this process has to leave room for the worker
    # pool, and an array task reads back the text of just its own shard (see `read_texts`).
    frame = load_documents(args.source, dist,
                           ["doc_id", "author_id", "language_primary"], args.defense)
    documents = select_documents(frame, language=language, limit=args.limit)
    if documents.empty:
        raise SystemExit(f"no documents in {args.source} match --language {args.language}.")
    print(f"[{args.source}] {len(documents):,} of {len(frame):,} documents selected{selected}")

    doc_order = list(documents["doc_id"])  # split order, for the merge
    shard = select_shard(documents, shard_index, num_shards)
    out_path = merged_path if num_shards == 1 else shard_path(
        out_dir, stem, shard_index, num_shards)
    if num_shards > 1:
        print(f"[shard {shard_index}/{num_shards}] featurizing {len(shard):,} of them "
              f"-> {out_path.name}")

    texts = read_texts(args.source, dist, shard.index, defense=args.defense)

    if not texts:
        # More shards than documents: nothing to compute, but the (empty) shard file still has to
        # exist for the merge to see a complete set.
        write_parquet(shard[["doc_id", "author_id"]], out_path)
        print(f"wrote an empty shard file (no documents in this shard) -> {out_path}")
    else:
        report_window(featurizer, texts)
        try:
            vectors = compute_features(featurizer, texts, cache_dir=cache,
                                       chunk_size=args.chunk_size)
        finally:  # release worker processes (and their models) as soon as the work is done
            if hasattr(featurizer, "close"):
                featurizer.close()

        features = build_feature_frame(
            shard["doc_id"], shard["author_id"], vectors,
            feature_column_names(featurizer, vectors.shape[1]),
        )
        write_parquet(features, out_path)
        report_written(features, out_path, "features" if num_shards == 1 else "shard features")

    if num_shards > 1 and not args.no_auto_merge:
        # Every task tries this; only the last one to finish finds a complete set of shards, so the
        # array assembles its own final file with no follow-up job.
        merged = merge_shards(out_dir, stem, doc_order)
        if merged is not None:
            write_parquet(merged, merged_path)
            report_written(merged, merged_path)


if __name__ == "__main__":
    main()
