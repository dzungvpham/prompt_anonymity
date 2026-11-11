r"""Embed the free-text topic labels -- ``subject`` and ``description`` -- one parquet per field.

Third companion to :mod:`~prompt_anonymity.data.label_topics` (which writes the labels) and
:mod:`~prompt_anonymity.data.compute_features` (which embeds the conversations themselves). This
script embeds the two free-text fields of ``<split>_topics.parquet`` with a registered featurizer
-- Gemini Embedding 2 by default -- so they can be compared, clustered and attacked like any other
feature.

Why embed the labels at all: they are the conversation's content restated by a third party in
fixed English, with none of the user's language, style or formatting. ``subject`` is the general
version (user-specific names replaced by their kind), so similarity between two subject vectors is
close to pure topic similarity; ``description`` keeps the names, so the gap between the two
measures how much shared names add. See :mod:`~prompt_anonymity.data.label_topics` for how the
fields are defined.

Where the output goes
---------------------

The labels are READ from ``dist/<split>_topics.parquet`` (where ``label_topics`` writes them) and
each field's vectors are WRITTEN to ``dist/<split>_topics_<field>_<feature>.parquet`` -- the
feature-file layout exactly: ``doc_id``, ``author_id``, then one float32 column per dimension, one
row per document in the topics file's order.

**A document with no label gets a row of NaN**, not no row. ``refused``, ``rejected`` and ``empty``
documents have no ``subject``, and dropping their rows would break the one-to-one ``doc_id`` join
every feature file is held to (a missing key is a hard error in the runners). NaN says plainly that
there is no vector; drop those rows before computing a similarity rather than letting a runner
coerce them to a zero vector.

A non-default ``--task`` is appended to the feature part of the name, as in ``compute_features``,
and so is a non-default ``--dimensions`` (``_d768``) -- two widths are two spaces and must never
share a file.

Cost and caching
----------------

Paid: Gemini Embedding 2 is billed per token through OpenRouter ($0.20 per million). The labels
are short, so the whole of both corpora's two fields is a couple of dollars; ``--dry-run`` counts
what a run would send without a request. Each run's spend is appended to the ledger
``tmp/topic_labels/embedding_spend.csv`` (one row per run that made a request, written even when
the run aborts) -- **never** to anything under ``data/``.

Vectors are cached content-addressed, so a label string many conversations share is embedded once
and a re-run is free -- but in **this script's own cache root**, ``<cache-dir>/topic_label_vectors/``,
**never the featurizer's shared one** that ``compute_features`` uses. That separation is not
tidiness. A featurizer's cache is keyed by a hash of its class source and, on open, deletes every
other version's directory; the first version of this script went through it, found the key moved
by an unrelated edit to ``features/base.py``, and began deleting the cached conversation vectors
(2026-11-09; stopped with 7,108 left, the feature parquets untouched). Here the key is the
featurizer's parameters plus :data:`EMBED_VERSION` only, and nothing is ever pruned -- bump the
version by hand when a change should re-embed, and delete a superseded namespace by hand if at all.

**The cache stores each vector as JSON, about 70 KB at 3,072 dimensions** -- WildChat's two fields
at native width are ~23 GB of cache plus ~6 GB of parquet. ``--dimensions 768`` cuts both
fourfold, and ``--cache-dir`` can put the cache on scratch.

Run (from the repo root)::

    python -m prompt_anonymity.data.embed_topic_labels --source swe_chat --dry-run   # no requests
    python -m prompt_anonymity.data.embed_topic_labels --source swe_chat             # PAID
    python -m prompt_anonymity.data.embed_topic_labels --source wildchat --fields subject \
        --dimensions 768
"""

from __future__ import annotations

import argparse
import csv
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from prompt_anonymity.caching import TransformCache, logic_hash, params_hash
from prompt_anonymity.features import FEATURIZERS

from .compute_features import (
    CHUNK_SIZE,
    SOURCES,
    build_feature_frame,
    build_featurizer,
    feature_column_names,
    feature_label,
    report_written,
    write_parquet,
)
from .config import cache_dir, dist_dir, project_root
from .label_topics import OUTPUT_LABEL, USAGE_SUBDIR

#: The free-text label columns this script can embed, in the order they are embedded by default.
FIELDS = ("subject", "description")

#: The featurizer used when ``--feature`` is not given: the one the conversations are embedded with.
DEFAULT_FEATURE = "gemini_embedding_2"

#: This script's own vector cache root under ``--cache-dir`` -- deliberately not ``features/``,
#: the featurizers' shared root (see "Cost and caching" for why).
CACHE_SUBDIR = "topic_label_vectors"

#: Manual logic version of the label-vector cache. The featurizer's source is deliberately not
#: hashed, so this is the only lever that re-embeds on a code change.
EMBED_VERSION = "1"

#: The embedding spend ledger, under the label usage directory -- a different schema from
#: ``label_topics``'s own ``spend.csv``, so a separate file.
SPEND_LEDGER = "embedding_spend.csv"

#: The ledger's columns: one row per run that made at least one request.
SPEND_COLUMNS = ["finished_at_utc", "outcome", "source", "fields", "feature", "task",
                 "dimensions", "requests", "texts_embedded", "tokens", "cost_usd"]

#: Tokens the ``task: clustering | query: `` prefix adds to every input, for ``--dry-run`` only --
#: a large share of a five-word subject, so it is counted rather than ignored.
DRY_RUN_PREFIX_TOKENS = 7


def topics_path(source: str, topics_dir: Path) -> Path:
    """Where ``label_topics`` wrote ``source``'s labels."""
    return Path(topics_dir) / f"{source}_{OUTPUT_LABEL}.parquet"


def output_path(out_dir: Path, source: str, field: str, label: str,
                dimensions: int | None) -> Path:
    """``<split>_topics_<field>_<feature label>[_d<dimensions>].parquet`` under ``out_dir``."""
    width = f"_d{dimensions}" if dimensions else ""
    return Path(out_dir) / f"{source}_{OUTPUT_LABEL}_{field}_{label}{width}.parquet"


def open_label_vector_cache(featurizer, root) -> TransformCache:
    """The label vectors' cache: ``<root>/topic_label_vectors/<featurizer>/<version>/<params>/``.

    Keyed by :data:`EMBED_VERSION` and the featurizer's parameters (model, task, width) -- not by
    its source, which would re-key on any edit to it or its base class -- and opened with
    ``prune_stale=False``, so opening it can never delete anything, here or in the featurizers'
    shared ``features/`` root.
    """
    return TransformCache(Path(root) / CACHE_SUBDIR, featurizer.name,
                          logic_hash([], version=EMBED_VERSION), params_hash(featurizer.params()),
                          prune_stale=False)


def embed_field(featurizer, labels: pd.Series, *, cache: TransformCache,
                chunk_size: int) -> np.ndarray:
    """``(len(labels), n_features)`` float32 vectors, a row of NaN wherever the label is missing.

    Only the present labels are sent, ``chunk_size`` at a time through ``cache``, so a refused
    document costs nothing and cannot be embedded as an empty string, and an interrupted run
    resumes from the last finished chunk.
    """
    present = labels.notna().to_numpy()
    texts = [str(text) for text in labels[present]]
    if not texts:
        raise SystemExit(f"no {labels.name} labels to embed.")
    step = chunk_size if chunk_size and chunk_size > 0 else len(texts)
    chunks, computed, reused = [], 0, 0
    for start in tqdm(range(0, len(texts), step), desc=f"  embedding {labels.name}"):
        chunks.append(np.asarray(featurizer.transform(texts[start:start + step], cache),
                                 dtype=np.float32))
        computed, reused = computed + cache.misses, reused + cache.hits
    print(f"[{featurizer.name}] {labels.name}: {computed:,} vectors computed, {reused:,} served "
          f"from {cache.dir}")
    vectors = np.concatenate(chunks, axis=0)
    full = np.full((len(labels), vectors.shape[1]), np.nan, dtype=np.float32)
    full[present] = vectors
    return full


def dry_run(topics: pd.DataFrame, fields: list[str]) -> None:
    """Print how many labels and tokens a real run would send, without a request.

    Counts with the featurizer's own tokenizer family (tiktoken ``o200k_base``) and prices at
    Gemini Embedding 2's $0.20 per million; the vector cache is not consulted, so this is the cost
    of an empty cache -- an upper bound.
    """
    import tiktoken

    encoding = tiktoken.get_encoding("o200k_base")
    total = 0.0
    for field in fields:
        texts = topics[field].dropna().astype(str).unique().tolist()
        tokens = sum(len(t) for t in encoding.encode_batch(texts)) + DRY_RUN_PREFIX_TOKENS * len(texts)
        cost = tokens / 1e6 * 0.20
        total += cost
        print(f"[dry run] {field}: {len(texts):,} distinct labels, ~{tokens / 1e6:.2f}M tokens, "
              f"~${cost:.2f}")
    print(f"[dry run] all fields: ~${total:.2f} at an empty cache")


def append_spend(path: Path, row: dict) -> None:
    """Append one run's row to the embedding spend ledger, writing the header when it is new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SPEND_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="swe_chat", choices=sorted(SOURCES),
                   help="which split's topic labels to embed (default: swe_chat)")
    p.add_argument("--fields", nargs="+", default=list(FIELDS), choices=FIELDS,
                   help="label fields to embed, one output file each (default: both)")
    p.add_argument("--feature", default=DEFAULT_FEATURE, choices=sorted(FEATURIZERS),
                   help=f"registered featurizer to embed with (default: {DEFAULT_FEATURE})")
    p.add_argument("--task", default=None,
                   help="embedding task for featurizers that take one (gemini_embedding_2 default: "
                        "clustering); a non-default task is appended to the output name")
    p.add_argument("--dimensions", type=int, default=None,
                   help="truncate the embedding to this width (e.g. 768); appended to the output "
                        "name. Cuts the vector cache and the parquet in proportion")
    p.add_argument("--workers", type=int, default=None,
                   help="concurrent API requests (default: the featurizer's own)")
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                   help=f"labels per featurizer call / cache checkpoint (default: {CHUNK_SIZE})")
    p.add_argument("--dry-run", action="store_true",
                   help="count the labels and tokens a run would send; no request, nothing written")
    p.add_argument("--topics-dir", default=None,
                   help="where <split>_topics.parquet is (default: the project's data/dist)")
    p.add_argument("--out-dir", default=None,
                   help="where to write the vectors (default: the project's data/dist)")
    p.add_argument("--cache-dir", default=None,
                   help="vector cache root (default: data/.cache; ~70 KB per vector at 3,072 dims)")
    p.add_argument("--usage-dir", default=None,
                   help=f"where the {SPEND_LEDGER} ledger goes (default: <project root>/"
                        f"{USAGE_SUBDIR}); kept out of data/")
    args = p.parse_args()

    topics_dir = Path(args.topics_dir) if args.topics_dir else dist_dir()
    out_dir = Path(args.out_dir) if args.out_dir else dist_dir()
    cache = Path(args.cache_dir) if args.cache_dir else cache_dir()
    usage_dir = Path(args.usage_dir) if args.usage_dir else project_root() / USAGE_SUBDIR
    source_path = topics_path(args.source, topics_dir)
    if not source_path.exists():
        raise SystemExit(f"{source_path} not found -- run `python -m "
                         f"prompt_anonymity.data.label_topics --source {args.source}` first.")
    topics = pd.read_parquet(source_path, columns=["doc_id", "author_id", *args.fields])
    print(f"[{args.source}] {len(topics):,} documents in {source_path.name}; "
          + ", ".join(f"{field} present for {topics[field].notna().sum():,}" for field in args.fields))

    if args.dry_run:
        dry_run(topics, args.fields)
        return

    featurizer = build_featurizer(args.feature, workers=args.workers,
                                  dimensions=args.dimensions, task=args.task)
    vector_cache = open_label_vector_cache(featurizer, cache)
    print(f"[cache] {vector_cache.dir}")
    label = feature_label(args.feature, args.task)
    outcome = "aborted"
    # SIGTERM (a SLURM time limit or preemption, scancel) would otherwise skip the `finally`
    # below and lose this run's ledger row; the vectors already bought are cached either way.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(f"stopped by signal {signum}"))
    try:
        for field in args.fields:
            vectors = embed_field(featurizer, topics[field], cache=vector_cache,
                                  chunk_size=args.chunk_size)
            frame = build_feature_frame(topics["doc_id"], topics["author_id"], vectors,
                                        feature_column_names(featurizer, vectors.shape[1]))
            path = output_path(out_dir, args.source, field, label, args.dimensions)
            write_parquet(frame, path)
            report_written(frame, path, f"{field} vectors")
        outcome = "completed"
    finally:
        if hasattr(featurizer, "close"):
            featurizer.close()  # prints the featurizer's own spend line
        requests = int(getattr(featurizer, "total_requests", 0) or 0)
        if requests:
            cost = float(getattr(featurizer, "total_cost", 0.0) or 0.0)
            append_spend(usage_dir / SPEND_LEDGER, {
                "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "outcome": outcome, "source": args.source, "fields": " ".join(args.fields),
                "feature": args.feature, "task": getattr(featurizer, "task", ""),
                "dimensions": args.dimensions or getattr(featurizer, "native_dimensions", ""),
                "requests": requests,
                "texts_embedded": int(getattr(featurizer, "total_documents", 0) or 0),
                "tokens": int(getattr(featurizer, "total_tokens", 0) or 0),
                "cost_usd": cost,
            })
            print(f"[spend] this run's ${cost:.4f} recorded in {usage_dir / SPEND_LEDGER}")


if __name__ == "__main__":
    main()
