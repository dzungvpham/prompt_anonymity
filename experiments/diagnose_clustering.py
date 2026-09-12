#!/usr/bin/env python
"""Error analysis of one clustering partition, to decide what is worth building next.

``improve_clustering.py`` searches a space of methods; this asks a different question -- *given
the best partition we have, what exactly is it getting wrong?* -- and it is what a further method
should be chosen against. Everything here runs on the **tuning slice** and reads its labels only
to score, never to fit.

BCubed decomposes per document, so the loss decomposes too, and the two halves call for opposite
fixes:

* **Over-merging** costs precision: a cluster holding several authors. The fix is a better split.
* **Under-merging** costs recall: one author spread over several clusters. The fix is a better
  link, which is a *harder* problem -- the pair was never a candidate, or was ranked below a
  stranger.

The rest of the file measures things a *next method* would exploit, each stated as a hypothesis
with the number that would confirm or kill it: temporal proximity, shared-neighbour (second-order)
similarity, language agreement, and the reciprocal-rank structure the impostors method rests on.

Run::

    python experiments/diagnose_clustering.py --projection-file <path> --k 10 --edge-budget 87908
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from prompt_anonymity.attacks.clustering import build_neighbor_graph  # noqa: E402
from prompt_anonymity.attacks.clustering.projection import LinearProjection  # noqa: E402

from improve_clustering import (  # noqa: E402
    SLICES,
    fast_bcubed,
    per_document_bcubed,
    slice_indices,
)
from run_experiment import load_documents_and_features  # noqa: E402


def build_partition(features: np.ndarray, k: int, budget: int, width: int = 100):
    """The partition one frontier row describes, plus the graph it came from."""
    graph = build_neighbor_graph(features, width, metric="cosine")
    source, target, distance = graph.truncate(k).edges()
    finite = np.isfinite(distance)
    source, target, distance = source[finite], target[finite], distance[finite]
    order = np.argsort(distance, kind="stable")[:budget]
    adjacency = csr_matrix((np.ones(len(order), dtype=np.int8), (source[order], target[order])),
                           shape=(len(features),) * 2)
    return connected_components(adjacency, directed=False)[1], graph


def loss_decomposition(labels: np.ndarray, codes: np.ndarray, n_authors: int) -> pd.DataFrame:
    """How much of the BCubed shortfall is over-merging and how much is under-merging.

    Two counterfactuals on the *same* partition, each repairing one failure and leaving the other:

    * **split-only** -- cut every cluster along author lines. Precision goes to 1; recall is
      untouched. What is left below 1 is entirely under-merging.
    * **merge-only** -- union every cluster that shares an author. Recall rises to whatever
      linking through shared clusters can reach; precision falls.

    The two are not a partition of the loss (they interact), but their *sizes* say which failure
    the next method should attack.
    """
    rows = []
    precision, recall, f_score, _ = fast_bcubed(labels, codes, n_authors)
    rows.append({"partition": "as-is", "bcubed_precision": precision,
                 "bcubed_recall": recall, "bcubed_f": f_score})

    split = pd.factorize(pd.Series(list(zip(labels, codes))))[0]
    precision, recall, f_score, _ = fast_bcubed(split, codes, n_authors)
    rows.append({"partition": "oracle split (fixes over-merging)", "bcubed_precision": precision,
                 "bcubed_recall": recall, "bcubed_f": f_score})

    # Merge-only: connect two documents when they share an author AND share a cluster with some
    # document of that author, i.e. take the transitive closure of (cluster | author). That is the
    # best any *linking* method could do while only ever joining what this partition already
    # touches.
    n = len(labels)
    joins = csr_matrix((np.ones(2 * n, dtype=np.int8),
                        (np.concatenate([np.arange(n), np.arange(n)]),
                         np.concatenate([labels, labels.max() + 1 + codes]))),
                       shape=(n, labels.max() + 2 + n_authors))
    merged = connected_components(joins @ joins.T, directed=False)[1]
    precision, recall, f_score, _ = fast_bcubed(merged, codes, n_authors)
    rows.append({"partition": "oracle merge (fixes under-merging)", "bcubed_precision": precision,
                 "bcubed_recall": recall, "bcubed_f": f_score})
    return pd.DataFrame(rows)


def cluster_shape(labels: np.ndarray, codes: np.ndarray, n_authors: int) -> pd.DataFrame:
    """Per-cluster purity and per-author fragmentation, summarised."""
    frame = pd.DataFrame({"cluster": labels, "author": codes})
    per_cluster = frame.groupby("cluster")["author"].agg(
        size="size", authors="nunique",
        dominant=lambda column: column.value_counts().iloc[0])
    per_cluster["purity"] = per_cluster["dominant"] / per_cluster["size"]
    per_author = frame.groupby("author")["cluster"].agg(size="size", clusters="nunique")

    documents = len(labels)
    rows = [
        {"measure": "clusters", "value": len(per_cluster)},
        {"measure": "authors", "value": n_authors},
        {"measure": "singleton clusters", "value": int((per_cluster["size"] == 1).sum())},
        {"measure": "documents in singleton clusters",
         "value": int(per_cluster.loc[per_cluster["size"] == 1, "size"].sum())},
        {"measure": "pure clusters (1 author)", "value": int((per_cluster["authors"] == 1).sum())},
        {"measure": "documents in pure clusters",
         "value": int(per_cluster.loc[per_cluster["authors"] == 1, "size"].sum())},
        {"measure": "documents in pure NON-singleton clusters",
         "value": int(per_cluster.loc[(per_cluster["authors"] == 1) & (per_cluster["size"] > 1),
                                      "size"].sum())},
        {"measure": "largest cluster", "value": int(per_cluster["size"].max())},
        {"measure": "largest cluster share", "value": per_cluster["size"].max() / documents},
        {"measure": "mean cluster purity (document-weighted)",
         "value": float((per_cluster["purity"] * per_cluster["size"]).sum() / documents)},
        {"measure": "authors fully in one cluster", "value": int((per_author["clusters"] == 1).sum())},
        {"measure": "authors split over >1 cluster", "value": int((per_author["clusters"] > 1).sum())},
        {"measure": "mean clusters per author", "value": float(per_author["clusters"].mean())},
        {"measure": "mean clusters per multi-doc author",
         "value": float(per_author.loc[per_author["size"] > 1, "clusters"].mean())},
    ]
    return pd.DataFrame(rows)


def pair_signals(graph, frame: pd.DataFrame, codes: np.ndarray, k: int) -> pd.DataFrame:
    """Signals available on candidate edges, contrasted for same-author and different-author pairs.

    Each row is a hypothesis about what a *next* method could add to cosine. A signal is worth
    building on only if it separates the two populations on edges the graph already proposes --
    the edges where the current attack has to make a decision and is getting a third of them wrong.
    """
    source, target, distance = graph.truncate(k).edges()
    finite = np.isfinite(distance)
    source, target, distance = source[finite], target[finite], distance[finite]
    same = codes[source] == codes[target]

    signals = {"cosine distance": distance}

    times = pd.to_datetime(frame["ended_at"], errors="coerce", utc=True)
    seconds = (times - pd.Timestamp(0, tz="UTC")).dt.total_seconds().to_numpy()
    seconds[times.isna().to_numpy()] = np.nan
    signals["|time gap| hours"] = np.abs(seconds[source] - seconds[target]) / 3600.0

    # Shared nearest neighbours: how much two documents' own neighbourhoods overlap. The classic
    # second-order similarity (Jarvis-Patrick), and the cheap cousin of the impostors method --
    # if two documents are by one person, the *company they keep* should agree even where their
    # direct distance does not.
    view = graph.truncate(k)
    neighbours = view.indices
    valid = np.isfinite(view.distances)
    sets = [set(row[mask].tolist()) for row, mask in zip(neighbours, valid)]
    signals["shared neighbours (of k)"] = np.array(
        [len(sets[a] & sets[b]) for a, b in zip(source, target)], dtype=float)

    # Reciprocal rank: is each the other's *near* neighbour, or does one merely reach the other?
    # Hubs are asymmetric, and asymmetry is what a mutual-kNN filter throws away wholesale.
    rank_of = {}
    for row, (indices, mask) in enumerate(zip(neighbours, valid)):
        for position, column in enumerate(indices[mask]):
            rank_of[(row, int(column))] = position + 1
    forward = np.array([rank_of.get((a, b), k + 1) for a, b in zip(source, target)], dtype=float)
    backward = np.array([rank_of.get((b, a), k + 1) for a, b in zip(source, target)], dtype=float)
    signals["max rank (worse direction)"] = np.maximum(forward, backward)

    languages = frame["language_primary"].astype(str).to_numpy()
    signals["same language"] = (languages[source] == languages[target]).astype(float)

    # Which direction means "same author" differs per signal -- a small distance does, a large
    # shared-neighbour count does -- so it is declared rather than inferred. Reporting a raw AUROC
    # without it makes a strong signal look like a weak one below 0.5, which is how the
    # shared-neighbour count first read as 0.28 when it is really 0.72.
    smaller_is_same = {"cosine distance": True, "|time gap| hours": True,
                       "shared neighbours (of k)": False, "max rank (worse direction)": True,
                       "same language": False}
    rows = []
    for name, values in signals.items():
        usable = np.isfinite(values)
        oriented = values if smaller_is_same[name] else -values
        rows.append({
            "signal": name,
            "direction": "smaller = same author" if smaller_is_same[name]
                         else "larger = same author",
            "same-author median": float(np.median(values[usable & same])),
            "diff-author median": float(np.median(values[usable & ~same])),
            # AUROC over the candidate edges, always oriented so >0.5 means the signal favours
            # same-author. This is the number that decides whether a signal is worth building on:
            # 0.5 is nothing, and cosine's own value is the bar a *replacement* has to clear --
            # though a weaker signal that is INDEPENDENT of cosine can still add to it.
            "auroc": edge_auroc(oriented[usable], same[usable]),
            "n_pairs": int(usable.sum()),
        })
    return pd.DataFrame(rows)


def edge_auroc(values: np.ndarray, positive: np.ndarray) -> float:
    """AUROC of ``-values`` (smaller = more same-author) via the rank identity."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1)
    # Average ties so a signal with few distinct values (shared-neighbour counts) is not flattered.
    frame = pd.Series(values)
    ranks = frame.rank(method="average").to_numpy()
    n_positive, n_negative = positive.sum(), (~positive).sum()
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    auc = (ranks[positive].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)
    return float(1 - auc)                       # smaller value = more likely same author


def error_slices(labels: np.ndarray, codes: np.ndarray, n_authors: int,
                 frame: pd.DataFrame) -> pd.DataFrame:
    """Where the per-document BCubed loss sits, by language and by author document count."""
    precision, recall = per_document_bcubed(labels, codes, n_authors)
    f_score = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    counts = np.bincount(codes, minlength=n_authors)[codes]
    bins = pd.cut(counts, [0, 1, 2, 4, 8, 16, 10 ** 9],
                  labels=["1", "2", "3-4", "5-8", "9-16", "17+"])
    rows = []
    for name, group in (("language", frame["language_primary"].astype(str)), ("author docs", bins)):
        table = pd.DataFrame({"group": group, "p": precision, "r": recall, "f": f_score})
        summary = table.groupby("group", observed=True).agg(
            n_documents=("f", "size"), bcubed_precision=("p", "mean"),
            bcubed_recall=("r", "mean"), bcubed_f=("f", "mean")).reset_index()
        summary.insert(0, "axis", name)
        rows.append(summary.sort_values("n_documents", ascending=False).head(9))
    return pd.concat(rows, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="wildchat")
    parser.add_argument("--feature", default="gemini_embedding_2")
    parser.add_argument("--defense", default="base")
    parser.add_argument("--slice", default="eval", choices=sorted(SLICES))
    parser.add_argument("--projection-file", type=Path, default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--edge-budget", type=int, default=87908)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "hf")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "experiments" / "clustering_dev")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.slice != "eval":
        raise SystemExit("this is a development diagnostic; it runs on the tuning slice only.")

    frame, embeddings = load_documents_and_features(
        args.data_dir, args.source, args.feature,
        defense="none" if args.defense == "base" else args.defense)
    window = slice_indices(len(frame), args.slice)
    frame = frame.iloc[window].reset_index(drop=True)
    features = np.nan_to_num(embeddings[window]).astype(np.float32)
    del embeddings

    name = "raw"
    if args.projection_file is not None:
        features = LinearProjection(np.load(args.projection_file), "p").transform(features)
        name = args.projection_file.stem[:40]
    _, codes = np.unique(frame["author_id"].to_numpy(), return_inverse=True)
    codes = codes.ravel()
    n_authors = int(codes.max()) + 1

    labels, graph = build_partition(features, args.k, args.edge_budget)
    print(f"{args.source} [{args.slice}] {name}: {len(frame):,} documents, {n_authors:,} authors, "
          f"k={args.k}, budget={args.edge_budget:,}\n")

    output = args.output_dir / f"{args.source}_{args.defense}_{args.feature}"
    output.mkdir(parents=True, exist_ok=True)

    for title, table, stem in (
            ("LOSS DECOMPOSITION", loss_decomposition(labels, codes, n_authors), "loss"),
            ("CLUSTER SHAPE", cluster_shape(labels, codes, n_authors), "shape"),
            ("PAIR SIGNALS on candidate edges", pair_signals(graph, frame, codes, args.k),
             "signals"),
            ("ERROR BY SLICE", error_slices(labels, codes, n_authors, frame), "slices")):
        print(f"--- {title}")
        print(table.to_string(index=False), "\n")
        table.to_csv(output / f"diagnose_{stem}.csv", index=False)
    print(f"Wrote {output}/diagnose_*.csv")


if __name__ == "__main__":
    main()
