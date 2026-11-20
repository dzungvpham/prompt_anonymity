#!/usr/bin/env python
"""What do niche topics look like? Keywords of the smallest-crowd and largest-crowd topics.

Fits the same own-data codebook as the content-floor runs (k-means on the known side of one
window), then for every topic counts its conversations and distinct users. A *niche* topic has
many conversations but very few users (one person's recurring project); a *crowded* topic is used
by many people. For each, prints the most distinctive words of the conversations' first user
message (mean TF-IDF in the topic minus the corpus mean). Keywords only, no message text.

Usage::

    python experiments/thesis/niche_topics.py --source wildchat --k 1024 --out topics.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_experiment import known_configurations, load_documents_and_features  # noqa: E402


def unit(rows: np.ndarray) -> np.ndarray:
    rows = np.array(rows, dtype=np.float32)
    rows /= np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="wildchat")
    parser.add_argument("--data-dir", default="/Users/vandanasingh/prompt_anonymity/data/hf")
    parser.add_argument("--window", default="0075")
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--min-docs", type=int, default=20)
    parser.add_argument("--show", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    frame, embeddings = load_documents_and_features(args.data_dir, args.source, "gemini_embedding_2")
    _, known, _ = known_configurations(len(frame), [args.window], 0.25)[0]
    known_frame = frame.iloc[known]
    codebook = MiniBatchKMeans(n_clusters=args.k, random_state=args.seed, n_init=3, batch_size=4096)
    topics = codebook.fit_predict(unit(embeddings[known]))
    del embeddings

    texts = pd.read_parquet(Path(args.data_dir) / f"{args.source}.parquet", columns=["doc_id", "turns"])
    first = texts.set_index("doc_id")["turns"].map(lambda t: str(t[0]) if len(t) else "")
    first = first.reindex(known_frame["doc_id"]).fillna("").to_numpy()
    tfidf = TfidfVectorizer(max_features=50000, min_df=5, sublinear_tf=True,
                            token_pattern=r"(?u)\b[^\W\d_]{3,}\b", stop_words="english")
    matrix = tfidf.fit_transform(first)
    vocabulary = np.array(tfidf.get_feature_names_out())
    overall = np.asarray(matrix.mean(0)).ravel()

    table = pd.DataFrame({"topic": topics, "author": known_frame["author_id"].to_numpy()})
    stats = table.groupby("topic").agg(conversations=("author", "size"), users=("author", "nunique"),
                                       top_user_share=("author", lambda a: a.value_counts().iloc[0] / len(a)))
    words = {}
    for topic in stats.index:
        rows = np.flatnonzero(topics == topic)
        lift = np.asarray(matrix[rows].mean(0)).ravel() - overall
        words[topic] = " ".join(vocabulary[np.argsort(-lift)[:8]])
    stats["keywords"] = pd.Series(words)
    stats.to_csv(args.out)

    busy = stats[stats["conversations"] >= args.min_docs]
    print(f"{args.source} known{args.window} K={args.k}: {len(stats)} topics; "
          f"{len(busy)} with >= {args.min_docs} conversations")
    for label, rows in (("NICHE (fewest users)", busy.sort_values(["users", "conversations"],
                                                                   ascending=[True, False])),
                        ("CROWDED (most users)", busy.sort_values("users", ascending=False))):
        print(f"\n{label}")
        print(rows.head(args.show)[["conversations", "users", "top_user_share", "keywords"]]
              .round(2).to_string())
    one_user = busy[busy["top_user_share"] >= 0.8]
    share = one_user["conversations"].sum() / stats["conversations"].sum()
    print(f"\nTopics with >= {args.min_docs} conversations where one user wrote >= 80%: "
          f"{len(one_user)} ({share:.1%} of all known conversations)")


if __name__ == "__main__":
    main()
