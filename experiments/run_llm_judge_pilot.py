"""Pilot: does an LLM judge help on top of the new best deterministic attack?

Everything up to this point (run_wccn_fusion.py) established Gemini + char n-gram + POS n-gram,
scored via WCCN, as the strongest deterministic attack on both corpora. The open question this
answers: on the small slice of documents where that attack is LEAST confident -- the smallest
top-1/top-2 margin -- does a constrained LLM judge (EuclideanLLMJudgeAttack) recover any of them?

Scoped deliberately small and cheap. Rather than reranking every document in a window (the
existing swechat_llm_judge_eval.py's approach, which costs a real per-window API sweep), this:

  1. Fits WCCN on the fused features for one window (the same fit run_wccn_fusion.py would do).
  2. Projects both sides into WCCN's whitened space (WhitenedCentroid.project()), so the
     existing author_candidates() shortlist -- ordinarily a plain embedding-distance NN search --
     operates in the space WCCN actually discriminates in, rather than raw Gemini space.
  3. Picks the N documents with the smallest top-1/top-2 AUTHOR margin under that shortlist --
     the "hard cases" where the deterministic attack is guessing.
  4. Judges only those N with EuclideanLLMJudgeAttack(margin_quantile=1.0) -- every row passed in
     is already pre-filtered to be ambiguous, so nothing further is gated out.

N documents means N judge API calls, not N * n_documents_in_window -- the entire point of gating
before judging rather than judging everything and gating the application after, which is what
margin_quantile alone would do if run over the whole window.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from run_experiment import (  # noqa: E402
    DEFAULT_KNOWN_WINDOWS, DEFAULT_TEST_FRACTION,
    known_configurations, load_documents_and_features, standardize,
)
from run_wccn_fusion import (  # noqa: E402
    build_matrices, load_pos_tags, load_turns_text,
)
from prompt_anonymity.attacks.similarity.whitened_centroid import WhitenedCentroid  # noqa: E402
from prompt_anonymity.attacks.llm.candidates import author_candidates  # noqa: E402
from prompt_anonymity.attacks.llm.euclidean_llm_judge import EuclideanLLMJudgeAttack  # noqa: E402
from prompt_anonymity.core import AttackData  # noqa: E402
from prompt_anonymity.evaluation.metrics.ranking import (  # noqa: E402
    macro_top_k_accuracy, ranking_summary, true_author_ranks,
)

TURN_SEPARATOR = "\n\n"


def load_texts_by_doc_id(data_dir: str, source: str) -> dict[str, str]:
    table = pq.read_table(Path(data_dir) / f"{source}.parquet", columns=["doc_id", "turns"])
    doc_ids = table.column("doc_id").to_pylist()
    turns = table.column("turns").to_pylist()
    return {d: TURN_SEPARATOR.join(t) for d, t in zip(doc_ids, turns)}


def majority_vote_rerank(data, base_scores, judge_model, top_k, cache_dir, n_runs, base_seed):
    """Run the judge ``n_runs`` times (different candidate-shuffle seeds each), promote each
    document's MODE pick across runs rather than trusting a single call.

    Per "De-Anonymization at Scale via Tournament-Style Attribution" (arXiv:2601.12407), whose
    tournament rounds combine multiple independent runs by majority vote to cancel single-run
    noise -- the same idea applied here to one round of top-K reranking rather than a
    multi-round tournament. Each run's promoted author is recovered as that run's row-wise
    argmax (EuclideanLLMJudgeAttack boosts its pick to strictly the row max), so this needs no
    change to that class -- just calling it several times.
    """
    picks_per_run, authors = [], None
    for i in range(n_runs):
        judge = EuclideanLLMJudgeAttack(judge_model=judge_model, top_k=top_k,
                                        margin_quantile=1.0, seed=base_seed + i, verbose=True)
        boosted = judge.attack(data, cache_dir=cache_dir).to_numpy()
        if authors is None:
            authors = judge.authors
        picks_per_run.append(np.argmax(boosted, axis=1))
    picks = np.array(picks_per_run)  # (n_runs, n_docs)

    final = base_scores.copy()
    row_max = base_scores.max(axis=1)
    agreement = 0
    for j in range(picks.shape[1]):
        vote, count = Counter(picks[:, j].tolist()).most_common(1)[0]
        agreement += count == n_runs
        final[j, vote] = row_max[j] + 1.0
    print(f"  majority vote over {n_runs} runs: {agreement}/{picks.shape[1]} rows unanimous")
    return final, authors


def window_metrics(scores, candidate_authors, true_authors):
    ranks = true_author_ranks(scores, candidate_authors, true_authors)
    summary = ranking_summary(ranks, len(candidate_authors))
    return {"macro_conv_acc1": macro_top_k_accuracy(ranks, true_authors, k=1), "mrr": summary["mrr"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="swe_chat")
    parser.add_argument("--data-dir", default="data/hf")
    parser.add_argument("--feature-set", default="gemini_embedding_2+char_ngram+pos_ngram")
    parser.add_argument("--shrinkage", type=float, default=0.6)
    parser.add_argument("--known-windows", nargs="+", default=list(DEFAULT_KNOWN_WINDOWS))
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    parser.add_argument("--n-ambiguous", type=int, default=30,
                        help="documents judged per window (= API calls per window)")
    parser.add_argument("--judge-model", default="anthropic/claude-sonnet-5")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--majority-vote-runs", type=int, default=1,
                        help="judge each ambiguous row this many times (different candidate "
                             "shuffles) and promote the mode pick, instead of trusting one "
                             "call. 1 = single run (default).")
    parser.add_argument("--cache-dir", default="llm_judge_cache")
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()

    comps = args.feature_set.split("+")
    needs_char = "char_ngram" in comps
    needs_pos = "pos_ngram" in comps

    print(f"loading {args.source} features for {args.feature_set}...")
    precomputed_needed = sorted({c for c in comps if c not in ("char_ngram", "pos_ngram")})
    frame = None
    precomputed = {}
    for feat in precomputed_needed:
        frame_i, mat = load_documents_and_features(args.data_dir, args.source, feat)
        if frame is None:
            frame = frame_i
        else:
            assert list(frame["doc_id"]) == list(frame_i["doc_id"])
        precomputed[feat] = mat
    doc_ids = frame["doc_id"].tolist()
    authors = frame["author_id"].to_numpy()
    texts_by_id = load_texts_by_doc_id(args.data_dir, args.source)
    texts = np.array([texts_by_id[d] for d in doc_ids], dtype=object)

    pos_cache_glob = ("pos_tags_cache_swe_chat.jsonl" if args.source == "swe_chat"
                      else "pos_tags_cache_wildchat_shard*of4.jsonl")
    turns_by_id = load_turns_text(args.data_dir, args.source, doc_ids) if needs_char else None
    pos_by_id = load_pos_tags(pos_cache_glob, doc_ids) if needs_pos else None

    configs = known_configurations(len(frame), args.known_windows, args.test_fraction)

    results = {"before (embedding-distance NN in whitened space)": [],
              "after (LLM judge, ambiguous rows only)": []}
    total_judged = 0

    for config, known_slice, unknown_slice in configs:
        known_labels = authors[known_slice]
        unknown_labels_all = authors[unknown_slice]
        in_set = np.isin(unknown_labels_all, known_labels)
        unknown_labels_in = unknown_labels_all[in_set]

        k_feat, u_feat, _ = build_matrices(comps, known_slice, unknown_slice, precomputed,
                                           doc_ids, turns_by_id, pos_by_id)
        u_feat_in = u_feat[in_set]

        wccn = WhitenedCentroid(shrinkage=args.shrinkage).fit(k_feat, known_labels)
        proj_known = wccn.project(k_feat)
        proj_unknown = wccn.project(u_feat_in)

        candidates = author_candidates(proj_known, known_labels, proj_unknown,
                                       top_k=args.top_k, metric="cosine")
        m_before_all = window_metrics(candidates.scores, candidates.authors, unknown_labels_in)

        n_ambiguous = min(args.n_ambiguous, len(unknown_labels_in))
        ambiguous_idx = np.argsort(candidates.margin)[:n_ambiguous]

        known_texts = texts[known_slice]
        unknown_texts_in = texts[unknown_slice][in_set]

        data = AttackData(
            known_embeddings=proj_known, unknown_embeddings=proj_unknown[ambiguous_idx],
            known_labels=known_labels, unknown_labels=unknown_labels_in[ambiguous_idx],
            metric="cosine", known_texts=known_texts, unknown_texts=unknown_texts_in[ambiguous_idx],
        )

        print(f"\n=== {config.tag} ({config.label}) known={len(known_labels)} "
              f"unknown={in_set.sum()} in-set, judging {n_ambiguous} most-ambiguous ===")
        print(f"  whole-window baseline (before, all in-set): "
              f"macro_conv_acc1={m_before_all['macro_conv_acc1']:.3f} mrr={m_before_all['mrr']:.3f}")

        before_subset = window_metrics(candidates.scores[ambiguous_idx], candidates.authors,
                                       unknown_labels_in[ambiguous_idx])
        if args.majority_vote_runs > 1:
            boosted, judge_authors = majority_vote_rerank(
                data, candidates.scores[ambiguous_idx], args.judge_model, args.top_k,
                args.cache_dir, args.majority_vote_runs, args.seed)
        else:
            judge = EuclideanLLMJudgeAttack(judge_model=args.judge_model, top_k=args.top_k,
                                            margin_quantile=1.0, seed=args.seed, verbose=True)
            boosted = judge.attack(data, cache_dir=args.cache_dir).to_numpy()
            judge_authors = judge.authors
        after_subset = window_metrics(boosted, judge_authors, unknown_labels_in[ambiguous_idx])
        total_judged += n_ambiguous * args.majority_vote_runs

        results["before (embedding-distance NN in whitened space)"].append(before_subset)
        results["after (LLM judge, ambiguous rows only)"].append(after_subset)
        print(f"  ambiguous subset  before: macro_conv_acc1={before_subset['macro_conv_acc1']:.3f} "
              f"mrr={before_subset['mrr']:.3f}")
        print(f"  ambiguous subset  after:  macro_conv_acc1={after_subset['macro_conv_acc1']:.3f} "
              f"mrr={after_subset['mrr']:.3f}")

    print("\n" + "=" * 60)
    print(f"total documents judged (== API calls): {total_judged}")
    print(f"{'variant':<45} {'macro_conv_acc1':>16} {'mrr':>8}")
    print("-" * 60)
    for name, rows in results.items():
        acc = np.mean([r["macro_conv_acc1"] for r in rows])
        mrr = np.mean([r["mrr"] for r in rows])
        print(f"{name:<45} {acc:>16.3f} {mrr:>8.3f}")

    print("\nper-window (for paired significance testing):")
    for name, rows in results.items():
        key = "before" if "before" in name else "after"
        print(f"{key}_acc = {[round(r['macro_conv_acc1'], 4) for r in rows]}")
        print(f"{key}_mrr = {[round(r['mrr'], 4) for r in rows]}")


if __name__ == "__main__":
    main()
