"""Pilot: does an LLM judge help on top of the new best deterministic attack?

Everything up to this point (run_wccn_fusion.py) established Gemini + char n-gram + POS n-gram
(or + LUAR), scored via WCCN, as the strongest deterministic attack on both corpora. The open
question this answers: on the small slice of documents where that attack is LEAST confident --
the smallest top-1/top-2 margin, by WCCN's OWN score -- does a constrained LLM judge recover any
of them?

**Corrected from an earlier version of this script**, which built its shortlist and "before"
baseline from ``author_candidates()`` over WCCN-*projected* embeddings -- a nearest-known-
*document* search in the whitened space -- rather than from ``WhitenedCentroid.score()``, the
actual author-*centroid* score the rest of this project reports. The two are not the same
attack: a single document can sit far from its author's own centroid, so the earlier pilot's
"before" numbers ran a few points below the real WCCN baseline and its shortlist could differ
from WCCN's own top-k. Concretely, that version's ambiguity gate, top-5 authors and reported
"before" score all came from::

    proj_known, proj_unknown = wccn.project(k_feat), wccn.project(u_feat_in)
    candidates = author_candidates(proj_known, known_labels, proj_unknown, top_k=5, metric="cosine")

instead of::

    base_scores = wccn.score(u_feat_in)

This version does the latter throughout: authors are shortlisted by ``base_scores`` directly,
and only the *evidence text* per shortlisted author (which of their own known documents to show
the judge) still uses a nearest-document search in the whitened space -- restricted to that
author's own rows, which is a sensible way to pick a representative document once the author is
already chosen on the real WCCN score, not a way to choose the author.

Scoped deliberately small and cheap: only ``--n-ambiguous`` documents per window are judged (=
API calls per window), not every document in the window.

Every judge variant (plain / majority-vote / feature-grounded) is built from ONE shared shortlist
and evidence-selection pass (:func:`wccn_shortlist`) and ONE shared judge-calling core
(:func:`run_judge_once`) -- they differ only in the prompt-building function and, for majority
vote, in how many times :func:`run_judge_once` is called. This replaces three previously
separate, partially-duplicated implementations (one leaning on
:class:`~prompt_anonymity.attacks.llm.euclidean_llm_judge.EuclideanLLMJudgeAttack`'s own
internal shortlist, two reimplementing it ad hoc), which is also what let the shortlist mismatch
above go unnoticed in two of the three variants but not the third.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
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
from prompt_anonymity.attacks.llm.euclidean_llm_judge import (  # noqa: E402
    DEFAULT_SNIPPET_CHARS, JUDGE_SYSTEM_PROMPT_TEMPLATE, _judge_prompt, _parse_choice,
)
from prompt_anonymity.attacks.llm._openrouter import OpenRouterChat  # noqa: E402
from prompt_anonymity.caching import TransformCache, logic_hash, params_hash  # noqa: E402
from prompt_anonymity.evaluation.metrics.ranking import (  # noqa: E402
    macro_top_k_accuracy, ranking_summary, true_author_ranks,
)

TURN_SEPARATOR = "\n\n"

FEATURE_GROUNDED_SYSTEM_PROMPT_TEMPLATE = (
    "You are an authorship-attribution judge. You will be shown one QUERY text and "
    "{n} CANDIDATE texts, labeled 1 through {n}, each annotated with computed writing-style "
    "statistics (average word length, sentence length, vocabulary variety, punctuation "
    "density, capitalization rate). Use these statistics to ground your judgment in addition "
    "to reading the text itself -- they are the same signal a stylometrist would compute by "
    "hand, not a substitute for reading.\n"
    "Task:\n"
    "Choose the ONE candidate most likely written by the SAME author as the QUERY, weighting "
    "writing style over topic or subject matter.\n"
    "Rules:\n"
    "- This is a forced choice: you MUST pick exactly one candidate, the single closest "
    "stylistic match. Even if none is an obvious match, pick the best of the {n}. Do NOT "
    "refuse and do NOT answer 0.\n"
    "- Output ONLY the single digit (1-{n}) of your choice and nothing else -- no words, no "
    "punctuation, no explanation."
)


def quick_style_stats(text: str) -> dict[str, float]:
    """A handful of cheap, human-interpretable style statistics -- the SALA-style grounding
    (arXiv:2602.23079). Deliberately not StyloMetrix (197 opaque-coded dims meant for a
    classifier, not for reading in a prompt): these five are simple enough that a judge can
    sanity-check them against the text it's also shown."""
    words = text.split() or [""]
    n_words = len(words)
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()] or [text]
    n_chars = len(text) or 1
    return {
        "avg_word_length": round(sum(len(w) for w in words) / n_words, 2),
        "type_token_ratio": round(len(set(w.lower() for w in words)) / n_words, 3),
        "avg_sentence_length_words": round(n_words / len(sentences), 1),
        "punctuation_density": round(sum(c in ".,;:!?-()[]{}\"'" for c in text) / n_chars, 3),
        "uppercase_ratio": round(sum(c.isupper() for c in text) / n_chars, 3),
    }


def _format_stats(stats: dict[str, float]) -> str:
    return ", ".join(f"{k}={v}" for k, v in stats.items())


def plain_prompt_fn(query_text: str, candidate_texts: list[str]) -> str:
    return _judge_prompt(query_text, candidate_texts)


def feature_grounded_prompt_fn(query_text: str, candidate_texts: list[str]) -> str:
    query_stats = quick_style_stats(query_text)
    cand_stats = [quick_style_stats(c) for c in candidate_texts]
    lines = [f"QUERY (stats: {_format_stats(query_stats)}):\n{query_text}"]
    for i, (cand, stats) in enumerate(zip(candidate_texts, cand_stats), 1):
        lines.append(f"\nCANDIDATE {i} (stats: {_format_stats(stats)}):\n{cand}")
    n = len(candidate_texts)
    lines.append(f"\nWhich candidate is the closest stylistic match to the QUERY? "
                f"You MUST pick one. Answer with a single digit 1-{n}.")
    return "\n".join(lines)


def load_texts_by_doc_id(data_dir: str, source: str) -> dict[str, str]:
    table = pq.read_table(Path(data_dir) / f"{source}.parquet", columns=["doc_id", "turns"])
    doc_ids = table.column("doc_id").to_pylist()
    turns = table.column("turns").to_pylist()
    return {d: TURN_SEPARATOR.join(t) for d, t in zip(doc_ids, turns)}


def wccn_shortlist(base_scores_subset: np.ndarray, candidate_authors: np.ndarray,
                   proj_known: np.ndarray, proj_query_subset: np.ndarray,
                   known_labels: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """True-WCCN top-k AUTHORS for each row of an already-selected subset, each with its own
    nearest known document (in WCCN-projected space) as the evidence text shown to the judge.

    Authors are chosen on ``base_scores_subset`` -- WCCN's real author-centroid score -- not on
    any document-level proxy. Only the representative document per already-chosen author uses a
    nearest-neighbor search, and that search is restricted to the chosen author's own known
    rows, so it cannot change which authors are shortlisted.

    Returns
    -------
    col_index : (n, top_k) int
        Column indices into ``base_scores_subset`` / ``candidate_authors``, best first --
        boosting ``col_index[i, j]`` is how a judge's pick gets promoted back into the real
        WCCN score matrix.
    document_index : (n, top_k) int
        Row indices into the known-side arrays: the evidence text for each shortlisted author.
    """
    n = base_scores_subset.shape[0]
    col_index = np.argsort(base_scores_subset, axis=1)[:, -top_k:][:, ::-1]
    document_index = np.empty((n, top_k), dtype=int)
    for i in range(n):
        q = proj_query_subset[i]
        for j in range(top_k):
            author = candidate_authors[col_index[i, j]]
            idxs = np.where(known_labels == author)[0]
            document_index[i, j] = idxs[np.argmax(proj_known[idxs] @ q)]
    return col_index, document_index


def _batch_with_retry(fn, max_attempts: int = 4, wait_seconds: float = 130.0):
    """Retry ``fn()`` on OpenRouter's HTTP 402 "in-flight budget exhausted" -- a transient
    per-account concurrency cap (too many requests in flight at once across this whole batch
    or a recent one), not the ordinary "out of credits" 402 and not retried by
    :meth:`OpenRouterChat.complete`, which treats every non-429 4xx as permanent. Firing 100
    prompts through a thread pool for one window, then another 100 for the very next window
    moments later, is exactly the shape that trips this -- the account-wide budget can still be
    settling from the first burst. Any other error (including an ordinary 402) is not retried.
    """
    for attempt in range(max_attempts):
        try:
            return fn()
        except RuntimeError as err:
            msg = str(err)
            if "402" not in msg or "in_flight_budget" not in msg or attempt == max_attempts - 1:
                raise
            print(f"  OpenRouter in-flight budget exhausted (attempt {attempt + 1}/"
                  f"{max_attempts}); waiting {wait_seconds:.0f}s before retrying...")
            time.sleep(wait_seconds)


def run_judge_once(prompt_fn, system_prompt_template: str, known_texts: list[str],
                   query_texts: list[str], col_index: np.ndarray, document_index: np.ndarray,
                   judge_model: str, cache_dir, cache_namespace: str, seed: int,
                   snippet_chars: int, max_tokens: int = 8) -> tuple[np.ndarray, int]:
    """One judge pass over an already-built shortlist: shuffle presentation order (cancels
    position bias), build prompts, call the model (cached), parse each pick back to a column
    index in the real WCCN score matrix. A refusal / invalid digit falls back to WCCN's own #1
    (``col_index[i, 0]``) rather than being left unresolved.

    Returns ``(picks, n_refused)`` where ``picks[i]`` is the column of ``base_scores`` to
    promote for row ``i``.
    """
    n, k = col_index.shape
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(k) for _ in range(n)]
    present_cols = [col_index[i][perms[i]] for i in range(n)]
    present_docs = [document_index[i][perms[i]] for i in range(n)]

    prompts = [
        prompt_fn(query_texts[i][:snippet_chars],
                  [known_texts[j][:snippet_chars] for j in present_docs[i]])
        for i in range(n)
    ]
    system_prompt = system_prompt_template.format(n=k)
    client = OpenRouterChat(judge_model, system_prompt, max_tokens=max_tokens)
    if cache_dir is not None:
        cache = TransformCache(
            Path(cache_dir) / "attacks", cache_namespace,
            logic_hash([OpenRouterChat, prompt_fn], version="2"),
            params_hash({"judge_model": judge_model, "system_prompt": system_prompt,
                        "top_k": k, "snippet_chars": snippet_chars, "seed": seed}),
        )
        raw_choices = _batch_with_retry(lambda: cache.apply_batch(prompts, client.complete_batch))
    else:
        raw_choices = _batch_with_retry(lambda: client.complete_batch(prompts))
    choices = [_parse_choice(r) for r in raw_choices]

    picks = np.empty(n, dtype=int)
    refused = 0
    for i, choice in enumerate(choices):
        if 1 <= choice <= k:
            picks[i] = present_cols[i][choice - 1]
        else:
            picks[i] = col_index[i, 0]
            refused += 1
    return picks, refused


def promote(base_scores_subset: np.ndarray, picks: np.ndarray) -> np.ndarray:
    final = base_scores_subset.copy()
    row_max = base_scores_subset.max(axis=1)
    final[np.arange(len(picks)), picks] = row_max + 1.0
    return final


def window_metrics(scores, candidate_authors, true_authors):
    ranks = true_author_ranks(scores, candidate_authors, true_authors)
    summary = ranking_summary(ranks, len(candidate_authors))
    return {"macro_conv_acc1": macro_top_k_accuracy(ranks, true_authors, k=1), "mrr": summary["mrr"],
           "ranks": ranks}


def recall_at_k(ranks: np.ndarray, true_authors: np.ndarray, k: int) -> tuple[float, float]:
    """``(micro, macro)`` recall@k -- the oracle top-1 ceiling for a reranker restricted to
    WCCN's own top-k, at document weight and at the project's actual headline weight.

    ``macro_top_k_accuracy`` already averages per-author accuracy before averaging over
    authors; the earlier version of this reported only the micro (document-weighted) figure
    next to a macro headline metric (``macro_conv_acc1``), which is the wrong ceiling for that
    number -- a corpus with a few authors carrying many documents can have a much higher micro
    than macro recall, so the printed "oracle ceiling" overstated how much headroom the macro
    metric actually had.
    """
    micro = float((ranks <= k).mean())
    macro = macro_top_k_accuracy(ranks, true_authors, k=k)
    return micro, macro


def rescue_counts(ranks_before: np.ndarray, ranks_after: np.ndarray) -> dict[str, int]:
    """Per-document correctness transitions on a judged subset: how many rows the judge
    RESCUED (WCCN wrong at rank 1, judge right) versus DAMAGED (WCCN right, judge wrong).

    ``net = rescued - damaged`` is the number that actually says whether reranking is a net
    win -- a hard-subset accuracy increase can still hide a judge that overturns a lot of
    correct WCCN picks along the way, which the aggregate accuracy alone would not show.
    """
    before_correct = ranks_before == 1
    after_correct = ranks_after == 1
    rescued = int((~before_correct & after_correct).sum())
    damaged = int((before_correct & ~after_correct).sum())
    return {
        "rescued": rescued, "damaged": damaged, "net": rescued - damaged,
        "unchanged_correct": int((before_correct & after_correct).sum()),
        "unchanged_wrong": int((~before_correct & ~after_correct).sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="swe_chat")
    parser.add_argument("--data-dir", default="data/hf")
    parser.add_argument("--feature-set", default="gemini_embedding_2+char_ngram+pos_ngram")
    parser.add_argument("--shrinkage", type=float, default=0.6)
    parser.add_argument("--known-windows", nargs="+", default=list(DEFAULT_KNOWN_WINDOWS))
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    parser.add_argument("--n-ambiguous", type=int, default=30,
                        help="documents judged per window (= API calls per window, per "
                             "majority-vote run)")
    parser.add_argument("--judge-model", default="anthropic/claude-sonnet-5")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--majority-vote-runs", type=int, default=1,
                        help="judge each ambiguous row this many times (different candidate "
                             "shuffles) and promote the mode pick, instead of trusting one "
                             "call. 1 = single run (default).")
    parser.add_argument("--judge-variant", default="plain", choices=["plain", "feature_grounded"],
                        help="'plain': raw text only. 'feature_grounded': SALA-style prompt "
                             "with computed style stats alongside each text.")
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

    prompt_fn, system_prompt_template, cache_namespace = {
        "plain": (plain_prompt_fn, JUDGE_SYSTEM_PROMPT_TEMPLATE, "wccn_judge_plain"),
        "feature_grounded": (feature_grounded_prompt_fn, FEATURE_GROUNDED_SYSTEM_PROMPT_TEMPLATE,
                             "wccn_judge_feature_grounded"),
    }[args.judge_variant]

    results = {"whole_window_before": [], "whole_window_after": [],
              "ambiguous_before": [], "ambiguous_after": []}
    recall_rows = []
    rescue_rows = []
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
        candidate_authors = wccn.authors
        base_scores = wccn.score(u_feat_in)           # the REAL WCCN score -- used throughout
        proj_known = wccn.project(k_feat)
        proj_unknown = wccn.project(u_feat_in)

        m_before_all = window_metrics(base_scores, candidate_authors, unknown_labels_in)
        micro_r5, macro_r5 = recall_at_k(m_before_all["ranks"], unknown_labels_in, args.top_k)
        recall_rows.append((micro_r5, macro_r5))

        n_ambiguous = min(args.n_ambiguous, len(unknown_labels_in))
        top1 = base_scores.max(axis=1)
        top2 = np.partition(base_scores, -2, axis=1)[:, -2]
        margins = top1 - top2
        ambiguous_idx = np.argsort(margins)[:n_ambiguous]

        known_texts = texts[known_slice]
        unknown_texts_in = texts[unknown_slice][in_set]
        base_scores_subset = base_scores[ambiguous_idx]
        true_subset = unknown_labels_in[ambiguous_idx]

        col_index, document_index = wccn_shortlist(
            base_scores_subset, candidate_authors, proj_known, proj_unknown[ambiguous_idx],
            known_labels, args.top_k)

        print(f"\n=== {config.tag} ({config.label}) known={len(known_labels)} "
              f"unknown={in_set.sum()} in-set, judging {n_ambiguous}/{in_set.sum()} "
              f"({n_ambiguous / in_set.sum():.1%}) most-ambiguous ===")
        print(f"  WCCN whole-window: macro_conv_acc1={m_before_all['macro_conv_acc1']:.3f} "
              f"mrr={m_before_all['mrr']:.3f}  "
              f"micro_recall@{args.top_k}={micro_r5:.3f}  macro_recall@{args.top_k}={macro_r5:.3f} "
              f"(macro = the oracle top-1 ceiling for the headline macro_conv_acc1 metric)")

        before_subset = window_metrics(base_scores_subset, candidate_authors, true_subset)

        query_texts_subset = [unknown_texts_in[i] for i in ambiguous_idx]
        if args.majority_vote_runs > 1:
            all_picks, total_refused = [], 0
            for r in range(args.majority_vote_runs):
                picks, refused = run_judge_once(
                    prompt_fn, system_prompt_template, known_texts, query_texts_subset,
                    col_index, document_index, args.judge_model, args.cache_dir,
                    f"{cache_namespace}_run{r}", args.seed + r, DEFAULT_SNIPPET_CHARS)
                all_picks.append(picks)
                total_refused += refused
            all_picks = np.array(all_picks)
            final_picks = np.empty(n_ambiguous, dtype=int)
            unanimous = 0
            for i in range(n_ambiguous):
                counts = Counter(all_picks[:, i].tolist())
                best_count = max(counts.values())
                tied = {col for col, c in counts.items() if c == best_count}
                # A tie (including every run disagreeing) breaks toward WCCN's own ranking
                # among the tied columns, rather than Counter's arbitrary first-seen order.
                final_picks[i] = next(col for col in col_index[i] if col in tied)
                unanimous += best_count == args.majority_vote_runs
            print(f"  majority vote over {args.majority_vote_runs} runs: "
                  f"{unanimous}/{n_ambiguous} rows unanimous, {total_refused} refusals total")
            n_calls = n_ambiguous * args.majority_vote_runs
        else:
            final_picks, refused = run_judge_once(
                prompt_fn, system_prompt_template, known_texts, query_texts_subset,
                col_index, document_index, args.judge_model, args.cache_dir,
                cache_namespace, args.seed, DEFAULT_SNIPPET_CHARS)
            print(f"  judge: {refused}/{n_ambiguous} refusals forced to WCCN's own #1")
            n_calls = n_ambiguous

        boosted_subset = promote(base_scores_subset, final_picks)
        after_subset = window_metrics(boosted_subset, candidate_authors, true_subset)
        total_judged += n_calls

        full_after = base_scores.copy()
        full_after[ambiguous_idx] = boosted_subset
        m_after_all = window_metrics(full_after, candidate_authors, unknown_labels_in)

        rescue = rescue_counts(before_subset["ranks"], after_subset["ranks"])
        rescue_rows.append(rescue)

        results["whole_window_before"].append(m_before_all)
        results["whole_window_after"].append(m_after_all)
        results["ambiguous_before"].append(before_subset)
        results["ambiguous_after"].append(after_subset)
        print(f"  ambiguous subset   before: macro_conv_acc1={before_subset['macro_conv_acc1']:.3f} "
              f"mrr={before_subset['mrr']:.3f}")
        print(f"  ambiguous subset   after:  macro_conv_acc1={after_subset['macro_conv_acc1']:.3f} "
              f"mrr={after_subset['mrr']:.3f}")
        print(f"  rescued={rescue['rescued']}  damaged={rescue['damaged']}  "
              f"net={rescue['net']:+d}  (out of {n_ambiguous} judged; "
              f"unchanged_correct={rescue['unchanged_correct']} "
              f"unchanged_wrong={rescue['unchanged_wrong']})")
        print(f"  whole window       after:  macro_conv_acc1={m_after_all['macro_conv_acc1']:.3f} "
              f"mrr={m_after_all['mrr']:.3f}  "
              f"(vs {m_before_all['macro_conv_acc1']:.3f}/{m_before_all['mrr']:.3f} before)")

    print("\n" + "=" * 70)
    print(f"total documents judged (== API calls): {total_judged}")
    micro_recalls = [r[0] for r in recall_rows]
    macro_recalls = [r[1] for r in recall_rows]
    print(f"mean micro_recall@{args.top_k}: {np.mean(micro_recalls):.3f}  "
          f"mean macro_recall@{args.top_k}: {np.mean(macro_recalls):.3f} "
          f"(the ceiling for the macro_conv_acc1 headline)")
    total_rescued = sum(r["rescued"] for r in rescue_rows)
    total_damaged = sum(r["damaged"] for r in rescue_rows)
    print(f"total rescued={total_rescued}  damaged={total_damaged}  "
          f"net={total_rescued - total_damaged:+d}  across {total_judged} judged rows")
    print(f"{'variant':<24} {'macro_conv_acc1':>16} {'mrr':>8}")
    print("-" * 70)
    for name, rows in results.items():
        acc = np.mean([r["macro_conv_acc1"] for r in rows])
        mrr = np.mean([r["mrr"] for r in rows])
        print(f"{name:<24} {acc:>16.3f} {mrr:>8.3f}")

    print("\nper-window (for paired significance testing):")
    for name, rows in results.items():
        print(f"{name}_acc = {[round(r['macro_conv_acc1'], 4) for r in rows]}")
        print(f"{name}_mrr = {[round(r['mrr'], 4) for r in rows]}")
    print(f"micro_recall_at_{args.top_k} = {[round(r, 4) for r in micro_recalls]}")
    print(f"macro_recall_at_{args.top_k} = {[round(r, 4) for r in macro_recalls]}")
    print(f"rescued_per_window = {[r['rescued'] for r in rescue_rows]}")
    print(f"damaged_per_window = {[r['damaged'] for r in rescue_rows]}")
    print(f"net_per_window = {[r['net'] for r in rescue_rows]}")


if __name__ == "__main__":
    main()
