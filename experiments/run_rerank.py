#!/usr/bin/env python
"""Listwise reranking attacks: reorder the whole shortlist, not just its head.

``run_experiment.py`` drives the estimator-shaped attacks -- everything with ``fit``/``score`` over
feature vectors. The rerankers are a different shape: they take a whole
:class:`~prompt_anonymity.core.AttackData` because they read the conversation *text*, and they need
a cache directory, an API key and in one case a GPU. This script is their runner.

Two attacks, one shortlist
--------------------------
Both start from the same place :mod:`~prompt_anonymity.attacks.llm.euclidean_llm_judge` does -- the
top-K most likely authors under nearest-neighbor linkage, each represented by their own known
conversation nearest to the unknown one -- and both leave that shortlist's *membership* untouched.
What they change is the order within it, all of it:

``listwise_llm_rerank``   Claude Sonnet 5 over OpenRouter, reasoning on, ranking all K and writing
                          a sentence or two per position. Batched by default (~50% of the real-time
                          price, 24-hour window). Needs ``SONNET_OR_KEY`` in a ``.env`` at the repo
                          root.
``listwise_jina_rerank``  the same shortlist scored locally by ``jina-reranker-v3.5`` (0.6B). Free,
                          needs a GPU, and is the control: it prices the frontier model rather than
                          the idea of reranking.

Run both at ``--top-k 5`` and ``--top-k 10`` and the comparison is four curves against one baseline.

Reading the output
------------------
``rerank_summary.csv`` carries the reranked top-k accuracies beside the plain nearest-neighbor ones
on the same rows, so the gain is a subtraction rather than a second run. **Top-K at the shortlist
size is identical to the baseline by construction** -- the reranker cannot add an author the
distance metric did not shortlist -- and this script asserts that rather than trusting it; a
mismatch means the fold-back leaked outside the shortlist and every other number is suspect.

``rerank_detail_*.csv`` is the per-candidate table: where each candidate was shown, where the
reranker put it, what distance rank it came from, and -- for the LLM -- why. That ``reason`` column
is the point of the exercise as much as the accuracy is: a judge that is reading topic rather than
style says so in it, and no aggregate will.

Cost
----
The LLM arm is real money: roughly $0.01-0.015 per unknown document at ``--top-k 10``, half that at
5. Verdicts are cached by prompt text, so start with a small ``--limit``, read the reasons, and
raise it once the rubric looks right -- re-running only judges what is new. ``--no-wait`` submits
the batches and stops, which is what a login node wants: re-run the same command tomorrow to
collect, resuming the same batches rather than paying for them twice.

    # free, local, both shortlist sizes
    python experiments/run_rerank.py --source swe_chat --feature luar --variant jina \\
        --top-k 5 --top-k 10 --limit 200

    # paid; real-time and tiny first, to look at the reasons
    python experiments/run_rerank.py --source swe_chat --feature luar --variant llm \\
        --top-k 5 --limit 20 --no-batch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Both scripts live here; run_experiment owns the loaders, the window arithmetic and the
# standardizer, and none of that should exist twice.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_experiment import (  # noqa: E402
    DATA_DIR,
    NO_DEFENSE_TAG,
    known_configurations,
    load_documents_and_features,
    standardize,
)

from prompt_anonymity.attacks import NearestNeighbor  # noqa: E402
from prompt_anonymity.attacks.llm import (  # noqa: E402
    ListwiseJinaRerankAttack,
    ListwiseLLMRerankAttack,
)
from prompt_anonymity.attacks.llm.listwise_jina_rerank import (  # noqa: E402
    DEFAULT_SNIPPET_CHARS as JINA_SNIPPET_CHARS,
)
from prompt_anonymity.attacks.llm.listwise_llm_rerank import (  # noqa: E402
    DEFAULT_SNIPPET_CHARS as LLM_SNIPPET_CHARS,
)
from prompt_anonymity.core import AttackData  # noqa: E402
from prompt_anonymity.data import config as data_config  # noqa: E402
from prompt_anonymity.data.compute_features import read_texts, split_path  # noqa: E402
from prompt_anonymity.evaluation import true_author_ranks  # noqa: E402

#: The cutoffs the summary table reports. 10 is included even at ``--top-k 5``, where it is pinned
#: to the baseline along with 5 -- seeing both pinned is the cheapest confirmation that the
#: fold-back stayed inside the shortlist.
SUMMARY_KS = (1, 2, 3, 5, 10)

#: Linkage used to build the shortlist. Must match
#: :func:`prompt_anonymity.attacks.llm.candidates.author_candidates`'s default, or the baseline this
#: script reports is not the baseline the attacks reranked.
SHORTLIST_LINKAGE = "max"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerank a nearest-neighbor attack's top-K with an LLM or a local reranker.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", default="swe_chat", help="corpus to attack")
    parser.add_argument("--feature", default="luar", help="feature parquet to rank with")
    parser.add_argument("--defense", default="none",
                        help="which defended feature file to read ('none' for undefended)")
    parser.add_argument("--data-dir", default=str(DATA_DIR),
                        help="directory holding <source>.parquet and the feature parquets")
    parser.add_argument("--metric", default="cosine", help="distance metric for the base ranking")
    parser.add_argument("--known-window", default="0075",
                        help="known side as four digits, start then end in whole percents")
    parser.add_argument("--test-fraction", type=float, default=0.25,
                        help="share of the corpus held out as the shared test set")
    parser.add_argument("--standardize", action=argparse.BooleanOptionalAction, default=True,
                        help="z-score the features on the known side before ranking")

    parser.add_argument("--variant", choices=("llm", "jina", "both"), default="jina",
                        help="which reranker(s) to run; 'llm' and 'both' cost money")
    parser.add_argument("--top-k", type=int, action="append", dest="top_k",
                        help="shortlist size; repeatable (e.g. --top-k 5 --top-k 10). Default: 5")
    parser.add_argument("--margin-quantile", type=float, default=1.0,
                        help="rerank only rows whose top-1/top-2 margin is at or below this "
                             "quantile; 1.0 reranks every row")
    parser.add_argument("--seed", type=int, default=47,
                        help="seed for the candidate shuffle and the unknown-side sample")
    parser.add_argument("--limit", type=int, default=None,
                        help="score at most this many unknown documents, sampled across the whole "
                             "unknown side (not a head slice). None scores all of them")

    parser.add_argument("--judge-model", default=None,
                        help="OpenRouter slug for the LLM judge (default: the attack's own)")
    parser.add_argument("--reasoning-effort", default=None,
                        help="thinking depth for the LLM judge: low/medium/high/xhigh/max")
    parser.add_argument("--batch", action=argparse.BooleanOptionalAction, default=True,
                        help="submit through OpenRouter's Batch API (~50%% off, 24h window)")
    parser.add_argument("--wait", action=argparse.BooleanOptionalAction, default=True,
                        help="--no-wait submits the batches and stops; re-run later to collect. "
                             "It stops at the FIRST configuration that has uncached rows, so a "
                             "multi-variant or multi-k sweep needs one pass per configuration")

    parser.add_argument("--cache-dir", default=None,
                        help="cache root for verdicts and batch resume tickets "
                             "(default: the project's own cache directory)")
    parser.add_argument("--output-dir", default=None,
                        help="where to write the tables (default: experiments/results/rerank/<tag>)")
    args = parser.parse_args()
    args.top_k = sorted(set(args.top_k or [5]))
    return args


def document_texts(source: str, data_dir, defense: str | None, doc_ids, max_chars: int) -> np.ndarray:
    """Conversation text for ``doc_ids``, truncated to ``max_chars`` as it is read.

    ``turns`` is ~98% of the WildChat parquet -- several GiB once pandas has turned it into Python
    strings -- and a reranker reads the first few hundred characters of each document and nothing
    else. Truncating on the way in is therefore the difference between a few hundred MB and a job
    the scheduler kills, and it costs nothing: the attacks slice to ``snippet_chars`` anyway.

    Positions are row indices into the *split parquet*, while the frame the caller holds has been
    filtered, sorted and re-indexed, so the mapping is rebuilt here from the same file the text is
    about to be read out of -- not from the undefended split, which would silently misalign if a
    defense ever wrote its rows in a different order.
    """
    path = split_path(source, data_dir, defense)
    keys = pd.Index(pd.read_parquet(path, columns=["doc_id"])["doc_id"])
    positions = keys.get_indexer(pd.Index(doc_ids))
    missing = int((positions < 0).sum())
    if missing:
        raise SystemExit(f"{missing:,} of {len(positions):,} documents have no row in {path}.")
    texts = read_texts(source, data_dir, positions, defense=defense)
    return np.array([text[:max_chars] for text in texts], dtype=object)


def base_ranking(known_embeddings, known_labels, unknown_embeddings, metric: str):
    """The plain nearest-neighbor scores the rerankers were handed, and their author labels.

    Recomputed here rather than taken from an attack, so the baseline every table is read against
    is produced by the same code path whichever reranker ran -- and so a run with ``--variant jina``
    alone still reports it.
    """
    ranker = NearestNeighbor(metric=metric, linkage=SHORTLIST_LINKAGE)
    ranker.fit(known_embeddings, known_labels)
    return np.asarray(ranker.score(unknown_embeddings), dtype=float), ranker.authors


def ranks_of(scores, authors, unknown_labels, in_set) -> np.ndarray:
    """Each document's rank for its own author (1 = top), ``NaN`` where no correct answer exists.

    Documents whose author is not on the known side are unattributable by construction, so they get
    no rank -- the same convention ``run_experiment.py`` uses, which is what lets top-k accuracy be
    re-derived from any slice of the predictions table as the share with rank <= k.
    """
    ranks = np.full(len(unknown_labels), np.nan)
    if in_set.any():
        ranks[in_set] = true_author_ranks(np.asarray(scores)[in_set], authors,
                                          np.asarray(unknown_labels)[in_set])
    return ranks


def accuracies(ranks: np.ndarray, prefix: str) -> dict:
    """Top-k accuracy at every cutoff in :data:`SUMMARY_KS`, over the rows that have a rank."""
    scored = ranks[~np.isnan(ranks)]
    return {f"{prefix}top_{k}": (float((scored <= k).mean()) if len(scored) else float("nan"))
            for k in SUMMARY_KS}


def build_attack(variant: str, top_k: int, args: argparse.Namespace):
    """Construct one reranker, passing through only the options the user actually set."""
    shared = {"top_k": top_k, "margin_quantile": args.margin_quantile, "seed": args.seed}
    if variant == "jina":
        return ListwiseJinaRerankAttack(**shared)
    settings = dict(shared, batch=args.batch, wait=args.wait)
    if args.judge_model:
        settings["judge_model"] = args.judge_model
    if args.reasoning_effort:
        settings["reasoning_effort"] = args.reasoning_effort
    return ListwiseLLMRerankAttack(**settings)


def main() -> None:
    args = parse_args()
    defense = None if args.defense == NO_DEFENSE_TAG or args.defense == "none" else args.defense
    cache_dir = Path(args.cache_dir) if args.cache_dir else data_config.cache_dir()

    frame, embeddings = load_documents_and_features(
        args.data_dir, args.source, args.feature, defense=args.defense)
    config, known, unknown = known_configurations(
        len(frame), [args.known_window], args.test_fraction)[0]

    # Sample the unknown side before any text is read: on a paid run the sample IS the budget, and
    # a head slice would be a handful of people on one day rather than a look at the corpus.
    unknown_rows = np.arange(unknown.start, unknown.stop)
    if args.limit is not None and args.limit < len(unknown_rows):
        rng = np.random.default_rng(args.seed)
        unknown_rows = np.sort(rng.choice(unknown_rows, size=args.limit, replace=False))
    known_rows = np.arange(known.start, known.stop)

    known_embeddings, unknown_embeddings = embeddings[known_rows], embeddings[unknown_rows]
    if args.standardize:
        known_embeddings, unknown_embeddings = standardize(known_embeddings, unknown_embeddings)
    known_frame, unknown_frame = frame.iloc[known_rows], frame.iloc[unknown_rows]
    known_labels = known_frame["author_id"].to_numpy()
    unknown_labels = unknown_frame["author_id"].to_numpy()
    known_authors = np.unique(known_labels)
    in_set = np.isin(unknown_labels, known_authors)
    if not in_set.any():
        raise SystemExit(
            f"{config.tag}: none of the {len(unknown_labels):,} unknown documents has an author on "
            f"the known side, so nothing can be scored."
        )
    print(f"{config.tag}: {len(known_labels):,} known documents ({len(known_authors):,} authors), "
          f"{len(unknown_labels):,} unknown ({int(in_set.sum()):,} attributable)")

    # One read for both sides, truncated on arrival. The widest snippet any configured attack uses
    # bounds what is worth keeping; doubling it leaves room to raise --snippet-chars without a
    # re-read being silently wrong.
    max_chars = 2 * max(LLM_SNIPPET_CHARS, JINA_SNIPPET_CHARS)
    print(f"reading conversation text (truncated to {max_chars:,} chars per document) ...")
    data = AttackData(
        known_embeddings=known_embeddings,
        unknown_embeddings=unknown_embeddings,
        known_labels=known_labels,
        unknown_labels=unknown_labels,
        metric=args.metric,
        known_texts=document_texts(args.source, args.data_dir, defense,
                                   known_frame["doc_id"].to_numpy(), max_chars),
        unknown_texts=document_texts(args.source, args.data_dir, defense,
                                     unknown_frame["doc_id"].to_numpy(), max_chars),
        known_ids=known_frame["doc_id"].to_numpy(),
        unknown_ids=unknown_frame["doc_id"].to_numpy(),
    )

    base_scores, authors = base_ranking(known_embeddings, known_labels, unknown_embeddings,
                                        args.metric)
    base_ranks = ranks_of(base_scores, authors, unknown_labels, in_set)
    base_accuracy = accuracies(base_ranks, "base_")
    print("baseline (nearest neighbor): "
          + ", ".join(f"top-{k} {base_accuracy[f'base_top_{k}']:.3f}" for k in SUMMARY_KS))

    # `none` is spelled `base` everywhere downstream (run_experiment.NO_DEFENSE_TAG), so a results
    # directory for the undefended run is named the same way here as under any other script.
    tag = f"{args.source}_{args.defense if defense else NO_DEFENSE_TAG}_{args.feature}_{config.tag}"
    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).resolve().parent / "results" / "rerank" / tag)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = ("llm", "jina") if args.variant == "both" else (args.variant,)
    summary = []
    for variant in variants:
        for top_k in args.top_k:
            print(f"\n=== {variant} rerank, top-{top_k} ===")
            attack = build_attack(variant, top_k, args)
            scores = attack.attack(data, cache_dir=cache_dir).to_numpy()
            ranks = ranks_of(scores, attack.authors, unknown_labels, in_set)

            predictions = pd.DataFrame({
                "position": unknown_rows,
                "doc_id": unknown_frame["doc_id"].to_numpy(),
                "true_author": unknown_labels,
                "author_in_known": in_set,
                "best_author": np.asarray(attack.authors)[scores.argmax(axis=1)],
                "base_best_author": np.asarray(authors)[base_scores.argmax(axis=1)],
                "true_author_rank": ranks,
                "base_true_author_rank": base_ranks,
                "n_candidate_authors": len(attack.authors),
            })
            predictions.to_csv(output_dir / f"rerank_predictions_{variant}_k{top_k}.csv",
                               index=False)
            attack.detail.to_csv(output_dir / f"rerank_detail_{variant}_k{top_k}.csv", index=False)

            row = {"variant": variant, "top_k": top_k, "known_config": config.tag,
                   "source": args.source, "feature": args.feature, "defense": args.defense,
                   "metric": args.metric, "margin_quantile": args.margin_quantile,
                   "n_unknown": len(unknown_labels), "n_attributable": int(in_set.sum()),
                   "n_known_authors": len(authors),
                   "n_changed_top1": int((predictions["best_author"]
                                          != predictions["base_best_author"]).sum()),
                   "judge_cost_usd": getattr(attack, "cost_usd", float("nan")),
                   # Whether the judge actually thought, counted over the replies this run
                   # received (cached rows contribute nothing). nan for the jina arm.
                   **{f"judge_{name}": getattr(attack, "reasoning_stats", {}).get(name, float("nan"))
                      for name in ("replies", "replies_with_reasoning", "reasoning_tokens")},
                   **accuracies(ranks, ""), **base_accuracy}
            summary.append(row)
            print("  " + ", ".join(f"top-{k} {row[f'top_{k}']:.3f} "
                                   f"({row[f'top_{k}'] - row[f'base_top_{k}']:+.3f})"
                                   for k in SUMMARY_KS))

            # The shortlist is the reranker's recall ceiling and it never moves, so top-K at the
            # shortlist size must be the baseline's number to the last bit. If it is not, the
            # fold-back wrote outside the shortlist and every accuracy above is meaningless.
            if top_k in SUMMARY_KS and row[f"top_{top_k}"] != row[f"base_top_{top_k}"]:
                raise SystemExit(
                    f"{variant} top-{top_k} is {row[f'top_{top_k}']:.6f} but the baseline's is "
                    f"{row[f'base_top_{top_k}']:.6f}. Reranking cannot change top-K at the "
                    f"shortlist size -- fold_listwise has touched authors outside the shortlist."
                )

    # Merge into the summary rather than replacing it. The Sonnet job runs one shortlist size per
    # invocation -- a --no-wait submit can only queue one configuration at a time -- and the two
    # variants are separate SLURM jobs, so a run that overwrote this file would leave the summary
    # holding only whatever happened to finish last. Rows are keyed by (variant, top_k): a re-run of
    # a configuration replaces its own row and leaves every other one alone. Same rule as
    # experiments/eval_utility.py, which merges each metric's columns into one per-corpus table.
    summary_path = output_dir / "rerank_summary.csv"
    table = pd.DataFrame(summary)
    if summary_path.exists():
        previous = pd.read_csv(summary_path)
        rewritten = set(zip(table["variant"], table["top_k"]))
        mask = [(str(variant), int(k)) not in rewritten
                for variant, k in zip(previous["variant"], previous["top_k"])]
        # What was actually paid is a fact about the past, and a cached re-run pays nothing -- so it
        # reports nan and must not erase the figure. Carry the recorded cost forward instead, which
        # is the rule experiments/eval_utility.py states for its own judge_cost_usd column.
        if "judge_cost_usd" in previous.columns and "judge_cost_usd" in table.columns:
            paid = {(str(variant), int(k)): cost
                    for variant, k, cost in zip(previous["variant"], previous["top_k"],
                                                previous["judge_cost_usd"])
                    if pd.notna(cost)}
            table["judge_cost_usd"] = [
                paid.get((str(variant), int(k)), cost) if pd.isna(cost) else cost
                for variant, k, cost in zip(table["variant"], table["top_k"],
                                            table["judge_cost_usd"])]
        table = pd.concat([previous[mask], table], ignore_index=True)
    table = table.sort_values(["variant", "top_k"], kind="mergesort")
    table.to_csv(summary_path, index=False)
    print(f"\nwrote {len(summary)} configuration(s) to {output_dir} "
          f"({len(table)} in rerank_summary.csv)")


if __name__ == "__main__":
    main()
