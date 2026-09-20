"""Controlled feature x attack matrix for the WCCN-fusion result, in one reproducible runner.

Consolidates what had grown into ~8 near-duplicate scratch scripts (one per source x feature-set
combination, several with copy-pasted fusion logic) discovered while chasing one question: does
Gemini + {char n-gram, StyloMetrix, POS n-gram} fusion help, and does the answer depend on which
attack scores the fused vectors? It did -- fusion looked null or negative on WildChat under
``logistic_sgd`` (the only classifier attack feasible at WildChat's author count, see
``run_all_experiments.py``'s documented 20.4 GB logit-matrix constraint) but showed a real,
significant MRR gain under WCCN, which has no such constraint and is independently the strongest
single-feature attack on both corpora. One script that takes the feature set and attack as
arguments, rather than one file per combination, is what makes that comparison reproducible
instead of re-derived by hand each time.

Two feature kinds, two safety stories
--------------------------------------
* **Precomputed, no cross-document dependency** (``gemini_embedding_2``, ``stylometrix``): each
  document's vector depends only on its own text, so :func:`load_documents_and_features` reads
  the existing feature parquet and slicing by window is the only per-window work.
* **Per-split TF-IDF** (``char_ngram``, ``pos_ngram``): vocabulary and IDF weights are corpus
  statistics, so fitting them once over the whole timeline would leak future documents' vocabulary
  into the attacker's known-side representation (this is also why neither is used through its
  ``compute_features`` parquet -- see ``pos_ngram_tfidf.py``'s docstring). A fresh featurizer
  instance is constructed per window and fit on the known slice only; the unknown slice is
  transformed, never refit against.

Usage
-----
::

    python experiments/run_wccn_fusion.py --source wildchat \\
        --feature-sets gemini_embedding_2 gemini_embedding_2+char_ngram \\
                       gemini_embedding_2+stylometrix gemini_embedding_2+pos_ngram \\
        --attacks wccn --pos-cache "pos_tags_cache_wildchat_shard*of4.jsonl"

Each ``--feature-sets`` entry is one or more ``+``-joined components, scored separately; every
combination is run against every requested attack, on every window in ``--known-windows``. Output
is one CSV row per (window, feature set, attack) with the git SHA, WCCN shrinkage, per-component
dimensionality, candidate/known/unknown counts and wall time it took, plus the same per-window
Python-literal arrays the significance-testing step (paired t-test / Wilcoxon across the six
windows) has been reading from every prior scratch script's stdout.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_experiment import (  # noqa: E402
    DEFAULT_KNOWN_WINDOWS, DEFAULT_TEST_FRACTION,
    known_configurations, load_documents_and_features, standardize,
)
from prompt_anonymity.attacks import ATTRIBUTION_ATTACKS  # noqa: E402
from prompt_anonymity.evaluation.metrics.ranking import (  # noqa: E402
    macro_top_k_accuracy, ranking_summary, true_author_ranks,
)
from prompt_anonymity.features.char_ngram_tfidf import CharNgramTfidfFeaturizer  # noqa: E402
from prompt_anonymity.features.pos_ngram_tfidf import POSNgramTfidfFeaturizer  # noqa: E402

#: Attacks whose ``shrinkage`` this script's ``--shrinkage`` controls; every other attack is
#: constructed with no arguments, matching the defaults every prior scratch script used.
SHRINKAGE_ATTACKS = {"wccn", "plda"}

#: Components fit fresh per window from raw text/cached tags, rather than read from a feature
#: parquet. Maps the ``--feature-sets`` token to (featurizer class, text source key).
PER_SPLIT_FEATURIZERS = {
    "char_ngram": (CharNgramTfidfFeaturizer, "turns"),
    "pos_ngram": (POSNgramTfidfFeaturizer, "pos_tags"),
}

TURN_SEPARATOR = "\n\n"


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()


def load_turns_text(data_dir: str, source: str, doc_ids: list[str]) -> dict[str, str]:
    table = pq.read_table(Path(data_dir) / f"{source}.parquet", columns=["doc_id", "turns"])
    by_id = dict(zip(table.column("doc_id").to_pylist(), table.column("turns").to_pylist()))
    missing = [d for d in doc_ids if d not in by_id]
    if missing:
        raise SystemExit(f"{len(missing):,} documents missing from {source}.parquet's turns column")
    return {d: TURN_SEPARATOR.join(by_id[d]) for d in doc_ids}


def load_pos_tags(cache_glob: str, doc_ids: list[str]) -> dict[str, str]:
    paths = sorted(glob.glob(cache_glob))
    if not paths:
        raise SystemExit(f"--pos-cache {cache_glob!r} matched no files -- run the "
                         f"pos_tag_precompute script(s) first.")
    by_id = {}
    for path in paths:
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                by_id[row["doc_id"]] = row["pos_tags"]
    missing = [d for d in doc_ids if d not in by_id]
    if missing:
        raise SystemExit(f"{len(missing):,}/{len(doc_ids):,} documents have no cached POS tags "
                         f"in {cache_glob!r} (e.g. {missing[0]!r}).")
    return by_id


def build_component(name: str, known_slice: slice, unknown_slice: slice,
                    precomputed: dict[str, np.ndarray], doc_ids: list[str],
                    turns_by_id: dict[str, str] | None, pos_by_id: dict[str, str] | None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """One feature block's standardized (known, unknown) matrices for this window.

    A fresh featurizer instance per call for the per-split kind: ``featurize()`` fits on its
    first call, so calling known first (fit) then unknown second (transform-only) on this
    instance is what keeps the fit known-side-only. Never reuse an instance across windows.
    """
    if name in PER_SPLIT_FEATURIZERS:
        featurizer_cls, text_key = PER_SPLIT_FEATURIZERS[name]
        by_id = turns_by_id if text_key == "turns" else pos_by_id
        featurizer = featurizer_cls()
        k = featurizer.featurize([by_id[d] for d in doc_ids[known_slice]])
        u = featurizer.featurize([by_id[d] for d in doc_ids[unknown_slice]])
    else:
        k, u = precomputed[name][known_slice], precomputed[name][unknown_slice]
    return standardize(k, u)


def build_matrices(components: list[str], known_slice: slice, unknown_slice: slice,
                    precomputed: dict[str, np.ndarray], doc_ids: list[str],
                    turns_by_id: dict[str, str] | None, pos_by_id: dict[str, str] | None
                    ) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Early (concatenation) fusion: standardize each component, hstack, standardize again."""
    blocks = [build_component(name, known_slice, unknown_slice, precomputed, doc_ids,
                              turns_by_id, pos_by_id) for name in components]
    dims = {name: k.shape[1] for name, (k, u) in zip(components, blocks)}

    if len(blocks) == 1:
        k, u = blocks[0]
        return k, u, dims
    fused_known = np.hstack([k for k, u in blocks])
    fused_unknown = np.hstack([u for k, u in blocks])
    fused_known, fused_unknown = standardize(fused_known, fused_unknown)
    return fused_known, fused_unknown, dims


def zscore(matrix: np.ndarray) -> np.ndarray:
    """Whole-matrix z-score: puts one attack's raw score scale on par with another's."""
    return (matrix - matrix.mean()) / (matrix.std() + 1e-12)


def score_fusion_predict(components: list[str], known_slice: slice, unknown_slice: slice,
                         known_labels: np.ndarray, attack_name: str, attack_kwargs: dict,
                         alpha: float, precomputed: dict[str, np.ndarray], doc_ids: list[str],
                         turns_by_id: dict[str, str] | None, pos_by_id: dict[str, str] | None
                         ) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Late fusion: fit one attack per component, then combine z-scored author scores.

    ``combined = z(score_0) + alpha * sum(z(score_i) for i > 0)`` -- the first component anchors
    the scale, ``alpha`` weights every other component equally against it. Simpler than a
    per-pair alpha, and enough to test whether score-level combination beats concatenating raw
    features before the first component gets fit against as one attack (``run_wccn_fusion``'s
    default mode).
    """
    combined, authors, dims = None, None, {}
    for i, name in enumerate(components):
        k, u = build_component(name, known_slice, unknown_slice, precomputed, doc_ids,
                               turns_by_id, pos_by_id)
        dims[name] = k.shape[1]
        attack = ATTRIBUTION_ATTACKS[attack_name](**attack_kwargs).fit(k, known_labels)
        scores = zscore(attack.score(u))
        if authors is None:
            authors = attack.authors
        else:
            assert list(authors) == list(attack.authors), \
                "candidate order differs between components fit on the same known_labels"
        weight = 1.0 if i == 0 else alpha
        combined = scores * weight if combined is None else combined + scores * weight
    return combined, authors, dims


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="swe_chat")
    parser.add_argument("--data-dir", default="data/hf")
    parser.add_argument("--feature-sets", nargs="+", required=True,
                        help="one or more '+'-joined component lists, e.g. "
                             "gemini_embedding_2 gemini_embedding_2+char_ngram")
    parser.add_argument("--attacks", nargs="+", default=["wccn"],
                        choices=sorted(ATTRIBUTION_ATTACKS))
    parser.add_argument("--shrinkage", type=float, default=0.2,
                        help="passed to wccn/plda only")
    parser.add_argument("--fusion-mode", default="concat", choices=["concat", "score"],
                        help="'concat' (default): standardize each component, hstack, fit one "
                             "attack. 'score': fit one attack PER component, z-score each "
                             "component's score matrix, combine with --score-alpha. Only "
                             "meaningful for a --feature-sets entry with 2+ components; a "
                             "single-component entry is identical under both modes.")
    parser.add_argument("--score-alpha", type=float, default=1.0,
                        help="score mode only: weight on every component after the first "
                             "(the first anchors the scale). Select this on one held-out "
                             "window, then lock it -- do not pick it from the same windows "
                             "the final numbers are reported on.")
    parser.add_argument("--known-windows", nargs="+", default=list(DEFAULT_KNOWN_WINDOWS))
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    parser.add_argument("--pos-cache", default=None,
                        help="glob for cached POS-tag jsonl files, required if any --feature-sets "
                             "entry uses 'pos_ngram' (e.g. "
                             "'pos_tags_cache_wildchat_shard*of4.jsonl')")
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--feature-dir", action="append", default=[], metavar="FEATURE=DIR",
                        help="load one precomputed feature's parquet from a different "
                             "directory than --data-dir (repeatable), e.g. luar=data/dist "
                             "for a feature whose vectors were computed alongside a "
                             "different document filtering/build. Re-aligned by doc_id "
                             "rather than assumed to share row order with --data-dir.")
    return parser.parse_args()


def load_feature_by_doc_id(data_dir: str, source: str, feature: str,
                           doc_ids: list[str]) -> np.ndarray:
    """One feature's matrix, reordered to ``doc_ids`` -- for a feature computed against a
    different document build (different filtering/ordering, or no co-located ``<source>.parquet``
    at all) than the run's anchor feature, so row position cannot be trusted across the two and
    there may be no documents file in ``data_dir`` to join through in the first place. Reads the
    feature parquet directly rather than going through :func:`load_documents_and_features`."""
    path = Path(data_dir) / f"{source}_{feature}.parquet"
    if not path.exists():
        raise SystemExit(f"{path} not found.")
    table = pq.read_table(path)
    columns = [c for c in table.column_names if c not in ("doc_id", "author_id")]
    by_id = {d: i for i, d in enumerate(table.column("doc_id").to_pylist())}
    missing = [d for d in doc_ids if d not in by_id]
    if missing:
        raise SystemExit(f"{len(missing):,}/{len(doc_ids):,} documents have no {feature} vector "
                         f"in {path} (e.g. {missing[0]!r}).")
    rows = np.array([by_id[d] for d in doc_ids])
    return np.column_stack([table.column(c).to_numpy()[rows] for c in columns]).astype(np.float32)


def main():
    args = parse_args()
    feature_sets = [(label, label.split("+")) for label in args.feature_sets]
    precomputed_needed = sorted({c for _, comps in feature_sets for c in comps
                                 if c not in PER_SPLIT_FEATURIZERS})
    needs_char = any("char_ngram" in comps for _, comps in feature_sets)
    needs_pos = any("pos_ngram" in comps for _, comps in feature_sets)
    if needs_pos and not args.pos_cache:
        raise SystemExit("--pos-cache is required when a feature set uses 'pos_ngram'")

    feature_dirs = {}
    for spec in args.feature_dir:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--feature-dir {spec!r}: expected FEATURE=DIR")
        feature_dirs[name] = path

    sha = git_sha()
    print(f"git SHA {sha}  source={args.source}  attacks={args.attacks}  shrinkage={args.shrinkage}")
    if feature_dirs:
        print(f"feature directory overrides: {feature_dirs}")

    frame = None
    precomputed: dict[str, np.ndarray] = {}
    anchor_candidates = [f for f in precomputed_needed if f not in feature_dirs]
    anchor_feature = anchor_candidates[0] if anchor_candidates else "gemini_embedding_2"
    for feat in dict.fromkeys([anchor_feature, *anchor_candidates]):
        frame_i, mat = load_documents_and_features(args.data_dir, args.source, feat)
        if frame is None:
            frame = frame_i
        else:
            assert list(frame["doc_id"]) == list(frame_i["doc_id"]), \
                f"row order mismatch loading {feat}"
        if feat in precomputed_needed:
            precomputed[feat] = mat
    print(f"{len(frame):,} documents loaded")

    authors = frame["author_id"].to_numpy()
    doc_ids = frame["doc_id"].tolist()

    for feat in [f for f in precomputed_needed if f in feature_dirs]:
        precomputed[feat] = load_feature_by_doc_id(feature_dirs[feat], args.source, feat, doc_ids)
        print(f"{feat}: {precomputed[feat].shape[1]} dims, loaded from {feature_dirs[feat]} "
              f"(re-aligned by doc_id)")

    turns_by_id = load_turns_text(args.data_dir, args.source, doc_ids) if needs_char else None
    pos_by_id = load_pos_tags(args.pos_cache, doc_ids) if needs_pos else None

    configs = known_configurations(len(frame), args.known_windows, args.test_fraction)

    csv_rows = []
    per_window: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for config, known_slice, unknown_slice in configs:
        known_labels = authors[known_slice]
        unknown_labels_all = authors[unknown_slice]
        in_set = np.isin(unknown_labels_all, known_labels)
        unknown_labels_in = unknown_labels_all[in_set]
        n_candidates = int(np.unique(known_labels).size)
        print(f"\n=== {config.tag} ({config.label}) known={known_slice.stop - known_slice.start:,} "
              f"unknown={int(in_set.sum()):,}/{unknown_slice.stop - unknown_slice.start:,} in-set ===")

        use_score_fusion = args.fusion_mode == "score"
        for label, comps in feature_sets:
            for attack_name in args.attacks:
                t0 = time.time()
                kwargs = {"shrinkage": args.shrinkage} if attack_name in SHRINKAGE_ATTACKS else {}

                if use_score_fusion and len(comps) > 1:
                    scores_all, cand_authors, dims = score_fusion_predict(
                        comps, known_slice, unknown_slice, known_labels, attack_name, kwargs,
                        args.score_alpha, precomputed, doc_ids, turns_by_id, pos_by_id)
                    scores = scores_all[in_set]
                else:
                    k, u, dims = build_matrices(comps, known_slice, unknown_slice, precomputed,
                                                doc_ids, turns_by_id, pos_by_id)
                    attack = ATTRIBUTION_ATTACKS[attack_name](**kwargs).fit(k, known_labels)
                    scores = attack.score(u[in_set])
                    cand_authors = attack.authors
                runtime = time.time() - t0

                ranks = true_author_ranks(scores, cand_authors, unknown_labels_in)
                summary = ranking_summary(ranks, len(cand_authors))
                acc1 = macro_top_k_accuracy(ranks, unknown_labels_in, k=1)

                mode_tag = f"{label}[score,a={args.score_alpha}]" if use_score_fusion and len(comps) > 1 else label
                m = {"macro_conv_acc1": acc1, "mrr": summary["mrr"]}
                per_window[(mode_tag, attack_name)].append(m)
                print(f"  {mode_tag:<40} {attack_name:<14} "
                      f"macro_conv_acc1={acc1:.4f}  mrr={summary['mrr']:.4f}")

                csv_rows.append({
                    "source": args.source, "known_config": config.tag,
                    "feature_set": label, "fusion_mode": args.fusion_mode,
                    "score_alpha": args.score_alpha if use_score_fusion and len(comps) > 1 else "",
                    "attack": attack_name,
                    "shrinkage": args.shrinkage if attack_name in SHRINKAGE_ATTACKS else "",
                    "macro_conv_acc1": round(acc1, 6), "mrr": round(summary["mrr"], 6),
                    "n_known": known_slice.stop - known_slice.start,
                    "n_unknown_in_set": int(in_set.sum()), "n_candidates": n_candidates,
                    "dims": json.dumps(dims), "runtime_sec": round(runtime, 2),
                    "seed": args.seed, "git_sha": sha,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

    output_csv = Path(args.output_csv) if args.output_csv else (
        REPO_ROOT / "experiments" / "results" /
        f"wccn_fusion_{args.source}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.csv"
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nwrote {len(csv_rows)} rows to {output_csv}")

    print("\n" + "=" * 70)
    print(f"{'feature_set':<32} {'attack':<14} {'macro_conv_acc1':>16} {'mrr':>8}")
    print("-" * 70)
    summary_rows = []
    for (label, attack_name), rows in per_window.items():
        acc = np.mean([r["macro_conv_acc1"] for r in rows])
        mrr = np.mean([r["mrr"] for r in rows])
        summary_rows.append((acc, mrr, label, attack_name))
    for acc, mrr, label, attack_name in sorted(summary_rows, reverse=True):
        print(f"{label:<32} {attack_name:<14} {acc:>16.3f} {mrr:>8.3f}")

    print("\nper-window (for paired significance testing):")
    for (label, attack_name), rows in per_window.items():
        key = f"{label.replace('+', '_')}__{attack_name}"
        print(f"{key}_acc = {[round(r['macro_conv_acc1'], 4) for r in rows]}")
        print(f"{key}_mrr = {[round(r['mrr'], 4) for r in rows]}")


if __name__ == "__main__":
    main()
