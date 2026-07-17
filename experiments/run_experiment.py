#!/usr/bin/env python
"""Linkage re-identification experiment driver (WildChat / SWE-chat).

Selects a dataset, an optional defense, and an attack -- all by name -- using the
``prompt_anonymity`` package, and writes CSV results and PDF plots::

    python experiments/run_experiment.py --dataset swe-chat
    python experiments/run_experiment.py --dataset wildchat --language English \
        --attack nearest_neighbor --defense none

Pipeline (identical regardless of dataset/attack/defense)::

    load_dataset -> apply_defense -> apply_featurizer -> run_attack -> LinkageRanking
                 -> headline_accuracy + pool_size_sweep -> CSVs + plots

Features are (re)computed *after* the defense, so a defense that rewrites text is reflected in
the vectors. The loaded dataset's committed features are passed as the featurizer ``reference``,
so text the defense left unchanged keeps its precomputed vector (no GPU) and only rewritten
text is recomputed and cached -- with no defense this is an exact, GPU-free pass-through.

Attacks, defenses and featurizers are pluggable through the package registries
(``prompt_anonymity.attacks.ATTACKS``, ``prompt_anonymity.defenses.DEFENSES``,
``prompt_anonymity.features.FEATURIZERS``): the ``--attack`` / ``--defense`` choices below are
read straight from them, so registering a new attack or defense makes it selectable here with
no change to this script.

Outputs (under ``--output-dir``): ``headline_results.csv``, ``sweep_results.csv``,
``topk_accuracy.pdf``, ``poolsize_sweep_top{k}.pdf``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from prompt_anonymity.attacks import ATTACKS, run_attack
from prompt_anonymity.data import load_dataset
from prompt_anonymity.defenses import DEFENSES, apply_defense
from prompt_anonymity.evaluation import LinkageRanking, headline_accuracy, pool_size_sweep
from prompt_anonymity.features import FEATURIZERS, get_featurizer, apply_featurizer
from prompt_anonymity.viz import plot_headline_topk, plot_pool_size_sweep

# Dataset files live in the repo next to this script; the package itself is path-agnostic
# and takes the directory as an argument.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIRS = {"wildchat": REPO_ROOT / "wildchat", "swe-chat": REPO_ROOT / "swe-chat"}
FEATURE_LABELS = {"stylometrix": "StyloMetrix"}  # legend name for the attack/feature series
# StyloMetrix language model code per WildChat language subset (SWE-chat is English-only).
STYLOMETRIX_LANGUAGE_CODES = {"English": "en", "Russian": "ru"}


def build_featurizer(args: argparse.Namespace):
    """Construct the featurizer for this run, configured to match the loaded feature space.

    The featurizer must produce vectors in the same space as the loaded dataset's committed
    features (its ``reference``), so StyloMetrix is built with the matching language code.
    """
    options = {}
    if args.feature == "stylometrix":
        language = args.language if args.dataset == "wildchat" else "English"
        options["language_code"] = STYLOMETRIX_LANGUAGE_CODES[language]
    return get_featurizer(args.feature, **options)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(DATA_DIRS), help="Dataset to attack.")
    parser.add_argument("--feature", default="stylometrix", choices=sorted(FEATURIZERS), help="Conversation representation (only StyloMetrix is wired up; Gemini is planned).")
    parser.add_argument("--attack", default="nearest_neighbor", choices=sorted(ATTACKS), help="Attack to run.")
    parser.add_argument("--defense", default="none", choices=sorted(DEFENSES), help="Defense applied before the attack.")
    parser.add_argument("--language", default="English", choices=["English", "Russian"], help="WildChat language subset.")
    parser.add_argument(
        "--model-owner", default="Anthropic",
        help="SWE-chat: restrict the pool to one agent provider (model_owner), or 'all'.",
    )
    parser.add_argument("--output-dir", default=None, help="Where to write outputs (default: experiments/results/<tag>).")
    parser.add_argument(
        "--cache-dir", default=str(REPO_ROOT / "experiments" / ".cache"),
        help="Directory for cached expensive transforms (defense rewrites under 'defenses/', "
             "featurizer vectors under 'features/').",
    )
    parser.add_argument("--top-ks", type=int, nargs="+", default=[1, 5, 10], help="k values for the headline table.")
    parser.add_argument("--sweep-top-k", type=int, default=1, help="k used for the pool-size sweep.")
    parser.add_argument("--pool-step", type=int, default=25, help="Increment between candidate-pool sizes in the sweep.")
    parser.add_argument("--n-sims", type=int, default=100, help="Random sub-pools drawn per pool size.")
    parser.add_argument("--seed", type=int, default=47, help="Seed for the sweep's random sub-sampling.")
    return parser.parse_args()


def output_tag(args: argparse.Namespace) -> str:
    """Short, self-describing directory name for this run's outputs."""
    if args.dataset == "wildchat":
        scope = f"wildchat_{args.feature}_{args.language.lower()}"
    else:
        owner = "" if args.model_owner.lower() == "all" else f"_{args.model_owner.lower()}"
        scope = f"swe-chat_{args.feature}{owner}"
    defense = "" if args.defense == "none" else f"_{args.defense}"
    return f"{scope}_{args.attack}{defense}"


def main() -> None:
    args = parse_args()

    # Dataset-specific loader options; everything after loading is dataset-agnostic.
    options = {"feature": args.feature}
    if args.dataset == "wildchat":
        options["language"] = args.language
    else:
        options["model_owner"] = args.model_owner

    # Load (text + committed reference features) -> defense (rewrites text) -> featurize.
    # `reference` is the loader's features for the ORIGINAL text; the featurizer reuses them
    # wherever the defense left text unchanged and recomputes/caches only the rewritten text.
    data = load_dataset(args.dataset, DATA_DIRS[args.dataset], **options)
    reference = data
    data = apply_defense(args.defense, data, cache_dir=args.cache_dir)
    featurizer = build_featurizer(args)
    data = apply_featurizer(featurizer, data, cache_dir=args.cache_dir, reference=reference)
    print(
        f"[{args.dataset}] {data.n_identities} identities | "
        f"{data.n_known} known + {data.n_unknown} unknown conversations | "
        f"attack={args.attack} defense={args.defense} feature={featurizer.name} metric={data.metric}"
    )

    # Attack -> distance matrix -> ranking reused by both the headline table and sweep.
    distances = run_attack(args.attack, data)
    ranking = LinkageRanking(distances, data.known_labels, data.unknown_labels)
    headline = headline_accuracy(ranking, top_ks=tuple(args.top_ks))
    print("\nHeadline (full pool):")
    for _, row in headline.iterrows():
        print(
            f"  top {int(row['top']):>2}: conv_acc={row['conv_acc']:.3f}  id_acc={row['id_acc']:.3f}  "
            f"random_id={row['random_id']:.3f}  advantage={row['advantage']:.3f}"
        )

    print(f"\nPool-size sweep (top-{args.sweep_top_k}, {args.n_sims} sims/size)...")
    sweep = pool_size_sweep(
        ranking, pool_step=args.pool_step, n_sims=args.n_sims, top_k=args.sweep_top_k, seed=args.seed
    )

    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "experiments" / "results" / output_tag(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    headline.to_csv(output_dir / "headline_results.csv", index=False)
    sweep.to_csv(output_dir / "sweep_results.csv", index=False)

    method_label = FEATURE_LABELS.get(args.feature, args.feature)
    plot_headline_topk(headline, method_label, output_dir / "topk_accuracy.pdf")
    plot_pool_size_sweep(sweep, method_label, args.sweep_top_k, output_dir / f"poolsize_sweep_top{args.sweep_top_k}.pdf")

    print(f"\nWrote results to {output_dir}/")
    print("  headline_results.csv, sweep_results.csv, topk_accuracy.pdf, "
          f"poolsize_sweep_top{args.sweep_top_k}.pdf")


if __name__ == "__main__":
    main()
