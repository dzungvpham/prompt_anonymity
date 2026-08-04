#!/usr/bin/env python
"""Linkage re-identification experiment driver (WildChat / SWE-chat).

Selects a dataset, an optional defense, and an attack -- all by name -- using the
``prompt_anonymity`` package, and writes CSV results::

    python experiments/run_experiment.py --dataset swe_chat
    python experiments/run_experiment.py --dataset wildchat --language English \
        --attack nearest_neighbor --defense none

Pipeline (identical regardless of dataset/attack/defense)::

    load_dataset -> apply_defense -> [--fidelity] -> apply_featurizer -> run_attack -> LinkageRanking
                 -> headline_accuracy + pool_size_sweep -> CSVs

``--fidelity`` optionally scores how much of the prompt the defense preserved (a utility axis
orthogonal to the attack) before featurizing; see ``prompt_anonymity.fidelity``.

Features are (re)computed *after* the defense, so a defense that rewrites text is reflected in
the vectors. The loaded dataset's committed features are passed as the featurizer ``reference``,
so text the defense left unchanged keeps its precomputed vector (no GPU) and only rewritten
text is recomputed and cached -- with no defense this is an exact, GPU-free pass-through.

Attacks, defenses and featurizers are pluggable through the package registries
(``prompt_anonymity.attacks.ATTRIBUTION_ATTACKS``, ``prompt_anonymity.defenses.DEFENSES``,
``prompt_anonymity.features.FEATURIZERS``): the ``--attack`` / ``--defense`` choices below are
read straight from them, so registering a new attack or defense makes it selectable here with
no change to this script.

Outputs (under ``--output-dir``): ``headline_results.csv``, ``sweep_results.csv``,
``predictions.csv``, and ``fidelity_{metric}.csv`` when ``--fidelity`` is set.

**No figures.** This script produces numbers only; every figure in the project is drawn by
``experiments/plot_results.py``, run separately with no arguments. It finds runs by their
directory name, which is why :func:`output_tag` spells the four axes out in full
(``<dataset>_<defense>_<feature>_<attack>``, with ``base`` for no defense).
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import numpy as np
import pandas as pd

from prompt_anonymity.attacks import ATTRIBUTION_ATTACKS, get_attribution_attack
from prompt_anonymity.data import load_dataset
from prompt_anonymity.defenses import DEFENSES, apply_defense
from prompt_anonymity.evaluation import LinkageRanking, headline_accuracy, pool_size_sweep
from prompt_anonymity.features import FEATURIZERS, get_featurizer, apply_featurizer
from prompt_anonymity.fidelity import FIDELITY_METRICS, run_fidelity

# Dataset files live in the repo next to this script; the package itself is path-agnostic
# and takes the directory as an argument.
REPO_ROOT = Path(__file__).resolve().parent.parent
# Dataset name -> the repo directory its CSVs live in. The name is the one every script and
# results directory uses (``swe_chat``); the *directory* it reads keeps its own older spelling.
DATA_DIRS = {"wildchat": REPO_ROOT / "wildchat", "swe_chat": REPO_ROOT / "swe-chat"}
# StyloMetrix language model code per WildChat language subset (SWE-chat is English-only).
STYLOMETRIX_LANGUAGE_CODES = {"English": "en", "Russian": "ru"}

#: How :func:`output_tag` spells "no defense". The results directory names all four axes
#: positionally, so the undefended case needs a name of its own rather than an empty slot;
#: ``experiments/plot_results.py`` parses the same word.
NO_DEFENSE_TAG = "base"



def build_featurizers(args: argparse.Namespace) -> list:
    """Construct the featurizer(s) for this run, configured to match the loaded feature space.

    ``--feature`` may name several featurizers to combine; this returns one built instance per
    name, in order. Each must produce vectors in the same space as the loaded dataset's committed
    features (its ``reference``), so StyloMetrix is built with the matching language code; the
    others take no configuration.
    """
    featurizers = []
    for name in args.feature:
        options = {}
        if name == "stylometrix":
            language = args.language if args.dataset == "wildchat" else "English"
            options["language_code"] = STYLOMETRIX_LANGUAGE_CODES[language]
        featurizers.append(get_featurizer(name, **options))
    return featurizers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(DATA_DIRS), help="Dataset to attack.")
    parser.add_argument(
        "--feature", nargs="+", default=["stylometrix"], choices=sorted(FEATURIZERS),
        help="Conversation representation(s); name several to concatenate their feature vectors.",
    )
    parser.add_argument("--attack", default="nearest_neighbor",
                        choices=sorted(ATTRIBUTION_ATTACKS), help="Attack to run.")
    parser.add_argument(
        "--metric", default="cosine",
        help="Distance metric the attack uses to compare vectors, e.g. 'cosine' (default) or "
             "'euclidean' (any scipy cdist metric).",
    )
    parser.add_argument("--defense", default="none", choices=sorted(DEFENSES), help="Defense applied before the attack.")
    parser.add_argument(
        "--fidelity", default="none", choices=["none", *sorted(FIDELITY_METRICS)],
        help="Score defense utility preservation before the attack: 'utility' (per-turn PASS/FAIL "
             "on generated answers) or 'conversation' (whole-conversation 1-5). 'none' (default) "
             "skips it and makes no API calls. Ignored when --defense none (nothing to score).",
    )
    parser.add_argument(
        "--fidelity-model", default=None,
        help="OpenRouter judge model for --fidelity (default: the metric's own -- gpt-4o for "
             "'utility', Sonnet for 'conversation').",
    )
    parser.add_argument(
        "--fidelity-side", default="unknown", choices=["unknown", "known"],
        help="Which side --fidelity scores (default: 'unknown', the side a defense rewrites).",
    )
    parser.add_argument(
        "--fidelity-limit", type=int, default=None,
        help="Score only a seeded random sample of this many CONVERSATIONS -- for calibrating a "
             "rubric cheaply before a full run. Counts conversations, not API calls: 'conversation' "
             "makes ~1 call each (so ~50 for 50 calls), 'utility' makes ~3 per changed turn (~18 "
             "per conversation, so ~3 for 50 calls). Uses --seed; writes to a *_sample<N>.csv so it "
             "never clobbers a full run.",
    )
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
    """Short, self-describing directory name for this run's outputs.

    A default run is exactly ``<dataset>_<defense>_<feature>_<attack>``, the four axes
    positionally, with :data:`NO_DEFENSE_TAG` standing in when there is no defense so the shape
    never changes. That is the name ``experiments/plot_results.py`` parses. Every non-default
    scope choice is then appended, which both keeps two runs from overwriting each other and
    takes the qualified run out of the comparable set -- a Russian-subset run is not a point on
    the same curve as an English one.
    """
    feature = "+".join(args.feature)  # combined runs list every feature, e.g. "stylometrix+function_words"
    defense = NO_DEFENSE_TAG if args.defense == "none" else args.defense
    if args.dataset == "wildchat":
        scope = "" if args.language == "English" else f"_{args.language.lower()}"
    else:
        scope = "" if args.model_owner.lower() == "all" else f"_{args.model_owner.lower()}"
    metric = "" if args.metric == "cosine" else f"_{args.metric}"  # only a non-default metric gets a suffix
    return f"{args.dataset}_{defense}_{feature}_{args.attack}{scope}{metric}"


def main() -> None:
    args = parse_args()

    # Dataset-specific loader options; everything after loading is dataset-agnostic. The loader
    # returns committed reference features for a single feature space (StyloMetrix today), so
    # pass the first requested feature -- the one whose committed vectors a featurizer can reuse.
    options = {"feature": args.feature[0]}
    if args.dataset == "wildchat":
        options["language"] = args.language
    else:
        options["model_owner"] = args.model_owner

    # Load (text + committed reference features) -> defense (rewrites text) -> featurize.
    # `reference` is the loader's features for the ORIGINAL text; each featurizer reuses them
    # wherever the defense left text unchanged (only the one matching the committed space) and
    # recomputes/caches only the rewritten text. Multiple --feature values are concatenated.
    data = load_dataset(args.dataset, DATA_DIRS[args.dataset], **options)
    reference = data
    data = apply_defense(args.defense, data, cache_dir=args.cache_dir)

    # Fidelity (optional) runs on the defended text vs. the pre-defense `reference`, before
    # featurization -- so a bad API key or an over-long input fails fast, before the GPU work. Held
    # in a variable and written alongside the attack outputs below. Skipped with --defense none:
    # nothing was rewritten, so every conversation is trivially faithful.
    fidelity_result = None
    if args.fidelity != "none" and args.defense != "none":
        fidelity_kwargs = {"judge_model": args.fidelity_model} if args.fidelity_model else {}
        fidelity_result = run_fidelity(
            args.fidelity, data, cache_dir=args.cache_dir, reference=reference,
            side=args.fidelity_side, limit=args.fidelity_limit, **fidelity_kwargs,
        )
        print(f"\n{fidelity_result.summary()}")

    featurizers = build_featurizers(args)
    data = apply_featurizer(featurizers, data, cache_dir=args.cache_dir, reference=reference)
    # The distance metric is an attack-level choice, decoupled from the featurizer: set it here
    # so every featurizer (single or combined) feeds the same, caller-chosen metric to the attack.
    data.metric = args.metric
    print(
        f"[{args.dataset}] {data.n_identities} identities | "
        f"{data.n_known} known + {data.n_unknown} unknown conversations | "
        f"attack={args.attack} defense={args.defense} "
        f"feature={'+'.join(f.name for f in featurizers)} metric={data.metric}"
    )

    # Attack -> author score matrix -> ranking reused by both the headline table and sweep.
    # Attacks score authors, not conversations, so each column of the matrix handed to
    # LinkageRanking *is* one identity; it is negated because LinkageRanking ranks ascending
    # (smaller = more similar) while an attack score means the opposite.
    attack_class = get_attribution_attack(args.attack)
    accepted = inspect.signature(attack_class).parameters
    model = attack_class(**({"metric": data.metric} if "metric" in accepted else {}))
    model.fit(data.known_embeddings, data.known_labels)
    scores = model.score(data.unknown_embeddings)
    ranking = LinkageRanking(-np.asarray(scores, dtype=float), model.authors, data.unknown_labels)

    # Save the individual top-10 matches for qualitative/topic analysis.
    predictions = []

    for u in range(len(data.unknown_labels)):
        true_identity = data.unknown_labels[u]

        # ranked known conversations (identity codes)
        ranked_codes = ranking._ranked_known_codes[u]

        seen = set()
        rank = 1

        for code in ranked_codes:
            identity = ranking._identities[code]

            # keep first occurrence of each identity only
            if identity not in seen:
                predictions.append(
                    {
                        "unknown_index": u,
                        "true_identity": true_identity,
                        "predicted_identity": identity,
                        "rank": rank,
                        "correct": identity == true_identity,
                    }
                )

                seen.add(identity)
                rank += 1

            if rank > 10:
                break
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

    pd.DataFrame(predictions).to_csv(
        output_dir / "predictions.csv",
        index=False
    )

    headline.to_csv(output_dir / "headline_results.csv", index=False)
    sweep.to_csv(output_dir / "sweep_results.csv", index=False)

    fidelity_csv = None
    if fidelity_result is not None:
        # A sampled (--fidelity-limit) run goes to its own file so a quick calibration pass never
        # overwrites a full run's table in the same output dir.
        suffix = f"_sample{args.fidelity_limit}" if args.fidelity_limit is not None else ""
        fidelity_csv = output_dir / f"fidelity_{args.fidelity}{suffix}.csv"
        fidelity_result.to_csv(fidelity_csv)

    print(f"\nWrote results to {output_dir}/")
    outputs = ["headline_results.csv", "sweep_results.csv", "predictions.csv"]
    if fidelity_csv is not None:
        outputs.append(fidelity_csv.name)
    print("  " + ", ".join(outputs))
    print("\nNo figures were drawn. To (re)draw every figure in the project from the CSVs:")
    print("  python experiments/plot_results.py")


if __name__ == "__main__":
    main()
