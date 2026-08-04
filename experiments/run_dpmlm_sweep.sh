#!/usr/bin/env bash
#
# run_dpmlm_sweep.sh -- run the DP-MLM privacy-budget (epsilon) sweep, one experiment per epsilon.
#
# Each epsilon is a registered defense `dp_mlm_eps<eps>` (see DPMLM_SWEEP_EPSILONS in
# src/prompt_anonymity/defenses/__init__.py), so every epsilon gets its own defended parquet, its
# own feature parquet, its own results directory (run_experiment.py's output_tag embeds the
# defense name) and its own cache entries (epsilon is in the defense's params()).
#
# THREE STAGES PER EPSILON. This used to be a single command, because the old fixed-split runner
# applied the defense to the text as part of the run. It does not exist any more: defending is a
# build step of its own, and run_experiment.py only ever *reads* vectors. So each epsilon is
#
#   apply_defenses   --defense dp_mlm_eps<e>   -> data/dist/<source>_dp_mlm_eps<e>.parquet
#   compute_features --defense dp_mlm_eps<e>   -> data/dist/<source>_dp_mlm_eps<e>_<feature>.parquet
#   run_experiment.py --defense dp_mlm_eps<e>  -> experiments/results/<source>_dp_mlm_eps<e>_...
#
# and the run reads --data-dir data/dist, because that is where the first two stages write (the
# runner's own default is data/hf, the published mirror, which will not have these files).
#
# Usage:
#   bash experiments/run_dpmlm_sweep.sh                                  # swe_chat, stylometrix
#   SOURCE=wildchat FEATURE=gemini_embedding_2 bash experiments/run_dpmlm_sweep.sh
#
# Any extra arguments are forwarded verbatim to run_experiment.py only (not to the build stages),
# so attack and scoring flags go there:
#   bash experiments/run_dpmlm_sweep.sh --attacks nearest_neighbor logistic
#
# Run a subset (must be registered epsilons -- the values in DPMLM_SWEEP_EPSILONS above):
#   DPMLM_SWEEP_EPSILONS="100 250 500" bash experiments/run_dpmlm_sweep.sh
#
# NOTE: each epsilon is a full DP-MLM pass over the corpus (hours on a large source, no cache
# sharing across epsilons), so the whole default sweep is long. Start with a couple to validate.
# The stages are individually resumable -- apply_defenses checkpoints its cache -- so re-running
# this script after an interruption picks up rather than restarting.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
SOURCE="${SOURCE:-swe_chat}"
FEATURE="${FEATURE:-stylometrix}"
DIST_DIR="${DIST_DIR:-$REPO_ROOT/data/dist}"
# Keep this default in sync with DPMLM_SWEEP_EPSILONS in the defenses registry.
EPSILONS="${DPMLM_SWEEP_EPSILONS:-10 25 50 100 250 500 1000}"

for eps in $EPSILONS; do
  defense="dp_mlm_eps${eps}"
  echo ""
  echo "=================== DP-MLM sweep: epsilon=${eps} ==================="

  "$PYTHON" -m prompt_anonymity.data.apply_defenses \
      --source "$SOURCE" --defense "$defense"

  "$PYTHON" -m prompt_anonymity.data.compute_features \
      --source "$SOURCE" --defense "$defense" --feature "$FEATURE" --dist-dir "$DIST_DIR"

  "$PYTHON" "$REPO_ROOT/experiments/run_experiment.py" \
      --source "$SOURCE" --defense "$defense" --feature "$FEATURE" \
      --data-dir "$DIST_DIR" "$@"
done

echo ""
echo "Sweep complete. Each epsilon wrote to its own experiments/results/*_dp_mlm_eps<eps>_*/ directory."
echo "Draw the figures with: $PYTHON $REPO_ROOT/experiments/plot_results.py"
