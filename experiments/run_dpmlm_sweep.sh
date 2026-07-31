#!/usr/bin/env bash
#
# run_dpmlm_sweep.sh -- run the DP-MLM privacy-budget (epsilon) sweep, one experiment per epsilon.
#
# Each epsilon is a registered defense `dp_mlm_eps<eps>` (see DPMLM_SWEEP_EPSILONS in
# src/prompt_anonymity/defenses/__init__.py). run_experiment.py's output_tag embeds the defense name,
# so every epsilon writes to its OWN results directory -- its own headline_results.csv,
# sweep_results.csv, predictions.csv and plots -- and each caches separately (epsilon is in the
# defense's params()). All other flags are forwarded verbatim to run_experiment.py.
#
# Usage (pass the usual run_experiment.py flags; they apply to every epsilon):
#   bash experiments/run_dpmlm_sweep.sh --dataset wildchat --language English --attack nearest_neighbor
#
# Run a subset (must be registered epsilons -- the values in DPMLM_SWEEP_EPSILONS above):
#   DPMLM_SWEEP_EPSILONS="100 250 500" bash experiments/run_dpmlm_sweep.sh --dataset wildchat ...
#
# NOTE: each epsilon is a full DP-MLM pass (hours on a large unknown side, no cache sharing across
# epsilons), so the whole default sweep is long. Start with a couple of epsilons to validate.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
# Keep this default in sync with DPMLM_SWEEP_EPSILONS in the defenses registry.
EPSILONS="${DPMLM_SWEEP_EPSILONS:-10 25 50 100 250 500 1000}"

for eps in $EPSILONS; do
  echo ""
  echo "=================== DP-MLM sweep: epsilon=${eps} ==================="
  "$PYTHON" "$REPO_ROOT/experiments/run_experiment.py" --defense "dp_mlm_eps${eps}" "$@"
done

echo ""
echo "Sweep complete. Each epsilon wrote to its own experiments/results/*_dp_mlm_eps<eps>/ directory."
