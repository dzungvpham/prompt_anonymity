#!/bin/bash
#
# run_dpmlm_var_a10.sbatch -- one GPU job that runs the full three-stage pipeline for the
# adaptive-length DP-MLM defense `dp_mlm_var_a10` (eps=100, A=0.10, D=0.05).
#
#   sbatch experiments/run_dpmlm_var_a10.sbatch
#
# SUBMIT FROM THE REPO ROOT. --output below is a relative path, and SLURM resolves it against the
# directory you submit from.
#
# ONE-TIME SETUP -- SLURM fails a job whose --output directory does not exist:
#
#   mkdir -p scripts/logs
#
# Overrides (sbatch exports your submitting environment, so these are read at run time):
#
#   SOURCE=wildchat sbatch experiments/run_dpmlm_var_a10.sbatch          # default swe_chat
#   FEATURE=gemini_embedding_2 sbatch experiments/run_dpmlm_var_a10.sbatch   # default stylometrix
#   DEFENSE=dp_mlm_var_a25 sbatch experiments/run_dpmlm_var_a10.sbatch   # the A=0.25 variant
#   sbatch --account=my-account experiments/run_dpmlm_var_a10.sbatch     # sbatch flags win over
#                                                                        # the #SBATCH lines below
# Anything after the script name is forwarded to run_experiment.py only (not to the build stages),
# so attack/scoring flags go there, as in run_dpmlm_sweep.sh:
#
#   sbatch experiments/run_dpmlm_var_a10.sbatch --attacks nearest_neighbor logistic
#
# WHY THIS FILE HAS #SBATCH DIRECTIVES, unlike scripts/slurm_job.sh: that file is a wrapper the
# run_all_experiments launcher submits, so its resources come from scripts/slurm.toml. There is no
# launcher for the *defense* pipeline -- it is a one-off, so the resources live here. They mirror
# [profiles.gpu] in scripts/slurm.toml, minus its --qos=short (that QOS caps at 4 hours and a
# DP-MLM pass over a full source takes longer). Change the cluster-specific values here.
#
# PREEMPTION IS SAFE. gpu-preempt jobs get killed; --requeue puts this back in the queue, and all
# three stages resume rather than restart -- apply_defenses checkpoints its cache every 200
# conversations, compute_features caches per text, and run_experiment skips finished output. A
# requeued job simply picks up where it stopped.
#
#SBATCH --job-name=dp_mlm_var_a10
#SBATCH --output=scripts/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --partition=gpu-preempt
#SBATCH --gpus=1
#SBATCH --constraint=a100|l40s|h100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=0-12:00:00
#SBATCH --requeue

set -euo pipefail

PROJECT="${PROMPT_ANONYMITY_PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT"

SOURCE="${SOURCE:-swe_chat}"
FEATURE="${FEATURE:-stylometrix}"
DEFENSE="${DEFENSE:-dp_mlm_var_a10}"
DIST_DIR="${DIST_DIR:-$PROJECT/data/dist}"

# Each stage goes through scripts/slurm_job.sh, the same env bootstrap the experiment launcher
# uses, so how a batch shell here finds conda stays defined in exactly one file. It ends in
# `exec "$@"`, so calling it as a command runs the stage and returns here.
#
# Deliberately NOT `srun`: since SLURM 22.05 a step does not inherit --cpus-per-task from the batch
# allocation, so `srun <stage>` would hand these stages a single CPU and slow the tokenizer down for
# no reason. This is one process on one node, which the batch shell already has all 8 CPUs for.
RUN=(bash scripts/slurm_job.sh)

echo "=================================================================="
echo "DP-MLM adaptive length: $DEFENSE  (source=$SOURCE feature=$FEATURE)"
echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
echo "=================================================================="

# 1. Defend: the long one (a full DP-MLM pass over the source, hours). This is the stage that
#    actually applies the add/delete coins; it prints the realized counters when it finishes --
#    check them against A=0.10 / D=0.05 before trusting the run.
"${RUN[@]}" python -m prompt_anonymity.data.apply_defenses \
    --source "$SOURCE" --defense "$DEFENSE"

# 2. Featurize the defended text. Note this cannot reuse the fixed-length eps=100 features: the
#    whole point of this variant is that the text (and so its length/word-count features) changed.
"${RUN[@]}" python -m prompt_anonymity.data.compute_features \
    --source "$SOURCE" --defense "$DEFENSE" --feature "$FEATURE" --dist-dir "$DIST_DIR"

# 3. Attack it. --data-dir is data/dist because that is where the two stages above write; the
#    runner's own default is data/hf, the published mirror, which will not have these files.
"${RUN[@]}" python experiments/run_experiment.py \
    --source "$SOURCE" --defense "$DEFENSE" --feature "$FEATURE" \
    --data-dir "$DIST_DIR" "$@"

echo ""
echo "Done $(date). Results in experiments/results/${SOURCE}_${DEFENSE}_${FEATURE}_*/"
echo "Compare against the fixed-length run at the same eps (${SOURCE}_dp_mlm_eps100_${FEATURE}_*)"
echo "to read off what breaking the length signature bought. Draw figures with:"
echo "  python experiments/plot_results.py"
