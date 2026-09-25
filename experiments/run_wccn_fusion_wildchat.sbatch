#!/bin/bash
#
# run_wccn_fusion_wildchat.sbatch -- the full feature x attack matrix on WildChat, one job.
#
#   sbatch experiments/run_wccn_fusion_wildchat.sbatch
#
# SUBMIT FROM THE REPO ROOT. --output below is a relative path, resolved against the submitting
# directory. One-time: mkdir -p scripts/logs
#
# CPU-only, like the SWE-chat sibling -- WCCN, TF-IDF and StyloMetrix are all CPU work, so a GPU
# would sit idle. --mem follows this repo's wildchat_cpu_large precedent (scripts/slurm.toml):
# 24g was enough for the single-feature WCCN/StyloMetrix cells this project has already run at
# this scale; this job holds one extra feature block (Gemini's 172,509 x 3,072 matrix, ~2.1 GB) in
# memory for the whole run rather than loading it per window, so 32g is the safety margin, not the
# measured peak. If a cell still OOMs, shard by feature set (`--feature-sets <one>` as its own
# job) rather than raising this further -- see the module docstring in run_wccn_fusion.py for why
# each feature set's per-window work is already independent.
#
# --time budgets ~1-1.5h per feature-set cell (measured for single fusion combos earlier in this
# project) x 7 feature sets = well under the 8h asked here; --requeue plus per-window printing to
# the log is what makes an 8h ask on the preemptible partition safe to leave running unattended.
#
#SBATCH --job-name=wccn-fusion-wildchat
#SBATCH --output=scripts/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --partition=cpu-preempt
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=0-08:00:00
#SBATCH --requeue

set -euo pipefail

PROJECT="${PROMPT_ANONYMITY_PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT"

echo "=================================================================="
echo "WCCN fusion matrix: wildchat"
echo "Job ${SLURM_JOB_ID:-local} on $(hostname), started $(date)"
echo "=================================================================="

bash scripts/slurm_job.sh python experiments/run_wccn_fusion.py \
    --source wildchat \
    --feature-sets gemini_embedding_2 \
                  gemini_embedding_2+char_ngram \
                  gemini_embedding_2+stylometrix \
                  gemini_embedding_2+pos_ngram \
                  gemini_embedding_2+char_ngram+stylometrix+pos_ngram \
                  char_ngram stylometrix pos_ngram \
    --attacks wccn \
    --pos-cache "pos_tags_cache_wildchat_shard*of4.jsonl" \
    "$@"

echo ""
echo "Done $(date)."
echo "  results CSV: experiments/results/wccn_fusion_wildchat_*.csv"
