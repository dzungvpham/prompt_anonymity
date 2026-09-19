#!/bin/bash
#
# run_wccn_fusion_swe_chat.sbatch -- the full feature x attack matrix on SWE-chat, one job.
#
#   sbatch experiments/run_wccn_fusion_swe_chat.sbatch
#
# SUBMIT FROM THE REPO ROOT. --output below is a relative path, resolved against the submitting
# directory. One-time: mkdir -p scripts/logs
#
# CPU-only (WCCN, TF-IDF and StyloMetrix are all CPU work) and small: SWE-chat is 4,334 documents,
# so every window x feature-set x attack cell here is a matter of seconds. The whole matrix
# (7 feature sets x 6 windows) previously ran in well under the time a single WildChat fusion cell
# takes; --time is generous anyway since queue latency, not compute, is the binding cost here.
#
# Forward extra args to the runner, e.g. to add logistic as a comparison baseline:
#
#   sbatch experiments/run_wccn_fusion_swe_chat.sbatch --attacks wccn logistic
#
#SBATCH --job-name=wccn-fusion-swe
#SBATCH --output=scripts/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --partition=cpu-preempt
#SBATCH --cpus-per-task=8
#SBATCH --mem=16g
#SBATCH --time=0-02:00:00
#SBATCH --requeue

set -euo pipefail

PROJECT="${PROMPT_ANONYMITY_PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT"

echo "=================================================================="
echo "WCCN fusion matrix: swe_chat"
echo "Job ${SLURM_JOB_ID:-local} on $(hostname), started $(date)"
echo "=================================================================="

bash scripts/slurm_job.sh python experiments/run_wccn_fusion.py \
    --source swe_chat \
    --feature-sets gemini_embedding_2 \
                  gemini_embedding_2+char_ngram \
                  gemini_embedding_2+stylometrix \
                  gemini_embedding_2+pos_ngram \
                  gemini_embedding_2+char_ngram+stylometrix+pos_ngram \
                  char_ngram stylometrix pos_ngram \
    --attacks wccn \
    --pos-cache "pos_tags_cache_swe_chat.jsonl" \
    "$@"

echo ""
echo "Done $(date)."
echo "  results CSV: experiments/results/wccn_fusion_swe_chat_*.csv"
