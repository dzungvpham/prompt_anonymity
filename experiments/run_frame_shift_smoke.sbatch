#!/bin/bash
#
# run_frame_shift_smoke.sbatch -- push N real prompts (default 30) through frame_shift and print
# every rewrite, before and after, with the contract scored.
#
#   sbatch experiments/run_frame_shift_smoke.sbatch
#
# SUBMIT FROM THE REPO ROOT (--output is relative and SLURM resolves it against the submit dir).
# One-time: mkdir -p scripts/logs
#
# WHAT THIS IS FOR. Deciding whether the full arm is worth launching, in about a minute and for a
# fraction of a cent. It answers the three questions the corpus run cannot answer cheaply:
#
#   1. Does the rewrite actually restate the prompt, or does it just wrap it? A wrapper preserves
#      the author's exact wording, which is the one failure that produces output looking perfectly
#      defended while defending nothing. Watch `reuse` -- near 0% is right, high is fatal.
#   2. Does code survive byte-identical? `code kept` must be n/n.
#   3. Is the frame bulky enough to dilute? `expansion` well above 1.
#
# It takes ONE turn from each of N documents rather than N turns from one, so you see up to N
# different frames -- breadth is what is informative here, and a SWE-chat document is an agent
# session that can carry hundreds of turns.
#
# NO GPU, NO DATA WRITTEN. The defense stage is network-bound, so this asks for a CPU node and a
# short wall clock, which is what gets it scheduled almost immediately on a busy cluster. It writes
# nothing: no cache, no parquet. The output lives in the job log, and nothing here can corrupt or
# invalidate a real run's cache.
#
# REQUIRES a reachable OPENROUTER_API_KEY. load_dotenv() walks UP from the working directory, so the
# key must be in a .env at the REPO ROOT (or above it) -- a key in DS_env/.env is NOT found from the
# root. Fix with: cp DS_env/.env .env
#
# Overrides (sbatch exports your submitting environment, so these are read at run time):
#
#   N=100 sbatch experiments/run_frame_shift_smoke.sbatch            # more prompts
#   SOURCE=wildchat sbatch experiments/run_frame_shift_smoke.sbatch  # default swe_chat
#   SINGLE=podcast_transcript sbatch ...                             # preview the ablation's frame
#   # the cheap-model bake-off: identical prompts, several models, scored side by side.
#   # On WildChat the gap between these is roughly $15 vs $95 for the same corpus.
#   MODELS=inclusionai/ling-2.6-flash,qwen/qwen3.7-flash,deepseek/deepseek-v4-flash-0731 \
#       sbatch experiments/run_frame_shift_smoke.sbatch
#
#SBATCH --job-name=frame_shift_smoke
#SBATCH --output=scripts/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --partition=cpu-preempt
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH --time=0-00:30:00
#SBATCH --requeue

set -euo pipefail

PROJECT="${PROMPT_ANONYMITY_PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT"

N="${N:-30}"
SOURCE="${SOURCE:-swe_chat}"
DIST_DIR="${DIST_DIR:-$PROJECT/data/dist}"

RUN=(bash scripts/slurm_job.sh)

echo "=================================================================="
echo "Frame shift smoke test: $N prompts from $SOURCE"
echo "  models: ${MODELS:-<the configured default>}"
echo "Job ${SLURM_JOB_ID:-local} on $(hostname), started $(date)"
echo "=================================================================="

# 0. Offline invariants first: seconds, no API key, no network. A broken build should fail here
#    rather than after the first paid call.
"${RUN[@]}" python -m prompt_anonymity.defenses.frame_shift --selftest

# 1. The rewrites. One turn per document across N documents -> up to N distinct frames.
#    --preview-max-jobs is the hard ceiling, so N is exactly what gets billed (per model).
PREVIEW=(--preview --source "$SOURCE" --dist-dir "$DIST_DIR"
         --limit "$N" --preview-turns-per-doc 1 --preview-max-jobs "$N")
[ -n "${MODELS:-}" ] && PREVIEW+=(--models "$MODELS")
[ -n "${SINGLE:-}" ] && PREVIEW+=(--single "$SINGLE")

"${RUN[@]}" python -m prompt_anonymity.defenses.frame_shift "${PREVIEW[@]}"

echo ""
echo "Done $(date)."
echo ""
echo "READ THE SUMMARY TABLE AT THE BOTTOM, not just the rewrites:"
echo "  reuse      share of the original's 8-word runs surviving verbatim. THE number. A model"
echo "             that wraps instead of restating scores high here and must not be used, however"
echo "             good the prose looks."
echo "  code kept  fenced blocks and inline spans byte-identical. Anything less breaks utility"
echo "             across the whole corpus, silently."
echo "  expansion  chars out over chars in. Near 1 means a one-line wrapper, not a scene."
echo "  fallback   replies that were empty or truncated and fell back to the original turn. A few"
echo "             is fine; many means raising FRAME_SHIFT_OUTPUT_RATIO."
echo ""
echo "Then judge by eye the one thing no metric covers: does the framed text still plainly ask for"
echo "a real, complete answer? If it reads as a request to write fiction, the utility judge will"
echo "score it as a failure and it should be fixed in the system prompt before the corpus run."
echo ""
echo "If it all looks right:  FRAME_SHIFT_MAX_WORKERS=32 sbatch experiments/run_frame_shift.sbatch"
