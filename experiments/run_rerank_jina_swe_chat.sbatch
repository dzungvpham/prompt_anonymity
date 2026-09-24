#!/bin/bash
#
# run_rerank_jina_swe_chat.sbatch -- rerank the top-5 nearest-neighbor author shortlist on SWE-chat
# (gemini_embedding_2 vectors) with the LOCAL jina reranker, then check the output is usable.
#
#   sbatch experiments/run_rerank_jina_swe_chat.sbatch
#
# SUBMIT FROM THE REPO ROOT (--output is relative and SLURM resolves it against the submit dir).
# One-time: mkdir -p scripts/logs
#
# STANDALONE ON PURPOSE. This does not go through run_rerank_jina.sbatch -- it calls run_rerank.py
# directly, and every stage prints a numbered, timestamped header and fails loudly, so when
# something breaks the log says which stage it was.
#
# WHAT IT MEASURES. The shortlist is the top-5 authors under nearest-neighbor (max linkage, see
# attacks/llm/candidates.py); the reranker only reorders it. So top-5 is pinned to the shortlist's
# own recall -- run_rerank.py checks that row by row; it equals the baseline except where base
# scores tie at the shortlist boundary -- and top-1/2/3 are the reranker's contribution.
#
# PREEMPTION IS SURVIVABLE. gpu-preempt can kill and requeue this at any moment; scores are flushed
# to the cache every JINA_RERANK_FLUSH_EVERY (default 8) shortlists, so a requeue resumes rather
# than starting over.
#
# Preflight on a login node (no GPU needed, nothing is reranked):
#
#     CHECK_ONLY=1 bash experiments/run_rerank_jina_swe_chat.sbatch
#
# Overrides (sbatch exports your submitting environment, so these are read at run time):
#   LIMIT=200 sbatch ...                   # score a seeded sample of the unknown side
#   TOP_K=10 sbatch ...                    # a different shortlist size
#   FEATURE=luar sbatch ...                # needs data/dist/swe_chat_luar.parquet first
#   WINDOW=5075 sbatch ...                 # a different known side
#   SAMPLE=8 sbatch ...                    # print more reranked documents at the end
#   OUT_DIR=... sbatch ...                 # write the tables somewhere else
#
#SBATCH --job-name=rrk-jina-swe
#SBATCH --output=scripts/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --partition=gpu-preempt
#SBATCH --gpus=1
#SBATCH --constraint=a100|l40s|h100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=0-04:00:00
#SBATCH --requeue

set -euo pipefail

# Resolve the repo root, then CHECK it -- $SLURM_SUBMIT_DIR is inherited garbage in an Open
# OnDemand shell, so it is a candidate rather than an answer.
looks_like_repo() { [ -d "$1/src/prompt_anonymity" ] && [ -d "$1/experiments" ]; }

PROJECT=""
for candidate in "${PROMPT_ANONYMITY_PROJECT:-}" "${SLURM_SUBMIT_DIR:-}" "$PWD"; do
    if [ -n "$candidate" ] && looks_like_repo "$candidate"; then
        PROJECT="$candidate"
        break
    fi
done
if [ -z "$PROJECT" ]; then
    echo "ERROR: cannot find the repo root. Tried \$PROMPT_ANONYMITY_PROJECT" \
         "(${PROMPT_ANONYMITY_PROJECT:-unset}), \$SLURM_SUBMIT_DIR (${SLURM_SUBMIT_DIR:-unset})" \
         "and \$PWD ($PWD)." >&2
    echo "       Run this from the repo root, or set PROMPT_ANONYMITY_PROJECT=/path/to/repo." >&2
    exit 1
fi
cd "$PROJECT"

SOURCE="${SOURCE:-swe_chat}"
FEATURE="${FEATURE:-gemini_embedding_2}"
DEFENSE="${DEFENSE:-none}"
WINDOW="${WINDOW:-0075}"
TOP_K="${TOP_K:-5}"
SEED="${SEED:-47}"
SAMPLE="${SAMPLE:-3}"
DIST_DIR="${DIST_DIR:-$PROJECT/data/dist}"
CACHE_DIR="${CACHE_DIR:-$PROJECT/data/.cache}"
CHECK_ONLY="${CHECK_ONLY:-0}"

# `none` is spelled `base` in results/ (run_experiment.NO_DEFENSE_TAG); the split stem is the
# source alone for `none`, else <source>_<defense>.
if [ "$DEFENSE" = "none" ]; then TAG=base; STEM="$SOURCE"; else TAG="$DEFENSE"; STEM="${SOURCE}_$DEFENSE"; fi

# The `jina` subdirectory is the one submit_rerank_swe_chat.sh gives this arm, so
# finish_rerank.sbatch can later merge it with a Sonnet arm without racing on rerank_summary.csv.
OUT_DIR="${OUT_DIR:-$PROJECT/experiments/results/rerank/${SOURCE}_${TAG}_${FEATURE}_known${WINDOW}/jina}"

# Bare `python` when checking on a login node; the env bootstrap only when running as a job.
if [ "$CHECK_ONLY" = "1" ]; then
    RUN=()
else
    RUN=(bash scripts/slurm_job.sh)
fi
run() { if [ "${#RUN[@]}" -eq 0 ]; then "$@"; else "${RUN[@]}" "$@"; fi; }

echo "=================================================================="
echo "listwise rerank (jina) -- $SOURCE / $FEATURE / known$WINDOW / top-$TOP_K (nearest neighbor)"
if [ -n "${LIMIT:-}" ]; then
    echo "  unknown side: a seeded sample of $LIMIT documents"
else
    echo "  unknown side: all of it"
fi
echo "  output:       $OUT_DIR"
echo "Job ${SLURM_JOB_ID:-local} on $(hostname), $(date)"
echo "=================================================================="

# --- 1. environment --------------------------------------------------------
# Which interpreter, and whether it can see a GPU. A 0.6B reranker on CPU does not crash -- it just
# takes all night -- so a job without a GPU stops here instead.
echo ""
echo "[1/4] environment ($(date))"
run python -c '
import sys
print("  interpreter:", sys.executable)
import torch, transformers
print("  torch", torch.__version__, "| transformers", transformers.__version__)
cuda = torch.cuda.is_available()
print("  cuda available:", cuda, "|", torch.cuda.get_device_name(0) if cuda else "no device")
'
if [ "$CHECK_ONLY" != "1" ]; then
    run python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' || {
        echo "ERROR: no GPU visible to torch inside the job; refusing to rerank on CPU." >&2
        exit 1
    }
fi

# --- 2. inputs, self-test, checkpoint --------------------------------------
echo ""
echo "[2/4] inputs ($(date))"
if [ ! -f "$DIST_DIR/${STEM}.parquet" ]; then
    echo "ERROR: no $DIST_DIR/${STEM}.parquet -- the $SOURCE split is missing." >&2
    exit 1
fi
if [ ! -f "$DIST_DIR/${STEM}_${FEATURE}.parquet" ]; then
    echo "ERROR: no $DIST_DIR/${STEM}_${FEATURE}.parquet" >&2
    echo "  Compute the vectors first with:" >&2
    echo "    python -m prompt_anonymity.data.compute_features --source $SOURCE \\" >&2
    echo "        --feature $FEATURE --dist-dir $DIST_DIR" >&2
    exit 1
fi
echo "  split:    $DIST_DIR/${STEM}.parquet"
echo "  features: $DIST_DIR/${STEM}_${FEATURE}.parquet"

echo "  attack self-test:"
run python -m prompt_anonymity.attacks.llm.listwise --selftest

# The checkpoint is allowed to download on an unconfigured machine; print where it resolves so a
# download shows up here rather than as a job that looks hung.
run python -c "
from prompt_anonymity.attacks.llm.listwise_jina_rerank import (
    DEFAULT_MODEL_ID, MODEL_CONFIG_SECTION, MODEL_ENV_VAR)
from prompt_anonymity.evaluation.utility._local import resolve_checkpoint
print('  jina checkpoint:', resolve_checkpoint(MODEL_CONFIG_SECTION, MODEL_ENV_VAR, DEFAULT_MODEL_ID))"

if [ "$CHECK_ONLY" = "1" ]; then
    echo ""
    echo "(CHECK_ONLY=1 -- preflight only, nothing reranked)"
    exit 0
fi

# --- 3. the attack ---------------------------------------------------------
ARGS=(--source "$SOURCE" --feature "$FEATURE" --defense "$DEFENSE"
      --data-dir "$DIST_DIR" --known-window "$WINDOW" --seed "$SEED"
      --cache-dir "$CACHE_DIR" --output-dir "$OUT_DIR"
      --variant jina --top-k "$TOP_K")
if [ -n "${LIMIT:-}" ]; then
    ARGS+=(--limit "$LIMIT")
fi

echo ""
echo "[3/4] reranking ($(date))"
run python -u experiments/run_rerank.py "${ARGS[@]}"

# --- 4. is the output usable? ----------------------------------------------
# Finishing is not the same as being good: a reranker that scores every candidate identically still
# writes a complete, well-formed table. This checks the structure and prints reranked documents.
echo ""
echo "[4/4] validating the output ($(date))"
run python experiments/check_rerank_output.py "$OUT_DIR" --sample "$SAMPLE" --seed "$SEED"

echo ""
echo "Done $(date). Wrote, under $OUT_DIR:"
ls -l "$OUT_DIR"
echo ""
echo "rerank_summary.csv: top_1..3 against base_top_1..3 is the reranker's gain; top_$TOP_K must"
echo "equal shortlist_ceiling. Read the sampled documents above, not just the summary."
