#!/bin/bash
#
# run_rerank_jina.sbatch -- rerank a nearest-neighbor shortlist with the LOCAL jina reranker, at
# two shortlist sizes, then check that the output is actually usable.
#
#   sbatch experiments/run_rerank_jina.sbatch
#
# SUBMIT FROM THE REPO ROOT (--output is relative and SLURM resolves it against the submit dir).
# One-time: mkdir -p scripts/logs
#
# FREE. jina-reranker-v3.5 is 0.6B and runs on the local GPU -- no API, no key, no bill. Run this
# BEFORE the Sonnet job: it exercises exactly the same split, feature parquet, shortlist, fold-back
# and output tables, so anything structurally broken fails here where a mistake costs GPU minutes
# instead of dollars.
#
# WHAT IT MEASURES. The shortlist's *membership* never moves, only the order inside it, so top-K at
# the shortlist size is identical to the plain nearest-neighbor baseline by construction -- that is
# the recall ceiling the reranker was handed. Everything below it (top-1, top-2, top-3) is the
# reranker's own contribution. rerank_summary.csv carries both numbers on the same rows so the gain
# is a subtraction rather than a second run.
#
# THIS IS ALSO THE CONTROL FOR THE SONNET JOB. Identical shortlist, identical presentation, no bill.
# The gap between the two prices the frontier model rather than the idea of reranking -- and a
# retrieval reranker asked an authorship question may well score confidently in the wrong direction,
# which would make it a useful negative control rather than a failure.
#
# PREEMPTION IS SURVIVABLE. This asks for gpu-preempt, where a job can be killed at any moment and
# requeued from the top -- so scores are written to the cache every few shortlists as they are
# computed (JINA_RERANK_FLUSH_EVERY, default 8). A requeued job re-scores at most that many rows
# and carries on from where it was, which for a whole corpus is the difference between finishing
# across several preemptions and never finishing at all.
#
# SEE WHAT IS MISSING WITHOUT RUNNING ANYTHING -- a second, on the login node, no GPU:
#
#     CHECK_ONLY=1 bash experiments/run_rerank_jina.sbatch
#
# INPUTS, both of which must already exist (stage 0b names the command that builds each):
#   data/dist/<source>.parquet                  -- cut from wildchat by build_subset.py
#   data/dist/<source>_<feature>.parquet        -- compute_features.py
#
# LICENSE: jina-reranker-v3.5 is CC BY-NC 4.0 (non-commercial). Fine for this research; worth
# knowing before the weights travel anywhere else.
#
# Overrides (sbatch exports your submitting environment, so these are read at run time):
#   SOURCE=wildchat_small sbatch ...       # the 200-author cut instead of the 40-author one
#   FEATURE=luar sbatch ...                # needs data/dist/<source>_luar.parquet to exist first
#   TOP_KS="5" sbatch ...                  # one shortlist size instead of both
#   LIMIT=100 sbatch ...                   # score a seeded sample of the unknown side
#   WINDOW=5075 sbatch ...                 # a different known side
#   SAMPLE=8 sbatch ...                    # print more reranked documents at the end
#
#SBATCH --job-name=rrk-jina
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

# Resolve the repo root, then CHECK it -- do not just trust $SLURM_SUBMIT_DIR. Open OnDemand exports
# that variable into interactive shells pointing at its own dashboard app, so run by hand it is
# inherited garbage and `cd` fails before anything else has a chance to.
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

SOURCE="${SOURCE:-wildchat_tiny}"
FEATURE="${FEATURE:-gemini_embedding_2}"
DEFENSE="${DEFENSE:-none}"
WINDOW="${WINDOW:-0075}"
SEED="${SEED:-47}"
SAMPLE="${SAMPLE:-3}"
DIST_DIR="${DIST_DIR:-$PROJECT/data/dist}"
CACHE_DIR="${CACHE_DIR:-$PROJECT/data/.cache}"
CHECK_ONLY="${CHECK_ONLY:-0}"

read -r -a TOP_K_LIST <<< "${TOP_KS:-5 10}"

# `none` is spelled `base` in results/ everywhere else (run_experiment.NO_DEFENSE_TAG).
tag_of() { if [ "$1" = "none" ]; then echo base; else echo "$1"; fi; }
# split_stem(): the source alone for `none`, else <source>_<defense>.
stem_of() { if [ "$1" = "none" ]; then echo "$SOURCE"; else echo "${SOURCE}_$1"; fi; }

STEM="$(stem_of "$DEFENSE")"
OUT_DIR="${OUT_DIR:-$PROJECT/experiments/results/rerank/${SOURCE}_$(tag_of "$DEFENSE")_${FEATURE}_known${WINDOW}}"

# Bare `python` when checking on a login node; the env bootstrap only when running as a job. Not
# `srun`: SLURM >= 22.05 does not inherit --cpus-per-task into steps.
if [ "$CHECK_ONLY" = "1" ]; then
    RUN=()
else
    RUN=(bash scripts/slurm_job.sh)
fi
run() { if [ "${#RUN[@]}" -eq 0 ]; then "$@"; else "${RUN[@]}" "$@"; fi; }

echo "=================================================================="
echo "listwise rerank (jina) -- $SOURCE / $FEATURE / known$WINDOW"
echo "  shortlist sizes: ${TOP_K_LIST[*]}"
echo "  output:          $OUT_DIR"
echo "Job ${SLURM_JOB_ID:-local} on $(hostname), $(date)"
echo "=================================================================="

# --- 0a. offline invariants ------------------------------------------------
# Seconds, no GPU, no key. A build that cannot rank a synthetic shortlist correctly should fail here
# rather than after loading a checkpoint.
echo ""
echo "[0a/3] attack self-test ($(date))"
run python -m prompt_anonymity.attacks.llm.listwise --selftest

# --- 0b. inputs ------------------------------------------------------------
echo ""
echo "[0b/3] inputs ($(date))"
if [ ! -f "$DIST_DIR/${STEM}.parquet" ]; then
    echo "ERROR: no $DIST_DIR/${STEM}.parquet" >&2
    echo "  $SOURCE is cut from wildchat locally. Build it with:" >&2
    echo "    python -m prompt_anonymity.data.build_subset --k-min 5 --n-authors 40 \\" >&2
    echo "        --max-chars 60000 --out-source wildchat_tiny --data-dir data/hf --out-dir $DIST_DIR" >&2
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

# --- 0c. checkpoint --------------------------------------------------------
# Printed, because unlike the defenses' embedder this checkpoint is ALLOWED to download: an
# unconfigured machine fetches 0.6B rather than failing. That is the right default, and on a compute
# node it should still be visible in the first seconds rather than as a job that looks hung.
echo ""
echo "[0c/3] reranker checkpoint ($(date))"
run python -c "
from prompt_anonymity.attacks.llm.listwise_jina_rerank import (
    DEFAULT_MODEL_ID, MODEL_CONFIG_SECTION, MODEL_ENV_VAR)
from prompt_anonymity.evaluation.utility._local import resolve_checkpoint
print('  [jina] resolved to:', resolve_checkpoint(MODEL_CONFIG_SECTION, MODEL_ENV_VAR, DEFAULT_MODEL_ID))
print('  (set \$JINA_RERANKER_MODEL in .env to point at a local copy instead)')"

if [ "$CHECK_ONLY" = "1" ]; then
    echo ""
    echo "(CHECK_ONLY=1 -- preflight only, nothing run)"
    exit 0
fi

# --- 1. the attack ---------------------------------------------------------
ARGS=(--source "$SOURCE" --feature "$FEATURE" --defense "$DEFENSE"
      --data-dir "$DIST_DIR" --known-window "$WINDOW" --seed "$SEED"
      --cache-dir "$CACHE_DIR" --output-dir "$OUT_DIR" --variant jina)
for k in "${TOP_K_LIST[@]}"; do
    ARGS+=(--top-k "$k")
done
if [ -n "${LIMIT:-}" ]; then
    ARGS+=(--limit "$LIMIT")
fi

echo ""
echo "[1/3] reranking ($(date))"
run python experiments/run_rerank.py "${ARGS[@]}"

# --- 2. is the output usable? ----------------------------------------------
# The run finishing is not the same as the run being good: a reranker that scored every candidate
# identically produces a complete, well-formed, entirely uninformative table. This checks the
# structure and then prints whole reranked documents to read.
echo ""
echo "[2/3] validating the output ($(date))"
run python experiments/check_rerank_output.py "$OUT_DIR" --sample "$SAMPLE" --seed "$SEED"

echo ""
echo "Done $(date)."
echo ""
echo "Wrote, under $OUT_DIR:"
echo "  rerank_summary.csv                  one row per shortlist size, reranked vs baseline top-k"
echo "  rerank_predictions_jina_k*.csv      per document: predicted author, its rank, the base rank"
echo "  rerank_detail_jina_k*.csv           per candidate: rank, distance rank, relevance score"
echo ""
echo "READ THE SAMPLED DOCUMENTS ABOVE, not just the summary. The number that matters first is"
echo "'top pick by distance rank' in the run log: mass on 0-1 means the reranker agrees with the"
echo "vectors about the neighbourhood and is sharpening it; a flat spread means it found nothing the"
echo "embedding had not already found, and the rerank is closer to a shuffle than to an attack."
echo ""
echo "Then, for the paid arm:  sbatch experiments/run_rerank_sonnet.sbatch"
