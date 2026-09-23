#!/bin/bash
#
# run_dpmlm_wildchat.sbatch -- DP-MLM over all of WildChat on ONE GPU with WORKERS processes
# sharing it, then the usual pipeline: defend -> featurize -> attack.
#
#   sbatch experiments/run_dpmlm_wildchat.sbatch
#
# Writes:
#   data/dist/shards/wildchat_dp_mlm_eps100.000N-of-0005.parquet  one per worker (intermediate)
#   data/dist/wildchat_dp_mlm_eps100.parquet                      defended split (merged)
#   data/dist/wildchat_dp_mlm_eps100_stylometrix.parquet          features
#   experiments/results/wildchat_dp_mlm_eps100_stylometrix_*/     attack results
#
# To read the defended text as CSV afterwards (login node, minutes):
#   python experiments/export_defended_csv.py --source wildchat --defense dp_mlm_eps100
#
# SUBMIT FROM THE REPO ROOT, with the environment pinned -- scripts/slurm_job.sh otherwise
# activates $PROJECT/.venv if one exists, which on this checkout is not the real env:
#   export PROMPT_ANONYMITY_CONDA_ENV=$HOME/prompt_anonymity/env
#   mkdir -p scripts/logs
#
# WHY SEVERAL WORKERS ON ONE CARD: a DP-MLM process alternates a single-threaded CPU phase (NLTK
# detokenize + tokenizer encode per masked word) with a GPU phase and cannot overlap them, so one
# process leaves the GPU idle much of the time. Five processes overlap each other's CPU phases.
# Each worker defends every 5th document (interleaved), and the last one to finish merges.
#
# MEMORY: the longest WildChat turns (~100k words) are sampled in slices/streamed in waves
# (DPMLM_SAMPLE_SLICE in dp_mlm.py), so a worker peaks at a few GB regardless of turn length.
# The 80 GB constraint is a safety margin on top of that.
#
# PREEMPTION / TIMEOUT: each worker checkpoints its cache every DPMLM_CHECKPOINT_EVERY turns, so
# resubmitting the same command resumes rather than restarts. --requeue covers preemption; a
# TIMEOUT is not requeued by SLURM, so if it hits the wall clock, just sbatch it again.
# KEEP WORKERS THE SAME ACROSS RESUBMITS: the shard count is part of the cache path, so changing it
# starts over from zero.
#
# Overrides: WORKERS=2 ... | DEFENSE=dp_mlm_eps250 ... | FEATURE=gemini_embedding_2 ... | ATTACK=0
# Anything after the script name goes to run_experiment.py, e.g. --attacks nearest_neighbor
#
#SBATCH --job-name=dpmlm-wc
#SBATCH --output=scripts/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --partition=gpu-preempt
#SBATCH --gpus=1
#SBATCH --constraint=a100-80g|h100|h200
#SBATCH --cpus-per-task=12
#SBATCH --mem=80g
#SBATCH --time=1-00:00:00
#SBATCH --requeue

set -euo pipefail

PROJECT="${PROMPT_ANONYMITY_PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT"

SOURCE="${SOURCE:-wildchat}"
DEFENSE="${DEFENSE:-dp_mlm_eps100}"
FEATURE="${FEATURE:-stylometrix}"
ATTACK="${ATTACK:-1}"
WORKERS="${WORKERS:-5}"
READ_DIR="${DIST_DIR:-$PROJECT/data/hf}"
OUT_DIR="${OUT_DIR:-$PROJECT/data/dist}"
LOG_DIR="$PROJECT/scripts/logs"
STEM="${SOURCE}_${DEFENSE}"
MERGED="$OUT_DIR/${STEM}.parquet"
JOB="${SLURM_JOB_ID:-local}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DPMLM_CHECKPOINT_EVERY="${DPMLM_CHECKPOINT_EVERY:-1000}"
# Split the allocation's cores between workers instead of each torch runtime assuming the node.
CPUS="${SLURM_CPUS_PER_TASK:-12}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(( CPUS / WORKERS > 0 ? CPUS / WORKERS : 1 ))}"
export TOKENIZERS_PARALLELISM=false

RUN=(bash scripts/slurm_job.sh)

echo "DP-MLM on $SOURCE: defense=$DEFENSE feature=$FEATURE workers=$WORKERS  job $JOB on $(hostname), $(date)"

if [ ! -f "$READ_DIR/${SOURCE}.parquet" ]; then
    echo "ERROR: no $READ_DIR/${SOURCE}.parquet"
    exit 1
fi
mkdir -p "$LOG_DIR"

# Fail in seconds, not hours, if the job landed in the wrong environment or on no GPU -- or if the
# long-turn streaming is missing or disabled. Without it, WildChat's longest turns (~100k words)
# need ~19 GB in a single allocation and OOM the run hours in (job 64621446). The check reads the
# code that will actually run, so it also catches a cluster checkout that predates the fix.
"${RUN[@]}" python -c '
import sys, torch, transformers, nltk
from prompt_anonymity.defenses import dp_mlm
print("[preflight]", sys.executable, "| torch", torch.__version__, "| cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("[preflight] no GPU visible")
slice_ = getattr(dp_mlm, "DPMLM_SAMPLE_SLICE", None)
if slice_ is None or not hasattr(dp_mlm._DPMLMBackend, "_process_long_turn"):
    sys.exit("[preflight] this dp_mlm.py has no long-turn streaming -- sync the fixed file to the cluster")
if slice_ <= 0:
    sys.exit(f"[preflight] DPMLM_SAMPLE_SLICE={slice_} disables streaming; unset it or set it > 0")
print(f"[preflight] long-turn streaming ON: turns over {slice_} masked positions are streamed")
'

# Shard files from earlier attempts with a different worker count must be moved aside: the merge
# refuses to combine shards of two different counts, so they would fail this job at the very end.
# (Moved, not deleted. The merge only looks at the top of shards/, not at subdirectories.)
STALE=$(ls "$OUT_DIR/shards/${STEM}."*-of-*.parquet 2>/dev/null \
        | grep -v -- "-of-$(printf '%04d' "$WORKERS")\.parquet$" || true)
if [ -n "$STALE" ]; then
    mkdir -p "$OUT_DIR/shards/stale"
    echo "moving $(echo "$STALE" | wc -l) stale shard file(s) to $OUT_DIR/shards/stale/"
    echo "$STALE" | xargs -I{} mv {} "$OUT_DIR/shards/stale/"
fi

# 1. Defend: WORKERS processes on the one GPU, each on every WORKERS-th document. Each worker has
#    its own log (appended, so a requeue keeps the earlier attempt's). The last to finish merges.
echo "[1/3] defend with $WORKERS workers  ($(date))"
pids=()
for worker in $(seq 0 $((WORKERS - 1))); do
    log="$LOG_DIR/dpmlm-wc-${JOB}.shard${worker}.log"
    echo "  worker $worker -> $log"
    echo "=== worker $worker, job $JOB, attempt starting $(date) ===" >> "$log"
    "${RUN[@]}" python -m prompt_anonymity.data.apply_defenses \
        --source "$SOURCE" --defense "$DEFENSE" --dist-dir "$READ_DIR" --out-dir "$OUT_DIR" \
        --num-shards "$WORKERS" --shard-index "$worker" >> "$log" 2>&1 &
    pids+=($!)
    sleep 10   # stagger model loads
done

failed=0
for worker in "${!pids[@]}"; do
    if wait "${pids[$worker]}"; then
        echo "  worker $worker finished ($(date))"
    else
        echo "  worker $worker FAILED ($(date)) -- see $LOG_DIR/dpmlm-wc-${JOB}.shard${worker}.log"
        failed=1
    fi
done
if [ "$failed" != "0" ]; then
    echo "A worker failed. Fix the cause and resubmit with the same WORKERS -- finished turns are cached."
    exit 1
fi

# Merge explicitly, always. The last worker already auto-merges, but checking "does the merged
# file exist" is not enough: an older one (e.g. from a smoke test) would pass that check and be
# featurized instead of this run's output. Re-merging only concatenates the shard files -- a
# minute, no GPU -- and overwrites whatever was there.
echo "merging shards  ($(date))"
"${RUN[@]}" python -m prompt_anonymity.data.apply_defenses \
    --source "$SOURCE" --defense "$DEFENSE" --dist-dir "$READ_DIR" --out-dir "$OUT_DIR" --merge

echo "[2/3] featurize  ($(date))"
"${RUN[@]}" python -m prompt_anonymity.data.compute_features \
    --source "$SOURCE" --defense "$DEFENSE" --feature "$FEATURE" --dist-dir "$OUT_DIR"

if [ "$ATTACK" != "0" ]; then
    echo "[3/3] attack  ($(date))"
    "${RUN[@]}" python experiments/run_experiment.py \
        --source "$SOURCE" --defense "$DEFENSE" --feature "$FEATURE" --data-dir "$OUT_DIR" "$@"
fi

echo "Done $(date)."
echo "  defended: $MERGED"
echo "  results:  experiments/results/${SOURCE}_${DEFENSE}_${FEATURE}_*/"
