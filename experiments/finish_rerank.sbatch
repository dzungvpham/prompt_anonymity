#!/bin/bash
#
# finish_rerank.sbatch -- publish one comparison table from rerank arms that ran side by side.
#
#     BASE=experiments/results/rerank/swe_chat_base_gemini_embedding_2_known0075 \
#         sbatch experiments/finish_rerank.sbatch
#
# Normally you do not run this by hand: experiments/submit_rerank_swe_chat.sh submits it with a
# dependency on both arms.
#
# WHY THIS EXISTS. run_rerank.py MERGES rerank_summary.csv rather than overwriting it, keyed on
# (variant, top_k) -- which makes two SEQUENTIAL jobs compose correctly and two CONCURRENT ones
# race, because the merge is a read, a concat and a write with no lock around them. Running the
# arms in parallel is the whole point here (one wants a GPU, the other wants an API and neither
# wants to wait for the other), so each writes into its own subdirectory and this job reconciles
# them afterwards. Nothing else in the pipeline needs to know: the per-arm files are copied up
# under their own names, and the table that lands in $BASE is the one every downstream reader
# already expects.
#
# The per-arm output files carry the variant in their names -- rerank_predictions_jina_k5.csv vs
# rerank_predictions_llm_k5.csv -- so nothing collides on the way up.
#
# SUBMITTED WITH afterany, NOT afterok, deliberately: if the paid arm dies at hour three, the free
# arm's results should still be published rather than stranded in a subdirectory. Which arms were
# found, and which were not, is stated in the log and in the exit status.
#
# SUBMIT FROM THE REPO ROOT (--output is relative and SLURM resolves it against the submit dir).
# One-time: mkdir -p scripts/logs
#
# Overrides:
#   BASE=<dir>          (required) the parent directory the arms wrote into
#   ARMS="jina llm"     which subdirectories to look for
#   SAMPLE=8            reranked documents to print per configuration (0 prints none)
#   SEED=47             which documents get sampled
#
#SBATCH --job-name=rrk-finish
#SBATCH --output=scripts/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8g
#SBATCH --time=0-00:20:00
#SBATCH --requeue

set -euo pipefail

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
    exit 1
fi
cd "$PROJECT"

if [ -z "${BASE:-}" ]; then
    echo "ERROR: BASE is not set. It is the directory the arms wrote their subdirectories into," >&2
    echo "  e.g. BASE=\$PWD/experiments/results/rerank/swe_chat_base_gemini_embedding_2_known0075" >&2
    exit 1
fi
if [ ! -d "$BASE" ]; then
    echo "ERROR: BASE=$BASE does not exist. Neither arm got far enough to create it, so there is" >&2
    echo "  nothing to publish -- read their logs in scripts/logs/ instead." >&2
    exit 1
fi

read -r -a ARM_LIST <<< "${ARMS:-jina llm}"
SAMPLE="${SAMPLE:-3}"
SEED="${SEED:-47}"

RUN=(bash scripts/slurm_job.sh)

echo "=================================================================="
echo "publishing the rerank comparison table"
echo "  base: $BASE"
echo "  arms: ${ARM_LIST[*]}"
echo "Job ${SLURM_JOB_ID:-local} on $(hostname), $(date)"
echo "=================================================================="

# --- 1. which arms actually produced anything ------------------------------
echo ""
echo "[1/3] arms ($(date))"
# The subdirectory is named for the run_rerank.py VARIANT; the job that wrote it is named for the
# model. They differ for the paid arm -- variant `llm`, job `rrk-sonnet` -- and a hint that sends
# someone to a log file that does not exist is worse than no hint.
job_of() { if [ "$1" = "llm" ]; then echo rrk-sonnet; else echo "rrk-$1"; fi; }

FOUND=()
for arm in "${ARM_LIST[@]}"; do
    if [ -f "$BASE/$arm/rerank_summary.csv" ]; then
        echo "  $arm: found $(wc -l < "$BASE/$arm/rerank_summary.csv") line(s) of summary"
        FOUND+=("$arm")
    else
        echo "  $arm: NO rerank_summary.csv under $BASE/$arm -- that arm did not finish."
        echo "        Its log is scripts/logs/$(job_of "$arm")-*.out; re-running that arm and then"
        echo "        this job folds it in, and the cache means it re-judges only what is new."
    fi
done
if [ "${#FOUND[@]}" -eq 0 ]; then
    echo ""
    echo "ERROR: no arm produced a summary. Nothing to publish." >&2
    exit 1
fi

# --- 2. copy the per-arm tables up, then merge the summaries ---------------
# Copy first, merge second: check_rerank_output.py reads the summary and then opens the prediction
# and detail files named by its rows, so a summary row whose files have not arrived yet is a
# failure report rather than a missing file.
echo ""
echo "[2/3] merging ($(date))"
for arm in "${FOUND[@]}"; do
    for file in "$BASE/$arm"/rerank_predictions_*.csv "$BASE/$arm"/rerank_detail_*.csv; do
        [ -f "$file" ] || continue
        cp -f "$file" "$BASE/"
        echo "  $(basename "$file")"
    done
done

# Rows are keyed (variant, top_k) and later inputs win, which is the rule run_rerank.py itself
# follows. An existing merged table is the FIRST input, so a row from an arm that is not in this
# run's ARMS survives instead of being silently dropped.
"${RUN[@]}" python -c '
import sys
from pathlib import Path

import pandas as pd

base = Path(sys.argv[1])
arms = sys.argv[2:]

frames = []
merged = base / "rerank_summary.csv"
if merged.exists():
    frames.append(pd.read_csv(merged))
for arm in arms:
    frames.append(pd.read_csv(base / arm / "rerank_summary.csv"))

table = pd.concat(frames, ignore_index=True)
table = table.drop_duplicates(subset=["variant", "top_k"], keep="last")
table = table.sort_values(["variant", "top_k"], kind="mergesort")
table.to_csv(merged, index=False)

print(f"  wrote {len(table)} row(s) to {merged}")
for variant, top_k in zip(table["variant"], table["top_k"]):
    print(f"    {variant} k={top_k}")
' "$BASE" "${FOUND[@]}"

# --- 3. is the combined output usable? -------------------------------------
# Every arm validated its own directory when it finished; this re-checks the published copies, so
# a table assembled here is held to exactly the same standard as one a single job wrote.
echo ""
echo "[3/3] validating ($(date))"
"${RUN[@]}" python experiments/check_rerank_output.py "$BASE" --sample "$SAMPLE" --seed "$SEED"

echo ""
echo "Done $(date)."
echo ""
echo "  $BASE/rerank_summary.csv"
echo ""
echo "Read it as a subtraction, not as four numbers: each row carries the reranked top-k beside"
echo "the base_top_k the SAME shortlist scored under plain nearest-neighbor linkage. top_k at the"
echo "shortlist size is pinned to the baseline by construction -- the reranker cannot add an author"
echo "the distance metric did not shortlist -- so the gain lives entirely in top-1 through top-(k-1)."
echo ""
echo "The jina rows are the control. They price the frontier model rather than the idea of"
echo "reranking: if a free 0.6B retrieval reranker moves top-1 as far as Sonnet 5 does, the"
echo "interesting result is that the shortlist was the whole attack."
if [ "${#FOUND[@]}" -lt "${#ARM_LIST[@]}" ]; then
    echo ""
    echo "NOTE: this table holds only the ${FOUND[*]} arm(s) -- see stage 1 for what is missing."
fi
