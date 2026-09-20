#!/usr/bin/env bash
#
# submit_rerank_swe_chat.sh -- start BOTH listwise rerank arms on SWE-chat at once, unbatched,
# and publish one comparison table when they are done.
#
#     bash experiments/submit_rerank_swe_chat.sh
#     DRY=1 bash experiments/submit_rerank_swe_chat.sh        # print the sbatch lines, submit none
#
# RUN THE SMOKE TEST FIRST -- ten CPU minutes and about five cents against forty dollars:
#
#     sbatch experiments/run_rerank_smoke.sbatch
#
# THIS IS NOT AN SBATCH FILE. It is a submit wrapper, like experiments/run_dpmlm_sweep.sh, and it
# queues three jobs:
#
#   1. rrk-jina    GPU,  free.       jina-reranker-v3.5 locally. The control.
#   2. rrk-sonnet  CPU,  ~$35-45.    Claude Sonnet 5 over OpenRouter, reasoning on, NO Batch API.
#   3. rrk-finish  CPU,  free.       afterany on both: merges their tables and validates.
#
# WHY TWO JOBS AND NOT ONE. The arms want different hardware -- one an A100, the other nothing but
# a socket -- and running them in one allocation would hold a GPU idle for the Sonnet arm's hours
# and put both arms behind a single preemption. Two allocations started together IS the
# parallelism here; neither arm has an internal sharding axis to exploit (run_rerank.py has no
# --shard flags, and the jina attack scores one shortlist per forward pass), so the concurrency
# that is available is: the two arms against each other, and WORKERS requests in flight inside the
# Sonnet one.
#
# WHY A THIRD JOB. run_rerank.py merges rerank_summary.csv read-modify-write, which is safe for
# sequential jobs and a race for concurrent ones -- so each arm writes into its own subdirectory
# and rrk-finish reconciles them. See experiments/finish_rerank.sbatch.
#
# NO BATCH API, so the Sonnet arm answers in its own allocation instead of across a 24-hour
# window, and pays full real-time price for it ($2/$10 per MTok against the batch tier's $1/$5).
# Its key is SONNET_API_KEY in a .env AT THE REPO ROOT -- load_dotenv walks UP from the working
# directory, so DS_env/.env is never found. Verdicts are cached by prompt text, so a re-run judges
# only what is new and an interrupted arm resumes cheaply.
#
# Overrides:
#   FEATURE=luar bash ...              # needs data/dist/swe_chat_luar.parquet to exist first
#   TOP_KS="5" bash ...                # one shortlist size instead of both
#   LIMIT=200 bash ...                 # a seeded sample of the unknown side; bounds the bill
#   WORKERS=32 bash ...                # concurrent Sonnet requests (default 16)
#   EFFORT=medium bash ...             # thinking depth: low/medium/high/xhigh/max
#   SOURCE=wildchat_small bash ...     # despite the name, nothing here is swe_chat-specific
#   DRY=1 bash ...                     # print, do not submit

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
    echo "ERROR: cannot find the repo root. Run this from the repo root, or set" \
         "PROMPT_ANONYMITY_PROJECT=/path/to/repo." >&2
    exit 1
fi
cd "$PROJECT"

SOURCE="${SOURCE:-swe_chat}"
FEATURE="${FEATURE:-gemini_embedding_2}"
DEFENSE="${DEFENSE:-none}"
WINDOW="${WINDOW:-0075}"
TOP_KS="${TOP_KS:-5 10}"
WORKERS="${WORKERS:-16}"
SEED="${SEED:-47}"
SAMPLE="${SAMPLE:-3}"
DIST_DIR="${DIST_DIR:-$PROJECT/data/dist}"
DRY="${DRY:-0}"

tag_of() { if [ "$1" = "none" ]; then echo base; else echo "$1"; fi; }
BASE="${BASE:-$PROJECT/experiments/results/rerank/${SOURCE}_$(tag_of "$DEFENSE")_${FEATURE}_known${WINDOW}}"

mkdir -p scripts/logs

echo "=================================================================="
echo "listwise rerank, both arms, $SOURCE / $FEATURE / known$WINDOW"
echo "  shortlist sizes: $TOP_KS"
if [ -n "${LIMIT:-}" ]; then
    echo "  unknown side:    a seeded sample of $LIMIT documents"
else
    echo "  unknown side:    all of it"
fi
echo "  sonnet:          unbatched, $WORKERS concurrent, full real-time price"
echo "  published to:    $BASE"
echo "=================================================================="

# The scripts are steered by their environment, as every other job in this repo is; `env VAR=...`
# rather than a prefixed string so that TOP_KS="5 10" survives as one value. DRY prints the line
# instead of running it, quoted the way the shell would have to see it.
submit() {
    if [ "$DRY" = "1" ]; then
        # stderr, because the caller captures stdout to read the job id off it.
        { printf '  '; printf '%q ' "$@"; printf '\n'; } >&2
        echo "DRY"       # stands in for the job id, so the dependency line still prints
        return 0
    fi
    "$@"
}

# Shared by both arms: same split, same shortlist, same presentation -- which is the point of the
# control. Anything that differs between the arms is added below.
COMMON_ENV=(SOURCE="$SOURCE" FEATURE="$FEATURE" DEFENSE="$DEFENSE" WINDOW="$WINDOW"
            SEED="$SEED" SAMPLE="$SAMPLE" DIST_DIR="$DIST_DIR" TOP_KS="$TOP_KS")
[ -n "${LIMIT:-}" ] && COMMON_ENV+=(LIMIT="$LIMIT")

JINA_ENV=("${COMMON_ENV[@]}" OUT_DIR="$BASE/jina")

SONNET_ENV=("${COMMON_ENV[@]}" OUT_DIR="$BASE/llm" BATCH=0 WORKERS="$WORKERS")
[ -n "${EFFORT:-}" ] && SONNET_ENV+=(EFFORT="$EFFORT")
[ -n "${MODEL:-}" ] && SONNET_ENV+=(MODEL="$MODEL")

echo ""
echo "--- 1/3: jina (GPU, free) ---"
JINA=$(submit env "${JINA_ENV[@]}" sbatch --parsable experiments/run_rerank_jina.sbatch | tail -1)
echo "job $JINA"

echo ""
echo "--- 2/3: sonnet (CPU, paid) ---"
SONNET=$(submit env "${SONNET_ENV[@]}" sbatch --parsable experiments/run_rerank_sonnet.sbatch \
         | tail -1)
echo "job $SONNET"

# afterany, not afterok: one arm failing must not strand the other's results. The finisher reports
# which arms it found.
echo ""
echo "--- 3/3: finish (CPU, free, after both) ---"
FINISH=$(submit env BASE="$BASE" SAMPLE="$SAMPLE" SEED="$SEED" \
         sbatch --parsable --dependency="afterany:$JINA:$SONNET" \
         experiments/finish_rerank.sbatch | tail -1)
echo "job $FINISH"

if [ "$DRY" = "1" ]; then
    echo ""
    echo "(DRY=1 -- nothing was submitted)"
    exit 0
fi

echo ""
echo "=================================================================="
echo "Submitted $JINA (jina), $SONNET (sonnet), $FINISH (finish)."
echo ""
echo "  squeue -j $JINA,$SONNET,$FINISH"
echo "  tail -f scripts/logs/rrk-jina-$JINA.out"
echo "  tail -f scripts/logs/rrk-sonnet-$SONNET.out"
echo "  tail -f scripts/logs/rrk-finish-$FINISH.out"
echo ""
echo "Then read:"
echo "  column -s, -t < $BASE/rerank_summary.csv"
echo ""
echo "CANCEL ALL THREE, including the pending finisher:"
echo "  scancel $JINA $SONNET $FINISH"
echo "=================================================================="
