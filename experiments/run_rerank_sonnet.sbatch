#!/bin/bash
#
# run_rerank_sonnet.sbatch -- rerank a nearest-neighbor shortlist with Claude Sonnet 5, reasoning
# on, through OpenRouter, then check that the output is actually usable.
#
# TWO DELIVERY CHANNELS, and the choice is money against wall clock. Same model, same rubric, same
# effort, same cache either way -- only how the requests are delivered differs.
#
# BATCH=1 (the default): OpenRouter's Batch API at ~half price, in TWO PHASES, TWO SUBMISSIONS. A
# batch has a 24-hour completion window and the longest wall clock in this repo is 16h, so the job
# that submits is not the job that collects:
#
#     sbatch experiments/run_rerank_sonnet.sbatch                 # PHASE=submit: queues, exits
#     PHASE=collect sbatch experiments/run_rerank_sonnet.sbatch   # later: polls, writes, validates
#
# BATCH=0: the synchronous client, one job, no phases, answers in this allocation. FULL PRICE --
# $2/$10 per MTok against the batch tier's $1/$5, so roughly double the figures quoted below. Worth
# it when you want the result today rather than tomorrow. Its key is SONNET_API_KEY (not
# SONNET_OR_KEY), and WORKERS sets how many requests are in flight:
#
#     BATCH=0 sbatch experiments/run_rerank_sonnet.sbatch
#     BATCH=0 WORKERS=24 SOURCE=swe_chat sbatch experiments/run_rerank_sonnet.sbatch
#
# For the full swe_chat pairing of this job with the free jina control, both started at once, use
# the launcher rather than this file directly:  bash experiments/submit_rerank_swe_chat.sh
#
# SUBMIT FROM THE REPO ROOT (--output is relative and SLURM resolves it against the submit dir).
# One-time: mkdir -p scripts/logs
#
# WHY THIS IS ONE FILE AND NOT TWO, and the thing to be careful about if you edit it: the collect
# phase resumes the submitted batch only if it can reproduce the prompts byte for byte. The batch id
# is recorded in a ticket keyed on a hash of the exact prompts, so ANY drift between the phases --
# a different SOURCE, FEATURE, WINDOW, SEED, LIMIT or TOP_KS -- misses the ticket, submits a second
# copy of the same work, and bills you twice for it. Two separate scripts would let those defaults
# drift; one script cannot. If you override something at submit time, override it identically at
# collect time.
#
# NO GPU. This is an HTTP client that spends most of its life asleep between status checks.
#
# COST. ~$0.006 per document at --top-k 5 and ~$0.012 at --top-k 10, at the batch rate. wildchat_tiny
# has ~225 documents on the unknown side, so both shortlist sizes together are roughly $4. Verdicts
# are cached by prompt text, so a re-run judges only what is new and a fully-cached run makes no
# request at all. Start with LIMIT set if you want to read the reasons before committing the rest.
#
# THE DISCOUNT IS CARRIED BY THE MODEL SLUG, not by the endpoint: OpenRouter prices
# anthropic/claude-sonnet-5 at $2/$10 per MTok and anthropic/claude-sonnet-5:batch at $1/$5, and the
# batch endpoint accepts either. The client appends the :batch variant itself and prints the slug it
# actually submitted -- check that line in the log; a plain slug there means you are paying double.
#
# RUN THE JINA JOB FIRST. It exercises the same split, shortlist, fold-back and output tables for
# free, so anything structurally broken fails there rather than here.
#
# SEE WHAT IS MISSING WITHOUT SUBMITTING ANYTHING -- a second, on the login node:
#
#     CHECK_ONLY=1 bash experiments/run_rerank_sonnet.sbatch
#
# The submit phase takes minutes, not hours; the wall clock below is sized for collect. Shorten it:
#
#     sbatch --time=0-01:00:00 experiments/run_rerank_sonnet.sbatch
#
# BATCH=0 needs the whole wall clock, though: all of SWE-chat's unknown side at both shortlist
# sizes is ~2,200 requests, which is roughly 3-4h at WORKERS=16.
#
# INTERRUPTION IS CHEAP, which is not free and is worth knowing exactly. Each verdict is written
# to the cache the moment it arrives, so a job that dies -- requeued, out of wall clock, node
# failure, scancel -- loses only the requests that were in flight at that instant (at most
# WORKERS of them). Everything already answered is banked, and re-running the identical command
# re-buys nothing. This partition (`cpu`) does not preempt, so the realistic risks here are the
# wall clock and your own scancel rather than the scheduler.
#
# NEEDS A KEY in the environment or in a .env AT THE REPO ROOT (load_dotenv walks UP from the
# working directory, so DS_env/.env is not found). Which one depends on the channel --
# SONNET_OR_KEY for BATCH=1, SONNET_API_KEY for BATCH=0. Deliberately not OPENROUTER_API_KEY in
# either case: this attack's spend is meant to be separable from the defenses' and the other
# judges'.
#
# Overrides (sbatch exports your submitting environment; under BATCH=1, KEEP THEM IDENTICAL ACROSS
# BOTH PHASES -- see above for why drift there bills you twice):
#   SOURCE=wildchat_small sbatch ...
#   FEATURE=luar sbatch ...
#   TOP_KS="5" sbatch ...
#   LIMIT=50 sbatch ...                    # a seeded sample of the unknown side; bounds the bill
#   EFFORT=medium sbatch ...               # thinking depth: low/medium/high/xhigh/max
#   MODEL=anthropic/claude-opus-5 sbatch ...
#   WORKERS=24 sbatch ...                  # BATCH=0 only: concurrent requests (default 16)
#
#SBATCH --job-name=rrk-sonnet
#SBATCH --output=scripts/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16g
#SBATCH --time=0-12:00:00
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

BATCH="${BATCH:-1}"
if [ "$BATCH" != "0" ] && [ "$BATCH" != "1" ]; then
    echo "ERROR: BATCH must be 1 (the Batch API, two phases) or 0 (synchronous, one job)," \
         "got '$BATCH'." >&2
    exit 1
fi

if [ "$BATCH" = "0" ]; then
    # Not defaulted-and-ignored: a PHASE left over in the submitting environment would otherwise
    # read as "this is the collect half of something", and there is no other half.
    if [ -n "${PHASE:-}" ]; then
        echo "ERROR: PHASE=$PHASE was set, but BATCH=0 has no phases -- it submits nothing to" \
             "collect later. Drop PHASE, or use BATCH=1." >&2
        exit 1
    fi
    PHASE="none"
    KEY_ENV="SONNET_API_KEY"
else
    PHASE="${PHASE:-submit}"
    if [ "$PHASE" != "submit" ] && [ "$PHASE" != "collect" ]; then
        echo "ERROR: PHASE must be 'submit' or 'collect' (got '$PHASE')." >&2
        exit 1
    fi
    KEY_ENV="SONNET_OR_KEY"
fi

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
if [ "$BATCH" = "0" ]; then
    echo "listwise rerank (sonnet, UNBATCHED -- full real-time price)"
else
    echo "listwise rerank (sonnet, batched) -- PHASE=$PHASE"
fi
echo "  $SOURCE / $FEATURE / known$WINDOW"
echo "  shortlist sizes: ${TOP_K_LIST[*]}"
echo "  output:          $OUT_DIR"
if [ "$BATCH" = "0" ]; then
    echo "  concurrency:     ${WORKERS:-16} requests in flight"
else
    echo "  tickets:         $CACHE_DIR/attacks/_batches"
fi
echo "  key:             \$$KEY_ENV"
echo "Job ${SLURM_JOB_ID:-local} on $(hostname), $(date)"
echo "=================================================================="

# --- 0a. offline invariants ------------------------------------------------
# Both self-tests, and this is the right place for them: the second one checks the request shape,
# the :batch slug, the custom_id keying and the ticket logic without a key and without a cent. A
# payload OpenRouter would reject should fail here, not 24 hours after you paid for it.
echo ""
echo "[0a/4] self-tests ($(date))"
# Which interpreter, before anything is asked of it. A job that fails on an import is nearly
# always running in a different environment from the one you tested in, and without this line
# the log cannot tell you that -- compare it against the smoke test's own [env] interpreter.
run python -c 'import sys; print("  interpreter:", sys.executable)'
run python -m prompt_anonymity.attacks.llm.listwise --selftest
# The batch client's self-test is about the batch payload, the :batch slug and the ticket logic --
# none of which the synchronous channel has.
if [ "$BATCH" = "1" ]; then
    run python -m prompt_anonymity.attacks.llm._openrouter_batch --selftest
fi

# --- 0b. inputs ------------------------------------------------------------
echo ""
echo "[0b/4] inputs ($(date))"
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

# --- 0c. the key -----------------------------------------------------------
# Checked before submitting rather than discovered by the client mid-run, so a missing key costs a
# queue slot instead of a half-submitted sweep.
echo ""
echo "[0c/4] credentials ($(date))"
# Indirect expansion: which variable holds the key depends on the channel (see the KEY_ENV block).
if [ -z "${!KEY_ENV:-}" ] && [ ! -f "$PROJECT/.env" ]; then
    echo "ERROR: $KEY_ENV is not set and there is no $PROJECT/.env for load_dotenv to read." >&2
    echo "  Add it at the REPO ROOT (load_dotenv walks UP from the working directory, so a .env in" >&2
    echo "  a subdirectory -- DS_env/.env -- is never found):" >&2
    echo "    echo '$KEY_ENV=sk-or-...' >> $PROJECT/.env" >&2
    exit 1
fi
if [ -n "${!KEY_ENV:-}" ]; then
    echo "  $KEY_ENV: set in the environment"
else
    echo "  $KEY_ENV: expected in $PROJECT/.env (not read here, so it stays out of this log)"
fi

# Then actually USE it. The check above only proves a variable or a file exists, and a .env that
# exists but defines a different variable passes it happily -- which is how a run reached the
# thread pool and took 401 on every request. This asks OpenRouter, generates no tokens, and
# prints the key's length and last four characters rather than the key.
if [ "$BATCH" = "0" ]; then
    run python -c '
import sys
from prompt_anonymity.attacks.llm._openrouter import check_credentials
from prompt_anonymity.attacks.llm.listwise_llm_rerank import SYNC_API_KEY_ENV

ok, message = check_credentials(SYNC_API_KEY_ENV)
print("  " + message)
if not ok:
    sys.exit(1)
'
fi

if [ "$CHECK_ONLY" = "1" ]; then
    echo ""
    echo "(CHECK_ONLY=1 -- preflight only, nothing submitted)"
    exit 0
fi

# --- 1. rerank (unbatched), or submit, or collect ---------------------------
# COMMON is built once and shared by all three, which is the point: the batch phases differ by
# exactly one flag (--no-wait) and so build byte-identical prompts, and the unbatched channel
# builds the same ones again, so its verdicts and theirs are the same cache entries.
#
# The batch phases pass ONE shortlist size per invocation because --no-wait stops at the first
# configuration with uncached rows. The unbatched channel has no such stop and passes all of them.
COMMON=(--source "$SOURCE" --feature "$FEATURE" --defense "$DEFENSE"
        --data-dir "$DIST_DIR" --known-window "$WINDOW" --seed "$SEED"
        --cache-dir "$CACHE_DIR" --output-dir "$OUT_DIR" --variant llm)
if [ -n "${LIMIT:-}" ]; then
    COMMON+=(--limit "$LIMIT")
fi
if [ -n "${EFFORT:-}" ]; then
    COMMON+=(--reasoning-effort "$EFFORT")
fi
if [ -n "${MODEL:-}" ]; then
    COMMON+=(--judge-model "$MODEL")
fi

if [ "$BATCH" = "0" ]; then
    # ONE invocation with every shortlist size. The batch path splits them up only because
    # --no-wait stops at the first configuration with uncached rows; with nothing queued that
    # constraint is gone, and one call means one read-modify-write of rerank_summary.csv.
    export LISTWISE_RERANK_MAX_WORKERS="${WORKERS:-16}"
    TOP_K_ARGS=()
    for k in "${TOP_K_LIST[@]}"; do
        TOP_K_ARGS+=(--top-k "$k")
    done

    echo ""
    echo "[1/4] reranking, unbatched ($(date))"
    echo "  $LISTWISE_RERANK_MAX_WORKERS concurrent requests; 429s are retried with backoff, so a"
    echo "  value above the account's rate limit costs latency rather than failures."
    run python experiments/run_rerank.py "${COMMON[@]}" --no-batch "${TOP_K_ARGS[@]}"

elif [ "$PHASE" = "submit" ]; then
    echo ""
    echo "[1/4] submitting batches ($(date))"
    QUEUED=0
    CACHED=0
    for k in "${TOP_K_LIST[@]}"; do
        echo ""
        echo "--- top-k $k ---"
        # 3 is _openrouter_batch.BATCH_QUEUED_EXIT: "queued, come back and collect". It is distinct
        # from 1 precisely so this branch can exist -- an exit of 1 would be indistinguishable from
        # a crash, and a successful submission would show up in SLURM as a failed job.
        set +e
        run python experiments/run_rerank.py "${COMMON[@]}" --top-k "$k" --no-wait
        status=$?
        set -e
        case "$status" in
            0) echo "  top-k $k: nothing to submit -- every row was already cached."
               CACHED=$((CACHED + 1)) ;;
            3) echo "  top-k $k: queued."
               QUEUED=$((QUEUED + 1)) ;;
            *) echo "ERROR: top-k $k failed with status $status (not a queued batch)." >&2
               exit "$status" ;;
        esac
    done

    echo ""
    echo "Done $(date).  $QUEUED batch group(s) queued, $CACHED already cached."
    echo ""
    echo "The batch ids are printed above and recorded under $CACHE_DIR/attacks/_batches."
    echo "Most batches finish well inside an hour; the window is 24h. Then collect with the SAME"
    echo "overrides you used here:"
    echo ""
    echo "    PHASE=collect sbatch experiments/run_rerank_sonnet.sbatch"
    echo ""
    echo "Collecting early is safe -- it just polls until the batch is done."
    exit 0

else
    echo ""
    echo "[1/4] collecting batches ($(date))"
    echo "  (the same prompts as the submit phase, so each ticket resumes its batch rather than"
    echo "   submitting a second copy of it -- watch for 'resuming existing batch' below)"
    for k in "${TOP_K_LIST[@]}"; do
        echo ""
        echo "--- top-k $k ---"
        run python experiments/run_rerank.py "${COMMON[@]}" --top-k "$k"
    done
fi

# --- 2. is the output usable? ----------------------------------------------
# The run finishing is not the same as the run being good. A judge that refused every row degrades
# silently to the distance ordering BY DESIGN -- correct behaviour, and indistinguishable from
# success if you only read the accuracy columns. This checks the parse rate, the reason quality and
# the table structure, then prints whole reranked documents.
echo ""
echo "[2/4] validating the output ($(date))"
run python experiments/check_rerank_output.py "$OUT_DIR" --sample "$SAMPLE" --seed "$SEED"

echo ""
echo "Done $(date)."
echo ""
echo "Wrote, under $OUT_DIR:"
echo "  rerank_summary.csv                 one row per shortlist size, reranked vs baseline top-k,"
echo "                                     plus judge_cost_usd -- what OpenRouter actually billed"
echo "  rerank_predictions_llm_k*.csv      per document: predicted author, its rank, the base rank"
echo "  rerank_detail_llm_k*.csv           per candidate: rank, distance rank, and the judge's reason"
echo ""
echo "READ THE SAMPLED DOCUMENTS ABOVE. The accuracies cannot tell you whether the judge reasoned"
echo "about writing style or about subject matter, and a judge that read topic is measuring the"
echo "wrong thing however good its top-1 looks. Three printed rows can tell you; a CSV cannot."
echo ""
if [ "$BATCH" = "0" ]; then
    echo "judge_cost_usd is what OpenRouter reported on the replies themselves, and it is a FLOOR:"
    echo "a reply that arrives without a usage block contributes nothing to it."
    echo ""
fi
echo "Compare against the free control on the same rows:  experiments/run_rerank_jina.sbatch"
