#!/bin/bash
#
# run_rerank_smoke.sbatch -- prove the install, the data, the checkpoint and the API request shape
# before ~$40 of Sonnet calls are queued. CPU only, about ten minutes, about five cents.
#
#     sbatch experiments/run_rerank_smoke.sbatch
#     LLM=0 sbatch experiments/run_rerank_smoke.sbatch      # skip the one paid stage
#
# SUBMIT FROM THE REPO ROOT (--output is relative and SLURM resolves it against the submit dir).
# One-time: mkdir -p scripts/logs
#
# WHAT THIS IS FOR. experiments/submit_rerank_swe_chat.sh queues a GPU job and hours of paid API
# calls. Every way that can fail in the first thirty seconds is checked here first, on a node that
# schedules immediately:
#
#   1. Is the [rerank] extra even installed? install.sh SKIPS every GPU extra on a non-CUDA
#      machine, so an environment built there has no transformers and the jina arm dies on import
#      after waiting in the GPU queue.
#   2. Is the reranker checkpoint already in the HF cache, or will the GPU job spend its first
#      minutes downloading 0.6B of weights?
#   3. Do the swe_chat parquets exist, under the names the jobs will look for?
#   4. Does the UNBATCHED Sonnet request shape actually work? This is the expensive one to get
#      wrong: the synchronous client is a different code path from the batch client, with a
#      different key, and a reasoning model rejects the sampling controls a chat client would
#      normally send. Two documents' worth of calls answer it for about a nickel.
#
# NO GPU ON PURPOSE. The jina reranker is 0.6B and runs on a CPU perfectly well for twelve
# documents; asking for a GPU here would put the smoke test behind the queue it exists to
# de-risk. Slow on CPU is the point -- it is still faster than finding out on an A100.
#
# WRITES: a scratch results directory (experiments/results/rerank/_smoke_<source>), which no real
# run reads. It SHARES the real verdict cache, which is intentional and free: verdicts are keyed
# by prompt text, so these rows are reused by the full run rather than paid for twice.
#
# Overrides:
#   SOURCE=wildchat_tiny sbatch ...      # default swe_chat
#   FEATURE=luar sbatch ...              # default gemini_embedding_2
#   N=24 sbatch ...                      # documents through the free jina stage (default 12)
#   LLM=0 sbatch ...                     # skip the paid stage entirely
#   LLM_N=5 sbatch ...                   # documents through the paid stage (default 3). Raise it
#                                        #   if a stage reports 0 attributable documents: the
#                                        #   sample is seeded, so a few unlucky draws whose authors
#                                        #   are all absent from the known side stay unlucky.
#   FETCH=1 sbatch ...                   # download the reranker checkpoint here, not in the GPU job
#
#SBATCH --job-name=rrk-smoke
#SBATCH --output=scripts/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --partition=cpu-preempt
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=0-00:30:00
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
    echo "       Run this from the repo root, or set PROMPT_ANONYMITY_PROJECT=/path/to/repo." >&2
    exit 1
fi
cd "$PROJECT"

SOURCE="${SOURCE:-swe_chat}"
FEATURE="${FEATURE:-gemini_embedding_2}"
DEFENSE="${DEFENSE:-none}"
WINDOW="${WINDOW:-0075}"
SEED="${SEED:-47}"
N="${N:-12}"
LLM="${LLM:-1}"
LLM_N="${LLM_N:-3}"
FETCH="${FETCH:-0}"
DIST_DIR="${DIST_DIR:-$PROJECT/data/dist}"
CACHE_DIR="${CACHE_DIR:-$PROJECT/data/.cache}"
OUT_DIR="${OUT_DIR:-$PROJECT/experiments/results/rerank/_smoke_${SOURCE}}"

stem_of() { if [ "$1" = "none" ]; then echo "$SOURCE"; else echo "${SOURCE}_$1"; fi; }
STEM="$(stem_of "$DEFENSE")"

# No CHECK_ONLY switch here: the whole script is the check, and it has to run in the same
# environment the real jobs will, which is what scripts/slurm_job.sh sets up.
RUN=(bash scripts/slurm_job.sh)

echo "=================================================================="
echo "rerank smoke test -- $SOURCE / $FEATURE / known$WINDOW"
echo "  free stage:  $N documents through jina, on the CPU"
if [ "$LLM" = "1" ]; then
    echo "  paid stage:  $LLM_N documents through Sonnet 5, unbatched (~\$0.05)"
else
    echo "  paid stage:  SKIPPED (LLM=0)"
fi
echo "  scratch:     $OUT_DIR"
echo "Job ${SLURM_JOB_ID:-local} on $(hostname), $(date)"
echo "=================================================================="

# --- 1. the environment ----------------------------------------------------
# The interpreter is printed before anything is asked of it, because "module not found" on a
# cluster is nearly always the wrong environment rather than a missing package.
echo ""
echo "[1/6] environment ($(date))"
"${RUN[@]}" python -c '
import importlib.util
import sys

print("[env] interpreter:", sys.executable)

required = ("numpy", "pandas", "pyarrow", "requests", "dotenv")
extra = ("torch", "transformers")

missing = [m for m in required if importlib.util.find_spec(m) is None]
if missing:
    print("[env] MISSING (base install):", ", ".join(missing))
    print("[env] install the package into the interpreter printed above:")
    print("[env]   python -m pip install -e .")
    print("[env] if that is not the env you meant, export PROMPT_ANONYMITY_CONDA_ENV=<your env>")
    print("[env] and resubmit.")
    sys.exit(1)

missing = [m for m in extra if importlib.util.find_spec(m) is None]
if missing:
    print("[env] MISSING (the [rerank] extra):", ", ".join(missing))
    print("[env] install.sh SKIPS every GPU extra on a machine without CUDA, so an environment")
    print("[env] built on a login node without one has no transformers and the jina arm cannot")
    print("[env] run. Install it into the interpreter printed above:")
    print("[env]   python -m pip install -e \".[rerank]\"")
    sys.exit(1)

import torch
print("[env] torch", torch.__version__, "| cuda visible:", torch.cuda.is_available(),
      "(not needed here -- this is the CPU smoke test)")
import transformers
print("[env] transformers", transformers.__version__)

# The whole attacks package, not just the top-level one. attacks/__init__.py builds its registries
# by importing every attack family, so ANY of their dependencies missing breaks ANY entry point --
# a reranker included, and none of them touch xgboost or the trees. Importing it here is what
# makes that a named failure in a 30-second CPU job instead of a traceback four hours into a paid
# one. (The imports inside that chain should be lazy for anything optional; this is the canary for
# when one is not.)
import prompt_anonymity
print("[env] prompt_anonymity imports from", prompt_anonymity.__file__)
try:
    import prompt_anonymity.attacks
    import prompt_anonymity.attacks.llm
except ModuleNotFoundError as err:
    print("[env] the attacks package does not import:", err)
    print("[env] this breaks EVERY entry point, the rerankers included, even when the missing")
    print("[env] package is for an attack they never run. Sync this interpreter with pyproject:")
    print("[env]   python -m pip install -e \".[rerank]\"")
    print("[env] and if the import is for something optional, it belongs inside the function")
    print("[env] that uses it -- see multiclass/boosted_trees.py.")
    sys.exit(1)
print("[env] the attacks registries import clean")
'

# --- 2. offline invariants -------------------------------------------------
# Seconds, no key, no network. A build that cannot rank a synthetic shortlist, or that would send
# a payload OpenRouter rejects, should fail here rather than after the first paid call.
echo ""
echo "[2/6] self-tests ($(date))"
"${RUN[@]}" python -m prompt_anonymity.attacks.llm.listwise --selftest
"${RUN[@]}" python -m prompt_anonymity.attacks.llm._openrouter_batch --selftest

# --- 3. the reranker checkpoint --------------------------------------------
# Resolved the same way the jina job resolves it, then probed for locally. Unlike the defenses'
# embedder this checkpoint is ALLOWED to download, so a cold cache is not an error -- it is just
# the first few minutes of a GPU allocation, which is worth knowing in advance.
echo ""
echo "[3/6] reranker checkpoint ($(date))"
FETCH="$FETCH" "${RUN[@]}" python -c '
import os
import sys

from prompt_anonymity.attacks.llm.listwise_jina_rerank import (
    DEFAULT_MODEL_ID, MODEL_CONFIG_SECTION, MODEL_ENV_VAR)
from prompt_anonymity.evaluation.utility._local import resolve_checkpoint

checkpoint = resolve_checkpoint(MODEL_CONFIG_SECTION, MODEL_ENV_VAR, DEFAULT_MODEL_ID)
print("[ckpt] resolved to:", checkpoint)

if os.path.isdir(checkpoint):
    print("[ckpt] a local directory -- nothing to download")
    sys.exit(0)

from huggingface_hub import snapshot_download

if os.environ.get("FETCH") == "1":
    print("[ckpt] FETCH=1: downloading now, so the GPU job does not spend its allocation on it")
    print("[ckpt] cached at:", snapshot_download(checkpoint))
    sys.exit(0)

try:
    print("[ckpt] already in the HF cache:", snapshot_download(checkpoint, local_files_only=True))
except Exception:
    print("[ckpt] NOT in the HF cache. The jina job will download ~0.6B of weights on start,")
    print("[ckpt] which is fine but eats the first minutes of a GPU allocation. To fetch it on a")
    print("[ckpt] CPU node instead:  FETCH=1 sbatch experiments/run_rerank_smoke.sbatch")
    print("[ckpt] (or point $JINA_RERANKER_MODEL at a local copy in .env)")
'

# --- 4. the inputs ---------------------------------------------------------
echo ""
echo "[4/6] inputs ($(date))"
if [ ! -f "$DIST_DIR/${STEM}.parquet" ]; then
    echo "ERROR: no $DIST_DIR/${STEM}.parquet" >&2
    echo "  Build the split first:" >&2
    echo "    python -m prompt_anonymity.data.build_dataset --sources $SOURCE --out-dir $DIST_DIR" >&2
    exit 1
fi
if [ ! -f "$DIST_DIR/${STEM}_${FEATURE}.parquet" ]; then
    echo "ERROR: no $DIST_DIR/${STEM}_${FEATURE}.parquet" >&2
    echo "  Compute the vectors first:" >&2
    echo "    python -m prompt_anonymity.data.compute_features --source $SOURCE \\" >&2
    echo "        --feature $FEATURE --dist-dir $DIST_DIR" >&2
    exit 1
fi
echo "  split:    $DIST_DIR/${STEM}.parquet"
echo "  features: $DIST_DIR/${STEM}_${FEATURE}.parquet"

# The key is checked HERE rather than beside the stage that needs it, so a missing one costs
# seconds instead of the free stage's few minutes on a CPU.
if [ "$LLM" = "1" ]; then
    if [ -z "${SONNET_API_KEY:-}" ] && [ ! -f "$PROJECT/.env" ]; then
        echo "ERROR: SONNET_API_KEY is not set and there is no $PROJECT/.env to read it from," >&2
        echo "  so the paid stage would fail. Add it at the REPO ROOT (load_dotenv walks UP from" >&2
        echo "  the working directory, so a .env in a subdirectory -- DS_env/.env -- is never" >&2
        echo "  found), or pass LLM=0 to skip that stage:" >&2
        echo "    echo 'SONNET_API_KEY=sk-or-...' >> $PROJECT/.env" >&2
        exit 1
    fi
    if [ -n "${SONNET_API_KEY:-}" ]; then
        echo "  key:      SONNET_API_KEY set in the environment"
    else
        echo "  key:      SONNET_API_KEY expected in $PROJECT/.env (not read here, so it stays"
        echo "            out of this log)"
    fi
fi

# --- 5. free end to end ----------------------------------------------------
# A real run of the real entry point, just tiny: the split arithmetic, the shortlist, the text
# read, the fold-back, the output tables and the validator, all of it exercised for nothing.
COMMON=(--source "$SOURCE" --feature "$FEATURE" --defense "$DEFENSE"
        --data-dir "$DIST_DIR" --known-window "$WINDOW" --seed "$SEED"
        --cache-dir "$CACHE_DIR" --output-dir "$OUT_DIR")

echo ""
echo "[5/6] jina, $N documents, on the CPU ($(date))"
"${RUN[@]}" python experiments/run_rerank.py "${COMMON[@]}" --variant jina --top-k 5 --limit "$N"

# --- 6. paid end to end ----------------------------------------------------
# The ONLY stage that proves the unbatched Sonnet path, which is a different client, a different
# key and a different payload from the batched one: that SONNET_API_KEY authenticates, that
# omitting temperature/top_p clears the 400 a reasoning model answers them with, and that
# reasoning.effort is accepted. Two documents. If this fails with an HTTP 400, do not submit the
# full run -- it will fail the same way, hours in.
if [ "$LLM" = "1" ]; then
    echo ""
    echo "[6/6] sonnet, $LLM_N documents, unbatched ($(date))"
    "${RUN[@]}" python experiments/run_rerank.py "${COMMON[@]}" \
        --variant llm --top-k 5 --limit "$LLM_N" --no-batch
else
    echo ""
    echo "[6/6] sonnet -- SKIPPED (LLM=0) ($(date))"
    echo "  Nothing has checked the unbatched request shape or the key. The full run is where you"
    echo "  will find out instead."
fi

# --- the validator, over whatever ran --------------------------------------
echo ""
echo "validating the scratch output ($(date))"
"${RUN[@]}" python experiments/check_rerank_output.py "$OUT_DIR" --sample 2 --seed "$SEED"

echo ""
echo "Done $(date)."
echo ""
echo "WHAT TO CHECK BEFORE SUBMITTING THE REAL RUN:"
echo "  - the validator above reported no failures"
if [ "$LLM" = "1" ]; then
    echo "  - the llm stage parsed $LLM_N/$LLM_N rows, with a readable reason per position"
    echo "  - judge_cost_usd in $OUT_DIR/rerank_summary.csv is non-empty on the llm row. Empty"
    echo "    means OpenRouter returned no usage block, so the real run's spend will be"
    echo "    under-reported -- worth knowing, not worth stopping for."
fi
echo "  - 'top pick by distance rank' in the jina output has mass on 0-1. A flat spread means the"
echo "    reranker found nothing the embedding had not, which is a result, not a bug."
echo ""
echo "Then, both arms in parallel on the full unknown side:"
echo "    bash experiments/submit_rerank_swe_chat.sh"
echo ""
echo "This scratch directory is disposable and no real run reads it:"
echo "    rm -rf $OUT_DIR"
