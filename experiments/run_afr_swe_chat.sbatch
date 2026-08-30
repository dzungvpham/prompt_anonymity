#!/bin/bash
#
# run_afr_swe_chat.sbatch -- the full AFR arm on SWE-chat, end to end:
# defend (GPU) -> featurize with Gemini -> attack with logistic -> plot.
#
#   sbatch experiments/run_afr_swe_chat.sbatch
#
# SUBMIT FROM THE REPO ROOT. --output below is a relative path, and SLURM resolves it against the
# directory you submit from. One-time: mkdir -p scripts/logs
#
# READ THIS BEFORE SUBMITTING: HOW LONG STAGE 1 IS
# ------------------------------------------------
# SWE-chat is 4,334 documents / 157 authors and AFR spends roughly 11 generations on each. That is
# ~48,000 generations -- but they are BATCHED: authors are stepped through their timelines in
# lockstep, so each stage is one vLLM call across every author in the current chunk
# ($AFR_AUTHOR_BATCH, default 32) rather than one call per document. An earlier version issued
# batch-1 calls and could not finish; see the module docstring in defenses/afr.py.
#
# It still CANNOT be sharded across nodes -- the cascade needs an author's timeline intact and
# `apply_defenses --num-shards` splits by document -- so all the parallelism is inside one GPU.
#
# The wall clock below may still be one preemptible slice of several. --requeue plus the cache's
# per-chunk checkpointing is what makes that safe. Unlike before, the edit log now flushes per
# DOCUMENT, so progress is visible within minutes instead of after a whole author.
#
# MEASURE FIRST. Before committing to the full run, do a timed subset and extrapolate:
#
#     LIMIT=20 sbatch experiments/run_afr_swe_chat.sbatch
#
# LIMIT is a SMOKE-TEST KNOB ONLY. It truncates the defended parquet to the first N documents, and
# stages 2-4 then join a partial defended split against the full one -- so the attack numbers from a
# LIMIT run are meaningless. Use it to time stage 1 and to check the edit log, nothing else.
#
# WHY ONE JOB RATHER THAN TWO
# ---------------------------
# Stages 2-4 want a CPU node (a Gemini embedding is a network call, logistic is scikit-learn,
# plotting is matplotlib) and holding a GPU through them wastes it. They are also minutes against
# stage 1's many hours on this corpus, so splitting them into a second submission would cost more in
# queue time and operator attention than the idle GPU is worth. If you ever re-featurize this same
# defended parquet on another channel, do NOT re-run this script -- copy the pattern in
# run_dpmlm_var_a10_gemini.sbatch, which starts from the parquet and never defends.
#
# COST. Stage 2 is the only stage that spends money: one OpenRouter embedding call per document at
# $0.20/M tokens. SWE-chat is cents. It caches by text content, so a requeue re-embeds nothing.
#
# Overrides (sbatch exports your submitting environment, so these are read at run time):
#
#   DEFENSE=afr_stage1 sbatch experiments/run_afr_swe_chat.sbatch   # the loop ablation -- run it
#   DEFENSE=afr_a50 sbatch experiments/run_afr_swe_chat.sbatch      # a weaker residual target
#   FEATURE=harrier sbatch experiments/run_afr_swe_chat.sbatch      # free, and the overfit check
#   LIMIT=20 sbatch experiments/run_afr_swe_chat.sbatch             # smoke only, see above
#   PLOT=0 sbatch experiments/run_afr_swe_chat.sbatch               # skip the figures
#
# Anything after the script name is forwarded to run_experiment.py only, after this script's own
# --attacks, so it wins:
#
#   sbatch experiments/run_afr_swe_chat.sbatch --attacks logistic nearest_neighbor
#
# RESOURCES. An 80 GB card: the bf16 27B agent is ~54 GB of weights and Harrier-0.6B co-resides,
# loading FIRST (it embeds the reference pool before any generation) so it sits OUTSIDE vLLM's
# gpu_memory_utilization budget. bf16 rather than the FP8 sibling because an A100 has no native FP8
# and would dequantize through Marlin on every forward pass -- spending compute to save memory on
# the one job whose bottleneck is compute. On an H100/H200, FP8 is the better choice: set
# AFR_MODEL=<...-FP8> and drop AFR_GPU_MEM_UTIL back to 0.75.
#
# Knobs, all environment-only and all OUTSIDE the cache key, so tuning them does not invalidate
# completed authors: AFR_GPU_MEM_UTIL (0.90), AFR_MAX_MODEL_LEN (16384), AFR_AUTHOR_BATCH (32,
# the batch width), AFR_CASCADE_DEPTH (32), AFR_ENFORCE_EAGER (0).
#
#SBATCH --job-name=afr-swe
#SBATCH --output=scripts/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --partition=gpu-preempt
#SBATCH --gpus=1
#SBATCH --constraint=a100-80g|h100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=0-16:00:00
#SBATCH --requeue

set -euo pipefail

PROJECT="${PROMPT_ANONYMITY_PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT"

# Must be exported BEFORE anything initializes CUDA. The embedder and vLLM share one card and vLLM's
# reservation is permanent, so the embedder works in a shrinking remainder; expandable segments let
# the allocator grow a block instead of failing on a fragmented heap. vLLM's own OOM message
# recommends exactly this.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SOURCE="swe_chat"                       # this script is the SWE-chat arm; see run_afr.sbatch for others

# This script is SINGLE-TASK. Submitted with --array it would do the wrong thing quietly: stage 1
# would correctly defend one author shard, but stages 2-4 resolve their own sharding from the same
# SLURM_ARRAY_* variables, so every task would featurize and attack only its own eighth and write
# partial results over each other. run_afr_wildchat_small.sbatch is the array-shaped one -- it
# merges the shards and elects a single task for the downstream stages. Copy that pattern here if
# this arm needs it too.
if [ "${SLURM_ARRAY_TASK_COUNT:-1}" -gt 1 ]; then
    echo "ERROR: this script is not array-safe (stages 2-4 would shard themselves)." >&2
    echo "       Use experiments/run_afr_wildchat_small.sbatch as the template for an array run." >&2
    exit 1
fi
FEATURE="${FEATURE:-gemini_embedding_2}"
DEFENSE="${DEFENSE:-afr}"
DIST_DIR="${DIST_DIR:-$PROJECT/data/dist}"
PLOT="${PLOT:-1}"
LIMIT="${LIMIT:-}"

LIMIT_ARGS=()
if [ -n "$LIMIT" ]; then
    LIMIT_ARGS=(--limit "$LIMIT")
fi

# Same env bootstrap the experiment launcher uses, so how a batch shell finds conda stays defined in
# exactly one file. It ends in `exec "$@"`, so calling it as a command runs the stage.
#
# Deliberately NOT `srun`: since SLURM 22.05 a step does not inherit --cpus-per-task from the batch
# allocation, so `srun <stage>` would hand these stages a single CPU -- throttling both the embedding
# client's concurrency and plot_results' worker pool.
RUN=(bash scripts/slurm_job.sh)

echo "=================================================================="
echo "AFR on SWE-chat: $DEFENSE"
echo "  source:  $SOURCE"
echo "  feature: $FEATURE"
echo "  attack:  logistic"
[ -n "$LIMIT" ] && echo "  LIMIT:   $LIMIT  (SMOKE TEST -- attack numbers will be meaningless)"
echo "Job ${SLURM_JOB_ID:-local} on $(hostname), started $(date)"
echo "=================================================================="

# 0. Fail before the GPU stage, not after it. A missing key discovered at stage 2 would mean the
#    hours already spent defending were spent to reach an error. Checking for the file as well as
#    the variable because load_dotenv is what usually supplies it, from a .env at the repo root.
if [ "$FEATURE" = "gemini_embedding_2" ] \
   && [ -z "${OPENROUTER_API_KEY:-}" ] && [ ! -f "$PROJECT/.env" ]; then
    echo "ERROR: $FEATURE needs an OPENROUTER_API_KEY and there is none in the environment,"
    echo "       nor a $PROJECT/.env for load_dotenv to read."
    echo "  cp DS_env/.env .env        (or run with FEATURE=harrier, which is local and free)"
    exit 1
fi

# 0b. Resolve both checkpoints now, so a mis-pointed model is a first-second failure naming the
#     variable to set rather than an OSError an hour into the cascade. This also prints WHERE the
#     weights come from -- if that line says anything about downloading, stop the job.
"${RUN[@]}" python -c "
from prompt_anonymity.defenses import get_defense
from prompt_anonymity.features.harrier import HarrierFeaturizer
get_defense('${DEFENSE}').agent_checkpoint()
print('[harrier] embedder checkpoint:', HarrierFeaturizer().checkpoint())"

# 1. Defend. The long stage. Authors advance in lockstep so every model call is batched across the
#    chunk; the log prints the chunk size as it starts each one. The edit log grows per DOCUMENT,
#    so `wc -l` on it is a live throughput reading.
echo ""
echo "------------------------------------------------------------------"
echo "[1/4] defend  ($(date))"
echo "------------------------------------------------------------------"
"${RUN[@]}" python -m prompt_anonymity.data.apply_defenses \
    --source "$SOURCE" --defense "$DEFENSE" --dist-dir "$DIST_DIR" "${LIMIT_ARGS[@]}"

# 2. Featurize the defended text. The ATTACKER's encoder, and deliberately not the one AFR optimized
#    against -- that mismatch is what makes this a transfer result rather than self-grading. This is
#    the stage that costs money; it caches by text content, so a requeue re-embeds nothing.
echo ""
echo "------------------------------------------------------------------"
echo "[2/4] featurize  ($(date))"
echo "------------------------------------------------------------------"
"${RUN[@]}" python -m prompt_anonymity.data.compute_features \
    --source "$SOURCE" --defense "$DEFENSE" --feature "$FEATURE" --dist-dir "$DIST_DIR"

# 3. Attack. --data-dir is data/dist because that is where stages 1-2 write; the runner's own
#    default is data/hf, the published mirror, which lacks these files. "$@" comes last so a
#    forwarded --attacks overrides the one here.
echo ""
echo "------------------------------------------------------------------"
echo "[3/4] attack  ($(date))"
echo "------------------------------------------------------------------"
"${RUN[@]}" python experiments/run_experiment.py \
    --source "$SOURCE" --defense "$DEFENSE" --feature "$FEATURE" \
    --data-dir "$DIST_DIR" --attacks logistic "$@"

# 4. Draw the figures. Cheap and idempotent: recomputes no attack, reads the result CSVs on disk,
#    and its cache skips every figure nothing has moved under. --jobs is pinned to the allocation
#    because its own default reads the node's affinity mask, and each worker holds a replicate
#    weight matrix.
if [ "$PLOT" != "0" ]; then
    echo ""
    echo "------------------------------------------------------------------"
    echo "[4/4] plot  ($(date))"
    echo "------------------------------------------------------------------"
    "${RUN[@]}" python experiments/plot_results.py --jobs "${SLURM_CPUS_PER_TASK:-8}"
fi

TAG="${SOURCE}_${DEFENSE}_${FEATURE}"
echo ""
echo "Done $(date)."
echo "  defended: $DIST_DIR/${SOURCE}_${DEFENSE}.parquet"
echo "  results:  experiments/results/${TAG}_logistic/"
echo "  figures:  experiments/plots/${SOURCE}/"
echo ""
echo "Read the edit log BEFORE the results -- it says whether the defense did what it claims."
echo "The defend stage printed its exact path ('[afr] edit log -> ...'); the two hash levels are"
echo "the defense's logic and params fingerprints, so glob past them:"
echo "  jq -s 'group_by(.stop_reason)|map({(.[0].stop_reason):length})|add' \\"
echo "     data/.cache/defended/${SOURCE}/defenses/afr/*/*/afr_logs/edits_a*.jsonl"
echo "  target_met dominating -> the loop is working"
echo "  ladder_exhausted      -> the agent stalls; the propose prompt needs work"
echo "  too_long              -> emitted UNDEFENDED; raise AFR_MAX_MODEL_LEN (see prompt_tokens)"
echo "  max_level always 0    -> escalation never fires; the trigger is mistuned"
echo ""
echo "What to compare, and what each comparison answers:"
echo "  vs ${SOURCE}_base_${FEATURE}_logistic        -- how much attribution the defense removed"
echo "  vs ${SOURCE}_afr_stage1_${FEATURE}_logistic  -- THE control. Same model, same abstraction"
echo "                                                  pass, zero probes. This is what isolates the"
echo "                                                  measurement loop; vs base only shows that a"
echo "                                                  27B model rewrote the text."
echo "  vs ${SOURCE}_loo_unlink_b30_${FEATURE}_logistic -- the greedy-rule alternative at a similar"
echo "                                                  target: does an agent beat a fixed rule?"
echo "  vs ${SOURCE}_${DEFENSE}_harrier_logistic     -- the surrogate AFR optimized against. A big"
echo "                                                  drop there with none on Gemini is transfer"
echo "                                                  failure, and must be reported as such."
echo ""
echo "Utility (free metrics first; the paid judge only once those look sane):"
echo "  python experiments/eval_utility.py --source $SOURCE --defense $DEFENSE"
