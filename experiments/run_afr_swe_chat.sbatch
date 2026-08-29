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
# SWE-chat is 4,334 documents and AFR spends roughly 11 generations on each -- the abstraction pass,
# a rolling author signature, ~4 proposal rounds, and the answer/judge pair -- against a 27B model.
# That is ~48,000 generations, and it CANNOT be sharded across nodes: the defense cascades each
# author's timeline in order, so `apply_defenses --num-shards` is refused outright.
#
# So the wall clock below is not "the job will finish in 16 hours". It is one preemptible slice of a
# job that will very likely need several. --requeue plus the defense's own checkpointing (every 5
# AUTHORS -- the atomic unit of the cascade, so a resumed job never restarts an author mid-timeline)
# is what makes that safe: each requeue picks up where the last one stopped and recomputes nothing.
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
# RESOURCES. An 80 GB card, because the FP8 27B agent (~27 GB) and Harrier-0.6B co-reside and
# Harrier loads FIRST -- it embeds the reference pool before the first generation -- so it sits
# outside vLLM's gpu_memory_utilization budget. If your allocation lands somewhere smaller, the
# knobs are AFR_GPU_MEM_UTIL and AFR_MAX_MODEL_LEN (environment only; neither is in the cache key,
# so tuning them does not invalidate completed authors).
#
#SBATCH --job-name=afr-swe
#SBATCH --output=scripts/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --partition=gpu-preempt
#SBATCH --gpus=1
#SBATCH --constraint=sm_90|a100-80g
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=0-16:00:00
#SBATCH --requeue

set -euo pipefail

PROJECT="${PROMPT_ANONYMITY_PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT"

SOURCE="swe_chat"                       # this script is the SWE-chat arm; see run_afr.sbatch for others
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

# 1. Defend. The long stage: ~11 generations per document, cascaded per author, no sharding.
#    Checkpoints every 5 authors, so a preemption costs at most that much work.
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
