#!/bin/bash
#
# run_afr_wildchat_small.sbatch -- the full AFR arm on wildchat_small, end to end:
# defend (GPU) -> featurize with Gemini -> attack with logistic -> plot.
#
#   sbatch experiments/run_afr_wildchat_small.sbatch
#
# SUBMIT FROM THE REPO ROOT. --output below is a relative path, and SLURM resolves it against the
# directory you submit from. One-time: mkdir -p scripts/logs
#
# THIS IS AFR'S DESIGN TARGET, and the run to do first
# ----------------------------------------------------
# ~1,000 documents / 200 authors, floored at 5 documents each by build_subset's --k-min. Two
# properties make it the corpus AFR was built for, both the opposite of swe_chat's:
#
#   * SHORT, UNIFORM TIMELINES. The cascade's serial dimension is an author's timeline length, so
#     ~5 documents means ~5 lockstep steps -- against swe_chat's mean of 27.6 and a tail reaching
#     into the hundreds, where the last steps run with one or two authors still active and the GPU
#     mostly idle. AFR_CASCADE_DEPTH (32) exists to bound that tail and is inert here.
#   * MORE AUTHORS THAN swe_chat (200 vs 157) IN A QUARTER THE DOCUMENTS. Authors are the unit of
#     parallelism, so batches stay wide for the whole run rather than decaying.
#
# ~1,000 documents x ~11 generations is ~11,000 generations, roughly a quarter of swe_chat's work
# and far better shaped. Hence the shorter wall clock below.
#
# PREREQUISITE: the split must exist. It is not downloaded -- it is cut locally from wildchat:
#
#     python -m prompt_anonymity.data.build_subset --data-dir data/hf --out-dir data/dist
#
# That writes data/dist/wildchat_small.parquet, which is why $DIST_DIR defaults to data/dist here
# (swe_chat, by contrast, is a published split that normally lives in data/hf).
#
# MEASURE FIRST, even though this corpus is small:
#
#     LIMIT=100 sbatch experiments/run_afr_wildchat_small.sbatch
#
# LIMIT is a SMOKE-TEST KNOB ONLY. It truncates the defended parquet to the first N documents, and
# stages 2-4 then join a partial defended split against the full one -- so the attack numbers from a
# LIMIT run are meaningless. Use it to time stage 1 and read the edit log, nothing else. Because the
# split is sorted by (author_id, started_at), the first N documents are whole author timelines,
# which is what the cascade needs to be exercised properly.
#
# COST. Stage 2 is the only stage that spends money: one OpenRouter embedding call per document at
# $0.20/M tokens. At ~1,000 documents this is well under a dollar. It caches by text content, so a
# requeue re-embeds nothing.
#
# Overrides (sbatch exports your submitting environment, so these are read at run time):
#
#   DEFENSE=afr_stage1 sbatch experiments/run_afr_wildchat_small.sbatch  # the ablation -- run it
#   DEFENSE=afr_a50 sbatch experiments/run_afr_wildchat_small.sbatch     # weaker residual target
#   FEATURE=harrier sbatch experiments/run_afr_wildchat_small.sbatch     # free, the overfit check
#   LIMIT=100 sbatch experiments/run_afr_wildchat_small.sbatch           # smoke only, see above
#   PLOT=0 sbatch experiments/run_afr_wildchat_small.sbatch              # skip the figures
#
# Anything after the script name is forwarded to run_experiment.py only, after this script's own
# --attacks, so it wins:
#
#   sbatch experiments/run_afr_wildchat_small.sbatch --attacks logistic nearest_neighbor
#
# THIS IS AN ARRAY JOB, and that is the whole reason it finishes
# ---------------------------------------------------------------
# Measured on the first run that got far enough to produce a number: 458 documents in 7.5 hours,
# i.e. 61 documents/hour, i.e. ~76 hours for all 4,673. One GPU cannot do this inside any
# reasonable wall clock.
#
# So stage 1 runs as an 8-task array, each task defending a disjoint set of WHOLE AUTHORS. Authors
# are independent in AFR -- nothing crosses between them -- so this changes not one output token.
# It is pure parallelism, unlike lowering max_probes, which would trade away the defense itself.
# `apply_defenses` picks the author split automatically because the defense declares
# `shardable_by = "author"`; the shard index and count come from SLURM_ARRAY_TASK_ID/COUNT.
#
# THE REFERENCE POOL IS THE SUBTLE PART. A task holds an eighth of the split, but the pool defines
# r_med and therefore the target every document is optimized to. A per-shard pool would give each
# task a different criterion and the arms would silently stop being comparable. So the pool is
# selected over the WHOLE split before sharding and handed to every task identically -- see
# `AgenticFootprintDefense.reference_pool_ids`. The selftest asserts that the resulting pool_digest
# matches what an unsharded run computes, which is the invariant that makes this safe.
#
# STAGES 2-4 RUN ONCE, in whichever task finishes last: the array auto-merges its shards, and the
# task that writes the merged parquet claims the downstream stages under a lock.
#
# CHANGING THE ARRAY SIZE RESTARTS THE DEFEND CACHE. The shard layout is part of the cache path
# (two shards sharing a table would overwrite each other), so `--array=0-3` after `--array=0-7`
# recomputes everything. Pick a size and keep it.
#
# Sharding does not scale linearly, because the two halves of a round scale differently: prefill is
# compute-bound and falls with the batch, decode is bandwidth-bound and does not (batch 25 reads the
# same 54 GB per token step as batch 64, for fewer tokens). Expect the gain to flatten past ~8 tasks
# -- which is why speculative decoding is on (AFR_SPEC_TOKENS): it attacks the decode term that
# causes the flattening. MEASURE BOTH TOGETHER with LIMIT=100 before committing to the full run.
#
# RESOURCES. An 80 GB card: the bf16 27B agent is ~54 GB of weights and Harrier-0.6B co-resides,
# loading FIRST (it embeds the reference pool before any generation) so it sits OUTSIDE vLLM's
# gpu_memory_utilization budget. bf16 rather than the FP8 sibling because an A100 has no native FP8
# and would dequantize through Marlin on every forward pass. On an H100/H200 prefer FP8:
# AFR_MODEL=<...-FP8> with AFR_GPU_MEM_UTIL=0.75.
#
# AFR_AUTHOR_BATCH stays wide (64) so each task defends its ~25 authors as ONE lockstep chunk, which
# is the widest batch available to it. The cost: `checkpoint_every` is both the flush cadence and the
# batch width, so a preempted task loses its chunk and restarts. Exposure shrinks with more tasks
# (shorter each), not with a narrower batch -- narrowing it slows every task down to protect against
# an event that may not happen. Lower it only if the KV cache cannot hold 25 authors in flight.
#
# Knobs, all environment-only and all OUTSIDE the cache key, so tuning them never invalidates
# completed authors: AFR_GPU_MEM_UTIL, AFR_MAX_MODEL_LEN, AFR_AUTHOR_BATCH, AFR_CASCADE_DEPTH,
# AFR_ENFORCE_EAGER, AFR_SPEC_TOKENS.
#
#SBATCH --job-name=afr-wcs
#SBATCH --output=scripts/logs/%x-%A_%a.out
#SBATCH --open-mode=append
#SBATCH --array=0-7
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

SOURCE="wildchat_small"          # this script is the wildchat_small arm; see run_afr_swe_chat.sbatch
FEATURE="${FEATURE:-gemini_embedding_2}"
DEFENSE="${DEFENSE:-afr}"
DIST_DIR="${DIST_DIR:-$PROJECT/data/dist}"
PLOT="${PLOT:-1}"
LIMIT="${LIMIT:-}"

# Wider batches than the swe_chat arm: 200 authors with ~5 documents each, so there is always
# plenty in flight and each author leaves the pool quickly.
export AFR_AUTHOR_BATCH="${AFR_AUTHOR_BATCH:-64}"

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

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
NSHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"

echo "=================================================================="
echo "AFR on wildchat_small: $DEFENSE"
echo "  source:       $SOURCE"
echo "  feature:      $FEATURE"
echo "  attack:       logistic"
echo "  author batch: $AFR_AUTHOR_BATCH"
echo "  shard:        $SHARD of $NSHARDS  (whole authors; the pool is shared)"
echo "  speculative:  ${AFR_SPEC_TOKENS:-5} draft tokens (0 disables)"
[ -n "$LIMIT" ] && echo "  LIMIT:        $LIMIT  (SMOKE TEST -- attack numbers will be meaningless)"
echo "Job ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}} task ${SHARD} on $(hostname), $(date)"
echo "=================================================================="

# 0a. The split is cut locally, not downloaded, so its absence is the most likely way this job dies
#     -- and it would die AFTER the queue wait. Check it first, and say how to make it.
if [ ! -f "$DIST_DIR/${SOURCE}.parquet" ]; then
    echo "ERROR: no $DIST_DIR/${SOURCE}.parquet"
    echo "wildchat_small is cut from wildchat locally; build it first with:"
    echo "  python -m prompt_anonymity.data.build_subset --data-dir data/hf --out-dir $DIST_DIR"
    exit 1
fi

# 0b. Fail before the GPU stage, not after it. A missing key discovered at stage 2 would mean the
#     hours already spent defending were spent to reach an error. Checking for the file as well as
#     the variable because load_dotenv is what usually supplies it, from a .env at the repo root.
if [ "$FEATURE" = "gemini_embedding_2" ] \
   && [ -z "${OPENROUTER_API_KEY:-}" ] && [ ! -f "$PROJECT/.env" ]; then
    echo "ERROR: $FEATURE needs an OPENROUTER_API_KEY and there is none in the environment,"
    echo "       nor a $PROJECT/.env for load_dotenv to read."
    echo "  cp DS_env/.env .env        (or run with FEATURE=harrier, which is local and free)"
    exit 1
fi

# 0c. Resolve both checkpoints now, so a mis-pointed model is a first-second failure naming the
#     variable to set rather than an OSError an hour into the cascade. This also prints WHERE the
#     weights come from -- if that line mentions downloading, stop the job.
"${RUN[@]}" python -c "
from prompt_anonymity.defenses import get_defense
from prompt_anonymity.features.harrier import HarrierFeaturizer
get_defense('${DEFENSE}').agent_checkpoint()
print('[harrier] embedder checkpoint:', HarrierFeaturizer().checkpoint())"

# 1. Defend THIS SHARD. Authors advance in lockstep, so every model call is batched across the chunk;
#    the log prints the chunk size as it starts each one. The edit log grows per DOCUMENT, so `wc -l`
#    on it is a live throughput reading rather than something that only moves once an author
#    finishes. No --num-shards flag: apply_defenses reads SLURM_ARRAY_TASK_ID/COUNT itself, so this
#    same script is an unsharded run when submitted without --array.
echo ""
echo "------------------------------------------------------------------"
echo "[1/4] defend shard $SHARD/$NSHARDS  ($(date))"
echo "------------------------------------------------------------------"
"${RUN[@]}" python -m prompt_anonymity.data.apply_defenses \
    --source "$SOURCE" --defense "$DEFENSE" --dist-dir "$DIST_DIR" "${LIMIT_ARGS[@]}"

# 1b. Stages 2-4 must run EXACTLY ONCE, over the merged split, and only after every shard exists.
#     The last task to finish is the one whose merge succeeds and writes the merged parquet, so
#     "does it exist" is the signal. `mkdir` is the claim because it is atomic on POSIX: if two
#     tasks somehow race here, exactly one creates the directory. The lock is keyed to this array's
#     job id, so a fresh submission is never blocked by a previous one's lock, and re-entrant for
#     its owner, so a requeued winner resumes the downstream stages instead of skipping them.
MERGED="$DIST_DIR/${SOURCE}_${DEFENSE}.parquet"
LOCK="$DIST_DIR/.afr_downstream_${SOURCE}_${DEFENSE}_${SLURM_ARRAY_JOB_ID:-solo}"
CLAIMED=0
if [ -f "$MERGED" ]; then
    if mkdir "$LOCK" 2>/dev/null; then
        echo "$SHARD" > "$LOCK/owner"
        CLAIMED=1
    elif [ "$(cat "$LOCK/owner" 2>/dev/null)" = "$SHARD" ]; then
        CLAIMED=1                      # this task owns the downstream stages and was requeued
    fi
fi

if [ "$CLAIMED" != "1" ]; then
    echo ""
    echo "Shard $SHARD defended. Another task runs featurize/attack/plot once every shard is in."
    echo "  shards so far: $(ls "$DIST_DIR/shards/${SOURCE}_${DEFENSE}."*.parquet 2>/dev/null | wc -l) of $NSHARDS"
    echo "Done $(date)."
    exit 0
fi

echo ""
echo "This task wrote the merged split, so it runs the downstream stages."

# From here on the SLURM array variables must be OUT of the environment. compute_features resolves
# its own sharding from exactly those variables, so leaving them set would make stage 2 featurize
# only an eighth of the documents and write a shard file nobody merges.
RUN=(env -u SLURM_ARRAY_TASK_ID -u SLURM_ARRAY_TASK_COUNT -u SLURM_ARRAY_JOB_ID
     bash scripts/slurm_job.sh)

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
#    because its own default reads the node's affinity mask.
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
echo "On THIS corpus expect many cold-start documents: build_subset floors authors at 5 documents,"
echo "so one in five is an author's first and takes the genericness path rather than the linkage"
echo "one. A large 'no_priors' count is correct here, not a bug."
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
