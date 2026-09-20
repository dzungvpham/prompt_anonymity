#!/bin/bash
#
# run_dpmlm_wildchat.sbatch -- DP-MLM over the whole of WildChat, then the usual pipeline on top:
# defend -> featurize -> attack. A CSV of every defended text drops out on the side, between the
# merge and the featurize stage.
#
#   sbatch experiments/run_dpmlm_wildchat.sbatch
#
# Produces, with the defaults below:
#
#   data/dist/shards/wildchat_dp_mlm_eps100.NNNN-of-0048.parquet   one per worker
#   data/dist/wildchat_dp_mlm_eps100.parquet                       merged defended split
#   data/dist/wildchat_dp_mlm_eps100_stylometrix.parquet           features
#   experiments/results/wildchat_dp_mlm_eps100_stylometrix_*/      the attack
#   data/dist/csv/wildchat_dp_mlm_eps100.csv                       SIDE OUTPUT: every turn,
#                                                                  original beside defended
#
# The CSV is for reading the text, nothing else -- no stage consumes it, and CSV=0 skips it without
# changing a single number downstream. It exists because DP-MLM's output quality *is* the question
# at a given epsilon, and a parquet of list-of-strings is not something you can eyeball.
#
# SUBMIT FROM THE REPO ROOT. --output below is a relative path and SLURM resolves it against the
# directory you submit from.
#
# ONE-TIME SETUP
# --------------
#   mkdir -p scripts/logs          # SLURM fails a job whose --output directory does not exist
#
#   # Warm the caches on the login node. 48 worker processes starting at once would otherwise all
#   # reach for roberta-base and the NLTK data (punkt/punkt_tab/stopwords/wordnet) simultaneously --
#   # a download storm on nodes that may have no network at all. Under a minute:
#   python -c "from prompt_anonymity.defenses import get_defense; get_defense('dp_mlm_eps100')._get_backend()"
#
#   # And check the split is there (it is READ from data/hf, the published mirror):
#   ls -l data/hf/wildchat.parquet   # else: python -m prompt_anonymity.data.download
#
# THE SHAPE OF THIS JOB: MANY WORKERS PER CARD, NOT MANY CARDS
# ------------------------------------------------------------
# DP-MLM is the lightest model in the registry -- roberta-base, 125M parameters, one masked forward
# pass per content word, no generation. The whole corpus is roughly:
#
#   172,509 documents x ~1,180 words of user text (mean; the median is ~90, the tail is what
#   costs) x ~45% non-stopword, non-punctuation  ~=  90M masked positions
#
# At a measured-order 1-4k positions/s per card that is 6-25 GPU-HOURS for all of WildChat. Handing
# that to 64 whole GPUs would mean ~15 minutes of compute per card behind a queue wait of hours,
# and a defend cache layout too wide to ever re-use. So this job goes the other way:
#
#   * 8 array tasks, one GPU each -- few enough to schedule together, even on gpu-preempt.
#   * 6 WORKER PROCESSES per GPU, each defending its own shard, sharing the card.
#   * 48 shards in total (TASKS x WORKERS), which is the number that defines the cache layout.
#
# WHY WORKERS PER CARD, AND NOT JUST A BIGGER BATCH. One DP-MLM process alternates between a
# single-threaded CPU phase and a GPU phase, and cannot overlap them: `_process_chunk` builds every
# masked input in Python first (NLTK detokenize + a tokenizer pair-encode PER WORD -- for a turn
# longer than the 224-token context window, that work is redone for every word in it), then runs the
# MLM. So a lone process leaves the card idle for a large fraction of its wall clock, and raising
# DPMLM_BATCH_SIZE does nothing about it -- it makes the GPU phase shorter, not the CPU phase.
# Concurrent processes fix exactly that: one worker's CPU phase overlaps another's GPU phase.
#
# HOW MANY FIT. Per worker, on the card: ~0.25 GB of bf16 weights, ~0.8 GB for the chunk logits
# buffer (DPMLM_FLUSH_POSITIONS=8192 x 50,265 vocab x fp16), ~1-1.5 GB of activations at
# DPMLM_BATCH_SIZE=128, plus ~0.5 GB of CUDA context -- about 3 GB, so an 80 GB A100 has room for
# 20+ and even a 40 GB card for a dozen. MEMORY IS NOT THE LIMIT. What binds first is CPU cores
# (~2-3 per worker for that Python phase) and then the card's own throughput, which is why the
# default is 6 workers on 16 CPUs rather than the 20 that would fit. Raise WORKERS if the smoke
# test below shows the GPU still idling; the failure mode is mild (time-slicing, not OOM).
#
# No CUDA MPS. Processes time-slicing one device is fine here because each worker submits large
# kernels (a 128 x ~450-token batch), so the scheduler has plenty to interleave.
#
# MEASURE BEFORE COMMITTING TO A LAYOUT:
#
#   LIMIT=2000 WORKERS=6 sbatch --array=0 experiments/run_dpmlm_wildchat.sbatch
#   # then, per worker log, words/s from the tqdm line, and:
#   nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 10
#
# Utilization well under ~90% means raise WORKERS (or CPUS); ~100% with the per-worker rate falling
# proportionally means the card is full and more workers buy nothing. A LIMIT run is a smoke test
# only: it defends the first N documents, so its attack numbers would be meaningless and ATTACK is
# forced off.
#
# CHANGING THE SHARD LAYOUT THROWS AWAY THE DEFEND CACHE. The layout is TASKS x WORKERS and it is
# part of the cache path (`defense_cache_dir`: two shards sharing one table would each drop the
# other's rows), so changing EITHER number recomputes everything. Throttling the array with
# `sbatch --array=0-7%2` does NOT change the layout and is therefore free. Pick a pair and keep it.
#
# WHAT A PREEMPTION COSTS
# -----------------------
# gpu-preempt jobs get killed; --requeue puts the task back in the queue, it re-runs this script
# from the top, and each of its workers picks up where it stopped. Concretely, per stage:
#
#   defend     RESUMES, losing at most DPMLM_CHECKPOINT_EVERY distinct turns (1,000 here, a few
#              minutes). The defense flushes its cache table every that many turns and omits the
#              unfinished ones, so on resume a finished turn is a hit and an unfinished one simply
#              recomputes. Writes are temp-file-plus-rename with an fsync, so a kill DURING a flush
#              leaves the previous table intact rather than a truncated one -- there is no state a
#              preemption can corrupt, only work it can discard.
#              The cache is per (split, shard layout, defense params), so a requeued worker reads
#              back its OWN table; this is also why the layout must not change between attempts.
#              Note the shard parquet is written only when a shard finishes -- the cache, not the
#              parquet, is what makes this resumable.
#   featurize  RESUMES: vectors are cached per document, content-addressed, flushed every 256.
#   attack     DOES NOT RESUME. run_experiment.py deletes its own predictions_*.csv on entry and
#              starts over. That is survivable for stylometrix (minutes to an hour) and is the
#              reason scripts/slurm.toml puts WildChat+gemini attacks on the non-preempt `gpu`
#              partition instead -- for that feature, prefer ATTACK=0 here and submit the attack
#              separately (see the note at the bottom of this header).
#
# STAGES 2-5 RUN ONCE, in whichever task holds the last worker to finish: that worker's
# `apply_defenses` merges the 48 shards, and the task that finds the merged parquet claims the
# downstream stages under a lock.
#
# EPSILON IS THE READABILITY KNOB. The default is dp_mlm_eps100: at the paper's eps=25 the sampled
# words are mostly unrelated to the originals and the output is not usable as prompt text -- the
# CSV is then perfectly well-formed and full of gibberish, which is the mechanism working as
# specified, not a bug. Raise it (dp_mlm_eps250, dp_mlm_eps500, dp_mlm_eps1000) for more readable
# text at weaker privacy. Each epsilon is a separate defense with its own cache, parquet, CSV and
# results directory, so arms never overwrite each other.
#
# Overrides (sbatch exports your submitting environment, so these are read at run time):
#
#   WORKERS=10 sbatch --cpus-per-task=24 ...   # more shards per card (see MEASURE above)
#   DEFENSE=dp_mlm_eps250 sbatch ...       # a different budget
#   DEFENSE=dp_mlm_var_a10 sbatch ...      # adaptive length (the paper's Algorithm 3)
#   FEATURE=gemini_embedding_2 sbatch ...  # the paid encoder -- read the memory note below first
#   SOURCE=wildchat_small sbatch --array=0-1 ...   # a smaller cut; mind the layout note above
#   LIMIT=2000 sbatch --array=0 ...        # SMOKE TEST (forces ATTACK=0)
#   CSV=0 sbatch ...                       # skip the side output entirely
#   SHARD_CSV=1 sbatch ...                 # also export each shard's CSV as it finishes
#   CSV_NEWLINES=escape sbatch ...         # one physical line per CSV row, for grep
#   ATTACK=0 sbatch ...                    # stop after featurizing (see the memory note)
#   MERGE=0 sbatch ...                     # shards only; merge and run the rest by hand later
#   DPMLM_BATCH_SIZE=256 sbatch ...        # bigger GPU batches; output-neutral and not in the
#   DPMLM_FLUSH_POSITIONS=16384 sbatch ... # cache key, so tuning these never invalidates work
#   DPMLM_CHECKPOINT_EVERY=250 sbatch ...  # flush the cache more often: less work lost to a
#                                          # preemption, more table rewrites (see below)
#
# Anything after the script name is forwarded to run_experiment.py only (not to the build stages),
# so attack and scoring flags go there:
#
#   sbatch experiments/run_dpmlm_wildchat.sbatch --attacks nearest_neighbor logistic
#
# MEMORY, AND WHEN TO RUN THE ATTACK SEPARATELY. --mem below is one number for all 8 tasks, so it
# is sized for 6 workers plus the merge, not for a WildChat attack on 3,072-d embeddings:
# scripts/slurm.toml asks 96g for wildchat+gemini_embedding_2 and 24g for stylometrix. The default
# FEATURE=stylometrix fits here (measured 14.6 GB peak for nearest_neighbor). With
# FEATURE=gemini_embedding_2, submit ATTACK=0 and run the attack as its own job afterwards:
#
#   python experiments/run_all_experiments.py --slurm --datasets wildchat \
#       --defenses dp_mlm_eps100 --features gemini_embedding_2
#
#SBATCH --job-name=dpmlm-wc
#SBATCH --output=scripts/logs/%x-%A_%a.out
#SBATCH --open-mode=append
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --partition=gpu-preempt
#SBATCH --gpus=1
# Any card with ~24 GB will hold six 3 GB workers; the list is the repo's usual set minus the small
# ones. Pre-Ampere would land on fp16 rather than bf16 (see `gpu_dtype`), which an MLM forward pass
# is fine with -- widen this if the queue for these is long and you would rather have more cards.
#SBATCH --constraint=a100|l40s|h100|h200
# ~2.7 cores per worker. This, not VRAM, is what caps WORKERS: each worker has a single-threaded
# Python phase (NLTK detokenize + tokenizer encode per masked word) that runs flat out.
#SBATCH --cpus-per-task=16
# 6 workers x ~3 GB of host memory, plus headroom for the merge that one task also runs:
# `merge_shards` reads all 48 shards into pandas and concatenates them, and WildChat's text is
# ~3.8 GiB as Python strings, so that step alone peaks near 8 GiB -- and the attack stage after it
# measured 14.6 GB on this corpus. If the merge still OOMs, submit MERGE=0 and finish on a fat CPU
# node (the commands are printed at the end of every log).
#SBATCH --mem=48g
# Expected 1-4 hours at 8 cards x 6 workers on a ~90M-position corpus; this is the margin for being
# wrong about the rate, and --requeue means a preemption costs queue time rather than work.
#SBATCH --time=0-12:00:00
#SBATCH --requeue

set -euo pipefail

PROJECT="${PROMPT_ANONYMITY_PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT"

SOURCE="${SOURCE:-wildchat}"
DEFENSE="${DEFENSE:-dp_mlm_eps100}"
FEATURE="${FEATURE:-stylometrix}"
# Read the split from the published mirror, write everything this job makes to data/dist. Passed
# explicitly rather than left to the defaults so the pre-flight check below tests the same path the
# stages will use. OUT_DIR is also what run_experiment.py reads as --data-dir: stages 1-4 write
# there, while the runner's own default is data/hf, which will not have these files.
READ_DIR="${DIST_DIR:-$PROJECT/data/hf}"
OUT_DIR="${OUT_DIR:-$PROJECT/data/dist}"
CSV_DIR="${CSV_DIR:-$OUT_DIR/csv}"
LOG_DIR="${LOG_DIR:-$PROJECT/scripts/logs}"
LIMIT="${LIMIT:-}"
MERGE="${MERGE:-1}"
CSV="${CSV:-1}"
SHARD_CSV="${SHARD_CSV:-0}"
CSV_NEWLINES="${CSV_NEWLINES:-keep}"
CSV_LEVEL="${CSV_LEVEL:-turn}"
ATTACK="${ATTACK:-1}"
STAGGER="${STAGGER:-10}"

# The shard layout. TASKS x WORKERS is the total shard count, and it is baked into both the shard
# filenames and the defend cache path -- see the header. TASKS defaults to the array's size, but
# RE-RUNNING ONE FAILED TASK must state it, because `sbatch --array=3` sets
# SLURM_ARRAY_TASK_COUNT=1 and the run would otherwise recompute the corpus under six wrong names:
#
#     TASKS=8 sbatch --array=3 experiments/run_dpmlm_wildchat.sbatch
#
TASK="${SLURM_ARRAY_TASK_ID:-0}"
TASKS="${TASKS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
WORKERS="${WORKERS:-6}"
NSHARDS=$((TASKS * WORKERS))

CPUS="${SLURM_CPUS_PER_TASK:-16}"
# Split the allocation's cores between the workers instead of letting every process think it owns
# the node: six torch runtimes each spawning $(nproc) OpenMP threads is pure contention, and the
# tokenizer's own pool is worse than useless here (one short pair-encode per call).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(( CPUS / WORKERS > 0 ? CPUS / WORKERS : 1 ))}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# HOW MUCH A PREEMPTION COSTS, in one number. The defense flushes its cache table every this many
# DISTINCT TURNS -- turns, not conversations, because `apply_defenses` hands the cache one row per
# turn (`<doc_id>#<n>`) -- and everything already in the table comes back as a hit on resume. So
# the work at risk at any instant is one batch of this size, and nothing else.
#
# Why not leave it at the module default of 200: each flush rewrites the WHOLE table for the shard
# (`_write_checkpoint`), which grows toward ~80 MB here, and 200 turns is ~20 seconds of work --
# that is a rewrite every 20 seconds, on a shared filesystem, times 48 workers, to protect against
# losing 20 seconds. 1,000 turns puts a flush every few minutes and cuts the write volume ~5x.
# Read at import time, and NOT part of the defense's params(), so changing it never invalidates a
# cached turn. Lower it if preemptions on this partition are frequent enough to matter.
export DPMLM_CHECKPOINT_EVERY="${DPMLM_CHECKPOINT_EVERY:-1000}"

# A LIMIT run defends the first N documents only, so stages 4-5 would join a partial defended split
# against the full one and produce attack numbers that mean nothing. The CSV is unaffected -- it
# describes exactly the text that was defended -- so a smoke test still ends in something readable.
if [ -n "$LIMIT" ]; then
    ATTACK=0
fi

LIMIT_ARGS=()
if [ -n "$LIMIT" ]; then
    LIMIT_ARGS=(--limit "$LIMIT")
fi

MERGE_ARGS=()
if [ "$MERGE" = "0" ]; then
    MERGE_ARGS=(--no-auto-merge)
fi

CSV_ARGS=(--level "$CSV_LEVEL" --newlines "$CSV_NEWLINES")

# Each stage goes through scripts/slurm_job.sh, the same env bootstrap the experiment launcher
# uses, so how a batch shell here finds conda stays defined in exactly one file. It ends in
# `exec "$@"`, so calling it as a command runs the stage and returns here.
#
# Deliberately NOT `srun`: since SLURM 22.05 a step does not inherit --cpus-per-task from the batch
# allocation, so `srun <worker>` would hand each one a single CPU -- which is precisely the resource
# this job is short of.
RUN=(bash scripts/slurm_job.sh)

STEM="${SOURCE}_${DEFENSE}"
MERGED="$OUT_DIR/${STEM}.parquet"
RUN_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"

echo "=================================================================="
echo "DP-MLM on WildChat: defend -> featurize -> attack (+ CSV on the side)"
echo "  source:   $SOURCE"
echo "  defense:  $DEFENSE"
echo "  feature:  $FEATURE"
echo "  layout:   task $TASK of $TASKS x $WORKERS workers = $NSHARDS shards"
echo "  threads:  $OMP_NUM_THREADS per worker (of $CPUS CPUs)"
echo "  reads:    $READ_DIR/${SOURCE}.parquet"
echo "  writes:   $MERGED"
[ "$CSV" != "0" ] && echo "  csv:      $CSV_DIR/${STEM}.csv  (side output, nothing reads it)"
[ -n "$LIMIT" ] && echo "  LIMIT:    $LIMIT documents (SMOKE TEST -- attack stage disabled)"
echo "Job $RUN_ID task $TASK on $(hostname), $(date)"
echo "=================================================================="

# 0. Fail in the first second, not after the queue wait, if the split is not where this will look.
if [ ! -f "$READ_DIR/${SOURCE}.parquet" ]; then
    echo "ERROR: no $READ_DIR/${SOURCE}.parquet"
    echo "Mirror the published dataset first:  python -m prompt_anonymity.data.download"
    echo "(or point DIST_DIR at a local build, e.g. DIST_DIR=$PROJECT/data/dist)"
    exit 1
fi

mkdir -p "$LOG_DIR"

# 1. Defend this task's shards -- WORKERS processes on the one GPU, in parallel.
#
#    Each worker is handed its shard EXPLICITLY (--num-shards/--shard-index), which
#    `resolve_sharding` prefers over the SLURM array variables: those describe the array, and the
#    array is now only half of the layout. Each also gets its own log, because six tqdm progress
#    bars multiplexed into one file are unreadable -- follow one with
#    `tail -f scripts/logs/dpmlm-wc-<jobid>_<task>.shard0000.log`.
echo ""
echo "------------------------------------------------------------------"
echo "[1/5] defend shards $((TASK * WORKERS))-$((TASK * WORKERS + WORKERS - 1)) of $NSHARDS  ($(date))"
echo "------------------------------------------------------------------"

pids=()
worker_shards=()
worker_logs=()
for worker in $(seq 0 $((WORKERS - 1))); do
    shard=$((TASK * WORKERS + worker))
    log="$LOG_DIR/${SLURM_JOB_NAME:-dpmlm-wc}-${RUN_ID}_${TASK}.shard$(printf '%04d' "$shard").log"
    echo "  worker $worker -> shard $shard/$NSHARDS  ($log)"
    # APPENDED, never truncated: a requeued task re-runs this loop, and `>` would erase the log of
    # the attempt that was preempted -- exactly the log you need to see how far it got. Matches
    # --open-mode=append on the task's own log.
    echo "=== shard $shard, job $RUN_ID, attempt starting $(date) ===" >> "$log"
    "${RUN[@]}" python -m prompt_anonymity.data.apply_defenses \
        --source "$SOURCE" --defense "$DEFENSE" \
        --dist-dir "$READ_DIR" --out-dir "$OUT_DIR" \
        --num-shards "$NSHARDS" --shard-index "$shard" \
        "${LIMIT_ARGS[@]}" "${MERGE_ARGS[@]}" >> "$log" 2>&1 &
    pids+=($!)
    worker_shards+=("$shard")
    worker_logs+=("$log")
    # Stagger the starts: six processes importing torch and initializing CUDA on the same second
    # is a thundering herd on a shared filesystem, and buys nothing. Seconds, so it is noise next
    # to the run; STAGGER=0 turns it off.
    [ "$STAGGER" = "0" ] || sleep "$STAGGER"
done

failed=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        echo "  shard ${worker_shards[$index]} finished ($(date))"
    else
        echo "  shard ${worker_shards[$index]} FAILED -- see ${worker_logs[$index]}"
        failed=1
    fi
done

if [ "$failed" != "0" ]; then
    echo ""
    echo "At least one worker failed; not merging or running anything downstream from a partial"
    echo "set of shards. Fix the cause and re-submit this task -- finished shards are cached, so"
    echo "it resumes rather than restarts:"
    echo "  TASKS=$TASKS WORKERS=$WORKERS sbatch --array=$TASK experiments/run_dpmlm_wildchat.sbatch"
    exit 1
fi

# 2. Optional side output, per shard: each finished shard's own CSV. Off by default -- the merged
#    export in stage 3 covers the same text in split order, and on WildChat writing both doubles
#    the disk. Turn it on to read defended text before the slowest worker has finished, or to
#    salvage shards whose siblings failed.
if [ "$CSV" != "0" ] && [ "$SHARD_CSV" = "1" ]; then
    echo ""
    echo "------------------------------------------------------------------"
    echo "[2/5] side output: shard CSVs  ($(date))"
    echo "------------------------------------------------------------------"
    for shard in "${worker_shards[@]}"; do
        file="$OUT_DIR/shards/${STEM}.$(printf '%04d' "$shard")-of-$(printf '%04d' "$NSHARDS").parquet"
        [ -f "$file" ] || continue
        "${RUN[@]}" python experiments/export_defended_csv.py \
            --source "$SOURCE" --defense "$DEFENSE" \
            --defended "$file" --out "$CSV_DIR/$(basename "${file%.parquet}").csv" \
            --dist-dir "$READ_DIR" --out-dir "$OUT_DIR" "${CSV_ARGS[@]}"
    done
fi

# 2b. Stages 3-5 must run EXACTLY ONCE, over the merged split, and only after every shard exists.
#     The last worker anywhere in the array auto-merges, so "the merged parquet is on disk" is the
#     signal for the task that holds it. `mkdir` is the claim because it is atomic on POSIX: if two
#     tasks race here, exactly one creates the directory. The lock is keyed to this array's job id,
#     so a fresh submission is never blocked by an old lock, and re-entrant for its owner, so a
#     requeued winner resumes the downstream stages instead of skipping them.
CLAIMED=0
if [ "$MERGE" != "0" ] && [ -f "$MERGED" ]; then
    LOCK="$OUT_DIR/.dpmlm_downstream_${STEM}_${RUN_ID}"
    if mkdir "$LOCK" 2>/dev/null; then
        echo "$TASK" > "$LOCK/owner"
        CLAIMED=1
    elif [ "$(cat "$LOCK/owner" 2>/dev/null)" = "$TASK" ]; then
        CLAIMED=1
    fi
fi

if [ "$CLAIMED" != "1" ]; then
    echo ""
    echo "Task $TASK's $WORKERS shards are defended."
    echo "  shards so far: $(ls "$OUT_DIR/shards/${STEM}."*.parquet 2>/dev/null | wc -l) of $NSHARDS"
    if [ "$MERGE" = "0" ]; then
        echo "MERGE=0: no task will merge. Run the merge, export and downstream stages by hand"
        echo "(the commands are at the bottom of this log)."
    else
        echo "Another task merges, exports the CSV and runs featurize/attack once the last shard lands."
    fi
    echo "Done $(date)."
    exit 0
fi

echo ""
echo "This task holds the worker that merged the split, so it runs the remaining stages."

# From here on the SLURM array variables must be OUT of the environment. compute_features resolves
# its own sharding from exactly those variables, so leaving them set would make stage 4 featurize
# only 1/8th of the documents and write a shard file nobody merges. Every worker above was given
# its shard explicitly, so nothing below needs them.
RUN=(env -u SLURM_ARRAY_TASK_ID -u SLURM_ARRAY_TASK_COUNT -u SLURM_ARRAY_JOB_ID
     bash scripts/slurm_job.sh)

# 3. THE SIDE OUTPUT: every defended text in one readable CSV, one row per turn, original beside
#    defended, in split order. Streamed a parquet batch at a time, so it costs minutes and almost
#    no memory however big the corpus is. Deliberately placed BEFORE the featurize and attack
#    stages, which are the long ones: if DP-MLM produced gibberish at this epsilon, the CSV says so
#    now rather than after hours of attacking text nobody looked at.
if [ "$CSV" != "0" ]; then
    echo ""
    echo "------------------------------------------------------------------"
    echo "[3/5] side output: CSV of every defended text  ($(date))"
    echo "------------------------------------------------------------------"
    "${RUN[@]}" python experiments/export_defended_csv.py \
        --source "$SOURCE" --defense "$DEFENSE" \
        --dist-dir "$READ_DIR" --out-dir "$OUT_DIR" --out "$CSV_DIR/${STEM}.csv" \
        "${CSV_ARGS[@]}"
fi

# 4. Featurize the defended text. --dist-dir is where stage 1 wrote, not the mirror.
echo ""
echo "------------------------------------------------------------------"
echo "[4/5] featurize  ($(date))"
echo "------------------------------------------------------------------"
"${RUN[@]}" python -m prompt_anonymity.data.compute_features \
    --source "$SOURCE" --defense "$DEFENSE" --feature "$FEATURE" --dist-dir "$OUT_DIR"

# 5. Attack it. "$@" is forwarded straight through, so attack/scoring flags given at submit time
#    land here and nowhere else.
if [ "$ATTACK" != "0" ]; then
    echo ""
    echo "------------------------------------------------------------------"
    echo "[5/5] attack  ($(date))"
    echo "------------------------------------------------------------------"
    "${RUN[@]}" python experiments/run_experiment.py \
        --source "$SOURCE" --defense "$DEFENSE" --feature "$FEATURE" \
        --data-dir "$OUT_DIR" "$@"
fi

echo ""
echo "Done $(date)."
echo "  defended: $MERGED"
echo "  features: $OUT_DIR/${STEM}_${FEATURE}.parquet"
[ "$ATTACK" != "0" ] && echo "  results:  experiments/results/${STEM}_${FEATURE}_*/"
[ "$CSV" != "0" ] && echo "  csv:      $CSV_DIR/${STEM}.csv  (side output)"
echo ""
echo "Read the text before trusting the numbers -- DP-MLM's output quality is the whole question at"
echo "a given epsilon:"
echo "  python -c \"import pandas as pd; d=pd.read_csv('$CSV_DIR/${STEM}.csv', nrows=20); print(d[['original_text','defended_text']].to_string())\""
echo ""
echo "Compare against the undefended baseline (${SOURCE}_base_${FEATURE}_*) for how much"
echo "attribution the defense removed, and against the neighbouring epsilons for the privacy/"
echo "utility curve. Draw the figures with:  python experiments/plot_results.py --jobs 8"
echo ""
echo "If a task died and the merge never happened, finish it by hand (CPU only, no GPU needed):"
echo "  python -m prompt_anonymity.data.apply_defenses --source $SOURCE --defense $DEFENSE \\"
echo "      --dist-dir $READ_DIR --out-dir $OUT_DIR --merge"
echo "  python experiments/export_defended_csv.py --source $SOURCE --defense $DEFENSE \\"
echo "      --dist-dir $READ_DIR --out-dir $OUT_DIR"
