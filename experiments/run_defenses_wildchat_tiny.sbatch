#!/bin/bash
#
# run_defenses_wildchat_tiny.sbatch -- every defense worth comparing, on the 40-author split.
# One ARRAY TASK PER DEFENSE: defend -> featurize -> attack. Plot once, afterwards, on the login
# node (it is CPU-only and takes seconds).
#
#   sbatch experiments/run_defenses_wildchat_tiny.sbatch
#   # ...then, once the array is done:
#   python experiments/plot_results.py --jobs 8 --force
#
# SUBMIT FROM THE REPO ROOT. --output is relative and SLURM resolves it against the submit dir.
# One-time: mkdir -p scripts/logs
#
# PREREQUISITE -- the split must exist, and must be capped so nothing is emitted undefended:
#
#     python -m prompt_anonymity.data.build_subset --k-min 5 --n-authors 40 \
#         --max-chars 60000 --out-source wildchat_tiny --data-dir data/hf --out-dir data/dist
#
# --max-chars is not optional for this corpus. `afr` refuses to emit a document that exceeds its
# context window (AFR_TOO_LONG_PASSTHROUGH), because untouched text inside a defended split keeps
# its full authorship signal and is then charged to the defense that never saw it. Capping in the
# CORPUS means every arm below reads the same text and the truncation is a property of the split,
# recorded in its manifest -- rather than an artifact of one defense.
#
# ONE TASK PER DEFENSE, so a defense that fails takes only its own task down and the rest of the
# grid still lands. The array index is an index into DEFENSES below; keep that list stable while an
# array is in flight, since the mapping is positional.
#
# WHAT IS IN, AND THE ONE OMISSION
# -------------------------------
#   dp_mlm*        excluded on request. It is a sweep of 11 registered variants and would dominate
#                  both the array and the figures.
#
# THE OPENANONYMITY REDACTOR RUNS THROUGH OPENROUTER HERE, not locally. Its models.toml entry is a bare repo id
# for a 120B checkpoint that is NOT mirrored under /datasets/ai, so a local run would download
# ~60 GB into $HF_HOME -- which has filled a disk quota once already. $OPENANON_API_MODEL (exported
# below) switches both `openanonymity` and the redact stage of `styleremix_openanon` to the API.
# It is part of those defenses' params(), so an API run never shares a cache entry with a local one.
# Unset it to go back to local weights.
#
# The rest, in increasing cost. The first six are CPU-only and finish in minutes; they are given a
# GPU anyway because an array shares one resource request, and they release it quickly.
#
#   none                   the undefended baseline -- spelled 'base' in results/ and on figures
#   example_normalization  trivial text normalization; the floor any defense must beat
#   rtt_argos              round-trip translation
#   collision_seeding      collision seeding at the default K
#   frame_shift            scene rewrite, 50-frame codebook
#   frame_pad              adds a scene turn and rewrites nothing -- splits dilution from rewriting
#   qwen_rewrite           a small local model rewriting each turn
#   styleremix             llama3 from /datasets/ai (mirrored; no download)
#   styleremix_openanon    restyle THEN gpt-oss-120b redaction via OpenRouter (see above).
#                          OpenAnonymity is measured only in this combined form: solo `openanonymity`
#                          is registered and works the same way, so add it to $DEFENSES if the
#                          redaction stage ever needs isolating from the restyle.
#   loo_unlink_b30         the greedy-rule alternative to afr at a similar target
#   afr_stage1             THE CONTROL: afr's abstraction pass, zero probes
#   afr                    the full defense
#
# `afr` vs `afr_stage1` is the comparison the whole thing exists for. `afr` vs `none` only shows
# that a 27B model rewrote the text.
#
# COST. Only the featurize stage spends money: one OpenRouter embedding call per document at
# $0.20/M tokens. ~900 documents x 12 arms is well under a dollar, and it caches by text content,
# so a requeued task re-embeds nothing.
#
# Overrides (sbatch exports your submitting environment):
#
#   SOURCE=wildchat_small sbatch ...      # a different split (keep --array in step with DEFENSES)
#   FEATURE=harrier sbatch ...            # local and free; also the surrogate-overfit check for afr
#   DEFENSES="afr afr_stage1" sbatch --array=0-1 ...   # a subset; array MUST match the list length
#
#SBATCH --job-name=def-wct
#SBATCH --output=scripts/logs/%x-%A_%a.out
#SBATCH --open-mode=append
#SBATCH --array=0-11
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

# Before anything initializes CUDA: the embedder and vLLM share one card and vLLM's reservation is
# permanent, so the embedder works in a shrinking remainder. Expandable segments let the allocator
# grow a block instead of failing on a fragmented heap.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SOURCE="${SOURCE:-wildchat_tiny}"
FEATURE="${FEATURE:-gemini_embedding_2}"
DIST_DIR="${DIST_DIR:-$PROJECT/data/dist}"

read -r -a DEFENSE_LIST <<< "${DEFENSES:-none example_normalization rtt_argos collision_seeding frame_shift frame_pad qwen_rewrite styleremix styleremix_openanon loo_unlink_b30 afr_stage1 afr}"

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$INDEX" -ge "${#DEFENSE_LIST[@]}" ]; then
    echo "task $INDEX has no defense: the list holds ${#DEFENSE_LIST[@]} entries " \
         "(${DEFENSE_LIST[*]}). Fix --array to match." >&2
    exit 1
fi
DEFENSE="${DEFENSE_LIST[$INDEX]}"

# Redact through OpenRouter rather than downloading a 120B checkpoint. Applies to `openanonymity`
# and to the second stage of `styleremix_openanon`; both put it in params(), so this is part of the
# cache key rather than a serving knob. Unset it to use local weights instead.
export OPENANON_API_MODEL="${OPENANON_API_MODEL:-openai/gpt-oss-120b}"

# 40 authors is one lockstep chunk. Deliberately NOT sharded: sharding would narrow afr's batch and
# make it slower per document, which is what produced 10 documents/hour on the 200-author split.
export AFR_AUTHOR_BATCH="${AFR_AUTHOR_BATCH:-64}"

RUN=(bash scripts/slurm_job.sh)

echo "=================================================================="
echo "defense $INDEX/${#DEFENSE_LIST[@]}: $DEFENSE   ($SOURCE / $FEATURE / logistic)"
echo "Job ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}} task $INDEX on $(hostname), $(date)"
echo "=================================================================="

# 0a. The split is cut locally, not downloaded, so its absence is the likeliest way this dies --
#     and it would die AFTER the queue wait.
if [ ! -f "$DIST_DIR/${SOURCE}.parquet" ]; then
    echo "ERROR: no $DIST_DIR/${SOURCE}.parquet. Build it first:"
    echo "  python -m prompt_anonymity.data.build_subset --k-min 5 --n-authors 40 \\"
    echo "      --max-chars 60000 --out-source ${SOURCE} --data-dir data/hf --out-dir $DIST_DIR"
    exit 1
fi

# 0b. Fail before the GPU stage, not after it: a missing key found at stage 2 would mean the hours
#     spent defending were spent to reach an error.
# The key is needed by the featurizer AND, for the two OpenAnonymity arms, by the defense itself.
# Written as plain `if` blocks, NOT `[ test ] && VAR=1`: under `set -e` an AND-list whose test
# fails returns non-zero and kills the script, so the harrier path would exit here instead of
# skipping the check.
NEEDS_KEY=0
if [ "$FEATURE" = "gemini_embedding_2" ]; then
    NEEDS_KEY=1
fi
if [ -n "$OPENANON_API_MODEL" ]; then
    case "$DEFENSE" in
        openanonymity|styleremix_openanon) NEEDS_KEY=1 ;;
    esac
fi
if [ "$NEEDS_KEY" = 1 ] && [ -z "${OPENROUTER_API_KEY:-}" ] && [ ! -f "$PROJECT/.env" ]; then
    echo "ERROR: this task needs an OPENROUTER_API_KEY and there is none in the environment," >&2
    echo "       nor a $PROJECT/.env for load_dotenv to read." >&2
    echo "       (feature=$FEATURE, defense=$DEFENSE, OPENANON_API_MODEL=$OPENANON_API_MODEL)" >&2
    exit 1
fi

# 1. Defend. Skipped entirely for `none`: the baseline IS the split, and apply_defenses has nothing
#    to write for it -- compute_features reads <split>.parquet directly when given no --defense.
if [ "$DEFENSE" != "none" ]; then
    echo ""
    echo "--------------------------------------------------------------"
    echo "[1/3] defend  ($(date))"
    echo "--------------------------------------------------------------"
    "${RUN[@]}" python -m prompt_anonymity.data.apply_defenses \
        --source "$SOURCE" --defense "$DEFENSE" --dist-dir "$DIST_DIR"
    FEATURE_ARGS=(--defense "$DEFENSE")
else
    echo ""
    echo "[1/3] defend: skipped -- 'none' is the undefended split itself"
    FEATURE_ARGS=()
fi

# 2. Featurize with the ATTACKER's encoder, deliberately not the one afr optimized against. That
#    mismatch is what makes the number a transfer result rather than self-grading.
echo ""
echo "--------------------------------------------------------------"
echo "[2/3] featurize  ($(date))"
echo "--------------------------------------------------------------"
"${RUN[@]}" python -m prompt_anonymity.data.compute_features \
    --source "$SOURCE" --feature "$FEATURE" --dist-dir "$DIST_DIR" \
    "${FEATURE_ARGS[@]+"${FEATURE_ARGS[@]}"}"

# 3. Attack. --data-dir is data/dist because that is where stages 1-2 write; the runner's own
#    default is data/hf, the published mirror, which lacks these files.
echo ""
echo "--------------------------------------------------------------"
echo "[3/3] attack  ($(date))"
echo "--------------------------------------------------------------"
"${RUN[@]}" python experiments/run_experiment.py \
    --source "$SOURCE" --defense "$DEFENSE" --feature "$FEATURE" \
    --data-dir "$DIST_DIR" --attacks logistic "$@"

echo ""
echo "Done $DEFENSE at $(date)."
echo "  results: experiments/results/${SOURCE}_$([ "$DEFENSE" = none ] && echo base || echo "$DEFENSE")_${FEATURE}_logistic/"
echo ""
echo "When the WHOLE array has finished, draw the figures (CPU, seconds):"
echo "  python experiments/plot_results.py --jobs 8 --force"
echo "  -> experiments/plots/${SOURCE}/accuracy/{doc,author}/by_defense/${FEATURE}_logistic.pdf"
echo "     one line per defense, which is the comparison this grid exists to make."
echo ""
echo "Read afr's edit log before its results -- it says whether the loop did anything:"
echo "  jq -s 'group_by(.stop_reason)|map({(.[0].stop_reason):length})|add' \\"
echo "     data/.cache/defended/${SOURCE}/**/afr/*/*/afr_logs/edits_a*.jsonl"
echo "  target_met dominating -> the loop is working"
echo "  ladder_exhausted      -> the agent stalls; the propose prompt needs work, not more probes"
