#!/bin/bash
#
# run_frame_shift.sbatch -- the whole frame-shift arm in one job: defend -> featurize -> attack.
#
#   sbatch experiments/run_frame_shift.sbatch
#
# SUBMIT FROM THE REPO ROOT. --output below is a relative path, and SLURM resolves it against the
# directory you submit from.
#
# ONE-TIME SETUP -- SLURM fails a job whose --output directory does not exist:
#
#   mkdir -p scripts/logs
#
# THIS ARM COSTS MONEY. Unlike collision seeding (pure Python string work) the defense stage here is
# one hosted-model rewrite per (frame, turn) through OpenRouter, so:
#
#   * an OPENROUTER_API_KEY must be in a .env the job can reach (load_dotenv walks up from the repo
#     root), or stage 1 dies immediately -- which is the good failure, before any featurization;
#   * READ THE PREVIEW BEFORE THE FULL RUN. Stage 0b rewrites a handful of real documents and prints
#     before/after for a few cents, with the measurable part of the contract scored: `reuse` (share
#     of the original's 8-word runs surviving verbatim -- a rewrite that quotes the user preserves
#     exactly the signal this defense exists to destroy, so this must be LOW), `expansion`, and
#     whether code spans came through byte-identical. What you still have to judge by eye is whether
#     the text plainly asks for a real answer. Set SKIP_PREVIEW=1 only once you have read one for
#     this corpus;
#   * TO PICK A CHEAPER MODEL, run the bake-off first -- same inputs, several models, scored:
#
#       python -m prompt_anonymity.defenses.frame_shift --preview --limit 3 \
#           --models inclusionai/ling-2.6-flash,qwen/qwen3.7-flash,deepseek/deepseek-v4-flash-0731
#
#     then set MODEL= (or $FRAME_SHIFT_MODEL) to the cheapest one whose reuse stays low and whose
#     code survives. On WildChat that choice is worth roughly $15 vs $95;
#   * stage 1 prints the true call count -- distinct (frame, turn) pairs -- before its first request.
#     That number is HIGHER than apply_defenses' "distinct non-blank" line, because the frame is part
#     of the dedup key: one turn appearing under three frames is three paid calls. Kill the job there
#     if it is not the number you expected.
#
# The defense stage is network- and CPU-bound and would be happy on any CPU node; the GPU request is
# for stage 2 (StyloMetrix runs a spaCy transformer). One job for the whole arm because the expensive
# thing is bought once and then cached: a killed run resumes from the last checkpoint rather than
# re-buying its rewrites.
#
# Overrides (sbatch exports your submitting environment, so these are read at run time):
#
#   SOURCE=wildchat sbatch experiments/run_frame_shift.sbatch          # default swe_chat
#   DEFENSES="frame_shift frame_shift_single" sbatch experiments/run_frame_shift.sbatch
#   FEATURES="char_ngram_tfidf" sbatch experiments/run_frame_shift.sbatch
#   SKIP_PREVIEW=1 sbatch experiments/run_frame_shift.sbatch
#
# Anything after the script name is forwarded to run_experiment.py only:
#
#   sbatch experiments/run_frame_shift.sbatch --attacks nearest_neighbor logistic
#
#SBATCH --job-name=frame_shift
#SBATCH --output=scripts/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --partition=gpu-preempt
#SBATCH --gpus=1
#SBATCH --constraint=a100|l40s|h100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=0-16:00:00
#SBATCH --requeue

set -euo pipefail

PROJECT="${PROMPT_ANONYMITY_PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT"

SOURCE="${SOURCE:-swe_chat}"
DIST_DIR="${DIST_DIR:-$PROJECT/data/dist}"
DEFENSES="${DEFENSES:-frame_shift}"
FEATURES="${FEATURES:-char_ngram_tfidf stylometrix gemini_embedding_2}"
PREVIEW_LIMIT="${PREVIEW_LIMIT:-3}"

# Same env bootstrap the experiment launcher uses, so how a batch shell finds conda stays defined
# in exactly one file. It ends in `exec "$@"`, so calling it as a command runs the stage.
RUN=(bash scripts/slurm_job.sh)

echo "=================================================================="
echo "Frame shift: source=$SOURCE"
echo "  defenses: $DEFENSES"
echo "  features: $FEATURES"
echo "Job ${SLURM_JOB_ID:-local} on $(hostname), started $(date)"
echo "=================================================================="

# 0a. Correctness checks. Seconds, no data, no API key, and every one of them is a property the
#     defense's validity rests on (determinism, shard invariance, one frame per document, the
#     composed cache source, truncation fallback). Run first so a broken build fails here rather
#     than after hours of featurization -- or after a corpus of paid rewrites.
"${RUN[@]}" python -m prompt_anonymity.defenses.frame_shift --selftest

# 0b. Quality preview -- see the header. Costs cents; it is the only thing standing between a bad
#     system prompt and a corpus rewritten badly.
if [ -z "${SKIP_PREVIEW:-}" ]; then
    "${RUN[@]}" python -m prompt_anonymity.defenses.frame_shift \
        --preview --source "$SOURCE" --dist-dir "$DIST_DIR" --limit "$PREVIEW_LIMIT"
fi

# 0c. The doc_id -> frame assignment. Nothing in the defense needs this (assignment is a pure
#     function of the doc id, recomputed on demand), but the analysis does: it is what lets you ask
#     whether an attacker is recovering the FRAME rather than the author, i.e. whether misattributed
#     documents land disproportionately inside their own surface cluster.
"${RUN[@]}" python -m prompt_anonymity.defenses.frame_shift \
    --manifest --source "$SOURCE" --dist-dir "$DIST_DIR" --out-dir "$DIST_DIR" --limit 0

for DEFENSE in $DEFENSES; do
    echo ""
    echo "------------------------------------------------------------------"
    echo "[$DEFENSE] defending $SOURCE  ($(date))"
    echo "------------------------------------------------------------------"

    # 1. Defend. This is the stage that spends money. It checkpoints every 500 rows, so a preempted
    #    job resumes from the last flush instead of re-buying its rewrites -- and it reports how
    #    many rewrites fell back to the original turn when it finishes. A high fallback rate means
    #    replies are being truncated: raise FRAME_SHIFT_OUTPUT_RATIO and re-run.
    "${RUN[@]}" python -m prompt_anonymity.data.apply_defenses \
        --source "$SOURCE" --defense "$DEFENSE"

    for FEATURE in $FEATURES; do
        echo ""
        echo "[$DEFENSE / $FEATURE] featurize + attack  ($(date))"

        # 2. Featurize the defended text.
        "${RUN[@]}" python -m prompt_anonymity.data.compute_features \
            --source "$SOURCE" --defense "$DEFENSE" --feature "$FEATURE" --dist-dir "$DIST_DIR"

        # 3. Attack it. --data-dir is data/dist because that is where the two stages above write;
        #    the runner's own default is data/hf, the published mirror, which lacks these files.
        "${RUN[@]}" python experiments/run_experiment.py \
            --source "$SOURCE" --defense "$DEFENSE" --feature "$FEATURE" \
            --data-dir "$DIST_DIR" "$@"
    done
done

echo ""
echo "Done $(date). Results in experiments/results/${SOURCE}_frame_shift*/"
echo ""
echo "What to compare, and what each comparison answers:"
echo "  vs ${SOURCE}_base_*              -- does burying the prompt in a heavy frame reduce"
echo "                                      attribution at all?"
echo "  frame_shift vs _single           -- dilution vs convergence. The default rotates 50 frames"
echo "                                      so an author's documents disagree with each other;"
echo "                                      _single puts the whole corpus in ONE register. _single"
echo "                                      is also the control for plain length inflation, since"
echo "                                      both arms lengthen documents the same way."
echo "  vs ${SOURCE}_openanonymity_*     -- rewriting the SURFACE against rewriting the CONTENT"
echo "  gemini_embedding_2 vs char_ngram_tfidf -- the scope claim, and the mirror image of collision"
echo "                                      seeding's: a frame is topical and semantic, so the"
echo "                                      embedding channel should move MOST here."
echo ""
echo "Also check utility -- this defense trades it. A framed prompt that stopped asking for a real"
echo "answer scores as a utility failure, which is the number that makes the privacy gain readable."
echo ""
echo "Draw figures with:  python experiments/plot_results.py"
