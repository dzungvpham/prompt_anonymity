#!/bin/bash
#
# run_frame_pad_gemini.sbatch -- run the frame-pad arm on an EXISTING passage bank, end to end on
# Gemini embeddings: defend -> featurize -> attack.
#
#   sbatch experiments/run_frame_pad_gemini.sbatch
#
# SUBMIT FROM THE REPO ROOT. --output below is a relative path, and SLURM resolves it against the
# directory you submit from. One-time: mkdir -p scripts/logs
#
# HOW THIS DIFFERS FROM run_frame_pad.sbatch: that script builds the passage bank if one is missing.
# This one NEVER calls a generation model -- it fails immediately if there is no bank -- because
# building the bank has been the flaky, slow, expensive part and there is no reason to re-enter it
# once a usable bank exists on disk. Everything downstream of the bank is deterministic string work
# plus embeddings.
#
# A PARTIAL BANK IS FINE. A build whose calls partly failed still writes what it got, and a bank
# holding 3-6 passages per scene instead of 8 costs only pad diversity: with K scenes x P passages
# the corpus circulates K*P distinct pads, so 250 passages means ~250 groups instead of 400. Stage 0b
# prints the real coverage. If some scenes have NO passages the defense restricts its draw to the
# covered ones, so every document is still padded -- it says so when it does, and the consequence is
# that the scene assignment no longer lines up with frame_shift's at the same seed.
#
# REQUIRES an OPENROUTER_API_KEY the job can reach -- not for the defense (no model runs), but for
# gemini_embedding_2, which is an OpenRouter embeddings call per document. load_dotenv walks UP from
# the working directory, so the key must be in a .env at the REPO ROOT: cp DS_env/.env .env
#
# COST is entirely stage 2: one embedding call per document at $0.20/M tokens. SWE-chat is 4,334
# documents, so each defense arm is cents; WildChat is 172,509 and is not.
#
# NO GPU. gemini_embedding_2 is a network call and the attacks are scikit-learn, so this asks for a
# CPU node -- which is also what gets it scheduled quickly. If you add stylometrix to $FEATURES you
# need a GPU partition back (it runs a spaCy transformer); copy the #SBATCH block from
# run_collision_seeding.sbatch.
#
# Overrides (sbatch exports your submitting environment, so these are read at run time):
#
#   SOURCE=wildchat sbatch experiments/run_frame_pad_gemini.sbatch        # default swe_chat
#   DEFENSES="frame_pad" sbatch experiments/run_frame_pad_gemini.sbatch   # skip the single-scene arm
#   FEATURES="gemini_embedding_2 char_ngram_tfidf" sbatch ...             # see the note below
#   BANK=/path/to/frame_pad_bank.json sbatch ...                          # a bank kept elsewhere
#
# Anything after the script name is forwarded to run_experiment.py only:
#
#   sbatch experiments/run_frame_pad_gemini.sbatch --attacks nearest_neighbor logistic
#
# READ THIS BEFORE INTERPRETING A NULL RESULT. gemini_embedding_2 reads only each document's first
# 8,192 tokens and silently discards the rest, and a document's text is its turns joined by a blank
# line -- so on any document already longer than that window, the APPENDED turn is never embedded.
# Stage 0c reports what share of the corpus is in that state. If it is large, "frame_pad did not move
# the embedding" and "the embedding never saw the pad" are the same number, and char_ngram_tfidf
# (which reads the whole document) is the channel that actually tests the hypothesis.
#
#SBATCH --job-name=frame_pad_gemini
#SBATCH --output=scripts/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --partition=cpu-preempt
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=0-08:00:00
#SBATCH --requeue

set -euo pipefail

PROJECT="${PROMPT_ANONYMITY_PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT"

SOURCE="${SOURCE:-swe_chat}"
DIST_DIR="${DIST_DIR:-$PROJECT/data/dist}"
DEFENSES="${DEFENSES:-frame_pad frame_pad_single}"
FEATURES="${FEATURES:-gemini_embedding_2}"
BANK="${BANK:-$DIST_DIR/frame_pad_bank.json}"
export FRAME_PAD_BANK="$BANK"

# Every defense CLI here runs with `python -m` after the registry has already imported the module,
# so runpy re-executes it as __main__ and warns. Two copies exist but nothing crosses between them
# (the CLI builds its own defense instance; the registry's is untouched). Filtered by message, not
# by category, so a real RuntimeWarning still reaches the log.
export PYTHONWARNINGS="ignore:.*found in sys.modules:RuntimeWarning"

# Same env bootstrap the experiment launcher uses, so how a batch shell finds conda stays defined
# in exactly one file. It ends in `exec "$@"`, so calling it as a command runs the stage.
RUN=(bash scripts/slurm_job.sh)

echo "=================================================================="
echo "Frame pad on an existing bank: source=$SOURCE"
echo "  bank:     $BANK"
echo "  defenses: $DEFENSES"
echo "  features: $FEATURES"
echo "Job ${SLURM_JOB_ID:-local} on $(hostname), started $(date)"
echo "=================================================================="

# 0a. Fail before anything else if there is no bank. This script does not build one on purpose (see
#     the header), and finding out here costs a second while finding out in stage 1 costs the queue.
if [ ! -f "$BANK" ]; then
    echo "ERROR: no passage bank at $BANK"
    echo "Build one first, on a login node so you can watch it:"
    echo "  python -m prompt_anonymity.defenses.frame_pad --build-bank"
    echo "Or point BANK= at a bank you already have."
    exit 1
fi

# 0b. Correctness checks, then the bank's real coverage. Seconds, no API key, no network -- and the
#     coverage line is what tells you how much pad diversity this run actually has: scenes covered,
#     passages held, and the contract violations to eyeball (a passage that asks a question gets
#     answered, and a thin bank makes this arm indistinguishable from "the documents got longer").
"${RUN[@]}" python -m prompt_anonymity.defenses.frame_pad --selftest
"${RUN[@]}" python -m prompt_anonymity.defenses.frame_pad --show-bank --show-frames 0

# 0c. What the padding will look like on real documents, and -- the line that decides how to read a
#     null result -- what share of them already exceed the embedding window before padding.
"${RUN[@]}" python -m prompt_anonymity.defenses.frame_pad \
    --preview --source "$SOURCE" --dist-dir "$DIST_DIR" --limit 200 | tail -25

# 0d. The doc_id -> scene/pad assignment. Nothing in the defense needs it (assignment is a pure
#     function of the doc id), but the analysis does: it is what lets you ask whether misattributed
#     documents land disproportionately inside their own PAD group rather than their own author.
"${RUN[@]}" python -m prompt_anonymity.defenses.frame_pad \
    --manifest --source "$SOURCE" --dist-dir "$DIST_DIR" --out-dir "$DIST_DIR" --limit 0

for DEFENSE in $DEFENSES; do
    echo ""
    echo "------------------------------------------------------------------"
    echo "[$DEFENSE] defending $SOURCE  ($(date))"
    echo "------------------------------------------------------------------"

    # 1. Defend. Seconds: a keyed hash per document and a dictionary lookup, no model, no API. It
    #    prints how many turns and characters it appended, and warns if any document went unpadded.
    "${RUN[@]}" python -m prompt_anonymity.data.apply_defenses \
        --source "$SOURCE" --defense "$DEFENSE"

    for FEATURE in $FEATURES; do
        echo ""
        echo "[$DEFENSE / $FEATURE] featurize + attack  ($(date))"

        # 2. Featurize the defended text. THIS is the stage that spends money, and it caches by text
        #    content -- so a re-run costs nothing and an interrupted one resumes.
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
echo "Done $(date). Results in experiments/results/${SOURCE}_frame_pad*/"
echo ""
echo "What to compare, and what each comparison answers:"
echo "  vs ${SOURCE}_base_*          -- does appending shared off-topic content reduce attribution"
echo "                                  at all, with the user's own words untouched?"
echo "  vs ${SOURCE}_frame_shift_*   -- THE comparison. frame_shift adds a frame AND rewrites the"
echo "                                  request; frame_pad only adds. The gap is what the rewrite"
echo "                                  bought, against ~\$95-180 a corpus for frame_shift and"
echo "                                  ~2 cents for this."
echo "  frame_pad vs _single         -- does pad DIVERSITY matter, or only pad presence? _single"
echo "                                  puts the whole corpus on one scene's passages."
echo "  vs ${SOURCE}_collision_seeding_* -- the same collision idea one channel over: shared"
echo "                                  character-level quirks against shared paragraphs."
echo ""
echo "Then check the truncation share printed in stage 0c before reading any of it as a null result."
echo ""
echo "Draw figures with:  python experiments/plot_results.py"
