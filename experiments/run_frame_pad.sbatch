#!/bin/bash
#
# run_frame_pad.sbatch -- the whole frame-pad arm in one job, on Gemini embeddings:
# build the passage bank -> defend -> featurize -> attack.
#
#   sbatch experiments/run_frame_pad.sbatch
#
# SUBMIT FROM THE REPO ROOT. --output below is a relative path, and SLURM resolves it against the
# directory you submit from.
#
# ONE-TIME SETUP -- SLURM fails a job whose --output directory does not exist:
#
#   mkdir -p scripts/logs
#
# WHAT THIS ARM IS. frame_shift does two things at once: it ADDS a heavy topical frame and it
# REWRITES the user's request into that frame's register. frame_pad is the first half alone. Every
# user turn is copied through byte-identical and one dense, off-topic turn -- a parish bulletin
# notice, a reef-tank log, zoning board minutes -- is APPENDED to the document. So the comparison
# frame_pad vs frame_shift asks: how much of frame_shift's effect was the added content, and how
# much needed the rewrite?
#
# WHY IT IS ALMOST FREE, unlike frame_shift's ~$95-180 per corpus. The padding text is generated
# ONCE: 50 scenes x 8 passages = 400 passages, one call per scene on the same model frame_shift
# rewrites with, about two cents in total (stage 0b) -- 50 calls buy the padding for an entire
# corpus. It is written to data/dist/frame_pad_bank.json and every later run just reads
# that file -- each document draws its pad by a keyed hash of its doc_id, with no API call at corpus
# time. Nothing about the defense stage scales with the corpus.
#
# COSTS THAT REMAIN. gemini_embedding_2 is an OpenRouter call per document, so the FEATURIZE stage
# is what this job actually spends money on, exactly as it does for any other defense -- and padding
# makes each document longer, so it embeds slightly more tokens than the base arm. An
# OPENROUTER_API_KEY must be in a .env the job can reach (load_dotenv walks UP from the repo root,
# so DS_env/.env is NOT found from the root -- fix with: cp DS_env/.env .env). Both the bank and the
# embeddings use it.
#
# NO GPU. Unlike run_collision_seeding.sbatch and run_frame_shift.sbatch, this asks for a CPU node:
# the defense is a dictionary lookup, gemini_embedding_2 is a network call, and the attacks are
# scikit-learn. If you add stylometrix to $FEATURES you need the GPU partition back -- it runs a
# spaCy transformer -- so copy the SBATCH block from run_collision_seeding.sbatch in that case.
#
# READ STAGE 0c BEFORE TRUSTING THE NUMBERS. It prints the bank: sample passages plus the measures
# that decide whether this arm means anything.
#
#   density    proper nouns + numbers per 100 words. The pad is supposed to be CONTENT -- named
#              people, dates, quantities, procedures. If the model wrote mood and weather instead,
#              this arm degenerates into "the documents got longer", which is the null hypothesis
#              it is meant to beat. Thin passages are counted; the count should be 0.
#   questions / imperatives
#              must be 0. The pad lands inside a real user message, so a passage that asks for
#              something WILL be answered by the response model, and that shows up as a utility
#              failure caused by the measurement rather than by the defense.
#   technical  must be 0. Nothing in the pad may share vocabulary with software or assistant chat;
#              passages carrying it are dropped at build time, so a non-zero count here means the
#              bank predates that filter and should be rebuilt with --force.
#
# Overrides (sbatch exports your submitting environment, so these are read at run time):
#
#   SOURCE=wildchat sbatch experiments/run_frame_pad.sbatch            # default swe_chat
#   DEFENSES="frame_pad" sbatch experiments/run_frame_pad.sbatch       # skip the single-scene arm
#   FEATURES="gemini_embedding_2 char_ngram_tfidf" sbatch experiments/run_frame_pad.sbatch
#   FRAME_PAD_MODEL=qwen/qwen3.7-flash sbatch experiments/run_frame_pad.sbatch   # bank writer;
#       the default matches frame_shift's rewriter so neither arm has the better writer. Changing it
#       does NOT rebuild an existing bank -- pass REBUILD_BANK=1 as well.
#   REBUILD_BANK=1 sbatch experiments/run_frame_pad.sbatch             # regenerate and REPLACE it
#
# Anything after the script name is forwarded to run_experiment.py only:
#
#   sbatch experiments/run_frame_pad.sbatch --attacks nearest_neighbor logistic
#
#SBATCH --job-name=frame_pad
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
SHOW_FRAMES="${SHOW_FRAMES:-4}"

# Same env bootstrap the experiment launcher uses, so how a batch shell finds conda stays defined
# in exactly one file. It ends in `exec "$@"`, so calling it as a command runs the stage.
RUN=(bash scripts/slurm_job.sh)

echo "=================================================================="
echo "Frame pad: source=$SOURCE"
echo "  defenses: $DEFENSES"
echo "  features: $FEATURES"
echo "  bank writer: ${FRAME_PAD_MODEL:-<the configured default>}"
echo "Job ${SLURM_JOB_ID:-local} on $(hostname), started $(date)"
echo "=================================================================="

# 0a. Correctness checks. Seconds, no data, no API key, and every one of them is a property this
#     defense's validity rests on: the pad is a function of the doc_id and NEVER of the text, one
#     turn is added and the existing turns come through untouched, the assignment is shard-invariant
#     and agrees with frame_shift's, and the bank measures actually discriminate. Run first so a
#     broken build fails here rather than after a corpus of embeddings has been paid for.
"${RUN[@]}" python -m prompt_anonymity.defenses.frame_pad --selftest

# 0b. The passage bank -- the only paid step of the defense, about two cents, and a no-op once the
#     file exists. Every later run and every array task reads this same file, which is what makes
#     the padding identical across shards and reproducible across re-runs.
BUILD=(--build-bank)
[ -n "${REBUILD_BANK:-}" ] && BUILD+=(--force)
"${RUN[@]}" python -m prompt_anonymity.defenses.frame_pad "${BUILD[@]}"

# 0c. Read the bank -- see the header. This is the text that is about to be appended to every
#     document in the corpus, and it is small enough to actually look at.
"${RUN[@]}" python -m prompt_anonymity.defenses.frame_pad --show-bank --show-frames "$SHOW_FRAMES"

# 0d. The doc_id -> scene/pad assignment. Nothing in the defense needs it (the assignment is a pure
#     function of the doc id, recomputed on demand), but the analysis does: it is what lets you ask
#     whether an attacker is recovering the PAD rather than the author, i.e. whether misattributed
#     documents land disproportionately among the documents sharing their pad. Note the group here
#     is the pad (K x P of them), not the scene (K).
"${RUN[@]}" python -m prompt_anonymity.defenses.frame_pad \
    --manifest --source "$SOURCE" --dist-dir "$DIST_DIR" --out-dir "$DIST_DIR" --limit 0

for DEFENSE in $DEFENSES; do
    echo ""
    echo "------------------------------------------------------------------"
    echo "[$DEFENSE] defending $SOURCE  ($(date))"
    echo "------------------------------------------------------------------"

    # 1. Defend. Seconds: a lookup per document, no model, no cache, no network. It reports how many
    #    documents were padded -- if any were NOT, their scene is missing from the bank and the bank
    #    needs rebuilding with --force.
    "${RUN[@]}" python -m prompt_anonymity.data.apply_defenses \
        --source "$SOURCE" --defense "$DEFENSE"

    for FEATURE in $FEATURES; do
        echo ""
        echo "[$DEFENSE / $FEATURE] featurize + attack  ($(date))"

        # 2. Featurize the defended text. This is the stage that spends money.
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
echo "                                  at all, with the user's own words completely untouched?"
echo "  vs ${SOURCE}_frame_shift_*   -- THE comparison this arm exists for. frame_shift adds a frame"
echo "                                  AND rewrites; frame_pad only adds. The gap between them is"
echo "                                  what the rewrite bought -- and frame_pad costs ~2 cents"
echo "                                  against frame_shift's ~\$95-180 per corpus."
echo "  frame_pad vs _single         -- 400 pads in circulation versus 8. Does pad DIVERSITY matter,"
echo "                                  or only pad presence?"
echo "  vs ${SOURCE}_collision_seeding_* -- the same collision idea at a different scale: shared"
echo "                                  character-level quirks versus a shared paragraph."
echo ""
echo "ONE CAVEAT TO READ THE EMBEDDING NUMBERS WITH. gemini_embedding_2 reads only a document's"
echo "first 8,192 tokens and silently discards the rest, and the pad is appended at the END -- so"
echo "for any document already longer than that, the pad is never embedded and this defense cannot"
echo "move its vector at all. Stage 0d's manifest plus"
echo "  python -m prompt_anonymity.defenses.frame_pad --preview --source $SOURCE --limit 200"
echo "report what share of the corpus is in that state. Expect the effect to sit in the short"
echo "documents; on char_ngram_tfidf and stylometrix, which read the whole text, it should not."
echo ""
echo "Also check utility. A padded prompt still contains the user's exact request, so utility should"
echo "move far less than under frame_shift -- but the response model does see an unrelated paragraph"
echo "and may remark on it, which is a real cost this arm has to be honest about."
echo ""
echo "Draw figures with:  python experiments/plot_results.py"
