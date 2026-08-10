#!/bin/bash
#
# run_collision_seeding.sbatch -- the whole collision-seeding arm in one job: every variant, every
# featurizer, defend -> featurize -> attack.
#
#   sbatch experiments/run_collision_seeding.sbatch
#
# SUBMIT FROM THE REPO ROOT. --output below is a relative path, and SLURM resolves it against the
# directory you submit from.
#
# ONE-TIME SETUP -- SLURM fails a job whose --output directory does not exist:
#
#   mkdir -p scripts/logs
#
# WHY ONE JOB FOR THE WHOLE ARM, unlike run_dpmlm_var_a10.sbatch which does a single variant: the
# defense itself is pure Python string work -- no model, no GPU, seconds per source rather than the
# hours DP-MLM needs. So sweeping four variants costs four sweeps of the *featurize* stage and
# nothing else, and splitting them across jobs would only multiply queue time. The GPU request is
# for stage 2 (StyloMetrix runs a spaCy transformer); stage 1 would be happy on any CPU node.
#
# Overrides (sbatch exports your submitting environment, so these are read at run time):
#
#   SOURCE=wildchat sbatch experiments/run_collision_seeding.sbatch       # default swe_chat
#   FEATURES="char_ngram_tfidf" sbatch experiments/run_collision_seeding.sbatch   # one featurizer
#   DEFENSES="collision_seeding" sbatch experiments/run_collision_seeding.sbatch  # one variant
#   SKIP_AUDIT=1 sbatch experiments/run_collision_seeding.sbatch          # skip the marker audit
#
# Anything after the script name is forwarded to run_experiment.py only:
#
#   sbatch experiments/run_collision_seeding.sbatch --attacks nearest_neighbor logistic
#
# READ THE AUDIT BEFORE TRUSTING THE NUMBERS. Stage 0 prints each marker's natural base rate and
# trigger coverage on this corpus. A marker with a ZERO base rate has no background to hide in --
# it is a perfect group indicator and reads as synthetic -- and a marker with near-zero trigger
# coverage cannot defend the authors it is assigned to. The audit prints a ready-to-paste
# MARKER_KEYS tuple of what survives; feed it to CollisionSeedingDefense(marker_keys=...) in
# defenses/__init__.py and re-run if the default inventory is doing badly on this source.
#
#SBATCH --job-name=collision_seeding
#SBATCH --output=scripts/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --partition=gpu-preempt
#SBATCH --gpus=1
#SBATCH --constraint=a100|l40s|h100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=0-12:00:00
#SBATCH --requeue

set -euo pipefail

PROJECT="${PROMPT_ANONYMITY_PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT"

SOURCE="${SOURCE:-swe_chat}"
DIST_DIR="${DIST_DIR:-$PROJECT/data/dist}"
DEFENSES="${DEFENSES:-collision_seeding collision_seeding_k4 collision_seeding_full collision_seeding_indep}"
FEATURES="${FEATURES:-char_ngram_tfidf stylometrix gemini_embedding_2}"

# Same env bootstrap the experiment launcher uses, so how a batch shell finds conda stays defined
# in exactly one file. It ends in `exec "$@"`, so calling it as a command runs the stage.
RUN=(bash scripts/slurm_job.sh)

echo "=================================================================="
echo "Collision seeding: source=$SOURCE"
echo "  defenses: $DEFENSES"
echo "  features: $FEATURES"
echo "Job ${SLURM_JOB_ID:-local} on $(hostname), started $(date)"
echo "=================================================================="

# 0a. Correctness checks. Seconds, needs no data, and every one of them is a property the defense's
#     validity rests on (determinism, shard invariance, protected spans, profile cardinality). Run
#     first so a broken build fails here rather than after hours of featurization.
"${RUN[@]}" python -m prompt_anonymity.defenses.collision_seeding --selftest

# 0b. Marker audit -- see the header. Informational: it does not change what runs.
if [ -z "${SKIP_AUDIT:-}" ]; then
    "${RUN[@]}" python -m prompt_anonymity.defenses.collision_seeding \
        --audit --source "$SOURCE" --dist-dir "$DIST_DIR" || \
        echo "[warn] audit failed; continuing (it is informational)"
fi

# 0c. The author -> profile assignment. Nothing in the defense needs this (assignment is a pure
#     function of the author id, recomputed on demand), but the evaluation does: the within-group
#     confusion metric joins predictions against it to ask what share of MISATTRIBUTED documents
#     were assigned to an author sharing the true author's profile. Chance is 1/K; above chance is
#     the direct evidence that the seeded markers, not general confusion, are catching the attacker.
"${RUN[@]}" python -m prompt_anonymity.defenses.collision_seeding \
    --manifest --source "$SOURCE" --dist-dir "$DIST_DIR" --out-dir "$DIST_DIR"

for DEFENSE in $DEFENSES; do
    echo ""
    echo "------------------------------------------------------------------"
    echo "[$DEFENSE] defending $SOURCE  ($(date))"
    echo "------------------------------------------------------------------"

    # 1. Defend. Cheap, and it prints realized coverage per marker when it finishes -- check the
    #    `rate` column sits inside the variant's configured band and that `hit` is not 0% before
    #    trusting anything downstream.
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
echo "Done $(date). Results in experiments/results/${SOURCE}_collision_seeding*/"
echo ""
echo "What to compare, and what each comparison answers:"
echo "  vs ${SOURCE}_base_*            -- does seeding shared quirks reduce attribution at all?"
echo "  collision_seeding vs _full     -- does 40-70% beat 100%? (the inconsistency premise:"
echo "                                    perfect consistency is a perfectly reliable feature)"
echo "  collision_seeding vs _indep    -- does the profile codebook do the work? _indep is"
echo "                                    predicted to score WORSE than base, because independent"
echo "                                    marker draws give each author a near-unique fingerprint"
echo "  collision_seeding vs _k4       -- how does the collision-group size trade off?"
echo "  char_ngram_tfidf vs gemini_embedding_2 -- the scope claim: this defends the surface-form"
echo "                                    channel, not the semantic one"
echo ""
echo "Draw figures with:  python experiments/plot_results.py"
